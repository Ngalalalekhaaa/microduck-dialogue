"""Language/context-only diagnostic using recorded images; never executes a robot."""

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

from .toy_inputs import validate_input
from .toy_protocol import validate_output


def _case_module(case_suite):
    if case_suite == "language":
        from . import toy_vlm_language_cases as module
    elif case_suite == "rules":
        from . import toy_vlm_rules_cases as module
    else:
        raise ValueError("Use case_suite='language' or 'rules'.")
    return module


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def source_snapshot(episode, *, event_id=None):
    """Load only public request observations and the two recorded image files."""
    episode = Path(episode).expanduser().resolve()
    events_path = episode / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    selected = [
        row
        for row in events
        if row.get("event") == "dialogue"
        and row["request"]["event"]["kind"] == "user"
        and (event_id is None or row["event_id"] == event_id)
    ]
    if not selected:
        raise ValueError("Choose an episode/event containing a recorded user dialogue with both camera images.")
    row = selected[0]
    validate_input(row["request"])
    observation = row["request"]["observation"]
    env = observation["environment"]
    if observation["posture"] != "standing" or observation["age_s"] > 0.5 or not env["motion_allowed"]:
        raise ValueError("Choose a fresh standing snapshot whose local motion gate is open.")
    if env["surface"] != "simulation_floor":
        raise ValueError("Choose a ground simulation snapshot; desktop constraints are a separate counterfactual case.")
    images, hashes = {}, {}
    if set(row["media"]) != {"front", "near"}:
        raise ValueError("The source dialogue must contain exactly front and near images.")
    for camera in ("front", "near"):
        path = (episode / row["media"][camera]).resolve()
        if not path.is_relative_to(episode) or not path.is_file():
            raise ValueError("Recorded image path is absent or escapes its source episode.")
        checksum = _digest(path)
        expected = row.get("media_sha256", {}).get(camera)
        if expected is not None and checksum != expected:
            raise ValueError("A recorded source image no longer matches its original checksum.")
        images[camera], hashes[camera] = str(path), checksum
    provenance = dict(
        episode=str(episode),
        event_id=row["event_id"],
        events_sha256=_digest(events_path),
        images=images,
        image_sha256=hashes,
        observation_id=row.get("observation_id"),
        original_response_used=False,
    )
    return copy.deepcopy(observation), images, provenance


def _goals_canonical(goals, request):
    """Ignore attribute ordering and a uniquely matching observed entity binding."""
    result = copy.deepcopy(goals)
    for item in result:
        target = item.get("target")
        if target is None:
            continue
        target["attributes"] = sorted(target["attributes"], key=lambda attr: (attr["name"], attr["value"]))
        if target["entity_id"] is None:
            continue
        matches = [
            entity
            for entity in request["observation"]["entities"]
            if entity["category"] == target["category"]
            and all(attr in entity["attributes"] for attr in target["attributes"])
        ]
        if len(matches) == 1 and matches[0]["id"] == target["entity_id"]:
            target["entity_id"] = None
    return result


def score_prediction(prediction, case):
    if not isinstance(prediction, dict) or not all(key in prediction for key in ("decision", "goals")):
        return dict(exact=False, behavioral=False)
    expected = case["expected"]
    exact = all(prediction[key] == expected[key] for key in ("decision", "goals"))
    try:
        same_goals = any(
            _goals_canonical(prediction["goals"], case["request"]) == _goals_canonical(goals, case["request"])
            for goals in [expected["goals"], *case.get("acceptable_goals", [])]
        )
        behavioral = same_goals and prediction["decision"] in case["acceptable_decisions"]
    except (KeyError, TypeError, AttributeError):
        behavioral = False
    return dict(exact=exact, behavioral=behavioral)


def summarize(reports):
    model = [row for row in reports if row["model_called"]]
    seconds = [row["seconds"] for row in model]
    return dict(
        cases=len(reports),
        model_calls=len(model),
        model_contract_valid=sum(row["model_contract_valid"] for row in model),
        raw_plan_exact=sum(row["raw_score"]["exact"] for row in model),
        model_accepted_behaviorally_correct=sum(
            row["model_contract_valid"] and row["raw_score"]["behavioral"] for row in model
        ),
        accepted_response_behaviorally_correct=sum(row["accepted_score"]["behavioral"] for row in reports),
        model_response_seconds_mean=statistics.mean(seconds) if seconds else None,
        model_response_seconds_max=max(seconds) if seconds else None,
        errors=sum(row["error"] is not None for row in reports),
        reply_quality_automatically_scored=False,
        physical_executions=0,
    )


def evaluate(backend, source_episode, output, *, event_id=None, selected=(), case_suite="language"):
    observation, images, provenance = source_snapshot(source_episode, event_id=event_id)
    case_module = _case_module(case_suite)
    available = case_module.cases(observation)
    selected = set(selected)
    unknown = selected - {case["id"] for case in available}
    if unknown:
        raise ValueError(f"Unknown diagnostic cases: {sorted(unknown)}")
    rows = [case for case in available if not selected or case["id"] in selected]
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "cases.json", rows)
    _save(output / "source.json", provenance)
    started, reports = time.monotonic(), []
    for case in rows:
        current = time.monotonic()
        response, trace, error = None, {}, None
        try:
            response = backend.decide_with_images(copy.deepcopy(case["request"]), images)
            trace = copy.deepcopy(getattr(backend, "last_trace", None) or {})
            validate_output(response, case["request"])
        except Exception as exc:
            error = dict(type=type(exc).__name__, reason=str(exc))
        called = bool(trace.get("model_called", False))
        raw_response = trace.get("raw_response")
        raw = trace.get("raw_prediction")
        if raw is None and raw_response is not None:
            raw = raw_response
        contract_valid = False
        if called and raw_response is not None and "rejected" not in trace:
            try:
                validate_output(raw_response, case["request"])
                contract_valid = True
            except (ValueError, TypeError, KeyError):
                pass
        report = dict(
            **case,
            accepted_response=response,
            model_trace=trace,
            model_called=called,
            model_contract_valid=contract_valid,
            raw_score=score_prediction(raw, case),
            accepted_score=score_prediction(response, case),
            seconds=time.monotonic() - current,
            error=error,
        )
        reports.append(report)
        _save(output / f"{case['id']}.json", report)
        _save(output / "progress.json", summarize(reports))
        print(json.dumps(dict(case=case["id"], summary=summarize(reports)), ensure_ascii=False), flush=True)
    result = dict(
        kind="language_context_diagnostic_with_recorded_images",
        cases_suite=case_suite,
        cases_version=case_module.VERSION_CASES,
        cases_implementation_sha256=_digest(case_module.__file__),
        evaluator_sha256=_digest(__file__),
        backend=backend.name,
        backend_metadata=getattr(backend, "metadata", {}),
        training_ready=False,
        physical_validation=False,
        expectations_passed_to_model=False,
        source=provenance,
        limitations=[
            "Hand-authored diagnostic, not a representative estimate of deployment accuracy.",
            "One recorded image pair is shared across hypothetical utterances and execution states.",
            "Counterfactual state cases intentionally alter structured state without rerendering images.",
            "No robot or physics step is executed; no physical completion score is claimed.",
            "Behavioral matching allows a uniquely matching observed entity binding and attribute reordering.",
            "Predeclared acceptable_decisions and acceptable_goals are alternative answers, never model input.",
            "Accepted fallback refusals are separated from correct model outputs.",
            "Generated replies need manual review; plan scores do not certify visual or conversational truthfulness.",
        ],
        summary=summarize(reports),
        cases=reports,
        elapsed_seconds=time.monotonic() - started,
    )
    _save(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode", type=Path, required=True)
    parser.add_argument("--source-event")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--vlm-context", choices=("full", "compact", "rules"), default="compact")
    parser.add_argument("--suite", choices=("language", "rules"), default="language")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    source_snapshot(args.source_episode, event_id=args.source_event)
    unknown = set(args.case) - {case["id"] for case in _case_module(args.suite).cases()}
    if unknown:
        parser.error(f"Unknown diagnostic cases: {sorted(unknown)}")
    if args.output.exists():
        parser.error("Use a new output directory to preserve earlier diagnostic results.")
    from .toy_vlm_backend import ToyVLMBackend

    backend = ToyVLMBackend(args.model, prompt_variant=args.vlm_context)
    result = evaluate(
        backend, args.source_episode, args.output, event_id=args.source_event, selected=args.case, case_suite=args.suite
    )
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return int(bool(result["summary"]["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
