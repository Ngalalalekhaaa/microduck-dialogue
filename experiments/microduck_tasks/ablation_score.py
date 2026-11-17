"""Score first raw generations, including invalid plans and unauthorized starts."""

import copy

import numpy as np

from .ablation_contract import validate_context, wrap_plan
from .evaluate_toy_vlm_language import _goals_canonical
from .toy_protocol import validate_output
from .toy_vlm_json import parse_json_object


def score(raw, case, *, plan_only):
    expected = case["expected"]
    result = dict(
        json_parsed=False,
        schema_valid=False,
        context_valid=False,
        exact=False,
        semantic=False,
        decision_correct=False,
        goal_count_correct=False,
        goals_correct=False,
        attempted_false_start=False,
        accepted_false_start=False,
        parsed=None,
        envelope=None,
        error=None,
    )
    try:
        parsed = parse_json_object(raw)
        result.update(json_parsed=True, parsed=copy.deepcopy(parsed))
        nonaction = expected["decision"] not in ("replace", "append", "resume")
        attempted = parsed.get("decision") in ("replace", "append", "resume") or bool(parsed.get("goals"))
        result["attempted_false_start"] = nonaction and attempted
        result["decision_correct"] = parsed.get("decision") == expected["decision"]
        result["goals_correct"] = parsed.get("goals") == expected["goals"]
        result["goal_count_correct"] = isinstance(parsed.get("goals"), list) and len(parsed["goals"]) == len(
            expected["goals"]
        )
        result["exact"] = result["decision_correct"] and result["goals_correct"]
        envelope = wrap_plan(parsed) if plan_only else parsed
        result["envelope"] = envelope
        validate_output(envelope, max_goals=case["request"]["capabilities"]["max_goals"])
        result["schema_valid"] = True
        validate_context(envelope, case["request"])
        result["context_valid"] = True
        result["accepted_false_start"] = result["attempted_false_start"]
        same_goals = _goals_canonical(parsed["goals"], case["request"]) == _goals_canonical(
            expected["goals"], case["request"]
        )
        result["semantic"] = same_goals and parsed["decision"] in case["acceptable_decisions"]
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        result["error"] = dict(type=type(exc).__name__, reason=str(exc))
    result["accepted_exact"] = result["context_valid"] and result["exact"]
    return result


def fraction(rows, name):
    n = sum(bool(row["score"][name]) for row in rows)
    return dict(correct=n, total=len(rows), rate=n / len(rows) if rows else None)


def summarize(rows, *, expected_count):
    actions = [r for r in rows if r["expected"]["decision"] in ("replace", "append")]
    boundary = [r for r in rows if r["expected"]["decision"] not in ("replace", "append", "resume")]
    long = [r for r in actions if 3 <= len(r["expected"]["goals"]) <= 6]
    metrics = {
        name: fraction(rows, name)
        for name in ("json_parsed", "schema_valid", "context_valid", "accepted_exact", "semantic")
    }
    metrics.update(
        action_plan_exact=fraction(actions, "accepted_exact"),
        three_six_exact=fraction(long, "accepted_exact"),
        nonaction_attempted_start=fraction(boundary, "attempted_false_start"),
        nonaction_accepted_start=fraction(boundary, "accepted_false_start"),
    )
    groups = {}
    for field in ("group", "domain", "context"):
        groups[field] = {
            value: {
                k: fraction([r for r in rows if r[field] == value], k)
                for k in ("accepted_exact", "context_valid", "decision_correct", "goal_count_correct", "goals_correct")
            }
            for value in sorted({r[field] for r in rows})
        }
    seconds = [r["timing"]["seconds"] for r in rows if r.get("timing")]
    peak = [r["timing"].get("peak_allocated_mib", 0) for r in rows if r.get("timing")]
    costs = dict(
        p50_seconds=float(np.percentile(seconds, 50)) if seconds else None,
        p95_seconds=float(np.percentile(seconds, 95)) if seconds else None,
        peak_allocated_mib=max(peak, default=0),
        input_tokens=sum(r.get("timing", {}).get("input_tokens", 0) for r in rows),
        output_tokens=sum(r.get("timing", {}).get("output_tokens", 0) for r in rows),
    )
    eligible = expected_count >= 240 and len(rows) == expected_count and bool(actions) and bool(long) and bool(boundary)
    passes = eligible and metrics["context_valid"]["rate"] >= 0.99 and metrics["action_plan_exact"]["rate"] >= 0.95
    passes = (
        passes and metrics["three_six_exact"]["rate"] >= 0.90 and metrics["nonaction_attempted_start"]["correct"] == 0
    )
    return dict(
        completed=len(rows),
        expected=expected_count,
        metrics=metrics,
        groups=groups,
        costs=costs,
        prototype_gate_eligible=eligible,
        prototype_gate_pass=bool(passes),
        physical_executions=0,
        errors=sum(bool(r.get("generation_error")) for r in rows),
        visual_truth_scored=False,
        reply_quality_scored=False,
    )
