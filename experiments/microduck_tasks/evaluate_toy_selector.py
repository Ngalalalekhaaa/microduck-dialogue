"""Compare generated or learned plans and bounded execution on the same fixed cases."""

import argparse
import json
import statistics
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from .toy_selector_cases import cases
from .toy_selector_rollout import run_case, write_json


def summarize(reports):
    turns = [turn for report in reports for turn in report["turns"]]
    learned = [turn for turn in turns if turn["raw_selector_exact"] is not None]
    model = [turn for turn in turns if turn.get("raw_model_exact", turn["raw_selector_exact"]) is not None]
    physical = Counter()
    for report in reports:
        physical.update(report["goal_status"])
    completed = physical["succeeded"] + physical["failed"]
    finite = [report for report in reports if report["expected_finite_plan_verified"] is not None]
    latencies = [turn["model_trace"]["seconds"] for turn in model if "seconds" in turn["model_trace"]]
    controls = [report for report in reports if report.get("control_checks")]
    return dict(
        cases=len(reports),
        user_turns=len(turns),
        accepted_response_exact=sum(turn["semantic_exact"] for turn in turns) / len(turns) if turns else None,
        semantic_cases_correct=sum(report["semantic_all_correct"] for report in reports),
        learned_selector_turns=len(learned),
        learned_raw_plan_exact=sum(turn["raw_selector_exact"] for turn in learned) / len(learned) if learned else None,
        model_turns=len(model),
        model_raw_plan_exact=(
            sum(turn.get("raw_model_exact", turn["raw_selector_exact"]) for turn in model) / len(model)
            if model
            else None
        ),
        host_control_turns=sum(
            bool(turn["model_trace"] and turn["model_trace"].get("decision_source") in ("host_stop", "host_pause"))
            for turn in turns
        ),
        physical_goal_status=dict(physical),
        completed_physical_success_fraction=physical["succeeded"] / completed if completed else None,
        cancelled_attempts_excluded_from_completed_fraction=physical["cancelled"],
        expected_finite_plan_cases=len(finite),
        expected_finite_plans_verified=sum(report["expected_finite_plan_verified"] for report in finite),
        control_cases=len(controls),
        control_cases_verified=sum(all(report["control_checks"].values()) for report in controls),
        model_response_seconds_mean=statistics.mean(latencies) if latencies else None,
        model_response_seconds_median=statistics.median(latencies) if latencies else None,
        model_response_seconds_max=max(latencies) if latencies else None,
        errors=sum(report["error"] is not None for report in reports),
        program_errors=sum(
            report["error"] is not None and report["error"]["type"] != "TimeoutError" for report in reports
        ),
        runtime_failures=sum(bool(report["runtime_failure"]) for report in reports),
        timed_out=sum(report["timed_out"] for report in reports),
    )


def evaluate(backend, output, *, selected=(), base_seed=530000, wall_limit=120, sim_limit=100):
    selected = set(selected)
    available = cases()
    unknown = selected - {case["id"] for case in available}
    if unknown:
        raise ValueError(f"Unknown evaluation cases: {sorted(unknown)}")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, reports = time.monotonic(), []
    for index, case in enumerate(available):
        if selected and case["id"] not in selected:
            continue
        with (output / f"{case['id']}.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            report = run_case(
                case, backend, output / case["id"], seed=base_seed + index, wall_limit=wall_limit, sim_limit=sim_limit
            )
        reports.append(report)
        write_json(output / "progress.json", summarize(reports))
        print(json.dumps(dict(case=case["id"], summary=summarize(reports)), ensure_ascii=False), flush=True)
    metadata = getattr(backend, "metadata", {})
    model_evaluation = backend.name in ("frozen_vlm_toy_selector", "qwen35_toy_zero_shot")
    reply_source = metadata.get("reply_source", "template")
    result = dict(
        backend=backend.name,
        model_evaluation=model_evaluation,
        teacher_diagnostic_only=backend.name == "toy_rule_teacher",
        goal_source=metadata.get("goal_source", "rule_teacher" if not model_evaluation else "unknown"),
        reply_source=reply_source,
        low_level_execution="existing_ONNX_and_verified_rule_controllers",
        training_ready=False,
        deployment_ready=False,
        evaluation_expectations_never_model_input=True,
        limitations=[
            "Small simulation acceptance set; it does not estimate broad real-world accuracy.",
            "Several phrases overlap the collection curriculum; scene seeds are new.",
            "Completed physical success measures the predicted goals, even when the selected plan is wrong.",
            "Host stop/pause bypass model inference and are excluded from model_raw_plan_exact.",
            (
                "Replies are model generations; semantic plan metrics do not grade reply truthfulness or quality."
                if reply_source == "vlm_generation"
                else "Replies are templates; this run does not evaluate learned free-form conversation."
            ),
            "Watchdog cancels physics; a stalled model forward must return before its case can close.",
        ],
        summary=summarize(reports),
        cases=reports,
        elapsed_seconds=time.monotonic() - started,
        backend_metadata=metadata,
        time_bounds=dict(wall_seconds_per_case=wall_limit, simulation_seconds_per_case=sim_limit),
    )
    write_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("selector", "teacher", "vlm"), default="selector")
    parser.add_argument(
        "--vlm-context",
        choices=("full", "compact", "rules"),
        default="compact",
        help="Prompt variant for --backend vlm.",
    )
    parser.add_argument("--case", action="append", default=[], choices=[case["id"] for case in cases()])
    parser.add_argument("--base-seed", type=int, default=530000)
    parser.add_argument("--wall-limit", type=float, help="Seconds per case; default: vlm 300, other backends 120.")
    parser.add_argument("--sim-limit", type=float, default=100)
    args = parser.parse_args(argv)
    if args.backend == "selector" and args.checkpoint is None:
        parser.error("--backend selector requires --checkpoint.")
    if args.backend != "selector" and args.checkpoint is not None:
        parser.error("--checkpoint requires --backend selector.")
    if args.wall_limit is None:
        args.wall_limit = 300 if args.backend == "vlm" else 120
    if args.wall_limit <= 0 or args.sim_limit <= 0:
        parser.error("Evaluation time bounds must be positive.")
    from .bootstrap import configure

    configure()
    if args.backend == "selector":
        from .toy_selector_backend import ToySelectorBackend

        backend = ToySelectorBackend(args.checkpoint, args.model)
    elif args.backend == "vlm":
        from .toy_vlm_backend import ToyVLMBackend

        backend = ToyVLMBackend(args.model, prompt_variant=args.vlm_context)
    else:
        from .toy_teacher import ToyTeacher

        backend = ToyTeacher()
    result = evaluate(
        backend,
        args.output,
        selected=args.case,
        base_seed=args.base_seed,
        wall_limit=args.wall_limit,
        sim_limit=args.sim_limit,
    )
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return int(bool(result["summary"]["program_errors"] or result["summary"]["runtime_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
