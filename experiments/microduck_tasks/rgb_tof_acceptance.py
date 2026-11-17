"""Predeclared four-case sensor-only acceptance through isolated child runs.

Every case contributes to the denominator, including crashes, timeouts and
missing result files. Cases run once in the declared order without retries.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

CASES = (
    dict(name="center", x=0.4, y=0.02, seed=17),
    dict(name="left", x=0.45, y=0.08, seed=18),
    dict(name="right", x=0.5, y=-0.08, seed=19),
    dict(name="far", x=0.6, y=0.0, seed=20),
)
EXTENSION_CASES = (
    dict(name="near_left", x=0.35, y=0.12, seed=21),
    dict(name="near_right", x=0.35, y=-0.12, seed=22),
    dict(name="far_left", x=0.55, y=0.12, seed=23),
    dict(name="far_right", x=0.55, y=-0.12, seed=24),
    dict(name="near_center", x=0.3, y=0, seed=25),
    dict(name="far_center", x=0.7, y=0, seed=26),
    dict(name="small_right", x=0.4, y=-0.03, seed=27),
    dict(name="small_left", x=0.6, y=0.04, seed=28),
)
TIMEOUT_S = 900


def _utc():
    return datetime.now(UTC).isoformat()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant {value} is not valid evidence")


def _sources():
    return {
        path.name: path.read_bytes()
        for path in sorted(Path(__file__).parent.glob("*.py"))
        if not path.name.startswith("test_")
    }


def _hashes(sources):
    return {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}


def _changes(before):
    after = _hashes(_sources())
    return [
        dict(file=name, before_sha256=before.get(name), after_sha256=after.get(name))
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    ]


def _environment():
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        MUJOCO_GL="egl",
        LIBGL_ALWAYS_SOFTWARE="1",
        LP_NUM_THREADS="2",
        __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    return environment


def _run_case(output, case, args):
    case_output = output / case["name"]
    log_path = output / f"{case['name']}.log"
    command = [
        sys.executable,
        "-m",
        "experiments.microduck_tasks.rgb_tof_run",
        "--output",
        str(case_output),
        "--mode",
        args.mode,
        "--kick-strategy",
        args.kick_strategy,
        "--x",
        str(case["x"]),
        "--y",
        str(case["y"]),
        "--seed",
        str(case["seed"]),
        "--noise",
        str(args.noise),
        "--dropout",
        str(args.dropout),
    ]
    outcome = dict(
        **case,
        output=str(case_output),
        log=str(log_path),
        command=command,
        status="failed",
        success=False,
        controller_success=False,
        started_at_utc=_utc(),
        returncode=None,
    )
    started = time.monotonic()
    try:
        with log_path.open("x") as log:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_S,
                check=False,
            )
        outcome["returncode"] = completed.returncode
    except subprocess.TimeoutExpired:
        outcome["error"] = f"Process exceeded the predeclared {TIMEOUT_S} s timeout"
    except OSError as error:
        outcome["error"] = f"Process launch or log error: {error}"
    result_path = case_output / "result.json"
    try:
        result = json.loads(result_path.read_text(), parse_constant=_reject_constant)
        if not isinstance(result, dict):
            raise ValueError("result must be an object")
        for field in ("controller", "evaluation"):
            if not isinstance(result.get(field), dict) or type(result[field].get("success")) is not bool:
                raise ValueError(f"{field}.success must be a JSON boolean")
        outcome["reported_result"] = result
        clean_exit = outcome["returncode"] == 0 and "error" not in outcome
        outcome["controller_success"] = clean_exit and result["controller"]["success"]
        outcome["success"] = outcome["controller_success"] and result["evaluation"]["success"]
        outcome["status"] = "passed" if outcome["success"] else "failed"
    except (OSError, ValueError, TypeError) as error:
        outcome["result_error"] = f"No usable result: {error}"
    outcome["wall_seconds"] = time.monotonic() - started
    outcome["finished_at_utc"] = _utc()
    return outcome


def run(args):
    if not hasattr(args, "kick_strategy"):
        args.kick_strategy = "foot_relative"
    cases = EXTENSION_CASES if getattr(args, "suite", "baseline") == "extension" else CASES
    if not math.isfinite(args.noise) or args.noise < 0:
        raise ValueError("Noise must be finite and nonnegative")
    if not math.isfinite(args.dropout) or not 0 <= args.dropout <= 1:
        raise ValueError("Dropout must be finite and between zero and one")
    if args.mode not in ("navigate", "kick"):
        raise ValueError("Mode must be navigate or kick")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    sources = _sources()
    hashes = _hashes(sources)
    for name, content in sources.items():
        (snapshot / name).write_bytes(content)
    started, started_utc = time.monotonic(), _utc()
    _write(
        output / "predeclared_protocol.json",
        dict(
            cases=cases,
            total=len(cases),
            mode=args.mode,
            kick_strategy=args.kick_strategy,
            noise_m=args.noise,
            dropout=args.dropout,
            maximum_case_wall_seconds=TIMEOUT_S,
            attempts_per_case=1,
            execution_order="listed_order_sequential",
            denominator="all_declared_cases_including_process_and_result_failures",
            success_rule="exit_zero_and_boolean_controller_success_and_boolean_independent_evaluation_success",
            controller_success_rule="exit_zero_and_boolean_controller_success",
            source_snapshot_scope="all_non_test_python_modules_in_experiment_directory",
            source_sha256=hashes,
            executable=sys.executable,
            started_at_utc=started_utc,
            hardware=False,
            rendering="Mesa_software_EGL",
            gpu_disabled=True,
        ),
    )
    summary = dict(
        total=len(cases),
        completed=0,
        successes=0,
        controller_successes=0,
        case_results=[],
        status="running",
        mode=args.mode,
        kick_strategy=args.kick_strategy,
        noise_m=args.noise,
        dropout=args.dropout,
        started_at_utc=started_utc,
        wall_seconds=0.0,
        source_changed=False,
        source_changes_before_cases=_changes(hashes),
        source_changes_after_cases=[],
    )
    _write(output / "summary.json", summary)
    for case in cases:
        outcome = _run_case(output, case, args)
        outcome["source_changes_after_case"] = _changes(hashes)
        summary["case_results"].append(outcome)
        summary["completed"] += 1
        summary["successes"] += int(outcome["success"])
        summary["controller_successes"] += int(outcome["controller_success"])
        summary["wall_seconds"] = time.monotonic() - started
        summary["source_changes_after_cases"] = outcome["source_changes_after_case"]
        summary["source_changed"] = bool(
            summary["source_changes_before_cases"]
            or any(item["source_changes_after_case"] for item in summary["case_results"])
        )
        _write(output / "summary.json", summary)
        print(
            json.dumps(dict(case=case["name"], success=outcome["success"], completed=summary["completed"])), flush=True
        )
    summary["status"] = "completed"
    summary["finished_at_utc"] = _utc()
    _write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=("navigate", "kick"))
    parser.add_argument("--suite", choices=("baseline", "extension"), default="baseline")
    parser.add_argument("--kick-strategy", choices=("narrow_body", "foot_relative"), default="foot_relative")
    parser.add_argument("--noise", type=float, default=0.002)
    parser.add_argument("--dropout", type=float, default=0.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
