"""Monitor an existing extraction, then run CPU training and selector evaluation."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _process_identity(pid):
    """Distinguish a live worker from a zombie or a later process reusing its PID."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return None
        if fields[0] in ("Z", "X"):
            return None
        return f"{pid}:{fields[19]}"
    return str(pid)


def _ready(features):
    path = features / "manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("The feature manifest must be a JSON object.")
    return manifest.get("complete") is True


def _progress(features):
    path = features / "progress.json"
    return json.loads(path.read_text()) if path.exists() else None


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("The selector pipeline exceeded its time limit; inspect the last stage log.")
    return remaining


def _wait_features(features, pid, deadline, poll_seconds, update, log_path):
    log_path.write_text("")
    if _ready(features):
        return
    if pid is None:
        raise ValueError("Features are incomplete; provide the PID of the existing extraction with --feature-pid.")
    identity = _process_identity(pid)
    with log_path.open("a") as log:
        while not _ready(features):
            current = _process_identity(pid)
            if identity is None or current != identity:
                if _ready(features):
                    return
                raise RuntimeError(f"Feature process {pid} exited, became a zombie, or was replaced before completion.")
            record = {"feature_pid": pid, "feature_process_identity": identity, "feature_progress": _progress(features)}
            update("waiting_features", **record)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            time.sleep(min(poll_seconds, _remaining(deadline)))


def _run_stage(name, command, log_path, deadline, update, *, cpu_only=False):
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    env.setdefault("MUJOCO_GL", "egl")
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    update(name, command=command, log=str(log_path))
    with log_path.open("x") as log:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=_remaining(deadline),
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{name} exited with code {result.returncode}; inspect {log_path}.")


def _require_report(path):
    if not path.is_file():
        raise RuntimeError(f"The completed subprocess did not write {path}; inspect its stage log.")
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"Expected a JSON object report at {path}.")
    return report


def _summary(training, evaluation):
    test = training.get("evaluation", {}).get("test", {})
    decision = test.get("decision_user", {})
    return {
        "trained_free_text_reply": training.get("trained_free_text_reply", False),
        "deployment_ready": False,
        "training_seconds": training.get("training_seconds"),
        "trainable_parameters": training.get("trainable_parameters"),
        "selected": training.get("selected"),
        "test_user_decision": {key: decision.get(key) for key in ("accuracy", "macro_f1_present")},
        "test_action_plan_exact": test.get("action_plan_exact"),
        "test_action_by_length": test.get("action_by_length"),
        "test_action_by_plan_novelty": test.get("action_by_plan_novelty"),
        "test_false_start_user": test.get("false_start_user"),
        "test_missed_cancel_user": test.get("missed_cancel_user"),
        "test_false_start_all": test.get("false_start"),
        "test_missed_cancel_all": test.get("missed_cancel"),
        "baseline": training.get("baselines", training.get("baseline")),
        "simulation_evaluation": evaluation.get("summary"),
    }


def run_pipeline(
    features, dataset, output, run_dir, model, *, feature_pid=None, poll_seconds=10, timeout_seconds=14400
):
    """Never start feature extraction or overwrite an existing run or checkpoint."""
    if feature_pid is not None and (type(feature_pid) is not int or feature_pid < 1):
        raise ValueError("--feature-pid must identify a positive existing process.")
    if not 0 < poll_seconds <= 15 or not 0 < timeout_seconds <= 14400:
        raise ValueError("Use a poll interval in (0,15] seconds and a total timeout in (0,14400] seconds.")
    features, dataset, output, run_dir, model = (
        Path(path).resolve() for path in (features, dataset, output, run_dir, model)
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    started, stage = time.monotonic(), "starting"
    deadline = started + timeout_seconds
    status = {
        "started_at": datetime.now(UTC).isoformat(),
        "pipeline_pid": os.getpid(),
        "features": str(features),
        "dataset": str(dataset),
        "output": str(output),
        "run_dir": str(run_dir),
        "model": str(model),
        "feature_pid": feature_pid,
        "timeout_seconds": timeout_seconds,
        "complete": False,
        "deployment_ready": False,
    }

    def update(name, **fields):
        nonlocal stage
        stage = name
        status.update(stage=name, updated_at=datetime.now(UTC).isoformat(), elapsed_s=time.monotonic() - started)
        status.update(fields)
        _write_json(run_dir / "status.json", status)

    try:
        update("starting")
        if output.exists():
            raise FileExistsError("Training output already exists; choose new --output and --run-dir paths.")
        if output == run_dir or run_dir.is_relative_to(output):
            raise ValueError("Keep --run-dir outside the new training --output directory.")
        if not (dataset / "manifest.json").is_file() or not model.is_dir():
            raise FileNotFoundError("Use an exported --dataset and the existing pinned --model directory.")
        update("waiting_features")
        _wait_features(features, feature_pid, deadline, poll_seconds, update, run_dir / "feature_wait.log")
        update("features_complete", feature_progress=_progress(features))
        command = [
            sys.executable,
            "-m",
            "experiments.microduck_tasks.train_toy_selector",
            "--features",
            str(features),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ]
        _run_stage("training", command, run_dir / "train.log", deadline, update, cpu_only=True)
        if not (output / "selector.pt").is_file():
            raise RuntimeError("Training exited successfully without selector.pt; inspect train.log.")
        training = _require_report(output / "results.json")
        evaluation = run_dir / "evaluation"
        command = [
            sys.executable,
            "-m",
            "experiments.microduck_tasks.evaluate_toy_selector",
            "--checkpoint",
            str(output / "selector.pt"),
            "--model",
            str(model),
            "--output",
            str(evaluation),
        ]
        _run_stage("evaluating", command, run_dir / "evaluate.log", deadline, update)
        evaluation_report = _require_report(evaluation / "report.json")
        summary = _summary(training, evaluation_report)
        summary.update(training_report=str(output / "results.json"), evaluation_report=str(evaluation / "report.json"))
        _write_json(run_dir / "summary.json", summary)
        update(
            "complete",
            complete=True,
            checkpoint=str(output / "selector.pt"),
            training_report=str(output / "results.json"),
            evaluation_report=str(evaluation / "report.json"),
            summary_report=str(run_dir / "summary.json"),
            summary=summary,
        )
        return status
    except Exception as error:
        failed_stage = stage
        update("failed", complete=False, failed_stage=failed_stage, error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    args = parser.parse_args()
    try:
        result = run_pipeline(
            args.features,
            args.dataset,
            args.output,
            args.run_dir,
            args.model,
            feature_pid=args.feature_pid,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        print(json.dumps({"complete": False, "error": f"{type(error).__name__}: {error}"}), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
