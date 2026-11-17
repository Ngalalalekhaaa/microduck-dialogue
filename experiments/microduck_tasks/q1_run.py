"""Queue Q1 behind the current Q0 process, enforcing completion and fit guards."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .q1_schedule import require_q0
from .run_ablation import save
from .toy_dataset import digest

SOURCES = [
    "q1_run.py",
    "q1_training.py",
    "q1_verify.py",
    "q1_schedule.py",
    "q1_data.py",
    "q1_evaluate.py",
    "q0_model.py",
    "q0_data.py",
    "q0_evaluate.py",
    "q0_recover.py",
    "head_features.py",
    "h0_features.py",
    "ablation_contract.py",
    "ablation_prompt.py",
    "ablation_score.py",
    "toy_protocol.py",
    "toy_vlm_json.py",
]


def process_identity(pid):
    directory = Path(f"/proc/{pid}")
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        command = (directory / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    if not any(f"experiments.microduck_tasks.{name}".encode() in command for name in ("q0_training", "q0_recover")):
        raise ValueError("Q0 PID belongs to an unexpected process; do not overlap GPU jobs.")
    return fields[19]


def run(root, q0_pid, model_dir, attempt="initial"):
    work = root / "q1"
    if not attempt.replace("_", "").isalnum():
        raise ValueError("Use an alphanumeric queue attempt name.")
    if attempt != "initial":
        previous = work / f"prior_to_{attempt}"
        previous.mkdir(exist_ok=False)
        for name in ("queue.json", "status.json", "implementation_manifest.json", "planned_config.json"):
            (previous / name).write_bytes((work / name).read_bytes())
    # This exclusive directory prevents a second supervisor from launching a duplicate run.
    snapshot = work / ("queued_implementation" if attempt == "initial" else f"queued_implementation_{attempt}")
    snapshot.mkdir(exist_ok=False)
    for name in SOURCES:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    hashes = {n: digest(snapshot / n) for n in SOURCES}
    save(work / "implementation_manifest.json", hashes)
    queued_config_hash = digest(work / "planned_config.json")
    data_hash = digest(work / "data/manifest.json")
    identity = process_identity(q0_pid)
    status = dict(
        started_at=datetime.now(UTC).isoformat(),
        supervisor_pid=os.getpid(),
        q0_pid=q0_pid,
        q0_process_start_ticks=identity,
        complete=False,
        attempt=attempt,
        blind_access=False,
        planned_config_sha256=queued_config_hash,
        data_manifest_sha256=data_hash,
        authorization="User requested continuation after Q0 and a runtime estimate.",
    )

    def update(stage, *, current=False, **fields):
        status.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **fields)
        save(work / "queue.json", status)
        if current:
            path = root / "CURRENT_STATUS.json"
            old = json.loads(path.read_text())
            old.update(
                state=stage, Q1_status="q1/status.json", Q1_queue="q1/queue.json", updated_at=status["updated_at"]
            )
            if "next_stage" in fields:
                old["next_stage"] = fields["next_stage"]
            save(path, old)

    try:
        update("Q1_queued_waiting_for_Q0")
        save(work / "status.json", status)
        while identity is not None:
            observed = process_identity(q0_pid)
            if observed is None:
                break
            if observed != identity:
                raise ValueError("Q0 PID was reused; inspect process state before training.")
            time.sleep(5)
        q0 = json.loads((root / "q0/run100/results.json").read_text())
        require_q0(q0)
        q0_status = json.loads((root / "q0/status.json").read_text())
        q0_data = json.loads((root / "q0/data/manifest.json").read_text())
        if not q0_status["complete"] or q0["data_sha256"] != q0_data["data_sha256"]:
            raise ValueError("Q0 completion or frozen subset identity mismatch.")
        reload_check = json.loads((root / "q0/run100/reload_check.json").read_text())
        if (
            not reload_check["passed"]
            or len(reload_check["checks"]) != 3
            or not all(r["exactly_same_output"] for r in reload_check["checks"])
        ):
            raise ValueError("Q0 must reload and reproduce all three predeclared examples.")
        for name in ("q0_model.py", "q0_data.py", "q0_evaluate.py"):
            if digest(root / "q0/run100/implementation" / name) != hashes[name]:
                raise ValueError("Q0 model, supervision or generation changed before Q1.")
        if (
            digest(work / "planned_config.json") != queued_config_hash
            or digest(work / "data/manifest.json") != data_hash
        ):
            raise ValueError("Q1 data or configuration changed after queueing.")
        if any(digest(Path(__file__).with_name(n)) != h for n, h in hashes.items()):
            raise ValueError("Queued Q1 source changed; preserve the run and inspect the implementation.")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        for module, stage, log in (
            ("q1_training", "Q1_starting", "training.log"),
            ("q1_verify", "Q1_verifying_selected_adapter", "reload.log"),
        ):
            command = [
                sys.executable,
                "-m",
                f"experiments.microduck_tasks.{module}",
                "--root",
                str(root),
                "--model",
                str(model_dir),
            ]
            update(stage, current=True, command=command)
            with (work / log).open("x") as handle:
                subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        result = json.loads((work / "train1200/results.json").read_text())
        if not result["complete"] or not result["reload_pass"]:
            raise ValueError("Q1 final result or adapter reload verification incomplete.")
        q1_status = json.loads((work / "status.json").read_text())
        q1_status.update(stage="Q1_complete", complete=True, reload_pass=True, updated_at=datetime.now(UTC).isoformat())
        save(work / "status.json", q1_status)
        update(
            "Q1_complete",
            current=True,
            complete=True,
            selected_step=result["selected_step"],
            prototype_gate_pass=result["prototype_gate_pass"],
            result="q1/train1200/results.json",
            next_stage="Review Q1 development learning curve before Q2/visual stages; keep blind sealed.",
        )
    except Exception as error:
        update(
            "Q1_continuation_failed",
            current=True,
            error=f"{type(error).__name__}: {error}",
            next_stage="Diagnose recorded Q0/Q1 failure; do not lower gates or open blind data.",
        )
        failed = json.loads((work / "status.json").read_text())
        failed.update(
            stage="Q1_continuation_failed", complete=False, error=status["error"], updated_at=status["updated_at"]
        )
        save(work / "status.json", failed)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--q0-pid", type=int, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--attempt", default="initial")
    args = parser.parse_args()
    run(args.root.resolve(), args.q0_pid, args.model.resolve(), args.attempt)
