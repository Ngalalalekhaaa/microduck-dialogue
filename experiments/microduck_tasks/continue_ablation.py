"""Finish P5 and the two finalists only after the first matrix has completed."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .ablation_report import rank_key, report
from .run_ablation import save


def launch(root, stage, variants, size):
    command = [
        sys.executable,
        "-m",
        "experiments.microduck_tasks.run_ablation",
        "--suite",
        str(root / "suite"),
        "--output",
        str(root / stage),
        "--size",
        str(size),
        "--variants",
        *variants,
    ]
    save(
        root / "supervisor_status.json",
        dict(stage=stage, state="running", variants=variants, size=size, command=command),
    )
    with (root / f"{stage}.log").open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), check=True)


def finish(root, initial_pid):
    root = root.resolve()
    save(root / "supervisor_status.json", dict(stage="screen", state="waiting_for_initial_process", pid=initial_pid))
    while not (root / "screen/complete.json").exists():
        try:
            os.kill(initial_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError("Initial matrix stopped before completing; inspect screen.log.") from exc
        time.sleep(10)
    # The completed process must release the GPU before the next one loads weights.
    while True:
        try:
            os.kill(initial_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(2)
    if not (root / "constrained/complete.json").exists():
        launch(root, "constrained", ["P5"], 96)
    link = root / "screen/P5"
    if not link.exists():
        link.symlink_to(root / "constrained/P5", target_is_directory=True)
    comparison = report(root / "screen")
    if not comparison["complete_screen"]:
        raise ValueError("All seven frozen screen runs must complete before selecting finalists.")
    finalists = comparison["finalists"]
    validation = root / "validation"
    validation.mkdir(exist_ok=True)
    reused = {}
    for variant in finalists:
        destination = validation / variant
        destination.mkdir(exist_ok=True)
        for source in sorted((root / "screen" / variant).glob("*.json")):
            if source.stem in ("progress", "report"):
                continue
            target = destination / source.name
            if not target.exists():
                shutil.copyfile(source, target)
            if target.read_bytes() != source.read_bytes():
                raise ValueError("A reused screen prediction differs from its original.")
            reused[str(target)] = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    save(validation / "reused_screen_predictions.json", reused)
    if not (validation / "complete.json").exists():
        launch(root, "validation", finalists, 240)
    comparison = report(validation)
    ranked = sorted(comparison["variants"].items(), key=rank_key, reverse=True)
    name, selected = ranked[0]
    passed = selected["summary"]["prototype_gate_pass"]
    decision = dict(
        stage="P",
        state="complete",
        selected=name,
        prototype_gate_pass=passed,
        next_stage="V/E simulation preparation" if passed else "H0 corrected 1200-decision dataset",
        next_stage_started=False,
        blind_test_used=False,
        training_started=False,
        requirement=(
            "Freeze a separate blind480 and audit train/dev family and image disjointness before formal training."
        ),
        summary=selected["summary"],
    )
    save(root / "stage_decision.json", decision)
    save(root / "supervisor_status.json", decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--initial-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        finish(args.root, args.initial_pid)
    except Exception as exc:
        save(args.root / "supervisor_status.json", dict(state="failed", error=type(exc).__name__, reason=str(exc)))
        raise


if __name__ == "__main__":
    main()
