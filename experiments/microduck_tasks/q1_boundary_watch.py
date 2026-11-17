"""Run the frozen CPU boundary audit when each Q1 validation report becomes available."""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from .q1_boundary_audit import STEPS, run
from .run_ablation import save


def ready_steps(root):
    return tuple(step for step in STEPS if (root / f"q1/train1200/validation_{step}/report.json").exists())


def watch(root, output, poll_seconds=30):
    if poll_seconds <= 0 or poll_seconds > 60:
        raise ValueError("Polling interval must be between 0 and 60 seconds.")
    state = dict(
        started_at=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        complete=False,
        blind_access=False,
        gpu_used=False,
        training_modified=False,
        original_artifacts_modified=False,
        reload_checked=False,
        reload_note=(
            "This audit only rescored saved generations. If the audited winner changes, that checkpoint needs "
            "a later independent reload verification; do not reuse the original winner's reload result. "
            "Even an unchanged winner still requires the original Q1 reload check."
        ),
    )

    def notify(stage, **fields):
        state.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **fields)
        save(output / "watcher_status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    try:
        previous = None
        while True:
            ready = ready_steps(root)
            if ready != previous:
                result = run(root, output)
                previous = ready
                changed = result["complete"] and result["selected_step"] != result["original_selected_step"]
                notify(
                    "complete_reload_required" if changed else "complete" if result["complete"] else "waiting",
                    complete=result["complete"],
                    completed_steps=result["completed_steps"],
                    provisional_selected_step=result["provisional_selected_step"],
                    selected_step=result["selected_step"],
                    original_selected_step=result["original_selected_step"],
                    changed_winner_requires_independent_reload=changed,
                )
                if result["complete"]:
                    return state
            training_path = root / "q1/status.json"
            if training_path.exists():
                training = json.loads(training_path.read_text())
                if training.get("stage") == "Q1_failed":
                    notify("stopped_training_failed", training_error=training.get("error"))
                    return state
            time.sleep(poll_seconds)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        notify("audit_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    watch(args.root, args.output or args.root / "q1/boundary_audit", args.poll_seconds)
