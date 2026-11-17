"""Continue H2 after its feature worker exits, then train and record comparisons."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .run_ablation import save
from .toy_dataset import digest


def run(root, feature_pid):
    work = root / "h2"
    status = json.loads((work / "status.json").read_text())
    status.update(supervisor_pid=os.getpid(), feature_pid=feature_pid)

    def update(stage, **fields):
        status.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **fields)
        save(work / "status.json", status)
        path = root / "CURRENT_STATUS.json"
        current = json.loads(path.read_text())
        current.update(
            state=stage,
            updated_at=status["updated_at"],
            H2_status="h2/status.json",
            next_stage=status.get("next_stage", "H2 memory extraction then three-seed training"),
            blind_test_used=False,
        )
        save(path, current)

    update("H2_extracting_memory")
    snapshot = work / "training_implementation"
    snapshot.mkdir(exist_ok=False)
    names = [
        "h2_run.py",
        "h2_training.py",
        "h2_data.py",
        "h2_model.py",
        "h0_training.py",
        "toy_selector_model.py",
        "toy_selector_training.py",
        "toy_selector_labels.py",
        "ablation_score.py",
    ]
    for name in names:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    hashes = {n: digest(snapshot / n) for n in names}
    save(work / "training_implementation_manifest.json", hashes)
    try:
        process = Path(f"/proc/{feature_pid}/cmdline")
        while process.exists():
            if b"experiments.microduck_tasks.h2_features" not in process.read_bytes():
                raise ValueError("Feature PID belongs to a different process.")
            time.sleep(5)
        manifest = json.loads((work / "memory/manifest.json").read_text())
        if not manifest["complete"]:
            raise RuntimeError("Feature worker exited without completing memory; inspect features.log.")
        if any(digest(Path(__file__).with_name(n)) != h for n, h in hashes.items()):
            raise ValueError("Training source changed after snapshot; review before running.")
        command = [
            sys.executable,
            "-m",
            "experiments.microduck_tasks.h2_training",
            "--dataset",
            str(root / "h0/data"),
            "--features",
            str(root / "h0/features_p6"),
            "--memory",
            str(work / "memory"),
            "--output",
            str(work / "train128_attention"),
        ]
        update("H2_training", command=command)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        with (work / "training.log").open("w") as handle:
            subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        result = json.loads((work / "train128_attention/results.json").read_text())
        compare = {
            stage: json.loads((root / path / "results.json").read_text())
            for stage, path in (("H0", "h0/train128"), ("H1", "h1/train256"), ("H2", "h2/train128_attention"))
        }
        save(work / "comparison.json", dict(**compare, blind_test_used=False, input_view="P6"))
        update(
            "H2_complete",
            complete=True,
            result="h2/train128_attention/results.json",
            selected_seed=result["selected_seed"],
            prototype_gate_pass=result["prototype_gate_pass"],
            next_stage="Review H2 errors and costs before deciding Q0; keep blind sealed.",
        )
    except Exception as error:
        update("H2_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--feature-pid", type=int, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.feature_pid)
