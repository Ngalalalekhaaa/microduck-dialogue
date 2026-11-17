"""Run the width-only H1 comparison on the frozen H0 data and features."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .run_ablation import save
from .toy_dataset import digest


def run(root):
    h0, work = root / "h0", root / "h1"
    baseline = json.loads((h0 / "train128/results.json").read_text())
    if not baseline["complete"]:
        raise ValueError("Finish H0 before starting H1.")
    work.mkdir(exist_ok=False)
    snapshot = work / "implementation"
    snapshot.mkdir()
    names = [
        "h1_run.py",
        "h0_training.py",
        "h0_features.py",
        "toy_selector_model.py",
        "toy_selector_training.py",
        "toy_selector_labels.py",
        "ablation_score.py",
    ]
    hashes = {}
    for name in names:
        source = Path(__file__).with_name(name)
        (snapshot / name).write_bytes(source.read_bytes())
        hashes[name] = digest(source)
    save(work / "implementation_manifest.json", hashes)
    save(
        work / "input_manifest.json",
        dict(
            dataset=str(h0 / "data"),
            features=str(h0 / "features_p6"),
            dataset_manifest_sha256=digest(h0 / "data/manifest.json"),
            feature_manifest_sha256=digest(h0 / "features_p6/manifest.json"),
            changed_variable="GRU hidden width 128 -> 256 only",
            blind_access=False,
        ),
    )
    status = dict(
        started_at=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        complete=False,
        blind_test_used=False,
        physical_executions=0,
    )

    def update(stage, **fields):
        status.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **fields)
        save(work / "status.json", status)
        path = root / "CURRENT_STATUS.json"
        current = json.loads(path.read_text())
        current.update(
            state=stage,
            updated_at=status["updated_at"],
            H1_status="h1/status.json",
            next_stage=status.get("next_stage", "H1 width-only comparison"),
            blind_test_used=False,
        )
        save(path, current)

    command = [
        sys.executable,
        "-m",
        "experiments.microduck_tasks.h0_training",
        "--dataset",
        str(h0 / "data"),
        "--features",
        str(h0 / "features_p6"),
        "--output",
        str(work / "train256"),
        "--hidden",
        "256",
    ]
    update("H1_training", command=command)
    try:
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
            PYTHONUNBUFFERED="1",
        )
        with (work / "training.log").open("w") as handle:
            subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        result = json.loads((work / "train256/results.json").read_text())
        old = json.loads((h0 / "train128/training_config.json").read_text())
        new = json.loads((work / "train256/training_config.json").read_text())
        differences = {k: [old[k], new[k]] for k in old if old[k] != new[k]}
        assert set(differences) == {"hidden", "source_sha256"}
        assert old["source_sha256"]["toy_selector_model.py"] == new["source_sha256"]["toy_selector_model.py"]
        assert old["source_sha256"]["toy_selector_training.py"] == new["source_sha256"]["toy_selector_training.py"]
        save(
            work / "comparison.json",
            dict(H0=baseline, H1=result, config_differences=differences, blind_test_used=False),
        )
        update(
            "H1_complete",
            complete=True,
            result="h1/train256/results.json",
            selected_seed=result["selected_seed"],
            prototype_gate_pass=result["prototype_gate_pass"],
            next_stage="Review H1 before H2 multi-position feature extraction; keep blind sealed.",
        )
    except Exception as error:
        update("H1_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())
