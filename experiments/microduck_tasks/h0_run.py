"""Run H0 extraction then CPU fitting, with explicit progress and no blind inference."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .ablation_report import load_rows
from .ablation_score import summarize
from .run_ablation import save
from .toy_dataset import digest


def run(root, model):
    work = root / "h0"
    work.mkdir(exist_ok=True)
    status = dict(
        started_at=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        complete=False,
        blind_test_used=False,
        H0_started=False,
        physical_executions=0,
    )

    def update(stage, **fields):
        status.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **fields)
        save(work / "status.json", status)
        current_path = root / "CURRENT_STATUS.json"
        current = json.loads(current_path.read_text())
        current.update(
            state=stage,
            updated_at=status["updated_at"],
            H0_started=status["H0_started"],
            H0_status="h0/status.json",
            blind_test_used=False,
        )
        save(current_path, current)

    sources = [
        *Path(__file__).parent.glob("h0_*.py"),
        Path(__file__).with_name("ablation_prompt.py"),
        Path(__file__).with_name("toy_selector_model.py"),
        Path(__file__).with_name("toy_selector_training.py"),
        Path(__file__).with_name("toy_selector_labels.py"),
        Path(__file__).with_name("head_features.py"),
    ]
    snapshot = work / "implementation"
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for path in sources:
        data = path.read_bytes()
        destination = snapshot / path.name
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError("Source snapshot differs; keep this run immutable and choose a new experiment.")
        destination.write_bytes(data)
        hashes[path.name] = digest(path)
    save(work / "implementation_manifest.json", hashes)
    base = [sys.executable, "-m"]
    stages = [
        (
            "H0_extracting_features",
            base
            + [
                "experiments.microduck_tasks.h0_features",
                "--dataset",
                str(work / "data"),
                "--output",
                str(work / "features_p6"),
                "--model",
                str(model),
                "--batch-size",
                "1",
            ],
        ),
        (
            "H0_training",
            base
            + [
                "experiments.microduck_tasks.h0_training",
                "--dataset",
                str(work / "data"),
                "--features",
                str(work / "features_p6"),
                "--output",
                str(work / "train128"),
            ],
        ),
    ]
    try:
        for stage, command in stages:
            env = os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            if stage == "H0_training":
                env["CUDA_VISIBLE_DEVICES"] = ""
                status["H0_started"] = True
            logfile = work / f"{stage}.log"
            update(stage, log=str(logfile), command=command)
            with logfile.open("a") as handle:
                subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        result = json.loads((work / "train128/results.json").read_text())
        baseline = load_rows(root / "validation/P6")
        comparison = dict(
            input_view="P6",
            baseline_all240=summarize(baseline, expected_count=240),
            baseline_common236=summarize([r for r in baseline if r["domain"] == "common"], expected_count=236),
            H0=result,
            historical_head_is_not_a_data_only_control=True,
            blind_test_used=False,
        )
        save(work / "comparison.json", comparison)
        update(
            "H0_complete",
            complete=True,
            result="h0/train128/results.json",
            prototype_gate_pass=result["prototype_gate_pass"],
            selected_seed=result["selected_seed"],
            next_stage="Review train/development errors before H1; keep blind sealed.",
        )
    except Exception as error:
        update("H0_failed", error=f"{type(error).__name__}: {error}", complete=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    run(args.root.resolve(), args.model.resolve())


if __name__ == "__main__":
    main()
