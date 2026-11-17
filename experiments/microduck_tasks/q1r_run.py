"""Durable continuation supervisor: freeze sources, train, then reload and replay."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .q1r_training_state import Status
from .run_ablation import save
from .toy_dataset import digest

SOURCES = (
    "q1r_run.py",
    "q1r_training.py",
    "q1r_training_state.py",
    "q1r_encoding.py",
    "q1r_prompt.py",
    "q1r_evaluate.py",
    "q1r_verify.py",
    "q1r_compare.py",
    "q0_model.py",
    "head_features.py",
    "h0_features.py",
    "toy_protocol.py",
    "toy_inputs.py",
    "toy_vlm_prompt.py",
    "toy_catalog.py",
    "setup_skill_head.py",
    "toy_vlm_json.py",
    "ablation_contract.py",
    "run_ablation.py",
)


def run(root, parent, model_dir):
    snapshot = root / "frozen_implementation"
    snapshot.mkdir(exist_ok=False)
    hashes = {}
    for name in SOURCES:
        source = Path(__file__).with_name(name)
        (snapshot / name).write_bytes(source.read_bytes())
        hashes[name] = digest(source)
    for source in sorted(Path(__file__).parent.glob("q1r_data*.py")):
        (snapshot / source.name).write_bytes(source.read_bytes())
        hashes[source.name] = digest(source)
    language = Path(__file__).with_name("q1r_language.py")
    (snapshot / language.name).write_bytes(language.read_bytes())
    hashes[language.name] = digest(language)
    data_hash = digest(root / "encoded/manifest.json")
    save(root / "implementation_manifest.json", dict(sources=hashes, data_manifest_sha256=data_hash))
    status = dict(
        supervisor_pid=os.getpid(), complete=False, authorization="User explicitly requested Q1 continuation now"
    )
    save(root / "supervisor.json", status)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    try:
        for module in ("q1r_training", "q1r_verify"):
            if any(digest(Path(__file__).with_name(n)) != h for n, h in hashes.items()):
                raise ValueError("Frozen continuation source changed; preserve and inspect this run.")
            if digest(root / "encoded/manifest.json") != data_hash:
                raise ValueError("Frozen data manifest changed.")
            command = [
                sys.executable,
                "-m",
                f"experiments.microduck_tasks.{module}",
                "--root",
                str(root),
                "--model",
                str(model_dir),
            ]
            if module == "q1r_training":
                command += ["--parent", str(parent)]
            with (root / f"{module}.log").open("x") as stream:
                child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                status.update(stage=module, child_pid=child.pid, command=command)
                save(root / "supervisor.json", status)
                code = child.wait()
            if code:
                raise RuntimeError(f"{module} exited {code}; inspect {module}.log and status.json.")
        result = json.loads((root / "train/results.json").read_text())
        if not result["complete"] or not result["reload_pass"]:
            raise ValueError("Final continuation reload and replay did not complete.")
        status.update(stage="complete", complete=True, selected_step=result["selected_step"])
        save(root / "supervisor.json", status)
    except Exception as error:
        status.update(stage="failed", error=f"{type(error).__name__}: {error}")
        save(root / "supervisor.json", status)
        Status(root)("pipeline_failed", error=status["error"])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    run(args.root.resolve(), args.parent.resolve(), args.model.resolve())
