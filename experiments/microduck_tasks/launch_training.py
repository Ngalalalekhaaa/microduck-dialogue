"""Launch a local training process that survives closing this chat or terminal."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.microduck_tasks.train_vlm import digest, enough_disk, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--predict-limit", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the launch request without starting or writing anything."
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    python = repo / ".cache/microduck_train/venv/bin/python"
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if not python.is_file():
        raise FileNotFoundError("Run python -m experiments.microduck_tasks.setup_training --download first.")
    for name in ("train.jsonl", "val.jsonl", "manifest.json"):
        if not (dataset / name).is_file():
            raise FileNotFoundError(f"Missing {dataset / name}; finish and audit the dataset export before launching.")
    if (output / "training_config.json").exists() and not args.resume:
        raise FileExistsError("This output already contains a training run; use --resume or choose a new output.")
    command = [
        str(python),
        "-u",
        "-m",
        "experiments.microduck_tasks.train_vlm",
        "--dataset",
        str(dataset),
        "--output",
        str(output),
    ]
    for name in ("epochs", "max_steps", "grad_accum", "eval_limit", "predict_limit", "eval_every", "save_every"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve())])
    request = {
        "argv": command,
        "cwd": str(repo),
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "stdout_stderr": str(output / "train.log"),
        "stdin": "DEVNULL",
        "start_new_session": True,
    }
    if args.dry_run:
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    enough_disk(output)
    with (output / "train.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false"),
        )
    launched = datetime.now(timezone.utc)
    request.update({"pid": process.pid, "started_utc": launched.isoformat(), "status": "launched"})
    write_json(output / f"launch_{launched.strftime('%Y%m%dT%H%M%S%fZ')}.json", request)
    write_json(output / "launch.json", request)
    print(json.dumps(request, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
