"""Run the longest-sample full optimizer-step probe before starting Q0 fitting."""

import argparse
from pathlib import Path

from .q0_data import load
from .q0_model import load_model, smoke
from .run_ablation import save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    _, rows = load(args.data)
    try:
        model, processor, _ = load_model(args.model, args.output)
        report = smoke(model, processor, rows, args.output)
        save(args.output / "status.json", dict(complete=True, passed=report["passed"], blind_access=False))
    except Exception as error:
        save(
            args.output / "status.json",
            dict(complete=False, error=f"{type(error).__name__}: {error}", blind_access=False),
        )
        raise
