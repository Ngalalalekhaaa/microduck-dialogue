"""Pair six language commands with the same initial scene and robot state."""

import argparse
import math
import sys
from pathlib import Path

from .collect_training import MOVE_PHRASES, make_plan, run


def make_reset_plan(seed_start=3000, train_seeds=10, val_seeds=3, test_seeds=3):
    seeds, original = make_plan(seed_start, train_seeds, val_seeds, test_seeds)
    plan = []
    for case in original:
        if case["kind"] != "direct_dialogue":
            continue
        for skill, instruction in zip(MOVE_PHRASES, case["instructions"][1:7], strict=True):
            plan.append(
                {
                    **case,
                    "case_id": f"seed_{case['seed']}_{skill}_reset",
                    "scenario": "direct_reset",
                    "expected_skill": skill,
                    "instructions": [instruction],
                }
            )
    return seeds, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--train-seeds", type=int, default=10)
    parser.add_argument("--val-seeds", type=int, default=3)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--min-free-gib", type=float, default=8)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--recheck-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive.")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be finite and positive.")
        if not math.isfinite(args.min_free_gib) or args.min_free_gib < 8:
            raise ValueError("Preserve at least 8 GiB free while collecting data.")
        return run(args, plan_factory=make_reset_plan)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
