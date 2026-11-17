"""Collect predeclared nearby-ball cases to balance the two trained kick skills."""

import argparse
import math
import random
import sys
from pathlib import Path

from .collect_training import COLORS, make_plan, run


def _layout(seed, foot, target_color):
    randomizer = random.Random(seed * 2 + int(foot == "right"))
    goal_x, goal_y = (0.0895, 0.0411) if foot == "left" else (0.0881, -0.0429)
    target = [goal_x + randomizer.uniform(-0.004, 0.004), goal_y + randomizer.uniform(-0.004, 0.004)]
    layout = {target_color: target}
    for color in COLORS:
        if color == target_color:
            continue
        for _ in range(1000):
            radius, angle = randomizer.uniform(0.35, 0.6), randomizer.uniform(-math.pi, math.pi)
            point = [radius * math.cos(angle), radius * math.sin(angle)]
            if all(math.dist(point, other) >= 0.2 for other in layout.values()):
                layout[color] = point
                break
        else:
            raise RuntimeError("Failed to generate separated supplemental balls for the fixed seed.")
    return layout


def make_near_plan(seed_start=2000, train_seeds=20, val_seeds=5, test_seeds=5):
    seeds, original = make_plan(seed_start, train_seeds, val_seeds, test_seeds)
    ordered_seeds = list(dict.fromkeys(case["seed"] for case in original))
    lookup = {(case["seed"], case["target_color"]): case for case in original if case["kind"] == "kick"}
    plan = []
    for seed in ordered_seeds:
        for foot in ("left", "right"):
            for color in COLORS:
                case = lookup[seed, color]
                plan.append(
                    {
                        **case,
                        "case_id": f"seed_{seed}_{foot}_{color}_near",
                        "scenario": f"near_kick_{foot}",
                        "expected_foot": foot,
                        "layout_world_xy_m": _layout(seed, foot, color),
                    }
                )
    return seeds, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--train-seeds", type=int, default=20)
    parser.add_argument("--val-seeds", type=int, default=5)
    parser.add_argument("--test-seeds", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Maximum new cases this invocation; rerun to resume.")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--min-free-gib", type=float, default=8)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--recheck-only", action="store_true", help="Revalidate completed traces; do not run new cases."
    )
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive.")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be finite and positive.")
        if not math.isfinite(args.min_free_gib) or args.min_free_gib < 8:
            raise ValueError("Preserve at least 8 GiB free while collecting data.")
        return run(args, plan_factory=make_near_plan)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
