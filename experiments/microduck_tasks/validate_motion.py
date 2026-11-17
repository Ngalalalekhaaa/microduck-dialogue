"""Fit cross-axis feedforward with heading feedback and evaluate held-out trials."""

import argparse
import json
from pathlib import Path

import numpy as np

from .bootstrap import configure
from .calibrate_motion import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    from .runtime import Runtime
    from .skills import TWISTS

    args.output.mkdir(parents=True, exist_ok=False)
    grid = json.loads(args.grid.read_text())
    runtime = Runtime(args.output)
    trials, validation, commands = [], [], {}
    try:
        for skill, initial in grid["commands"].items():
            axis = 1 if skill in ("move_forward", "move_backward") else 0
            candidates = []
            for lateral in np.linspace(-0.06, 0.06, 7):
                command = list(initial)
                command[axis] = float(lateral)
                runtime.policy.motion.commands = {skill: command}
                cases = []
                for heading, warmup in ((0, 0.6), (1.3, 1.2)):
                    trial = measure(runtime, skill, command, seconds=5, warmup=warmup, heading=heading, feedback=True)
                    cases.append(trial)
                    trials.append(trial)
                score = max(t["max_cross_track_m"] + abs(t["yaw_change_deg"]) * 0.002 for t in cases)
                candidates.append((score, command))
            commands[skill] = min(candidates, key=lambda item: item[0])[1]
            runtime.policy.motion.commands = {skill: commands[skill]}
            for seconds, heading, warmup in ((3, 0.7, 0.4), (8, -1.1, 1.4), (12, 2.4, 0.8)):
                for label, command, feedback in (
                    ("nominal", TWISTS[skill], False),
                    ("calibrated_heading", commands[skill], True),
                ):
                    trial = measure(
                        runtime, skill, command, seconds=seconds, warmup=warmup, heading=heading, feedback=feedback
                    )
                    validation.append({"mode": label, **trial})
                    print(json.dumps(validation[-1]), flush=True)
        result = {
            "format": "microduck-sim-heading-calibration-1",
            "hardware": False,
            "policy_sha256": runtime.policy.hashes["walk"],
            "physics": runtime.physics,
            "commands": commands,
            "fitting_trials": trials,
            "held_out": validation,
        }
        (args.output / "calibration.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
