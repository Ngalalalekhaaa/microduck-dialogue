"""Measure command tracking in the local simulator; never connect hardware.

Run with the existing microduck_rl Python environment, using --output NEW_DIR.
Fitting trials and held-out durations/initial headings are recorded separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .bootstrap import configure


def measure(runtime, skill, twist, *, seconds=6.0, warmup=1.0, heading=0.0, feedback=False):
    import mujoco

    from .skills import TWISTS

    original = TWISTS[skill]
    enabled = runtime.policy.motion.enabled
    runtime.policy.motion.enabled = feedback
    TWISTS[skill] = tuple(twist)
    try:
        runtime.reset({"red": [3, 3], "yellow": [-3, 3], "blue": [3, -3]})
        with runtime.lock:
            runtime.data.qpos[runtime.qa + 3 : runtime.qa + 7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
            mujoco.mj_forward(runtime.model, runtime.data)
        if warmup:
            runtime.execute("stand", warmup)
        first = len(runtime.rows) - 1
        for _ in range(round(seconds / 0.02)):
            runtime._tick(skill)
        rows = runtime.rows[first:]
        xyz = np.array([row[7] for row in rows])
        yaw = np.unwrap([row[5] for row in rows])
        times = np.array([row[0] for row in rows])
        rotation = np.array([[np.cos(yaw[0]), np.sin(yaw[0])], [-np.sin(yaw[0]), np.cos(yaw[0])]])
        xy = (xyz[:, :2] - xyz[0, :2]) @ rotation.T
        steady = times >= times[0] + min(1.5, seconds / 3)
        velocity = np.polyfit(times[steady], xy[steady], 1)[0]
        axis = 0 if skill in ("move_forward", "move_backward") else 1
        return {
            "skill": skill,
            "twist": list(twist),
            "seconds": seconds,
            "heading_rad": heading,
            "settle_s": warmup,
            "velocity_body_initial_m_s": velocity.tolist(),
            "yaw_rate_rad_s": float(np.polyfit(times[steady], yaw[steady], 1)[0]),
            "yaw_change_deg": float(np.rad2deg(yaw[-1] - yaw[0])),
            "displacement_m": xy[-1].tolist(),
            "max_cross_track_m": float(np.max(np.abs(xy[:, 1 - axis]))),
            "path_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            "max_tilt_deg": max(row[4] for row in rows),
        }
    finally:
        TWISTS[skill] = original
        runtime.policy.motion.enabled = enabled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configure()
    from .runtime import Runtime
    from .skills import TWISTS

    args.output.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(args.output)
    trials, validation, commands = [], [], {}
    try:
        for skill in ("move_forward", "move_backward", "move_left", "move_right"):
            nominal = np.array(TWISTS[skill], dtype=float)
            axis = 0 if skill in ("move_forward", "move_backward") else 1

            def objective(parameters, nominal=nominal, axis=axis, skill=skill):
                command = nominal.copy()
                command[1 - axis], command[2] = parameters
                trial = measure(runtime, skill, command)
                trials.append(trial)
                print(json.dumps(trial), flush=True)
                return trial["max_cross_track_m"] + abs(trial["yaw_change_deg"]) * 0.004

            best = np.zeros(2)
            for index, offsets in (
                (1, np.linspace(-0.4, 0.4, 9)),
                (0, np.linspace(-0.08, 0.08, 9)),
                (1, np.linspace(-0.08, 0.08, 9)),
                (0, np.linspace(-0.02, 0.02, 5)),
            ):
                candidates = [best.copy() for _ in offsets]
                for candidate, offset in zip(candidates, offsets, strict=True):
                    candidate[index] += offset
                best = min(candidates, key=objective)
            command = nominal.copy()
            command[1 - axis], command[2] = best
            commands[skill] = command.tolist()
            for seconds, heading, settle in ((3.0, 0.7, 0.4), (8.0, -1.1, 1.4)):
                for label, twist in (("nominal", nominal), ("fitted", command)):
                    trial = measure(runtime, skill, twist, seconds=seconds, warmup=settle, heading=heading)
                    validation.append({"mode": label, **trial})
        result = {
            "format": "microduck-sim-motion-calibration-1",
            "hardware": False,
            "policy_sha256": runtime.policy.hashes,
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
