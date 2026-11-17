"""Measure foot contact reach in simulation, independent of navigation success."""

import argparse
import json
from pathlib import Path

from .bootstrap import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    import mujoco

    from .runtime import Runtime
    from .tasks import evaluate_kick

    args.output.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(args.output)
    trials = []
    try:
        for foot, sign in (("left", 1), ("right", -1)):
            for x in (0.065, 0.08, 0.095, 0.11, 0.125):
                for y in (0.0, 0.015, 0.03, 0.045, 0.06):
                    runtime.reset({"red": [3, 3], "yellow": [-3, 3], "blue": [3, -3]})
                    with runtime.lock:
                        rotation = runtime.data.xmat[runtime.body].reshape(3, 3)
                        point = runtime.data.xpos[runtime.body] + rotation @ [x, sign * y, 0]
                        runtime.data.qpos[runtime.ball_q["red"] : runtime.ball_q["red"] + 3] = [*point[:2], 0.035]
                        mujoco.mj_forward(runtime.model, runtime.data)
                    obs, _, _ = runtime.observe()
                    before = runtime.truth()
                    start, contacts, rows = runtime.snapshot()["time_s"], len(runtime.contacts), len(runtime.rows)
                    runtime.execute(f"kick_{foot}", 0.5)
                    runtime.execute("stand", 3.0)
                    evaluation = evaluate_kick(
                        "red", foot, before, runtime.truth(), start, runtime.contacts[contacts:], runtime.rows[rows:]
                    )
                    measured = next((d["center_body_m"] for d in obs["detections"] if d["color"] == "red"), None)
                    trial = {
                        "foot": foot,
                        "placed_xy": [x, sign * y],
                        "measured_xyz": measured,
                        "evaluation": evaluation,
                    }
                    trials.append(trial)
                    print(json.dumps(trial), flush=True)
        (args.output / "reach.json").write_text(
            json.dumps(
                {"hardware": False, "policy_sha256": runtime.policy.hashes, "trials": trials}, indent=2, allow_nan=False
            )
            + "\n"
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
