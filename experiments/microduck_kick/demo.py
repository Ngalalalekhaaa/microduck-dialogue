"""Test pinned official kicks on the existing HD1910 physics and walking controller."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from bootstrap import KICK_HASHES, KICK_MODELS, STAND_MODEL, configure


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_case(experiment, output, foot, duration, offset, *, control=False):
    import numpy as np
    from engine import DT
    from evaluation import assess

    output.mkdir()
    ball_xy = [0.09 + offset[0], (0.042 if foot == "left" else -0.042) + offset[1]]
    experiment.reset(ball_xy)
    warmup, recovery = 2.0, 3.0
    failure = None
    for step in range(round((warmup + duration + recovery) / DT)):
        time = step * DT
        skill = "warmup" if time < warmup else "recovery"
        if warmup <= time < warmup + duration - 1e-8:
            skill = "control_stand" if control else f"kick_{foot}"
        try:
            experiment.tick(skill)
        except (RuntimeError, ValueError) as exc:
            failure = str(exc)
            break
    arrays = experiment.arrays()
    np.savez_compressed(output / "trajectory.npz", **arrays)
    write_json(output / "contacts.json", experiment.contacts)
    result = assess(
        arrays,
        experiment.contacts,
        foot=foot,
        kick_seconds=duration,
        warmup_seconds=warmup,
        recovery_seconds=recovery,
        failure=failure,
        control=control,
        lag_seconds=experiment.physics["lag_seconds"],
    )
    result.update(
        {
            "foot": foot,
            "kick_seconds": duration,
            "ball_offset_xy_m": list(offset),
            "ball_initial_world_xy_m": ball_xy,
            "warmup_seconds": warmup,
            "recovery_seconds": recovery,
            "variant": experiment.physics["variant"],
            "output": str(output),
            "target_outside_xml_range_ticks": experiment.target_outside_ticks,
            "max_target_outside_xml_range_rad": float(experiment.target_exceedance.max()),
            "target_exceedance_by_joint_rad": experiment.target_exceedance.tolist(),
            "max_measured_joint_limit_excess_rad": float(experiment.actual_exceedance.max()),
        }
    )
    write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "foot",
                    "kick_seconds",
                    "ball_offset_xy_m",
                    "control",
                    "status",
                    "ball_forward_m",
                    "max_tilt_deg",
                )
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--foot", choices=["left", "right", "both"], default="both")
    parser.add_argument("--durations", type=float, nargs="+", default=[0.5])
    parser.add_argument("--grid", action="store_true", help="Nine ball positions per foot, offsets +/-1.5 cm")
    parser.add_argument("--controls", action="store_true", help="Repeat each scene with standing only")
    parser.add_argument("--lag-steps", type=int, default=4)
    parser.add_argument("--variant", choices=["hd1910", "original"], default="hd1910")
    parser.add_argument("--stand-model", type=Path, default=STAND_MODEL)
    args = parser.parse_args()
    import math

    if any(not math.isfinite(v) or not 0.1 <= v <= 3 or abs(v / 0.02 - round(v / 0.02)) > 1e-8 for v in args.durations):
        parser.error("Use durations between 0.1 and 3 seconds in multiples of 0.02.")
    if not 0 <= args.lag_steps <= 20:
        parser.error("Use lag-steps between 0 and 20.")
    configure()
    import mujoco
    import onnxruntime
    from engine import Experiment

    output = (args.out or Path(__file__).parent / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        parser.error(f"Choose a new output directory; {output} already exists.")
    output.mkdir(parents=True)
    experiment = Experiment(output, variant=args.variant, lag_steps=args.lag_steps, stand_model=args.stand_model)
    feet = ("left", "right") if args.foot == "both" else (args.foot,)
    offsets = [(x, y) for x in (-0.015, 0, 0.015) for y in (-0.015, 0, 0.015)] if args.grid else [(0, 0)]
    results = []
    for foot in feet:
        for duration in args.durations:
            for offset in offsets:
                for control in (False, True) if args.controls else (False,):
                    case = output / f"case_{len(results):03d}_{foot}{'_control' if control else ''}"
                    results.append(run_case(experiment, case, foot, duration, offset, control=control))
    actual = [r for r in results if not r["control"]]
    controls = [r for r in results if r["control"]]
    summary = {
        "physics": experiment.physics,
        "versions": {"mujoco": mujoco.__version__, "onnxruntime": onnxruntime.__version__},
        "kick_sha256": KICK_HASHES,
        "stand_model": str(args.stand_model.resolve()),
        "stand_sha256": hashlib.sha256(args.stand_model.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((KICK_MODELS / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
        "success_count": sum(r["success"] for r in actual),
        "kick_cases": len(actual),
        "clean_setup_count": sum(r["setup_clean"] for r in actual),
        "control_passes": sum(r["status"] == "control_stable" for r in controls),
        "control_cases": len(controls),
        "criteria": (
            "No pre-kick robot-ball contact or >1cm drift; designated foot contact in kick window + actual lag; "
            ">=3cm forward launch within that window, >=10cm total; no other/late robot-ball contact; no fall; "
            "last0.5s tilt<12deg, trunk>9cm, XYspeed<0.05m/s, gyro<0.4rad/s. "
            "Standing controls: no robot-ball contact and max XY ball drift<1cm."
        ),
        "scope": (
            "Fixed-ball action test only. Public HD1910 BAM nominal parameters, not calibrated hardware. "
            "Ball assigned once at reset, never repositioned. No vision/language/navigation or physical robot. "
            "Native targets, no clipping; previous action and controller delay preserved across switches."
        ),
        "results": results,
    }
    write_json(output / "summary.json", summary)
    print(
        f"Saved {output}; kicks {summary['success_count']}/{len(actual)}, "
        f"controls {summary['control_passes']}/{len(controls)}"
    )
    return 0 if all(r["success"] for r in actual) and all(r["status"] == "control_stable" for r in controls) else 2


if __name__ == "__main__":
    raise SystemExit(main())
