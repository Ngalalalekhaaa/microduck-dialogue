#!/usr/bin/env python3
"""Run image-based discrete microduck navigation in local CPU simulation."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from bootstrap import MODEL_PATH, configure


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_case(runtime, planner, target, output, *, hidden=False, max_actions=30, max_seconds=60.0, goal_tolerance=0.035):
    import numpy as np
    from microduck_deploy.policy import PolicyContractError
    from PIL import Image
    from planner import validate_command

    output.mkdir(parents=True)
    started = time.perf_counter()
    events, estimate_errors = [], []
    actions = confirmations = misses = 0
    status = "timeout"
    failure_reason = None
    try:
        runtime.reset(target, hidden=hidden)
        while runtime.data.time < max_seconds:
            observation, rgb, depth = runtime.observe_target()
            image_name = f"camera_{len(events):03d}.png"
            Image.fromarray(rgb).save(output / image_name)
            np.savez_compressed(output / f"camera_{len(events):03d}_depth.npz", optical_z_m=depth)
            # Only RGB-D-derived observation crosses the planner boundary.
            command = validate_command(
                planner.decide(observation, rgb=rgb, instruction="Approach the red marker and stop in front of it.")
            )
            is_inside = (
                observation.found
                and planner.standoff - planner.tolerance
                <= observation.forward_m
                <= planner.standoff + planner.tolerance
                and abs(observation.left_m) <= planner.tolerance
            )
            root_velocity = runtime.data.qvel[int(runtime.model.joint("trunk_base_freejoint").dofadr[0]) :][:3]
            gyro = runtime.data.sensordata[runtime.gyro : runtime.gyro + 3]
            stationary = np.linalg.norm(root_velocity[:2]) < 0.025 and np.linalg.norm(gyro) < 0.3
            confirmations = confirmations + 1 if is_inside and stationary else 0
            misses = 0 if observation.found else misses + 1
            # Independent diagnostic only: this never determines the next command.
            oracle = runtime.oracle_target_body()
            if observation.found:
                estimate_errors.append(
                    float(np.linalg.norm(np.array([observation.forward_m, observation.left_m]) - oracle[:2]))
                )
            event = {
                "time_s": float(runtime.data.time),
                "image": image_name,
                "observation": asdict(observation),
                "command": asdict(command),
                "stationary": bool(stationary),
                "arrival_confirmations": confirmations,
            }
            events.append(event)
            with (output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            if confirmations >= 2:
                status = "arrived"
                break
            if misses >= 3:
                status = "target_lost"
                break
            if command.skill != "stand" and actions >= max_actions:
                status = "action_limit"
                break
            if runtime.data.time + command.seconds + 1.0 > max_seconds + 1e-8:
                status = "timeout"
                break
            if command.skill != "stand":
                actions += 1
            runtime.execute(command)
    except RuntimeError as exc:
        status = str(exc)
        failure_reason = str(exc)
    except PolicyContractError as exc:
        status = "policy_error"
        failure_reason = str(exc)
    runtime.save_trace(output / "trajectory.npz")
    oracle = runtime.oracle_target_body()
    finite_state = bool(
        np.isfinite(runtime.data.qpos).all() and np.isfinite(runtime.data.qvel).all() and np.isfinite(oracle).all()
    )
    error = [float(oracle[0] - planner.standoff), float(oracle[1])] if finite_state else None
    truth_inside = finite_state and all(abs(v) <= goal_tolerance for v in error)
    result = {
        "status": status,
        "success": status == "arrived" and truth_inside,
        "failure_reason": failure_reason,
        "false_arrival": status == "arrived" and not truth_inside,
        "target_world_xy_m": list(target),
        "target_hidden": hidden,
        "standoff_m": planner.standoff,
        "tolerance_per_axis_m": goal_tolerance,
        "vision_stop_tolerance_m": planner.tolerance,
        "final_oracle_target_body_m": oracle.tolist() if finite_state else None,
        "final_oracle_error_xy_m": error,
        "max_vision_position_error_m": max(estimate_errors, default=None),
        "actions": actions,
        "observations": len(events),
        "simulation_seconds": float(runtime.data.time),
        "wall_seconds": time.perf_counter() - started,
        "final_observation": events[-1] if events else None,
        "max_tilt_deg": max((r[4] for r in runtime.trace), default=0.0),
        "output": str(output),
    }
    write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "target_world_xy_m",
                    "status",
                    "success",
                    "actions",
                    "final_oracle_error_xy_m",
                    "max_vision_position_error_m",
                )
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, nargs=2, metavar=("FORWARD_M", "LEFT_M"), default=[0.4, 0.1])
    parser.add_argument(
        "--batch", type=int, choices=[20], help="Run a fixed 20-position grid and one hidden-target case"
    )
    parser.add_argument("--hide-target", action="store_true", help="Negative test: no visible red marker")
    parser.add_argument("--video", action="store_true", help="Render a continuous MP4 after the single run")
    parser.add_argument("--max-actions", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--standoff", type=float, default=0.2)
    parser.add_argument("--tolerance", type=float, default=0.035)
    parser.add_argument(
        "--vision-margin",
        type=float,
        default=0.01,
        help="Stop inside a tighter goal box to allow for RGB-D localization error",
    )
    parser.add_argument("--out", type=Path, help="New output directory; existing paths are refused")
    args = parser.parse_args()
    import math

    if args.max_actions <= 0 or not math.isfinite(args.max_seconds) or args.max_seconds < 3:
        parser.error("Use max-actions >0 and finite max-seconds >=3")
    if not all(math.isfinite(v) for v in args.target):
        parser.error("Target coordinates must be finite")
    if not (
        math.isfinite(args.tolerance) and math.isfinite(args.vision_margin) and 0 <= args.vision_margin < args.tolerance
    ):
        parser.error("Use finite tolerance > vision-margin >=0")
    if args.batch and (args.video or args.hide_target):
        parser.error("Use --video/--hide-target with a single run, separate from --batch")
    configure()
    import mujoco
    import onnxruntime
    from planner import MarkerPlanner
    from runtime import Runtime

    planner = MarkerPlanner(args.standoff, args.tolerance - args.vision_margin)
    output = (args.out or Path(__file__).parent / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new path: {output}")
    output.mkdir(parents=True)
    runtime = Runtime(output)
    results = []
    video_error = None
    try:
        targets = (
            [(x, y) for x in (0.17, 0.25, 0.35, 0.45, 0.55) for y in (-0.12, -0.04, 0.04, 0.12)]
            if args.batch
            else [args.target]
        )
        for i, target in enumerate(targets):
            results.append(
                run_case(
                    runtime,
                    planner,
                    target,
                    output / f"case_{i:02d}",
                    hidden=args.hide_target,
                    max_actions=args.max_actions,
                    max_seconds=args.max_seconds,
                    goal_tolerance=args.tolerance,
                )
            )
        if args.batch:
            results.append(
                run_case(
                    runtime,
                    planner,
                    (0.4, 0.1),
                    output / "hidden_target",
                    hidden=True,
                    max_actions=args.max_actions,
                    max_seconds=args.max_seconds,
                    goal_tolerance=args.tolerance,
                )
            )
        if args.video:
            from recording import render_video

            try:
                render_video(runtime, results[0], output / "demo.mp4")
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                video_error = f"{type(exc).__name__}: {exc}"
                print(f"Video failed; simulation results are preserved: {video_error}")
    finally:
        runtime.close()
    visible = [r for r in results if not r["target_hidden"]]
    summary = {
        "scope": (
            "RGB-D marker feedback + deterministic discrete planner + local ONNX/BAM. "
            "No VLM, cloth, jaw, audio or hardware. Independent Policy loop, not Strands Robot.run."
        ),
        "versions": {"mujoco": mujoco.__version__, "onnxruntime": onnxruntime.__version__},
        "policy": str(MODEL_PATH),
        "policy_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "physics": runtime.physics,
        "video_error": video_error,
        "camera": {
            "kind": "virtual body-mounted RGB-D; not the real IMX219",
            "resolution": [320, 240],
            "fovy_deg": 85,
            "position_body_m": [0.14, 0, 0.16],
            "depth": "MuJoCo metric optical-axis depth; perfect simulated calibration",
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
        "visible_successes": sum(r["success"] for r in visible),
        "visible_cases": len(visible),
        "false_arrivals": sum(r["false_arrival"] for r in results),
        "negative_checks_pass": all(
            r["status"] == "target_lost" and r["actions"] == 0 for r in results if r["target_hidden"]
        ),
        "results": results,
    }
    write_json(output / "summary.json", summary)
    print(f"Results: {output}")
    if any(not r["success"] for r in visible) or not summary["negative_checks_pass"]:
        return 2
    if video_error is not None:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
