"""Fixed-action physical foot reach calibration, with explicit truth fixtures.

This is a simulator calibration experiment, not sensor-only closed-loop task
acceptance. Truth places the ball on the ground and scores outcomes; neither
placement nor scoring is a planner input. All 18 cases run once as declared.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

CASES = tuple(
    dict(foot=foot, x_foot_m=x, y_outward_m=y)
    for foot in ("left", "right")
    for x in (0.06, 0.09, 0.12)
    for y in (-0.04, -0.01, 0.02)
)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _phase_state(runtime, foot):
    """Truth is read here exclusively to measure calibration fixture geometry."""
    from .rgb_tof_feet import feet_from_encoders
    from .runtime import Runtime

    feet = feet_from_encoders(runtime.model, runtime.data.qpos[runtime.qi])
    truth = Runtime.truth(runtime)
    rotation = runtime.data.xmat[runtime.body].reshape(3, 3)
    ball_body = rotation.T @ (truth["red"] - runtime.data.xpos[runtime.body])
    relative = feet[foot].to_foot(ball_body)
    return dict(
        time_s=float(runtime.data.time),
        head_command=runtime.policy.head_command.tolist(),
        joint_encoders=runtime.data.qpos[runtime.qi].tolist(),
        feet={
            name: dict(
                position_body_m=pose.position_body_m.tolist(),
                rotation_body_from_foot=pose.rotation_body_from_foot.tolist(),
            )
            for name, pose in feet.items()
        },
        true_ball_world_m=truth["red"].tolist(),
        true_ball_body_m=ball_body.tolist(),
        true_ball_foot_m=relative.tolist(),
        true_ball_outward_y_m=float(relative[1] * (1 if foot == "left" else -1)),
        estimated_body_state=runtime.snapshot(),
        source="independent_truth_for_calibration_measurement_only",
    )


def _place(runtime, case):
    import mujoco

    from .rgb_tof_feet import feet_from_encoders

    foot = feet_from_encoders(runtime.model, runtime.data.qpos[runtime.qi])[case["foot"]]
    lateral = case["y_outward_m"] * (1 if case["foot"] == "left" else -1)
    point_body = foot.to_body([case["x_foot_m"], lateral, 0.0])
    world = runtime.data.xpos[runtime.body] + runtime.data.xmat[runtime.body].reshape(3, 3) @ point_body
    world[2] = 0.035
    q, v = runtime.ball_q["red"], runtime.ball_v["red"]
    runtime.data.qpos[q : q + 7] = [*world, 1, 0, 0, 0]
    runtime.data.qvel[v : v + 6] = 0
    mujoco.mj_forward(runtime.model, runtime.data)
    runtime.cameras.last = runtime.packet = runtime.camera_pose = None
    runtime._record("calibration_ball_placement")
    return world.tolist()


def _execute(runtime, commands, skill, seconds, phase):
    record = dict(phase=phase, skill=skill, seconds=seconds, start_time_s=float(runtime.data.time), completed=False)
    commands.append(record)
    try:
        record["completed"] = bool(runtime.execute(skill, seconds))
        if not record["completed"]:
            raise RuntimeError(f"Fixed calibration action cancelled: {phase}")
    finally:
        record["end_time_s"] = float(runtime.data.time)


def _case(output, case, index):
    from .rgb_tof_feet import feet_from_encoders
    from .rgb_tof_runtime import SensorRuntime
    from .runtime import Runtime
    from .tasks import evaluate_kick

    output.mkdir()
    result = dict(**case, index=index, status="failed", commands=[], evaluation=dict(success=False))
    runtime = before = None
    kick_start = None
    started = time.monotonic()
    try:
        runtime = SensorRuntime(output, seed=17 + index, noise_m=0.002, dropout=0.0)
        runtime.head_tracking = False
        runtime.reset(dict(red=[1.0, 0.0], yellow=[1.5, 0.8], blue=[1.5, -0.8]))
        runtime.head_tracking = False
        runtime.policy.head_command[:] = [0, 0.7, 0, 0]
        _execute(runtime, result["commands"], "stand", 1.0, "head_down")
        result["fixture_world_m"] = _place(runtime, case)
        before = Runtime.truth(runtime)
        result["after_placement"] = _phase_state(runtime, case["foot"])
        try:
            observation, _, _ = runtime.observe()  # The only rendered observation in this case.
            foot = feet_from_encoders(runtime.model, runtime.data.qpos[runtime.qi])[case["foot"]]
            red = [d for d in observation["detections"] if d["color"] == "red"]
            result["after_placement"]["sensor_observation"] = observation
            result["after_placement"]["sensor_ball_foot_m"] = (
                foot.to_foot(max(red, key=lambda d: d["confidence"])["center_body_m"]).tolist() if red else None
            )
            result["after_placement"]["fusion_diagnostics"] = runtime.diagnostics
            np.savez_compressed(output / "placement_sensor.npz", **runtime.packet)
        except (RuntimeError, ValueError) as error:
            result["after_placement"]["observation_error"] = f"{type(error).__name__}: {error}"
        runtime.policy.head_command[:] = [0, 0.2, 0, 0]
        _execute(runtime, result["commands"], "stand", 0.6, "head_return")
        result["after_head_return"] = _phase_state(runtime, case["foot"])
        result["after_head_return"]["sensor_observation"] = None
        result["after_head_return"]["sensor_status"] = "not_acquired_one_render_per_case_budget"
        kick_start = float(runtime.data.time)
        _execute(runtime, result["commands"], f"kick_{case['foot']}", 0.5, "kick")
        _execute(runtime, result["commands"], "stand", 3.0, "recover")
        result["status"] = "completed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        (output / "error.log").write_text(traceback.format_exc())
    finally:
        if runtime is not None:
            result["kick_start_time_s"] = kick_start
            result["contacts"] = runtime.contacts
            result["simulation_seconds"] = float(runtime.data.time)
            if before is not None:
                result["evaluation"] = evaluate_kick(
                    "red",
                    case["foot"],
                    before,
                    Runtime.truth(runtime),
                    kick_start if kick_start is not None else float("nan"),
                    runtime.contacts,
                    runtime.rows,
                )
            runtime.save_trace()
            runtime.close()
    result["wall_seconds"] = time.monotonic() - started
    _write(output / "result.json", result)
    return result


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        content = path.read_bytes()
        (snapshot / path.name).write_bytes(content)
        hashes[path.name] = hashlib.sha256(content).hexdigest()
    _write(
        output / "predeclared_protocol.json",
        dict(
            total=18,
            cases=CASES,
            attempts_per_case=1,
            seed="17 + zero_based_index",
            classification="physical_simulation_reach_calibration_not_closed_loop_acceptance",
            fixture="body encoder FK foot x and outward-y converted using truth base; ground ball z fixed 0.035m",
            outward_convention="left foot +local_y; right foot -local_y",
            actions=[
                dict(head_command=[0, 0.7, 0, 0], skill="stand", seconds=1.0),
                dict(fixture="place_ball_then_capture_one_sensor_frame"),
                dict(head_command=[0, 0.2, 0, 0], skill="stand", seconds=0.6),
                dict(skill="selected_foot_kick", seconds=0.5),
                dict(skill="stand", seconds=3.0),
            ],
            head_tracking=False,
            maximum_rendered_observations_per_case=1,
            truth_use="fixture_and_independent_scoring_only_no_truth_based_action_branch",
            source_sha256=hashes,
        ),
    )
    started = time.monotonic()
    summary = dict(total=18, completed=0, strict_successes=0, foot_contacts=0, case_results=[], status="running")
    _write(output / "summary.json", summary)
    for index, case in enumerate(CASES):
        result = _case(output / f"case_{index:02d}_{case['foot']}", case, index)
        summary["case_results"].append(
            dict(
                **case,
                index=index,
                status=result["status"],
                error=result.get("error"),
                evaluation=result["evaluation"],
                after_placement_foot_m=result.get("after_placement", {}).get("true_ball_foot_m"),
                after_head_return_foot_m=result.get("after_head_return", {}).get("true_ball_foot_m"),
                sensor_red_detected=result.get("after_placement", {}).get("sensor_ball_foot_m") is not None,
                wall_seconds=result["wall_seconds"],
            )
        )
        summary["completed"] += 1
        summary["strict_successes"] += int(result["evaluation"]["success"])
        summary["foot_contacts"] += int(result["evaluation"].get("target_foot_contact", False))
        summary["wall_seconds"] = time.monotonic() - started
        _write(output / "summary.json", summary)
        print(json.dumps(summary["case_results"][-1]), flush=True)
    summary["status"] = "completed"
    summary["source_changed_files"] = [
        name
        for name, digest in hashes.items()
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest
    ]
    _write(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "case_results"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        MUJOCO_GL="egl",
        LIBGL_ALWAYS_SOFTWARE="1",
        LP_NUM_THREADS="2",
        __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    from .bootstrap import configure

    configure()
    run(args)


if __name__ == "__main__":
    main()
