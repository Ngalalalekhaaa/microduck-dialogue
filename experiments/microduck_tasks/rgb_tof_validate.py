"""Predeclared static RGB/ToF layouts and missing/inconsistent-range probes.

Only the scorer reads simulator truth, after all observations and head motions
have finished. These are static sensor checks, not navigation or kick trials.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _forbidden_truth():
    raise AssertionError("Static controller attempted to read simulator truth")


def _score(runtime, observation):
    from .runtime import Runtime

    rotation = runtime.data.xmat[runtime.body].reshape(3, 3)
    truth_body = rotation.T @ (Runtime.truth(runtime)["red"] - runtime.data.xpos[runtime.body])
    tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))
    red = [item for item in observation["detections"] if item["color"] == "red"]
    best = max(red, key=lambda item: item["confidence"]) if red else None
    error = (np.asarray(best["center_body_m"]) - truth_body) * 1000 if best else None
    return dict(
        source="independent_simulation_truth_after_sensor_actions_not_controller_input",
        red_detected=best is not None,
        truth_center_body_m=truth_body.tolist(),
        detected_center_body_m=best["center_body_m"] if best else None,
        error_vector_mm=error.tolist() if best else None,
        euclidean_error_mm=float(np.linalg.norm(error)) if best else None,
        supporting_zones=best["tof_zone_count"] if best else 0,
        true_tilt_deg=tilt,
        estimated_tilt_deg=observation["tilt_deg"],
        tilt_absolute_error_deg=abs(tilt - observation["tilt_deg"]),
    )


def _case(output, index, x, y, args):
    from .rgb_tof_fusion import fuse_balls
    from .rgb_tof_geometry import RGBToFConfig
    from .rgb_tof_runtime import SensorRuntime

    output.mkdir()
    layout = dict(red=[x, y], yellow=[0.65, 0.45], blue=[0.65, -0.45])
    result = dict(index=index, layout=layout, status="failed", observations=[])
    runtime = None
    try:
        runtime = SensorRuntime(
            output,
            config=RGBToFConfig(fovy_deg=100, range_sigma_m=args.noise),
            seed=args.seed + index,
            noise_m=args.noise,
            dropout=args.dropout,
        )
        runtime.reset(layout)
        runtime.truth = _forbidden_truth
        for attempt in range(10):
            observation, _, _ = runtime.observe()
            result["observations"].append(
                dict(
                    attempt=attempt,
                    observation=observation,
                    diagnostics=runtime.diagnostics,
                    head_command=runtime.policy.head_command.tolist(),
                )
            )
            if any(item["color"] == "red" for item in observation["detections"]):
                break
            if attempt < 9:
                runtime.execute("stand", 0.2)
        packet, pose = runtime.packet, runtime.camera_pose
        common = dict(config=runtime.config, camera_position_body=pose[0], camera_rotation_body=pose[1])
        missing, missing_diag = fuse_balls(
            packet["rgb"], packet["ranges_m"], valid=np.zeros_like(packet["valid"]), **common
        )
        shifted, shifted_diag = fuse_balls(packet["rgb"], packet["ranges_m"] + 0.3, valid=packet["valid"], **common)
        result["missing_ranges"] = dict(
            red_rejected=not any(item["color"] == "red" for item in missing),
            detections=missing,
            diagnostics=missing_diag,
        )
        result["ranges_shifted_plus_0_3m"] = dict(
            red_rejected=not any(item["color"] == "red" for item in shifted),
            detections=shifted,
            diagnostics=shifted_diag,
        )
        result["evaluation"] = _score(runtime, observation)
        result["status"] = "completed"
        np.savez_compressed(
            output / "final_sensor_packet.npz",
            rgb=packet["rgb"],
            ranges_m=packet["ranges_m"],
            valid=packet["valid"],
            time_s=packet["time_s"],
            camera_position_body=pose[0],
            camera_rotation_body=pose[1],
        )
    except (RuntimeError, ValueError, AssertionError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if runtime is not None:
            runtime.save_trace()
            runtime.close()
    _write(output / "result.json", result)
    return result


def run(args):
    from .rgb_tof_geometry import RGBToFConfig

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    layouts = [(x, y) for x in (0.14, 0.25, 0.4, 0.6) for y in (-0.04, 0.04)]
    sources = sorted(Path(__file__).parent.glob("rgb_tof_*.py"))
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for path in sources:
        content = path.read_bytes()
        (snapshot / path.name).write_bytes(content)
        source_hashes[path.name] = hashlib.sha256(content).hexdigest()
    _write(
        output / "predeclared_protocol.json",
        dict(
            layouts=[dict(red=[x, y], yellow=[0.65, 0.45], blue=[0.65, -0.45]) for x, y in layouts],
            maximum_observations_per_layout=10,
            stand_between_observations_s=0.2,
            stop_observing_on_first_red_detection=True,
            noise_m=args.noise,
            dropout=args.dropout,
            seed=args.seed,
            case_seed="seed + zero_based_case_index",
            missing_range_test="same packet with valid set false in every zone",
            inconsistent_range_test="same packet with every range increased by 0.3 m",
            calibration=asdict(RGBToFConfig(fovy_deg=100, range_sigma_m=args.noise)),
            calibration_status="provisional_simulation_assumptions_not_measured_hardware",
            source_sha256=source_hashes,
        ),
    )
    start = time.monotonic()
    cases = []
    for index, (x, y) in enumerate(layouts):
        case = _case(output / f"case_{index:02d}", index, x, y, args)
        cases.append(case)
        print(json.dumps(dict(case=index, status=case["status"], evaluation=case.get("evaluation"))), flush=True)
    completed = [case for case in cases if case["status"] == "completed"]
    detected = [case for case in completed if case["evaluation"]["red_detected"]]
    errors = [case["evaluation"]["euclidean_error_mm"] for case in detected]
    summary = dict(
        total_cases=len(cases),
        completed_cases=len(completed),
        red_detected_cases=len(detected),
        max_position_error_mm=max(errors) if errors else None,
        mean_position_error_mm=float(np.mean(errors)) if errors else None,
        max_tilt_error_deg=max((case["evaluation"]["tilt_absolute_error_deg"] for case in completed), default=None),
        missing_ranges_rejected_cases=sum(case["missing_ranges"]["red_rejected"] for case in completed),
        shifted_ranges_rejected_cases=sum(case["ranges_shifted_plus_0_3m"]["red_rejected"] for case in completed),
        wall_seconds=time.monotonic() - start,
        classification="static_simulation_sensor_validation",
        failures=[
            dict(index=case["index"], error=case.get("error")) for case in cases if case["status"] != "completed"
        ],
    )
    _write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--noise", type=float, default=0.002)
    parser.add_argument("--dropout", type=float, default=0.0)
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
