"""Reproducible sensor-only navigation/kicking, scored outside the controller."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


class SensorServo:
    """Wait for head pointing to acquire range support before moving the feet."""

    def __init__(self, runtime, *, kick_strategy="narrow_body"):
        self.runtime = runtime
        if kick_strategy not in ("narrow_body", "foot_relative"):
            raise ValueError("Select narrow_body or foot_relative kick strategy.")
        self.kick_strategy = kick_strategy
        self.decisions = []
        self.last_target = None

    def execute(self, skill, seconds, obs, phase="navigate", heading=None):
        self.runtime.policy.motion.visual_heading = heading
        self.decisions.append(
            dict(
                time_s=obs["time_s"],
                phase=phase,
                skill=skill,
                seconds=seconds,
                heading=heading,
                detections=obs.get("detections", []),
                feet=obs.get("feet"),
            )
        )
        return self.runtime.execute(skill, seconds)

    def observe(self):
        for attempt in range(10):
            obs, _, _ = self.runtime.observe()
            targets = [d for d in obs["detections"] if d["color"] == self.runtime.target_color]
            if targets:
                self.last_target = max(targets, key=lambda d: d["confidence"])["center_body_m"]
                return obs
            candidates = self.runtime.diagnostics["rejected_rgb_candidates"]
            visible = any(
                d["color"] == self.runtime.target_color
                and d["reason"] in ("no_consistent_tof_support", "tof_surface_fit_inconsistent")
                for d in candidates
            )
            near = self.last_target is not None and self.last_target[0] < 0.22
            if not visible and not near:
                return obs
            if attempt == 9:
                raise RuntimeError(
                    "Target has no reliable RGB/ToF observation after stationary reacquisition; task stopped."
                )
            self.execute("stand", 0.2, obs, "head_acquire")
        return obs

    def navigate(self):
        from .toy_motion import ToyMotion

        target = dict(category="ball", attributes=[dict(name="color", value="red")], entity_id=None)

        def pulse(skill, seconds, obs):
            return self.execute(skill, seconds, obs, heading=self.runtime.policy.motion.visual_heading)

        return ToyMotion(self.runtime, self.observe, pulse).execute(
            dict(type="navigate", target=target, standoff_m=0.25)
        )

    def kick(self):
        from .rgb_tof_planner import SensorKickPlanner

        planner = SensorKickPlanner("red", foot_relative=self.kick_strategy == "foot_relative")
        while True:
            obs = self.observe()
            decision = planner.decide(obs)
            if decision is None:
                self.execute("stand", 0.8, obs, "failure_settle")
                return dict(success=False, reason=planner.status)
            handoff = None
            if decision.phase == "kick":
                obs, handoff = self.prepare_kick(obs, planner.foot)
                if not handoff["accepted"]:
                    return dict(success=False, reason="kick_handoff_rejected", handoff=handoff)
            heading = getattr(decision, "heading", None)
            if not self.execute(decision.skill, decision.seconds, obs, decision.phase, heading):
                return dict(success=False, reason="execution_cancelled")
            if decision.phase == "kick":
                kick_start = self.runtime.last_execution["start_time_s"]
                for _ in range(10):
                    obs, _, _ = self.runtime.observe()
                    if not self.execute("stand", 0.3, obs, "recover"):
                        return dict(
                            success=False, reason="recovery_cancelled", foot=planner.foot, kick_start=kick_start
                        )
                return dict(
                    success=True, reason="kick_executed", foot=planner.foot, kick_start=kick_start, handoff=handoff
                )

    def prepare_kick(self, observation, foot):
        from .rgb_tof_handoff import check_handoff

        locked = max((d for d in observation["detections"] if d["color"] == "red"), key=lambda d: d["confidence"])
        lock_time = observation["sensor_time_s"]
        self.runtime.head_tracking = False
        self.runtime.policy.head_command[:] = [0, 0.2, 0, 0]
        for attempt in range(6):
            if not self.execute("stand", 0.2, observation, "head_return_for_kick"):
                return observation, dict(accepted=False, reason="cancelled")
            observation, _, _ = self.runtime.observe()
            check = check_handoff(
                self.runtime.packet["rgb"],
                config=self.runtime.config,
                pose=self.runtime.camera_pose,
                locked=locked,
                lock_time=lock_time,
                observation=observation,
                foot=foot,
                foot_pose=observation["feet"][foot] if self.kick_strategy == "foot_relative" else None,
            )
            if attempt >= 2 and check["accepted"]:
                return observation, check
        return observation, check


def forbidden_truth():
    raise AssertionError("Controller attempted to read simulator truth.")


def score(runtime, mode, result, before):
    """This function runs only after all controller decisions have finished."""
    from .runtime import Runtime
    from .tasks import evaluate_kick

    after = Runtime.truth(runtime)
    if mode == "kick" and result.get("kick_start") is not None:
        return evaluate_kick("red", result["foot"], before, after, result["kick_start"], runtime.contacts, runtime.rows)
    rotation = runtime.data.xmat[runtime.body].reshape(3, 3)
    point = rotation.T @ (after["red"] - runtime.data.xpos[runtime.body])
    distance, bearing = float(np.linalg.norm(point[:2])), float(np.arctan2(point[1], point[0]))
    final_speed = float(np.linalg.norm(runtime.rows[-1][2][:2]))
    valid_stance = max(row[4] for row in runtime.rows) < 30 and runtime.rows[-1][7][2] > 0.09
    clean = not runtime.contacts
    return dict(
        success=bool(
            mode == "navigate"
            and result["success"]
            and abs(distance - 0.25) <= 0.035
            and abs(bearing) < 0.2
            and final_speed < 0.025
            and valid_stance
            and clean
        ),
        target_body_m=point.tolist(),
        distance_m=distance,
        bearing_rad=bearing,
        final_planar_speed_m_s=final_speed,
        valid_stance=bool(valid_stance),
        no_robot_ball_contacts=clean,
        source="independent_simulation_ground_truth_not_controller_input",
    )


def run(args):
    from .rgb_tof_recording import Recorder
    from .rgb_tof_runtime import SensorRuntime
    from .runtime import Runtime

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    layout = dict(red=[args.x, args.y], yellow=[0.65, 0.45], blue=[0.65, -0.45])
    runtime = SensorRuntime(output, seed=args.seed, noise_m=args.noise, dropout=args.dropout)
    recorder = Recorder(output, runtime.config)
    servo = SensorServo(runtime, kick_strategy=args.kick_strategy)
    metadata = dict(
        mode=args.mode,
        kick_strategy=args.kick_strategy,
        layout=layout,
        seed=args.seed,
        range_noise_m=args.noise,
        dropout=args.dropout,
        calibration=asdict(runtime.config),
        camera_home_body_m=[0.14, 0, 0.16],
        camera_pitch_deg=55,
        calibration_status="provisional_simulation_assumptions_not_measured_hardware",
        range_model="nearest_cluster_of_at_least_2_of_25_subrays_within_12mm_per_zone",
        controller_inputs=["rgb", "64_radial_surface_ranges", "imu", "joint_encoders"],
        privileged_truth_guard=True,
        near_ball_recovery="fresh_RGB_arc_plus_at_least_two_current_ToF_zones_prior_search_age_at_most_0.8s",
        policy="existing_cpu_onnx_walk_and_kick_no_new_training",
        final_handoff="head +0.2rad; <=1.4s ToF memory checked by fresh known-radius RGB and IMU settle",
        policy_hashes=runtime.policy.hashes,
    )
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    wall_start = time.monotonic()
    result = dict(success=False, reason="not_started")
    try:
        runtime.reset(layout)
        before = Runtime.truth(runtime)
        runtime.truth = forbidden_truth
        runtime.observation_hook = recorder
        try:
            result = servo.navigate() if args.mode == "navigate" else servo.kick()
        except (RuntimeError, ValueError, AssertionError) as error:
            result = dict(success=False, reason=f"{type(error).__name__}: {error}")
        evaluation = score(runtime, args.mode, result, before)
        report = dict(
            controller=result,
            evaluation=evaluation,
            wall_seconds=time.monotonic() - wall_start,
            simulation_seconds=runtime.snapshot()["time_s"],
            decisions=len(servo.decisions),
        )
        (output / "result.json").write_text(json.dumps(report, indent=2))
        (output / "decisions.json").write_text(json.dumps(servo.decisions, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        runtime.save_trace()
        recorder.save()
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("navigate", "kick"), required=True)
    parser.add_argument("--kick-strategy", choices=("narrow_body", "foot_relative"), default="foot_relative")
    parser.add_argument("--x", type=float, default=0.4)
    parser.add_argument("--y", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--noise", type=float, default=0.002)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()
    # Preserve the GPU for the separately running language-model experiment.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["LP_NUM_THREADS"] = "2"
    from .bootstrap import configure

    configure()
    run(args)


if __name__ == "__main__":
    main()
