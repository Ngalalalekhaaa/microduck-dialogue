"""Face RGB/ToF runtime with IMU/encoder-derived control observations."""

from functools import partial

import mujoco
import numpy as np
from microduck_deploy.policy import HOME, JOINT_NAMES

from strands_robots import JointState, Observation, Pose

from .rgb_tof_body import BodyEstimator
from .rgb_tof_feet import feet_from_encoders
from .rgb_tof_fusion import fuse_balls
from .rgb_tof_geometry import RGBToFConfig
from .rgb_tof_policy import SensorPolicy
from .rgb_tof_scene import build_sensor_scene, camera_from_encoders
from .rgb_tof_sensor import RGBToFSensor
from .runtime import Runtime


class SensorRuntime(Runtime):
    def __init__(self, output, *, config=None, pitch_deg=55.0, seed=17, noise_m=0.002, dropout=0.0):
        if not np.isfinite(noise_m) or noise_m < 0 or not np.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("Use finite nonnegative range noise and a dropout probability between zero and one.")
        self.config = config or RGBToFConfig(fovy_deg=100.0, range_sigma_m=noise_m)
        self.estimator = BodyEstimator()
        self.control_rows = []
        self.body_estimate = None
        self.packet = self.diagnostics = self.camera_pose = None
        self.foot_poses = None
        self.last_fused_target = None
        self.target_color = "red"
        self.observation_hook = None
        self.head_tracking = True
        super().__init__(
            output,
            scene_builder=partial(build_sensor_scene, fovy_deg=self.config.fovy_deg, pitch_deg=pitch_deg),
            policy_factory=SensorPolicy,
            camera_factory=partial(RGBToFSensor, config=self.config, seed=seed, noise_m=noise_m, dropout=dropout),
        )
        self.accel = int(self.model.sensor("imu_accel").adr[0])

    def reset(self, layout):
        self.estimator.reset()
        self.control_rows = []
        self.body_estimate = None
        self.cameras.last = None
        self.packet = self.camera_pose = None
        self.foot_poses = None
        self.last_fused_target = None
        self.head_tracking = True
        super().reset(layout)

    def _update_body(self):
        now = float(self.data.time)
        if self.body_estimate is not None and now <= self.body_estimate.timestamp + 1e-10:
            return self.body_estimate
        self.body_estimate = self.estimator.update(
            now,
            self.data.sensordata[self.gyro : self.gyro + 3],
            self.data.sensordata[self.accel : self.accel + 3],
            self.data.qpos[self.qi],
        )
        return self.body_estimate

    def _tick(self, skill):
        # Startup uses a fixed HOME target until accelerometer-based initialization.
        # No ground-truth posture is used to pick or terminate a movement.
        if self.body_estimate is None:
            if self.data.time >= 0.6:
                try:
                    self._update_body()
                except ValueError:
                    if self.data.time > 1.5:
                        raise RuntimeError(
                            "IMU did not settle for initialization; inspect the recorded startup."
                        ) from None
            if self.body_estimate is None:
                self.controller.q_target = HOME
                for _ in range(4):
                    self.controller.update()
                    mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
                self._record("imu_startup_hold")
                return
        super()._tick(skill)

    def _policy_observation(self):
        state = self._update_body()
        self.control_rows.append(
            (
                state.timestamp,
                state.angular_velocity.copy(),
                self.data.sensordata[self.accel : self.accel + 3].copy(),
                self.data.qpos[self.qi].copy(),
                state.joint_velocities.copy(),
                state.orientation_wxyz.copy(),
                self.policy.head_command.copy(),
                self.policy.last_action.copy(),
            )
        )
        return Observation(
            state.timestamp,
            JointState(JOINT_NAMES, self.data.qpos[self.qi].copy(), state.joint_velocities, np.zeros(len(JOINT_NAMES))),
            Pose(np.zeros(3), state.orientation_wxyz, angular_velocity=state.angular_velocity),
        )

    def _posture_failed(self, skill, tilt):
        return self._update_body().tilt_deg > 50

    def save_trace(self):
        super().save_trace()
        names = (
            "time",
            "gyro",
            "accelerometer",
            "joint_encoders",
            "estimated_joint_velocities",
            "estimated_orientation",
            "head_command",
            "previous_action",
        )
        np.savez_compressed(
            self.output / "control_sensors.npz",
            **{name: np.asarray([row[i] for row in self.control_rows]) for i, name in enumerate(names)},
        )

    def snapshot(self):
        with self.lock:
            state = self._update_body()
            return dict(
                time_s=float(self.data.time),
                yaw=state.yaw,
                tilt_deg=state.tilt_deg,
                stationary=state.stationary,
                skill=self.rows[-1][3] if self.rows else "stand",
            )

    def observe(self):
        with self.lock:
            state = self.snapshot()
            packet = self.cameras.capture(self.data.qpos.copy(), self.data.qvel.copy(), state["time_s"])
            new_frame = self.packet is None or packet["time_s"] != self.packet["time_s"]
            if new_frame:
                pose = camera_from_encoders(self.model, self.data.qpos[self.qi].copy())
                self.foot_poses = feet_from_encoders(self.model, self.data.qpos[self.qi].copy())
            else:
                pose = self.camera_pose
        valid = packet["valid"].copy()
        sensor_age = state["time_s"] - packet["time_s"]
        rgb_time = packet.get("rgb_time_s", packet["time_s"])
        fresh = bool(
            np.isfinite([sensor_age, rgb_time]).all()
            and 0 <= sensor_age <= 0.12
            and abs(rgb_time - packet["time_s"]) <= 1e-9
        )
        if not fresh:
            valid[:] = False
        detections, diagnostics = fuse_balls(
            packet["rgb"],
            packet["ranges_m"],
            config=self.config,
            valid=valid,
            camera_position_body=pose[0],
            camera_rotation_body=pose[1],
        )
        targets = [d for d in detections if d["color"] == self.target_color]
        prior = getattr(self, "last_fused_target", None)
        if not targets and prior is not None:
            from .rgb_tof_tracking import recover_ball

            recovered, recovery = recover_ball(
                packet["rgb"],
                packet["ranges_m"],
                config=self.config,
                pose=pose,
                valid=valid,
                locked=prior[1],
                lock_age_s=packet["time_s"] - prior[0],
            )
            diagnostics["near_ball_recovery"] = recovery
            if recovered is not None:
                targets = [recovered]
                detections.append(recovered)
                diagnostics["associations"].append(recovery)
        if new_frame and fresh and targets:
            self.last_fused_target = (packet["time_s"], max(targets, key=lambda d: d["confidence"]))
        self.frame_seq += 1
        observation = dict(
            frame_id=f"frame_{self.frame_seq:05d}",
            **state,
            detections=detections,
            sensor_time_s=packet["time_s"],
            sensor_age_s=sensor_age,
            feet={
                foot: dict(
                    position_body_m=pose.position_body_m.tolist(),
                    rotation_body_from_foot=pose.rotation_body_from_foot.tolist(),
                )
                for foot, pose in self.foot_poses.items()
            },
            source="rgb_plus_64_radial_tof_ranges_and_imu_encoders",
        )
        self.packet, self.diagnostics, self.camera_pose = packet, diagnostics, pose
        if self.observation_hook:
            self.observation_hook(observation, packet, diagnostics)
        if self.head_tracking and new_frame and fresh:
            candidates = diagnostics["associations"] + diagnostics["rejected_rgb_candidates"]
            candidates = [
                item
                for item in candidates
                if item["color"] == self.target_color
                and item.get("reason")
                in (None, "no_consistent_tof_support", "tof_surface_fit_inconsistent", "fresh_rgb_arc_and_tof_verified")
            ]
            if candidates:
                candidate = max(candidates, key=lambda item: item["pixels"])
                box = candidate["box_uv"]
                vertical = (box[1] + box[3]) / 2
                error = np.arctan((vertical - (self.config.height - 1) / 2) / self.config.focal_px)
                self.policy.head_command[1] = np.clip(self.policy.head_command[1] + 0.45 * error, -0.7, 0.7)
                horizontal = (box[0] + box[2]) / 2
                error = np.arctan((horizontal - (self.config.width - 1) / 2) / self.config.focal_px)
                self.policy.head_command[2] = np.clip(self.policy.head_command[2] - 0.35 * error, -0.7, 0.7)
        return observation, {"face_rgb": packet["rgb"]}, {}
