"""Continuous simulated body with shared walking/kick state and RGB-D snapshots."""

import io
import threading
import time
from contextlib import redirect_stdout

import mujoco
import numpy as np
from microduck_deploy.policy import HOME, JOINT_NAMES
from render_hd1910_controller import load_scene

from strands_robots import JointState, Observation, Pose
from strands_robots._gl import gl_thread

from .perception import detect_balls
from .scene import CAMERAS, COLORS, build_scene
from .skills import TWISTS, SkillPolicy

DT = 0.02


class Cameras:
    @gl_thread
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, width=320, height=240)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0

    @gl_thread
    def capture(self, qpos, qvel, time_s):
        self.data.qpos[:], self.data.qvel[:], self.data.time = qpos, qvel, time_s
        mujoco.mj_forward(self.model, self.data)
        images, depths = {}, {}
        for name in CAMERAS:
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera=name, scene_option=self.option)
            images[name] = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            depths[name] = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return images, depths

    @gl_thread
    def close(self):
        self.renderer.close()


class Runtime:
    def __init__(
        self, output, *, scene_builder=build_scene, policy_factory=SkillPolicy, camera_factory=Cameras, variant="hd1910"
    ):
        self.output = output
        self.scene = output / "scene.xml"
        scene_builder(self.scene)
        log = io.StringIO()
        with redirect_stdout(log):
            self.model, self.data, self.controller, self.physics = load_scene(
                self.scene, variant, timestep=0.005, lag_steps=4
            )
        (output / "initialization.log").write_text(log.getvalue())
        self.model.opt.iterations, self.model.opt.ls_iterations = 30, 50
        self.body = self.model.body("trunk_base").id
        root = self.model.joint("trunk_base_freejoint")
        self.qa, self.va = int(root.qposadr[0]), int(root.dofadr[0])
        ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.qi, self.vi = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        self.limits = self.model.jnt_range[ids]
        self.gyro = int(self.model.sensor("imu_ang_vel").adr[0])
        self.ball_q = {c: int(self.model.joint(f"ball_{c}_free").qposadr[0]) for c in COLORS}
        self.ball_v = {c: int(self.model.joint(f"ball_{c}_free").dofadr[0]) for c in COLORS}
        self.ball_geoms = {self.model.geom(f"ball_{c}_geom").id: c for c in COLORS}
        self.floor = self.model.geom("floor").id
        self.camera_rotations = {}
        for name in CAMERAS:
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, self.model.camera(name).quat)
            self.camera_rotations[name] = rotation.reshape(3, 3)
        self.policy = policy_factory()
        self.cameras = camera_factory(self.model)
        self.lock = threading.RLock()
        self.worker = None
        self.shutdown = threading.Event()
        self.cancel = threading.Event()
        self.pulse_done = threading.Event()
        self.remaining = 0
        self.active_skill = "stand"
        self.failure = None
        self.last_execution = {"started": False, "completed": False, "start_time_s": None, "end_time_s": None}
        self._requested_ticks = self._executed_ticks = 0
        self.rows, self.contacts = [], []
        self.frame_seq = 0
        self._interactive_frames = None

    def enable_interactive_pacing(self):
        """Before desktop tasks, enable 1x ticks and independent presentation snapshots."""
        from .interactive_frames import InteractiveFrames

        with self.lock:
            if self.worker is not None:
                raise RuntimeError("Use interactive pacing without a background real-time physics worker.")
            if self._interactive_frames is None:
                self._interactive_frames = InteractiveFrames(DT)
                self._interactive_frames.publish(Runtime.snapshot(self))

    def display_snapshot(self):
        """Return the last completed tick without acquiring any world/session lock."""
        frames = self._interactive_frames
        if frames is None:
            raise RuntimeError("Call enable_interactive_pacing() before opening the desktop viewer.")
        return frames.snapshot()

    def reset(self, layout):
        if self.worker is not None:
            raise RuntimeError("Stop the real-time worker before resetting a scene.")
        if set(layout) != set(COLORS) or any(
            np.asarray(xy).shape != (2,) or not np.isfinite(xy).all() for xy in layout.values()
        ):
            raise ValueError("Provide finite x/y positions for exactly red, yellow and blue.")
        self.cancel.clear()
        self.remaining = 0
        self.active_skill = "stand"
        self.pulse_done.clear()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qa : self.qa + 7] = [0, 0, 0.12, 1, 0, 0, 0]
        self.data.qpos[self.qi] = HOME
        for color, xy in layout.items():
            self.data.qpos[self.ball_q[color] : self.ball_q[color] + 7] = [*xy, 0.035, 1, 0, 0, 0]
        self.controller.reset(self.data.qpos)
        self.controller.q_target = HOME
        self.policy.reset()
        self.failure = None
        self.rows, self.contacts = [], []
        self.frame_seq = 0
        self.target_excess = np.zeros(14)
        mujoco.mj_forward(self.model, self.data)
        self._record("initial")
        self.execute("stand", 2.0)

    def _record(self, skill):
        d = self.data
        rotation = d.xmat[self.body].reshape(3, 3)
        tilt = float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1, 1))))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.rows.append(
            (
                float(d.time),
                d.qpos.copy(),
                d.qvel.copy(),
                skill,
                tilt,
                yaw,
                np.array([d.qpos[a : a + 3].copy() for a in self.ball_q.values()]),
                d.qpos[self.qa : self.qa + 3].copy(),
                self.policy.last_action.copy(),
                self.controller.q_target.copy(),
            )
        )
        return tilt

    def _policy_observation(self):
        d = self.data
        return Observation(
            float(d.time),
            JointState(JOINT_NAMES, d.qpos[self.qi], d.qvel[self.vi], d.actuator_force[self.controller.act_indexes]),
            Pose(d.xpos[self.body], d.xquat[self.body], angular_velocity=d.sensordata[self.gyro : self.gyro + 3]),
        )

    def _tick(self, skill):
        with self.lock:
            if self.failure:
                raise RuntimeError(self.failure)
            d = self.data
            obs = self._policy_observation()
            self.policy.skill = skill
            targets = self.policy.act(obs)
            target = np.array([targets[name] for name in JOINT_NAMES])
            self.target_excess = np.maximum(
                self.target_excess, np.maximum(self.limits[:, 0] - target, target - self.limits[:, 1]).clip(min=0)
            )
            self.controller.q_target = target
            for _ in range(4):
                self._before_physics_step(skill)
                self.controller.update()
                mujoco.mj_step(self.model, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    self.failure = "nonfinite_state"
                    raise RuntimeError(self.failure)
                for index in range(d.ncon):
                    contact = d.contact[index]
                    pair = (int(contact.geom1), int(contact.geom2))
                    for geom in pair:
                        if geom not in self.ball_geoms:
                            continue
                        other = pair[1] if pair[0] == geom else pair[0]
                        if other == self.floor:
                            continue
                        force = np.zeros(6)
                        mujoco.mj_contactForce(self.model, d, index, force)
                        if force[0] > 1e-5:
                            self.contacts.append(
                                {
                                    "time_s": float(d.time - 0.005),
                                    "skill": skill,
                                    "color": self.ball_geoms[geom],
                                    "other_color": self.ball_geoms.get(other),
                                    "geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                                    or f"geom_{other}",
                                    "normal_force_n": float(force[0]),
                                }
                            )
            mujoco.mj_forward(self.model, d)
            tilt = self._record(skill)
            if self._interactive_frames is not None:
                self._interactive_frames.publish(Runtime.snapshot(self))
            if self._posture_failed(skill, tilt):
                self.failure = "fallen"
                raise RuntimeError(self.failure)

    def _before_physics_step(self, skill):
        """Optional simulated peripheral control, before every physics step."""

    def _posture_failed(self, skill, tilt):
        return tilt > 50 or self.data.qpos[self.qa + 2] < 0.065

    def supports_skill(self, skill):
        return skill in TWISTS or skill in ("kick_left", "kick_right")

    def _pulse_tick(self, skill):
        """Record physics advanced by this pulse, excluding idle balance ticks."""
        start = float(self.data.time)
        try:
            self._tick(skill)
        finally:
            end = float(self.data.time)
            if end > start:
                if not self.last_execution["started"]:
                    self.last_execution.update(started=True, start_time_s=start)
                self.last_execution["end_time_s"] = end
        self._executed_ticks += 1
        self.last_execution["completed"] = self._executed_ticks == self._requested_ticks

    def execute(self, skill, seconds):
        with self.lock:
            self.last_execution = {"started": False, "completed": False, "start_time_s": None, "end_time_s": None}
            self._requested_ticks = self._executed_ticks = 0
            if self.failure:
                raise RuntimeError(self.failure)
            if self.shutdown.is_set() or (self.worker is not None and not self.worker.is_alive()):
                raise RuntimeError("Simulation is closed or its worker has exited.")
            if not self.supports_skill(skill):
                raise ValueError(f"Unsupported skill {skill}.")
            if (
                not np.isfinite(seconds)
                or not 0.02 <= seconds <= 3
                or not np.isclose(seconds / DT, round(seconds / DT))
            ):
                raise ValueError("Use 0.02..3 seconds in multiples of 0.02.")
            ticks = self._requested_ticks = round(seconds / DT)
            if self.cancel.is_set():
                self._stop_locked()
                return False
            if self.worker is not None:
                self.pulse_done.clear()
                self.active_skill, self.remaining = skill, ticks
        if self.worker is None:
            for _ in range(ticks):
                with self.lock:
                    if self.cancel.is_set():
                        self._stop_locked()
                        return False
                    frames = getattr(self, "_interactive_frames", None)
                    tick_started = frames.clock() if frames is not None else None
                    self._pulse_tick(skill)
                if frames is not None:
                    frames.wait_after_tick(tick_started, self.cancel)
            return self.last_execution["completed"]
        if not self.pulse_done.wait(seconds + 10):
            self.request_stop()
            raise TimeoutError("Simulation worker did not finish its pulse.")
        if self.failure:
            raise RuntimeError(self.failure)
        return self.last_execution["completed"]

    def start_realtime(self):
        if getattr(self, "_interactive_frames", None) is not None:
            raise RuntimeError("Interactive pacing keeps physics lockstep; do not start a background physics worker.")
        if self.worker is not None:
            return
        self.shutdown.clear()

        def run():
            deadline = time.monotonic()
            try:
                while not self.shutdown.is_set():
                    with self.lock:
                        active = self.remaining > 0 and not self.cancel.is_set()
                        skill = self.active_skill if active else "stand"
                        if active:
                            self._pulse_tick(skill)
                            self.remaining = self._requested_ticks - self._executed_ticks
                        else:
                            self._tick(skill)
                        if self.remaining <= 0 or self.cancel.is_set():
                            self._stop_locked()
                    deadline += DT
                    self.shutdown.wait(max(0, deadline - time.monotonic()))
            except Exception as exc:
                self.failure = str(exc)
            finally:
                self.pulse_done.set()

        self.worker = threading.Thread(target=run, name="microduck-simulation", daemon=True)
        self.worker.start()

    def request_stop(self):
        self.cancel.set()
        if not self.lock.acquire(blocking=False):
            return
        try:
            self._stop_locked()
        finally:
            self.lock.release()

    def _stop_locked(self):
        """Call with the world lock; a busy worker observes the cancel latch next tick."""
        self.remaining = 0
        self.active_skill = "stand"
        self.pulse_done.set()

    def snapshot(self):
        with self.lock:
            d = self.data
            rotation = d.xmat[self.body].reshape(3, 3)
            gyro = d.sensordata[self.gyro : self.gyro + 3].copy()
            return {
                "time_s": float(d.time),
                "skill": self.rows[-1][3] if self.rows else "stand",
                "qpos": d.qpos.copy(),
                "qvel": d.qvel.copy(),
                "yaw": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                "tilt_deg": float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
                "stationary": bool(
                    np.linalg.norm(d.qvel[self.va : self.va + 2]) < 0.025 and np.linalg.norm(gyro) < 0.3
                ),
            }

    def observe(self):
        snap = self.snapshot()
        images, depths = self.cameras.capture(snap["qpos"], snap["qvel"], snap["time_s"])
        detected = []
        for name, calibration in CAMERAS.items():
            detected.extend(
                detect_balls(
                    images[name],
                    depths[name],
                    fovy_deg=calibration["fovy_deg"],
                    camera_position_body=calibration["position"],
                    camera_rotation_body=self.camera_rotations[name],
                    camera_name=name,
                )
            )
        # One ball of each color: prefer the better-conditioned available view.
        merged = {}
        for detection in detected:
            color = detection["color"]
            if color not in merged or detection["confidence"] > merged[color]["confidence"]:
                merged[color] = detection
        self.frame_seq += 1
        observation = {
            "frame_id": f"frame_{self.frame_seq:05d}",
            "time_s": snap["time_s"],
            "detections": list(merged.values()),
            "yaw": snap["yaw"],
            "stationary": snap["stationary"],
            "tilt_deg": snap["tilt_deg"],
        }
        return observation, images, depths

    def truth(self):
        """Independent evaluation only. Never pass this to perception/backend/planner."""
        with self.lock:
            d = self.data
            return {color: d.qpos[a : a + 3].copy() for color, a in self.ball_q.items()}

    def save_trace(self):
        with self.lock:
            names = (
                "time",
                "qpos",
                "qvel",
                "skill",
                "tilt_deg",
                "yaw",
                "balls_xyz",
                "trunk_xyz",
                "raw_action",
                "target",
            )
            arrays = {name: np.asarray([row[i] for row in self.rows]) for i, name in enumerate(names)}
        np.savez_compressed(self.output / "trajectory.npz", **arrays)

    def close(self):
        self.shutdown.set()
        if self.worker is not None:
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                raise RuntimeError("Simulation worker did not stop.")
            self.worker = None
        self.cameras.close()
