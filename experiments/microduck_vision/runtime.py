"""Independent CPU BAM execution with Strands Policy; no hardware or Robot.run driver."""

import io
from contextlib import redirect_stdout
from pathlib import Path

import mujoco
import numpy as np
from bootstrap import MODEL_PATH, ROBOT_ASSETS
from microduck_deploy.policy import HOME, JOINT_NAMES
from planner import validate_command
from policy import WalkingPolicy
from render_hd1910_controller import load_scene
from vision import detect_target

from strands_robots import JointState, Observation, Pose
from strands_robots._gl import gl_thread

# Virtual bracket in front of the head: the stock asset has no navigation camera.
CAMERA_POSITION = np.array([0.14, 0.0, 0.16])
CAMERA_FOV = 85.0
DT = 0.02
TWISTS = {
    "stand": (0.0, 0.0, 0.0),
    "move_forward": (0.2, 0.0, 0.0),
    "move_backward": (-0.2, 0.0, 0.0),
    "move_left": (0.0, 0.2, 0.0),
    "move_right": (0.0, -0.2, 0.0),
}


def build_scene(path: Path):
    spec = mujoco.MjSpec.from_file(str(ROBOT_ASSETS / "scene.xml"))
    spec.meshdir = str(ROBOT_ASSETS / "assets")
    # Camera local X points to robot right; Y points up/forward at a 45deg pitch.
    q = np.sqrt(0.5)
    spec.body("trunk_base").add_camera(
        name="navigation", pos=CAMERA_POSITION, xyaxes=[0, -1, 0, q, 0, q], fovy=CAMERA_FOV
    )
    spec.worldbody.add_geom(
        name="navigation_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.4, 0.1, 0.003],
        size=[0.025, 0.003, 0],
        rgba=[1, 0.015, 0.015, 1],
        contype=0,
        conaffinity=0,
        group=1,
    )
    spec.compile()
    path.write_text(spec.to_xml())


class Cameras:
    @gl_thread
    def __init__(self, model):
        self.model = model
        self.renderer = mujoco.Renderer(model, width=320, height=240)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0

    @gl_thread
    def rgbd(self, data):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera="navigation", scene_option=self.option)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    @gl_thread
    def close(self):
        self.renderer.close()


class Runtime:
    def __init__(self, output: Path):
        output.mkdir(parents=True, exist_ok=True)
        scene = output / "scene.xml"
        build_scene(scene)
        log = io.StringIO()
        with redirect_stdout(log):
            self.model, self.data, self.controller, self.physics = load_scene(
                scene, "hd1910", timestep=0.005, lag_steps=4
            )
        (output / "initialization.log").write_text(log.getvalue())
        self.model.opt.iterations, self.model.opt.ls_iterations = 30, 50
        self.body = self.model.body("trunk_base").id
        self.marker = self.model.geom("navigation_marker").id
        self.qa = int(self.model.joint("trunk_base_freejoint").qposadr[0])
        ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.qpos_ids, self.qvel_ids = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        self.gyro = int(self.model.sensor("imu_ang_vel").adr[0])
        camera = self.model.camera("navigation")
        self.camera_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(self.camera_rotation, camera.quat)
        self.camera_rotation = self.camera_rotation.reshape(3, 3)
        self.policy = WalkingPolicy(MODEL_PATH)
        self.cameras = Cameras(self.model)
        self.trace = []

    def reset(self, target_xy, *, hidden=False):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qa : self.qa + 7] = [0, 0, 0.12, 1, 0, 0, 0]
        self.data.qpos[self.qpos_ids] = HOME
        self.model.geom_pos[self.marker] = [*target_xy, 0.003]
        self.model.geom_rgba[self.marker] = [0.15, 0.15, 0.15, 1] if hidden else [1, 0.015, 0.015, 1]
        self.controller.reset(self.data.qpos)
        self.controller.q_target = HOME
        self.policy.reset()
        self.trace = []
        mujoco.mj_forward(self.model, self.data)
        self.hold(2.0)

    def tick(self, twist, label):
        d = self.data
        obs = Observation(
            float(d.time),
            JointState(JOINT_NAMES, d.qpos[self.qpos_ids], d.qvel[self.qvel_ids], d.actuator_force),
            Pose(d.xpos[self.body], d.xquat[self.body], angular_velocity=d.sensordata[self.gyro : self.gyro + 3]),
        )
        self.policy.command[:] = twist
        targets = self.policy.act(obs)
        self.controller.q_target = [targets[name] for name in JOINT_NAMES]
        for _ in range(4):
            self.controller.update()
            mujoco.mj_step(self.model, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("nonfinite_state")
        # mj_step integrates qpos after evaluating kinematics. Refresh derived
        # poses/sensors so the camera, joint state and timestamp describe one instant.
        mujoco.mj_forward(self.model, d)
        rotation = d.xmat[self.body].reshape(3, 3)
        tilt = float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1, 1))))
        self.trace.append((float(d.time), d.qpos.copy(), d.qvel.copy(), label, tilt))
        if tilt > 50 or d.qpos[self.qa + 2] < 0.065:
            raise RuntimeError("fallen")

    def hold(self, seconds):
        for _ in range(round(seconds / DT)):
            self.tick(TWISTS["stand"], "stand")

    def execute(self, command):
        validate_command(command)
        if command.skill != "stand":
            for _ in range(round(command.seconds / DT)):
                self.tick(TWISTS[command.skill], command.skill)
        self.hold(1.0)

    def observe_target(self):
        rgb, depth = self.cameras.rgbd(self.data)
        observation = detect_target(rgb, depth, CAMERA_FOV, CAMERA_POSITION, self.camera_rotation)
        return observation, rgb, depth

    def oracle_target_body(self):
        """Evaluation only: never pass this result to the planner or vision detector."""
        rotation = self.data.xmat[self.body].reshape(3, 3)
        return rotation.T @ (self.data.geom_xpos[self.marker] - self.data.xpos[self.body])

    def save_trace(self, path):
        np.savez_compressed(
            path,
            time=[r[0] for r in self.trace],
            qpos=[r[1] for r in self.trace],
            qvel=[r[2] for r in self.trace],
            skill=[r[3] for r in self.trace],
            tilt_deg=[r[4] for r in self.trace],
        )

    def close(self):
        self.cameras.close()
