"""Single-ball CPU BAM experiment; ball pose is assigned once at reset only."""

import io
from contextlib import redirect_stdout

import mujoco
import numpy as np
from bootstrap import ASSETS, KICK_MODELS, STAND_MODEL
from microduck_deploy.policy import HOME, JOINT_NAMES, OnnxPolicy, action_to_targets, build_obs61
from policy import KickPolicy
from render_hd1910_controller import load_scene

from strands_robots import JointState, Observation, Pose
from strands_robots.policy.onnx.graph import projected_gravity

DT = 0.02


class Experiment:
    def __init__(self, output, *, variant="hd1910", lag_steps=4, stand_model=STAND_MODEL):
        spec = mujoco.MjSpec.from_file(str(ASSETS / "scene_ball.xml"))
        spec.meshdir = str(ASSETS / "assets")
        spec.compile()
        self.scene = output / "scene.xml"
        self.scene.write_text(spec.to_xml())
        log = io.StringIO()
        with redirect_stdout(log):
            self.model, self.data, self.controller, self.physics = load_scene(
                self.scene, variant, timestep=0.005, lag_steps=lag_steps
            )
        (output / "initialization.log").write_text(log.getvalue())
        self.model.opt.iterations, self.model.opt.ls_iterations = 30, 50
        self.body = self.model.body("trunk_base").id
        self.qa = int(self.model.joint("trunk_base_freejoint").qposadr[0])
        self.va = int(self.model.joint("trunk_base_freejoint").dofadr[0])
        self.ba = int(self.model.joint("ball_free").qposadr[0])
        self.bv = int(self.model.joint("ball_free").dofadr[0])
        self.bg = self.model.geom("ball_geom").id
        self.floor = self.model.geom("floor").id
        self.gyro = int(self.model.sensor("imu_ang_vel").adr[0])
        ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.qi, self.vi = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        self.limits = self.model.jnt_range[ids]
        self.walk = OnnxPolicy(stand_model)
        self.kicks = {foot: KickPolicy(KICK_MODELS / f"ball_kick_{foot}.onnx", foot) for foot in ("left", "right")}

    def reset(self, ball_xy):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qa : self.qa + 7] = [0, 0, 0.12, 1, 0, 0, 0]
        self.data.qpos[self.qi] = HOME
        self.data.qpos[self.ba : self.ba + 7] = [*ball_xy, 0.035, 1, 0, 0, 0]
        self.controller.reset(self.data.qpos)
        self.controller.q_target = HOME
        self.last = np.zeros(14, np.float32)
        self.contacts = []
        self.rows = []
        self.target_exceedance = np.zeros(14)
        self.actual_exceedance = np.zeros(14)
        self.target_outside_ticks = 0
        mujoco.mj_forward(self.model, self.data)
        self.record("initial")

    def record(self, skill):
        d = self.data
        tilt = float(np.rad2deg(np.arccos(np.clip(d.xmat[self.body].reshape(3, 3)[2, 2], -1, 1))))
        self.rows.append(
            (
                float(d.time),
                d.qpos.copy(),
                d.qvel.copy(),
                skill,
                tilt,
                d.qpos[self.ba : self.ba + 3].copy(),
                d.qpos[self.qa : self.qa + 3].copy(),
                d.sensordata[self.gyro : self.gyro + 3].copy(),
                self.last.copy(),
                self.controller.q_target.copy(),
                d.qvel[self.va : self.va + 3].copy(),
            )
        )
        return tilt

    def tick(self, skill):
        d = self.data
        obs = Observation(
            float(d.time),
            JointState(JOINT_NAMES, d.qpos[self.qi], d.qvel[self.vi], d.actuator_force),
            Pose(d.xpos[self.body], d.xquat[self.body], angular_velocity=d.sensordata[self.gyro : self.gyro + 3]),
        )
        if skill.startswith("kick_"):
            policy = self.kicks[skill.removeprefix("kick_")]
            policy.last_action = self.last.copy()
            targets = policy.act(obs)
            self.last = policy.last_action.copy()
            target = np.array([targets[name] for name in JOINT_NAMES])
        else:
            vector = build_obs61(
                obs.pose.angular_velocity,
                projected_gravity(obs.pose.orientation),
                d.qpos[self.qi],
                d.qvel[self.vi],
                self.last,
            )
            self.last = self.walk.infer(vector)
            target = action_to_targets(self.last)
        violation = np.maximum(self.limits[:, 0] - target, target - self.limits[:, 1]).clip(min=0)
        self.target_exceedance = np.maximum(self.target_exceedance, violation)
        self.target_outside_ticks += int(np.any(violation > 0))
        self.controller.q_target = target  # Preserve native output; do not clamp or smooth.
        for _ in range(4):
            self.controller.update()
            mujoco.mj_step(self.model, d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise RuntimeError("nonfinite_state")
            measured = d.qpos[self.qi]
            excess = np.maximum(self.limits[:, 0] - measured, measured - self.limits[:, 1]).clip(min=0)
            self.actual_exceedance = np.maximum(self.actual_exceedance, excess)
            for index in range(d.ncon):
                contact = d.contact[index]
                pair = (int(contact.geom1), int(contact.geom2))
                if self.bg not in pair:
                    continue
                other = pair[1] if pair[0] == self.bg else pair[0]
                if other == self.floor:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, d, index, force)
                if force[0] > 1e-5:
                    self.contacts.append(
                        {
                            "time_s": float(d.time - self.model.opt.timestep),
                            "skill": skill,
                            "geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom_{other}",
                            "body": self.model.body(int(self.model.geom_bodyid[other])).name,
                            "normal_force_n": float(force[0]),
                            "normal_impulse_ns": float(force[0] * self.model.opt.timestep),
                        }
                    )
        mujoco.mj_forward(self.model, d)
        tilt = self.record(skill)
        if tilt > 50 or d.qpos[self.qa + 2] < 0.065:
            raise RuntimeError("fallen")

    def arrays(self):
        names = (
            "time",
            "qpos",
            "qvel",
            "skill",
            "tilt_deg",
            "ball_xyz",
            "trunk_xyz",
            "gyro",
            "raw_action",
            "target",
            "trunk_velocity",
        )
        return {name: np.asarray([row[i] for row in self.rows]) for i, name in enumerate(names)}
