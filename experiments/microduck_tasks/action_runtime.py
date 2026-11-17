"""Isolated action playground with physical mouth control and measured outcomes."""

import json
from functools import partial

import mujoco
import numpy as np

from .action_policy import DURATIONS, ActionPolicy
from .action_scene import MOUTH_OPEN, POCKET_LOCAL, build_action_scene
from .runtime import DT, Runtime

LAYOUT = {"red": [0.5, -0.25], "yellow": [0.7, 0.15], "blue": [-0.35, 0.3]}


class ActionRuntime(Runtime):
    def __init__(self, output, *, variant="hd1910", toy_xy=(0.3, 0.0), support_height=0.0, block_xy=(0.45, 0.32)):
        builder = partial(build_action_scene, toy_xy=toy_xy, support_height=support_height, block_xy=block_xy)
        super().__init__(output, scene_builder=builder, policy_factory=ActionPolicy, variant=variant)
        self.policy.limits = self.limits.copy()
        self.mouth_q = int(self.model.joint("passive_mouth").qposadr[0])
        self.mouth_act = self.model.actuator("mouth_servo").id
        self.head = self.model.body("jaw_soft").id
        self.toy_q = int(self.model.joint("grasp_toy_free").qposadr[0])
        self.toy_v = int(self.model.joint("grasp_toy_free").dofadr[0])
        self.toy_geom = self.model.geom("grasp_toy_geom").id
        self.block_geom = self.model.geom("push_block_geom").id
        self.block_q = int(self.model.joint("push_block_free").qposadr[0])
        self.robot_bodies = {self.body}
        for body in range(self.body + 1, self.model.nbody):
            if self.model.body_parentid[body] in self.robot_bodies:
                self.robot_bodies.add(body)
        self.grip_geoms = {g for g in range(self.model.ngeom) if (self.model.geom(g).name or "").startswith("grip_")}
        self.feet = [self.model.geom(f"{side}_foot_collision").id for side in ("left", "right")]
        self.mouth_target = MOUTH_OPEN
        self.mouth_command = MOUTH_OPEN
        self.metrics = []

    def reset(self, layout=None):
        self.metrics = []
        self.mouth_target = self.mouth_command = MOUTH_OPEN
        super().reset(LAYOUT if layout is None else layout)

    def supports_skill(self, skill):
        return skill in DURATIONS

    def _before_physics_step(self, skill):
        # Independent, torque-limited jaw servo; 2 rad/s command slew.
        self.mouth_command += float(np.clip(self.mouth_target - self.mouth_command, -0.01, 0.01))
        self.data.ctrl[self.mouth_act] = self.mouth_command

    def _posture_failed(self, skill, tilt):
        height = self.data.qpos[self.qa + 2]
        if skill in ("roll", "official_roll", "headstand"):
            return height < 0.02 or self.data.time - self.policy.start_time > DURATIONS[skill] + 0.1
        if skill in ("ground_pick", "official_pick", "sit_stand", "official_sit", "beak_throw"):
            return tilt > 120 or height < 0.03
        return super()._posture_failed(skill, tilt)

    def _record(self, skill):
        tilt = super()._record(skill)
        d = self.data
        pocket = d.xpos[self.head] + d.xmat[self.head].reshape(3, 3) @ POCKET_LOCAL
        support, grip_force, penetration, block_force = [], 0.0, 0.0, 0.0
        foot_ground = [False, False]
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, d, i, force)
            if force[0] < 1e-5:
                continue
            for foot, geom in enumerate(self.feet):
                if geom in pair and self.floor in pair:
                    foot_ground[foot] = True
            if self.block_geom in pair:
                other = next(g for g in pair if g != self.block_geom)
                if self.model.geom_bodyid[other] in self.robot_bodies:
                    block_force += float(force[0])
            if self.toy_geom in pair:
                other = next(g for g in pair if g != self.toy_geom)
                support.append(self.model.geom(other).name or f"geom_{other}")
                if other in self.grip_geoms:
                    grip_force += float(force[0])
                    penetration = max(penetration, -float(c.dist))
        self.metrics.append(
            {
                "time_s": float(d.time),
                "skill": skill,
                "toy_xyz": d.qpos[self.toy_q : self.toy_q + 3].tolist(),
                "pocket_xyz": pocket.tolist(),
                "mouth_rad": float(d.qpos[self.mouth_q]),
                "grip_force_n": grip_force,
                "grip_penetration_m": penetration,
                "toy_contacts": support,
                "feet_on_floor": foot_ground,
                "foot_z": [float(d.geom_xpos[g, 2]) for g in self.feet],
                "tilt_deg": tilt,
                "trunk_z": float(d.qpos[self.qa + 2]),
                "block_xyz": d.qpos[self.block_q : self.block_q + 3].tolist(),
                "block_contact_n": block_force,
            }
        )
        return tilt

    def perform(self, skill, *, recover=True):
        if skill not in DURATIONS:
            raise ValueError(f"Unknown action {skill}; use --list.")
        with self.lock:
            self.policy.start(skill, float(self.data.time))
            if skill in ("open_beak", "close_beak"):
                self.mouth_target = MOUTH_OPEN if skill == "open_beak" else 0.0
        remaining = round(DURATIONS[skill] / DT)
        while remaining and not self.cancel.is_set():
            ticks = min(remaining, 150)
            if not self.execute(skill, ticks * DT):
                return False
            remaining -= ticks
        if self.cancel.is_set():
            return False
        if recover:
            self.execute("stand", 3.0)
        return not self.failure and not self.cancel.is_set()

    def save_trace(self):
        super().save_trace()
        (self.output / "action_metrics.json").write_text(json.dumps(self.metrics, separators=(",", ":")))
