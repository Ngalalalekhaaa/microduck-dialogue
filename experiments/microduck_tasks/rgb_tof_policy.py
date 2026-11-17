"""Existing ONNX skills driven by estimated IMU/encoder observations.

The caller supplies sensor-derived orientation and joint velocities. No body
position, linear velocity, camera depth or simulator state is consumed here.
Head offsets go through the walking model's existing command slots; kick
graphs retain their inspected zero head/body command contract.
"""

import numpy as np
from microduck_deploy.policy import JOINT_NAMES, action_to_targets, build_obs61

from strands_robots.policy.onnx.graph import projected_gravity

from .skills import TWISTS, SkillPolicy


class SensorPolicy(SkillPolicy):
    def reset(self):
        super().reset()
        self.head_command = np.zeros(4)

    def act(self, obs, *, instruction=None):
        if self.skill not in TWISTS and self.skill not in ("kick_left", "kick_right"):
            raise ValueError(f"Unknown skill {self.skill}.")
        kind = self.skill if self.skill.startswith("kick_") else "walk"
        order = [obs.joints.names.index(name) for name in JOINT_NAMES]
        w, x, y, z = obs.pose.orientation
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        twist = self.motion.command(
            self.skill, TWISTS.get(self.skill, (0, 0, 0)), yaw, float(obs.pose.angular_velocity[2])
        )
        vector = build_obs61(
            obs.pose.angular_velocity,
            projected_gravity(obs.pose.orientation),
            obs.joints.positions[order],
            obs.joints.velocities[order],
            self.last_action,
            twist,
            self.head_command if kind == "walk" else np.zeros(4),
        )
        raw = self.sessions[kind].run(["actions"], {"obs": vector[None, :]})[0]
        if raw.shape != (1, 14) or not np.isfinite(raw).all():
            raise ValueError("Policy output is nonfinite or has the wrong shape.")
        target = action_to_targets(raw[0])
        self.last_action = raw[0].copy()
        return dict(zip(JOINT_NAMES, target, strict=True))
