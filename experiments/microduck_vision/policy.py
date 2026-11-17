"""Strands Policy adapter for the existing local HD1910 deployment contract."""

import numpy as np
from microduck_deploy.policy import JOINT_NAMES, OnnxPolicy, action_to_targets, build_obs61

from strands_robots import Policy
from strands_robots.policy.onnx.graph import projected_gravity


class WalkingPolicy(Policy):
    units = "rad"
    hz = 50.0

    def __init__(self, path):
        self.runtime = OnnxPolicy(path, policy_kind="walking")
        self.command = np.zeros(3, dtype=np.float32)
        self.reset()

    def reset(self):
        self.last_action = np.zeros(14, dtype=np.float32)

    def needs_observation(self, *, instruction=None):
        return False

    def act(self, obs, *, instruction=None):
        if obs.pose is None or obs.pose.angular_velocity is None:
            raise ValueError("Walking needs body gyro and orientation")
        indices = [obs.joints.names.index(name) for name in JOINT_NAMES]
        vector = build_obs61(
            obs.pose.angular_velocity,
            projected_gravity(obs.pose.orientation),
            obs.joints.positions[indices],
            obs.joints.velocities[indices],
            self.last_action,
            self.command,
        )
        action = self.runtime.infer(vector)
        target = action_to_targets(action)
        self.last_action = action.copy()
        return dict(zip(JOINT_NAMES, target, strict=True))
