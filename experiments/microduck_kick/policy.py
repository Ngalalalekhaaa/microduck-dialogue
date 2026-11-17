"""Pinned official kick policies using the local 61D observation convention."""

import hashlib

import numpy as np
import onnxruntime as ort
from bootstrap import KICK_HASHES
from microduck_deploy.policy import HOME, JOINT_NAMES, OBSERVATION_NAMES, action_to_targets, build_obs61

from strands_robots import Policy
from strands_robots.policy.onnx.graph import projected_gravity


class KickPolicy(Policy):
    units = "rad"
    hz = 50.0

    def __init__(self, path, foot):
        if foot not in KICK_HASHES or hashlib.sha256(path.read_bytes()).hexdigest() != KICK_HASHES[foot]:
            raise ValueError(f"Use the inspected official {foot} kick model; its SHA256 must match bootstrap.py.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        for name, expected in (
            ("joint_names", JOINT_NAMES),
            ("observation_names", OBSERVATION_NAMES),
            ("command_names", ("twist",)),
        ):
            if tuple(metadata.get(name, "").split(",")) != expected:
                raise ValueError(f"Kick {name} does not match the local observation contract.")
        home = np.array([float(v) for v in metadata["default_joint_pos"].split(",")])
        if home.shape != (14,) or not np.allclose(home, HOME, rtol=0, atol=0.000501):
            raise ValueError("Kick HOME differs from the deployment model.")
        if metadata.get("action_scale") != "1.0":
            raise ValueError("Kick action scale must be 1.0.")
        for nodes, name, width in ((self.session.get_inputs(), "obs", 61), (self.session.get_outputs(), "actions", 14)):
            if len(nodes) != 1 or (nodes[0].name, nodes[0].shape, nodes[0].type) != (name, [1, width], "tensor(float)"):
                raise ValueError(f"Kick requires {name}: float32[1,{width}].")
        self.reset()

    def reset(self):
        self.last_action = np.zeros(14, np.float32)

    def needs_observation(self, *, instruction=None):
        return False

    def act(self, obs, *, instruction=None):
        indices = [obs.joints.names.index(name) for name in JOINT_NAMES]
        vector = build_obs61(
            obs.pose.angular_velocity,
            projected_gravity(obs.pose.orientation),
            obs.joints.positions[indices],
            obs.joints.velocities[indices],
            self.last_action,
        )  # All 13 command values stay zero. No ball coordinates enter this policy.
        action = self.session.run(["actions"], {"obs": vector[None, :]})[0]
        if action.shape != (1, 14) or not np.isfinite(action).all():
            raise ValueError("Kick returned invalid joint actions; end this simulation case.")
        target = action_to_targets(action[0])
        self.last_action = action[0].copy()
        return dict(zip(JOINT_NAMES, target, strict=True))
