"""One persistent action history across walking, turning and pinned kick skills."""

import hashlib

import numpy as np
import onnxruntime as ort
from microduck_deploy.policy import HOME, JOINT_NAMES, OBSERVATION_NAMES, action_to_targets, build_obs61

from strands_robots import Policy
from strands_robots.policy.onnx.graph import projected_gravity

from .bootstrap import KICK_HASHES, KICK_MODELS, WALK_MODEL
from .motion import HeadingHold

TWISTS = {
    "stand": (0, 0, 0),
    "move_forward": (0.2, 0, 0),
    "move_backward": (-0.2, 0, 0),
    "move_left": (0, 0.2, 0),
    "move_right": (0, -0.2, 0),
    "turn_left": (0, 0, 1.0),
    "turn_right": (0, 0, -1.0),
}


class SkillPolicy(Policy):
    units = "rad"
    hz = 50.0

    def __init__(self):
        paths = {"walk": WALK_MODEL, **{f"kick_{f}": KICK_MODELS / f"ball_kick_{f}.onnx" for f in KICK_HASHES}}
        self.sessions = {}
        self.hashes = {}
        for kind, path in paths.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if kind.startswith("kick_") and digest != KICK_HASHES[kind.removeprefix("kick_")]:
                raise ValueError(f"Restore the inspected official model at {path}; its SHA256 differs.")
            self.hashes[kind] = digest
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            metadata = session.get_modelmeta().custom_metadata_map
            commands = ("twist", "head_pose", "body_pose") if kind == "walk" else ("twist",)
            for key, expected in (
                ("joint_names", JOINT_NAMES),
                ("observation_names", OBSERVATION_NAMES),
                ("command_names", commands),
            ):
                if tuple(metadata.get(key, "").split(",")) != expected:
                    raise ValueError(f"Model {kind} has incompatible {key}.")
            home = np.array([float(v) for v in metadata["default_joint_pos"].split(",")])
            if (
                home.shape != (14,)
                or not np.allclose(home, HOME, atol=0.000501, rtol=0)
                or metadata["action_scale"] != "1.0"
            ):
                raise ValueError(f"Model {kind} has incompatible HOME or action scale.")
            for nodes, name, width in ((session.get_inputs(), "obs", 61), (session.get_outputs(), "actions", 14)):
                if len(nodes) != 1 or (nodes[0].name, nodes[0].shape, nodes[0].type) != (
                    name,
                    [1, width],
                    "tensor(float)",
                ):
                    raise ValueError(f"Model {kind} requires {name}: float32[1,{width}].")
            self.sessions[kind] = session
        self.motion = HeadingHold.calibrated(self.hashes["walk"])
        self.reset()

    def reset(self):
        self.last_action = np.zeros(14, np.float32)
        self.skill = "stand"
        self.motion.reset()

    def needs_observation(self, *, instruction=None):
        return False

    def act(self, obs, *, instruction=None):
        kind = self.skill if self.skill.startswith("kick_") else "walk"
        if self.skill not in TWISTS and self.skill not in ("kick_left", "kick_right"):
            raise ValueError(f"Unknown skill {self.skill}.")
        order = [obs.joints.names.index(name) for name in JOINT_NAMES]
        w, x, y, z = obs.pose.orientation
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        twist = self.motion.command(
            self.skill, TWISTS.get(self.skill, (0, 0, 0)), float(yaw), float(obs.pose.angular_velocity[2])
        )
        vector = build_obs61(
            obs.pose.angular_velocity,
            projected_gravity(obs.pose.orientation),
            obs.joints.positions[order],
            obs.joints.velocities[order],
            self.last_action,
            twist,
        )
        raw = self.sessions[kind].run(["actions"], {"obs": vector[None, :]})[0]
        if raw.shape != (1, 14) or not np.isfinite(raw).all():
            raise ValueError("Policy output is nonfinite or has the wrong shape.")
        target = action_to_targets(raw[0])
        self.last_action = raw[0].copy()
        return dict(zip(JOINT_NAMES, target, strict=True))
