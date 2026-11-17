"""Finite, bounded action probes with explicit per-policy command contracts."""

import hashlib

import numpy as np
import onnxruntime as ort
from microduck_deploy.policy import HOME, JOINT_NAMES, OBSERVATION_NAMES, build_obs61

from strands_robots.policy.onnx.graph import projected_gravity

from .action_models import resolve
from .skills import TWISTS, SkillPolicy

DURATIONS = {
    **{name: 2.0 for name in TWISTS},
    "kick_left": 0.5,
    "kick_right": 0.5,
    "sit_stand": 7.0,
    "ground_pick": 4.0,
    "roll": 2.0,
    "official_sit": 7.0,
    "official_pick": 2.8,
    "official_roll": 1.0,
    "official_walk": 2.0,
    "official_stand": 2.0,
    "velstand": 2.0,
    "flamingo_left": 5.0,
    "flamingo_right": 5.0,
    "happy_hop": 3.0,
    "hop": 6.0,
    "run": 2.0,
    "bow": 4.0,
    "moonwalk": 3.0,
    "headstand": 3.0,
    "beak_throw": 2.4,
    "look_around": 4.0,
    "nod": 3.0,
    "body_sway": 4.0,
    "open_beak": 1.0,
    "close_beak": 1.0,
    "groove": 10.0,
    "pirouette": 4.0,
    "caught_dancing": 10.4,
}


def command(skill, elapsed):
    """Return model slot and twist/head/body in their training conventions."""
    t = max(0.0, elapsed)
    twist, head, body = np.zeros(3), np.zeros(4), np.zeros(6)
    slot = skill
    if skill in ("sit_stand", "official_sit"):
        twist[0] = min(t / 2, 1) if t < 3 else max(0, 1 - (t - 3))
    elif skill in ("ground_pick", "official_pick", "beak_throw"):
        phase = min(t / (2.4 if skill == "beak_throw" else 4), 0.999)
        twist[:2] = [np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)]
    elif skill.startswith("flamingo_"):
        slot = "flamingo"
        twist[:2] = [float(t < 3), 1 if skill.endswith("left") else -1]
    elif skill in ("run", "official_walk", "velstand", "moonwalk"):
        twist[0] = {"run": 0.6, "official_walk": 0.2, "velstand": 0.2, "moonwalk": -0.2}[skill]
    elif skill in ("look_around", "nod", "body_sway", "open_beak", "close_beak"):
        slot = "walk"
        envelope = np.sin(np.pi * min(t / DURATIONS[skill], 1)) ** 2
        if skill == "look_around":
            head[2] = 0.5 * np.sin(2 * np.pi * t / 4)
        elif skill == "nod":
            head[1] = 0.25 * np.sin(2 * np.pi * t) * envelope
        elif skill == "body_sway":
            slot = "velstand"
            body[3] = 0.12 * np.sin(2 * np.pi * t / 2) * envelope
    elif skill == "bow":
        envelope = np.sin(np.pi * min(t / 4, 1)) ** 2
        head[1], body[4] = 0.25 * envelope, 0.2 * envelope
    elif skill == "pirouette":
        twist[2] = float(t < 3)
    return slot, twist, head, body


class ActionPolicy(SkillPolicy):
    def __init__(self):
        super().__init__()
        self.limits = None
        self.max_raw_excess = 0.0
        self.raw_clip_ticks = self.action_ticks = 0

    def reset(self):
        super().reset()
        self.start_time = 0.0
        self.max_raw_excess = 0.0
        self.raw_clip_ticks = self.action_ticks = 0
        self.recurrent = {}

    def start(self, skill, time_s):
        self.skill, self.start_time = skill, time_s
        self.recurrent.pop(skill, None)

    def _load(self, slot):
        if slot in self.sessions:
            return self.sessions[slot]
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        path = resolve(slot)
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        meta = session.get_modelmeta().custom_metadata_map
        commands = ("twist", "head_pose", "body_pose")
        if slot in ("ground_pick", "official_pick", "roll", "official_roll", "happy_hop", "beak_throw"):
            commands = ("twist",)
        elif slot in ("sit_stand", "official_sit", "flamingo", "pirouette"):
            commands = ("twist", "head_pose")
        if tuple(meta.get("command_names", "").split(",")) != commands:
            raise ValueError(f"{slot}: incompatible command contract")
        for key, expected in (("joint_names", JOINT_NAMES), ("observation_names", OBSERVATION_NAMES)):
            if tuple(meta.get(key, "").split(",")) != expected:
                raise ValueError(f"{slot}: incompatible {key}")
        home = np.fromstring(meta.get("default_joint_pos", ""), sep=",")
        if home.shape != (14,) or not np.allclose(home, HOME, atol=0.000501, rtol=0):
            raise ValueError(f"{slot}: incompatible HOME")
        if meta.get("action_scale") != "1.0":
            raise ValueError(f"{slot}: unsupported action scale")
        expected_in = [("obs", [1, 61])]
        expected_out = [("actions", [1, 14])]
        if slot == "caught_dancing":
            expected_in += [("h_in", [1, 1, 1]), ("c_in", [1, 1, 1])]
            expected_out += [("h_out", [1, 1, 1]), ("c_out", [1, 1, 1])]
        for nodes, expected in ((session.get_inputs(), expected_in), (session.get_outputs(), expected_out)):
            if [(n.name, n.shape) for n in nodes] != expected or any(n.type != "tensor(float)" for n in nodes):
                raise ValueError(f"{slot}: incompatible ONNX input/output state contract")
        self.sessions[slot] = session
        self.hashes[slot] = hashlib.sha256(path.read_bytes()).hexdigest()
        return session

    def act(self, obs, *, instruction=None):
        if self.skill in TWISTS or self.skill in ("kick_left", "kick_right"):
            return super().act(obs, instruction=instruction)
        if self.skill not in DURATIONS:
            raise ValueError(f"Unknown action {self.skill}")
        slot, twist, head, body = command(self.skill, obs.t - self.start_time)
        session = self._load(slot)
        order = [obs.joints.names.index(name) for name in JOINT_NAMES]
        vector = build_obs61(
            obs.pose.angular_velocity,
            projected_gravity(obs.pose.orientation),
            obs.joints.positions[order],
            obs.joints.velocities[order],
            self.last_action,
            twist,
            head,
            body,
        )
        inputs = {"obs": vector[None, :]}
        if slot == "caught_dancing":
            h, c = self.recurrent.get(slot, (np.zeros((1, 1, 1), np.float32), np.zeros((1, 1, 1), np.float32)))
            inputs.update(h_in=h, c_in=c)
            raw, h_next, c_next = session.run(["actions", "h_out", "c_out"], inputs)
            if any(s.shape != (1, 1, 1) or not np.isfinite(s).all() for s in (h_next, c_next)):
                raise ValueError("Action policy returned an invalid recurrent state.")
        else:
            raw = session.run(["actions"], inputs)[0]
        if raw.shape != (1, 14) or not np.isfinite(raw).all():
            raise ValueError("Action policy returned an invalid/nonfinite action.")
        if slot == "caught_dancing":
            self.recurrent[slot] = (h_next.copy(), c_next.copy())
        target = HOME + raw[0]
        self.last_action = raw[0].copy()
        self.action_ticks += 1
        if self.limits is not None:
            excess = np.maximum(self.limits[:, 0] - target, target - self.limits[:, 1])
            self.max_raw_excess = max(self.max_raw_excess, float(excess.max()))
            self.raw_clip_ticks += int(excess.max() > 0)
            target = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        return dict(zip(JOINT_NAMES, target, strict=True))
