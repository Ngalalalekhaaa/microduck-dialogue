"""Measured simulator feedforward and bounded IMU heading hold (no world XY)."""

import json
import math
from pathlib import Path


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class HeadingHold:
    def __init__(self, commands=None, enabled=True):
        self.commands = commands or {}
        for value in self.commands.values():
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError("Motion calibration must contain finite velocity triples.")
            if abs(value[0]) > 0.3 or abs(value[1]) > 0.3 or abs(value[2]) > 0.6:
                raise ValueError("Motion calibration exceeds the inspected walking envelope.")
        self.enabled = enabled
        self.reset()

    @classmethod
    def calibrated(cls, policy_hash):
        path = Path(__file__).with_name("motion_calibration.json")
        if not path.is_file():
            return cls()
        value = json.loads(path.read_text())
        if value["policy_sha256"] != policy_hash or value["hardware"] is not False:
            raise ValueError("Recalibrate this simulator's walking model; calibration identity differs.")
        return cls(value["commands"])

    def reset(self):
        self.skill, self.reference, self.visual_heading = None, None, None
        self.integral = 0.0

    def command(self, skill, nominal, yaw, gyro_z):
        if not all(math.isfinite(v) for v in (*nominal, yaw, gyro_z)):
            raise ValueError("Motion sensors and commands must be finite; stop and inspect the sensor.")
        if self.visual_heading is not None and not math.isfinite(self.visual_heading):
            raise ValueError("Visual heading must be finite.")
        if not self.enabled or not skill.startswith("move_"):
            self.skill, self.reference, self.integral = None, None, 0.0
            return nominal
        if self.skill != skill:
            self.reference, self.integral = yaw, 0.0
        self.skill = skill
        target = self.reference if self.visual_heading is None else self.visual_heading
        error = wrap(target - yaw)
        self.integral = max(-0.4, min(0.4, self.integral + error * 0.02))
        command = list(self.commands.get(skill, nominal))
        correction = 2.0 * error + 0.5 * self.integral - 0.08 * gyro_z
        command[2] = max(-0.6, min(0.6, command[2] + correction))
        return tuple(command)
