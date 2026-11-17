"""Receding-horizon RGB-D teacher: face, visually track, settle, reachable kick."""

import math
from dataclasses import dataclass

from .kick_region import choose_foot, reachable
from .planner import KickPlanner


@dataclass(frozen=True)
class ServoDecision:
    phase: str
    skill: str
    seconds: float
    heading: float | None = None


class EfficientKickPlanner(KickPlanner):
    def __init__(self, color):
        super().__init__(color)
        self.settle_count = 0
        self.tracking = False

    def decide(self, observation):
        now, yaw = observation["time_s"], observation["yaw"]
        if not all(math.isfinite(v) for v in (now, yaw, observation["tilt_deg"])):
            self.status = "invalid_vision"
            return None
        targets = [d for d in observation["detections"] if d["color"] == self.color]
        if not targets:
            self.tracking = False
            return super().decide(observation)
        if self.first_time is None:
            self.first_time = now
        self.steps += 1
        if self.steps > 300 or now - self.first_time > 90:
            self.status = "timeout"
            return None
        self.last_yaw, self.last_search = yaw, False
        self.search_angle = self.search_yaw = self.search_min = self.search_max = 0.0
        point = max(targets, key=lambda d: d["confidence"])["center_body_m"]
        if len(point) != 3 or not all(math.isfinite(v) for v in point):
            self.status = "invalid_vision"
            return None
        x, y, _ = point
        foot = self.foot if self.foot and reachable(point, self.foot) else choose_foot(point)
        if foot:
            self.foot = foot
            if not self.aligned or not observation["stationary"]:
                self.aligned = True
                return ServoDecision("settle", "stand", 0.6)
            return ServoDecision("kick", f"kick_{foot}", 0.5)
        self.aligned = False
        obstacles = [
            d
            for d in observation["detections"]
            if d["color"] != self.color
            and 0 < d["center_body_m"][0] < min(x, 0.24)
            and abs(d["center_body_m"][1]) < 0.11
        ]
        if x > 0.14 and obstacles:
            obstacle = min(obstacles, key=lambda d: d["center_body_m"][0])
            return ServoDecision("avoid", "move_right" if obstacle["center_body_m"][1] >= 0 else "move_left", 0.2)
        bearing = math.atan2(y, x)
        if x > 0.22 or x < 0.03:
            threshold = 0.28 if self.tracking else 0.12
            if abs(bearing) > threshold:
                self.tracking = False
                duration = max(0.1, min(0.4, round(abs(bearing) * 0.65 / 0.02) * 0.02))
                return ServoDecision("face", "turn_left" if bearing > 0 else "turn_right", duration)
        self.tracking = True
        if x <= 0.18 and abs(y) < 0.018:
            # Put a center-line ball in a foot's lateral corridor before the
            # final forward step. Body position is measured from RGB-D only.
            foot = self.foot or ("left" if y > 0.005 else "right")
            self.foot = foot
            return ServoDecision("align", "move_right" if foot == "left" else "move_left", 0.1)
        if x <= 0.22 and abs(y) > 0.05:
            return ServoDecision("align", "move_left" if y > 0 else "move_right", 0.12)
        if x < 0.075:
            return ServoDecision("align", "move_backward", 0.1)
        if x <= 0.15 and not observation["stationary"]:
            return ServoDecision("settle", "stand", 0.4)
        duration = 0.3 if x > 0.3 else 0.2 if x > 0.2 else 0.22
        heading = yaw + max(-0.25, min(0.25, bearing)) if x > 0.2 else None
        return ServoDecision("approach", "move_forward", duration, heading)
