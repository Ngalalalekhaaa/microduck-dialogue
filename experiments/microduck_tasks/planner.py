"""Bounded visual servo decisions. No simulator object positions are accepted."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    phase: str
    skill: str
    seconds: float


class KickPlanner:
    def __init__(self, color):
        if color not in ("red", "yellow", "blue"):
            raise ValueError("Choose one ball color.")
        self.color = color
        self.foot = None
        self.last_yaw = None
        self.search_angle = 0.0
        self.search_yaw = 0.0
        self.search_min = self.search_max = 0.0
        self.last_search = False
        self.aligned = False
        self.steps = 0
        self.first_time = None
        self.status = "running"

    def decide(self, observation):
        now, yaw = observation["time_s"], observation["yaw"]
        if self.first_time is None:
            self.first_time = now
        if self.last_yaw is not None and self.last_search:
            self.search_yaw += math.atan2(math.sin(yaw - self.last_yaw), math.cos(yaw - self.last_yaw))
            self.search_min = min(self.search_min, self.search_yaw)
            self.search_max = max(self.search_max, self.search_yaw)
            self.search_angle = self.search_max - self.search_min
        self.last_yaw, self.last_search = yaw, False
        self.steps += 1
        if self.steps > 120 or now - self.first_time > 180:
            self.status = "timeout"
            return None
        targets = [d for d in observation["detections"] if d["color"] == self.color]
        if not targets:
            self.aligned = False
            if self.search_angle >= 2 * math.pi + 0.25:
                self.status = "not_found"
                return None
            self.last_search = True
            return Decision("search", "turn_left", 1.0)
        self.search_angle = 0
        self.search_yaw = self.search_min = self.search_max = 0.0
        x, y, _ = max(targets, key=lambda d: d["confidence"])["center_body_m"]
        if not all(math.isfinite(v) for v in (x, y)):
            self.status = "invalid_vision"
            return None
        # Turn to face a distant target; near the feet use lateral translation.
        angle = math.atan2(y, x)
        if (math.hypot(x, y) > 0.24 or x < 0.03) and abs(angle) > 0.35:
            self.aligned = False
            return Decision("face", "turn_left" if angle > 0 else "turn_right", 0.6)
        if self.foot is None and x < 0.24:
            self.foot = "left" if y >= 0 else "right"
        foot = self.foot or ("left" if y >= 0 else "right")
        goal_x, goal_y = (0.0895, 0.0411) if foot == "left" else (0.0881, -0.0429)
        dx, dy = x - goal_x, y - goal_y
        if abs(dx) <= 0.012 and abs(dy) <= 0.012:
            if not self.aligned or not observation["stationary"]:
                self.aligned = True
                return Decision("settle", "stand", 0.6)
            return Decision("kick", f"kick_{foot}", 0.5)
        self.aligned = False
        # Keep another visible ball out of the short forward stepping corridor.
        if dx > 0.012:
            obstacles = [
                d
                for d in observation["detections"]
                if d["color"] != self.color
                and 0 < d["center_body_m"][0] < min(x, 0.24)
                and abs(d["center_body_m"][1]) < 0.10
            ]
            if obstacles:
                oy = min(obstacles, key=lambda d: d["center_body_m"][0])["center_body_m"][1]
                return Decision("avoid", "move_right" if oy >= 0 else "move_left", 0.32)
        # Align sideways before entering the ball's contact region.
        tolerance_y = 0.05 if x > 0.24 else 0.012
        if abs(dy) > tolerance_y:
            duration = 0.32 if abs(dy) > 0.07 else 0.20 if abs(dy) > 0.028 else 0.06 if dy > 0 else 0.10
            return Decision("align", "move_left" if dy > 0 else "move_right", duration)
        if dx > 0.012:
            duration = 0.6 if dx > 0.12 else 0.32 if dx > 0.045 else 0.26
            return Decision("approach", "move_forward", duration)
        if dx < -0.012:
            return Decision("align", "move_backward", 0.32 if dx < -0.04 else 0.2)
        self.status = "unresolved_alignment"
        return None
