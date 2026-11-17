"""RGB/ToF approach with encoder-derived foot reach or legacy point alignment."""

import math

from .efficient_planner import EfficientKickPlanner, ServoDecision


class SensorKickPlanner(EfficientKickPlanner):
    """Stop adjusting once the ball lies in the selected foot's tested reach."""

    def __init__(self, color, *, foot_relative=False):
        super().__init__(color)
        self.foot_relative = foot_relative
        self.sensor_steps = 0
        self.sensor_start = None

    def decide(self, observation):
        if not all(math.isfinite(observation[key]) for key in ("time_s", "yaw", "tilt_deg")):
            self.status = "invalid_vision"
            return None
        if not 0 <= observation["tilt_deg"] < 25:
            self.status = "unstable_posture"
            return None
        self.sensor_steps += 1
        if self.sensor_start is None:
            self.sensor_start = observation["time_s"]
        if self.sensor_steps > 300 or observation["time_s"] - self.sensor_start > 90:
            self.status = "timeout"
            return None
        targets = [d for d in observation["detections"] if d["color"] == self.color]
        if not targets:
            return super().decide(observation)
        point = max(targets, key=lambda d: d["confidence"])["center_body_m"]
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            self.status = "invalid_vision"
            return None
        x, y, z = point
        if x < 0.2 and observation["stationary"] is not True:
            return ServoDecision("settle_before_foot_correction", "stand", 0.4)
        if self.foot_relative:
            return self._foot_decision(observation, point)
        if self.foot is None and x < 0.18:
            self.foot = "left" if y >= 0 else "right"
        if self.foot and x < 0.16:
            if self.first_time is None:
                self.first_time = observation["time_s"]
            self.steps += 1
            if self.steps > 300 or observation["time_s"] - self.first_time > 90:
                self.status = "timeout"
                return None
            target_y = 0.035 if self.foot == "left" else -0.035
            if x < 0.075 or (x < 0.115 and abs(y - target_y) > 0.012):
                self.aligned = False
                return ServoDecision("retreat_before_lateral_correction", "move_backward", 0.16)
            if abs(y - target_y) > 0.012:
                self.aligned = False
                skill, seconds = ("move_left", 0.12) if y > target_y else ("move_right", 0.16)
                return ServoDecision("foot_align", skill, seconds)
            if 0.08 <= x <= 0.101 and -0.11 <= z <= -0.04:
                if not self.aligned or not observation["stationary"]:
                    self.aligned = True
                    return ServoDecision("settle", "stand", 0.6)
                return ServoDecision("kick", f"kick_{self.foot}", 0.5)
            self.aligned = False
            if not observation["stationary"]:
                return ServoDecision("settle", "stand", 0.4)
            if x < 0.08:
                return ServoDecision("foot_align", "move_backward", 0.1)
            if x > 0.101:
                return ServoDecision("foot_approach", "move_forward", 0.16)
        return super().decide(observation)

    def _foot_decision(self, observation, point):
        from .rgb_tof_reach import MAX_FORWARD_M, MIN_FORWARD_M, choose_reachable_foot, foot_point, reachable_from_foot

        age = observation.get("sensor_age_s")
        if not isinstance(age, (int, float)) or not math.isfinite(age) or not 0 <= age <= 0.12:
            self.status = "stale_foot_observation"
            return None
        poses = observation.get("feet")
        if not poses:
            self.status = "missing_foot_encoders"
            return None
        try:
            foot = choose_reachable_foot(point, poses)
            if self.foot in poses and reachable_from_foot(point, self.foot, poses[self.foot]):
                foot = self.foot
            relative = {side: foot_point(point, pose) for side, pose in poses.items()}
        except (KeyError, TypeError, ValueError):
            self.status = "invalid_foot_encoders"
            return None
        if foot:
            changed = self.foot != foot
            self.foot = foot
            if not self.aligned or changed or observation["stationary"] is not True:
                self.aligned = True
                return ServoDecision("settle_in_foot_reach", "stand", 0.6)
            return ServoDecision("kick", f"kick_{foot}", 0.5)
        self.aligned = False
        x, y, _ = point
        if x >= 0.2 or x < 0.03:
            # The legacy approach handles distant heading/obstacles; its own
            # kick region is unreachable at these distances.
            return super().decide(observation)
        foot = min(relative, key=lambda side: abs(relative[side][1]))
        fx, fy, _ = relative[foot]
        outward = fy if foot == "left" else -fy
        if not -0.045 <= outward <= 0.025:
            self.foot = None
            if fx < 0.10:
                return ServoDecision("retreat_before_lateral_correction", "move_backward", 0.16)
            skill, seconds = ("move_left", 0.12) if fy > 0 else ("move_right", 0.16)
            return ServoDecision("foot_align", skill, seconds)
        if fx < MIN_FORWARD_M:
            return ServoDecision("foot_too_close", "move_backward", 0.16)
        if fx > MAX_FORWARD_M:
            return ServoDecision("foot_approach", "move_forward", 0.16)
        self.status = "ball_outside_foot_height"
        return None
