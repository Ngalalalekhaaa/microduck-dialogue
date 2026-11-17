"""Measured semantic motion for dialogue goals, isolated from the v2 executor."""

import math

from .action_policy import ActionPolicy
from .action_runtime import ActionRuntime
from .motion import wrap
from .toy_protocol import goal as validate_goal


class ToyPolicy(ActionPolicy):
    """Reuse the inspected sit policy with an explicit, persistent blend level."""

    def reset(self):
        super().reset()
        self.sit_level = self.sit_initial = 0.0

    def start(self, skill, time_s):
        super().start(skill, time_s)
        self.sit_initial = self.sit_level

    def act(self, obs, *, instruction=None):
        if self.skill not in ("sit_down", "sit_hold", "stand_up"):
            return super().act(obs, instruction=instruction)
        skill, start = self.skill, self.start_time
        elapsed = max(0.0, obs.t - start)
        if skill == "sit_down":
            self.sit_level = min(1.0, self.sit_initial + elapsed / 2)
        elif skill == "stand_up":
            self.sit_level = max(0.0, self.sit_initial - elapsed)
        # The original sit_stand ramp maps t=0..2 to twist[0]=0..1.
        self.skill, self.start_time = "sit_stand", obs.t - 2 * self.sit_level
        try:
            return super().act(obs, instruction=instruction)
        finally:
            self.skill, self.start_time = skill, start


class ToyRuntime(ActionRuntime):
    """Keep the reached sitting blend on cancellation and between user turns."""

    supports_sit_hold = True

    def __init__(self, output, *, variant="hd1910", toy_xy=(0.3, 0.0), support_height=0.0, block_xy=(0.45, 0.32)):
        super().__init__(output, variant=variant, toy_xy=toy_xy, support_height=support_height, block_xy=block_xy)
        self.policy = ToyPolicy()
        self.policy.limits = self.limits.copy()

    def supports_skill(self, skill):
        return skill in ("sit_down", "sit_hold", "stand_up") or super().supports_skill(skill)

    def _tick(self, skill):
        if skill == "stand" and self.policy.sit_level > 0:
            skill = "sit_hold"
        super()._tick(skill)

    def _posture_failed(self, skill, tilt):
        if skill in ("sit_down", "sit_hold", "stand_up"):
            return tilt > 120 or self.data.qpos[self.qa + 2] < 0.03
        return super()._posture_failed(skill, tilt)

    def posture(self):
        with self.lock:
            if self.failure:
                return "unknown"
            height = float(self.data.qpos[self.qa + 2])
            tilt = self.snapshot()["tilt_deg"]
            if 0.035 <= height < 0.08 and tilt < 25:
                return "sitting"
            if height >= 0.09 and tilt < 25:
                return "standing"
            return "unknown"


class ToyMotion:
    """Execute turn/navigation/posture using observations, never ball truth state."""

    def __init__(self, runtime, observe_callback=None, execute_callback=None):
        self.runtime = runtime
        self.observe_callback = observe_callback or runtime.observe
        self.execute_callback = execute_callback
        self.observations = self.pulses = 0

    def observe(self):
        value = self.observe_callback()
        obs = value[0] if isinstance(value, tuple) else value
        self.observations += 1
        required = (obs["time_s"], obs["yaw"], obs["tilt_deg"])
        if not all(math.isfinite(v) for v in required):
            raise ValueError("Nonfinite camera/IMU observation; stop the task.")
        age = self.runtime.snapshot()["time_s"] - obs["time_s"]
        if age < -0.001 or age > 0.5:
            raise ValueError("Stale camera observation; refresh before moving.")
        return obs

    def pulse(self, skill, seconds, obs, heading=None):
        if self.runtime.cancel.is_set():
            return False
        with self.runtime.lock:
            self.runtime.policy.motion.visual_heading = heading
        ticks = max(1, min(150, round(seconds / 0.02)))
        self.pulses += 1
        if self.execute_callback:
            return self.execute_callback(skill, ticks * 0.02, obs)
        return self.runtime.execute(skill, ticks * 0.02)

    def execute(self, goal):
        validate_goal(goal)
        self.observations = self.pulses = 0
        if self.runtime.cancel.is_set():
            return self.result(False, "cancelled")
        handlers = {"turn": self.turn, "navigate": self.navigate, "posture": self.posture, "move": self.move}
        if goal["type"] not in handlers:
            raise ValueError("ToyMotion handles turn, navigate, posture, and finite move goals.")
        try:
            return handlers[goal["type"]](goal)
        finally:
            with self.runtime.lock:
                self.runtime.policy.motion.visual_heading = None

    def result(self, success, reason, **metrics):
        return {
            "success": bool(success),
            "reason": reason,
            "metrics": {"observations": self.observations, "pulses": self.pulses, **metrics},
        }

    def turn(self, goal):
        sign = 1 if goal["direction"] == "left" else -1
        target = math.radians(goal["angle_deg"]) * sign
        obs = self.observe()
        start, previous, accumulated = obs["time_s"], obs["yaw"], 0.0
        settled = False
        while obs["time_s"] - start < 25:
            error = target - accumulated
            if abs(error) < math.radians(8 if settled else 4):
                if settled:
                    return self.result(
                        True,
                        "angle_reached",
                        angle_deg=math.degrees(accumulated),
                        angle_error_deg=math.degrees(error),
                        tolerance_deg=8.0,
                    )
                skill, duration, settled = "stand", 0.6, True
            else:
                skill = "turn_left" if error > 0 else "turn_right"
                duration, settled = min(0.24, max(0.04, abs(error) * 0.45)), False
            if not self.pulse(skill, duration, obs):
                return self.result(False, "cancelled", angle_deg=math.degrees(accumulated))
            obs = self.observe()
            accumulated += wrap(obs["yaw"] - previous)
            previous = obs["yaw"]
        return self.result(False, "turn_timeout", angle_deg=math.degrees(accumulated))

    def navigate(self, goal):
        attrs = {a["name"]: a["value"] for a in goal["target"]["attributes"]}
        color = attrs.get("color")
        entity_id = goal["target"]["entity_id"]
        if color is None and entity_id and entity_id.startswith("ball:"):
            color = entity_id.removeprefix("ball:")
        if color not in ("red", "yellow", "blue") or set(attrs) - {"color"}:
            return self.result(False, "unsupported_target_query")
        if entity_id and entity_id != f"ball:{color}":
            return self.result(False, "contradictory_target_query")
        obs = self.observe()
        start, previous, searched = obs["time_s"], obs["yaw"], 0.0
        settled, distance = False, None
        while obs["time_s"] - start < 45:
            seen = [d for d in obs["detections"] if d["color"] == color]
            heading = None
            if not seen:
                if searched > 2 * math.pi + 0.3:
                    return self.result(False, "target_not_found")
                skill, duration, settled = "turn_left", 0.32, False
            else:
                searched = 0.0
                point = max(seen, key=lambda d: d["confidence"])["center_body_m"]
                if len(point) != 3 or not all(math.isfinite(v) for v in point):
                    return self.result(False, "invalid_target_measurement")
                x, y, _ = point
                distance, bearing = math.hypot(x, y), math.atan2(y, x)
                error = distance - goal["standoff_m"]
                if abs(error) <= 0.035 and abs(bearing) < 0.2:
                    if settled and obs["stationary"]:
                        return self.result(True, "standoff_reached", distance_m=distance, bearing_rad=bearing)
                    skill, duration, settled = "stand", 0.6, True
                elif abs(bearing) > 0.18:
                    skill, duration, settled = "turn_left" if bearing > 0 else "turn_right", 0.12, False
                else:
                    blockers = [
                        d
                        for d in obs["detections"]
                        if d["color"] != color
                        and 0 < d["center_body_m"][0] < min(x, 0.24)
                        and abs(d["center_body_m"][1]) < 0.11
                    ]
                    if blockers and error > 0:
                        y_block = min(blockers, key=lambda d: d["center_body_m"][0])["center_body_m"][1]
                        skill, duration = "move_right" if y_block >= 0 else "move_left", 0.2
                    else:
                        skill = "move_forward" if error > 0 else "move_backward"
                        duration = min(0.3, max(0.1, abs(error) * 2))
                        heading = obs["yaw"] + bearing
                    settled = False
            if not self.pulse(skill, duration, obs, heading):
                return self.result(False, "cancelled", distance_m=distance)
            obs = self.observe()
            if not seen:
                searched += abs(wrap(obs["yaw"] - previous))
            previous = obs["yaw"]
        return self.result(False, "navigation_timeout", distance_m=distance)

    def posture(self, goal):
        if not isinstance(self.runtime, ToyRuntime):
            return self.result(False, "posture_hold_unavailable")
        skill = "sit_down" if goal["pose"] == "sitting" else "stand_up"
        with self.runtime.lock:
            self.runtime.policy.start(skill, self.runtime.data.time)
        for _ in range(15):
            obs = self.observe()
            if not self.pulse(skill, 0.2, obs):
                return self.result(False, "cancelled", posture=self.runtime.posture())
        obs = self.observe()
        if not self.pulse("stand", 1.0, obs):
            return self.result(False, "cancelled", posture=self.runtime.posture())
        pose = self.runtime.posture()
        return self.result(
            pose == goal["pose"], "posture_reached" if pose == goal["pose"] else "posture_unverified", posture=pose
        )

    def move(self, goal):
        if goal["limit"]["kind"] != "duration":
            return self.result(False, "continuous_move_requires_session_lease")
        remaining = round(goal["limit"]["seconds"] / 0.02)
        while remaining > 0:
            obs, ticks = self.observe(), min(remaining, 15)
            if not self.pulse(f"move_{goal['direction']}", ticks * 0.02, obs):
                return self.result(False, "cancelled")
            remaining -= ticks
        obs = self.observe()
        if not self.pulse("stand", 0.6, obs):
            return self.result(False, "cancelled")
        return self.result(True, "duration_completed")
