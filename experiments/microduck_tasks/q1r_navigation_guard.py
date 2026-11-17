"""Bound kick pursuit using observed range, simulation time, and measured yaw."""

import math


class NavigationGuard:
    """Reject endless search or pursuit without assuming privileged object state."""

    def __init__(self, color):
        self.color = color
        self.started = self.last_progress = self.missing_since = None
        self.best_range = math.inf
        self.last_yaw = None
        self.sweep = self.sweep_min = self.sweep_max = 0.0
        self.metrics = {}

    def check(self, observation):
        now, yaw = observation["time_s"], observation["yaw"]
        if not math.isfinite(now) or not math.isfinite(yaw):
            return "invalid_navigation_measurement"
        if self.started is None:
            self.started = self.last_progress = now
        targets = [d for d in observation["detections"] if d["color"] == self.color]
        elapsed = now - self.started
        self.metrics.update(elapsed_s=elapsed, target_color=self.color)
        if elapsed > 45:
            return "navigation_timeout"
        if not targets:
            if self.missing_since is None:
                self.missing_since = now
                self.sweep = self.sweep_min = self.sweep_max = 0.0
            elif self.last_yaw is not None:
                delta = yaw - self.last_yaw
                self.sweep += math.atan2(math.sin(delta), math.cos(delta))
                self.sweep_min = min(self.sweep_min, self.sweep)
                self.sweep_max = max(self.sweep_max, self.sweep)
            self.last_yaw = yaw
            coverage = self.sweep_max - self.sweep_min
            self.metrics.update(search_coverage_rad=coverage, missing_s=now - self.missing_since)
            if coverage >= 2 * math.pi + 0.25:
                return "target_not_found_after_full_search"
            if now - self.missing_since >= 10:
                return "target_search_timeout"
            return None
        self.last_yaw = yaw
        self.missing_since = None
        point = max(targets, key=lambda d: d["confidence"])["center_body_m"]
        if len(point) != 3 or not all(math.isfinite(v) for v in point):
            return "invalid_navigation_measurement"
        distance = math.hypot(point[0], point[1])
        if distance < self.best_range - 0.025:
            self.best_range, self.last_progress = distance, now
        self.metrics.update(
            distance_m=distance, best_distance_m=self.best_range, no_progress_s=now - self.last_progress
        )
        if now - self.last_progress >= 10:
            return "target_not_approaching"
        return None
