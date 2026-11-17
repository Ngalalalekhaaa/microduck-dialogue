"""Optional presentation snapshots and wall-clock pacing; never advances physics."""

import copy
import math
import threading
import time


class InteractiveFrames:
    """Publish one completed physics tick, without exposing the world's mutable arrays."""

    def __init__(self, period_s=0.02, *, clock=time.monotonic):
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("Use a finite, positive presentation tick period.")
        self.period_s, self.clock = period_s, clock
        self._lock = threading.Lock()
        self._frame, self._sequence = None, 0

    def publish(self, snapshot):
        frame = copy.deepcopy(snapshot)
        with self._lock:
            self._sequence += 1
            frame.update(display_frame_id=self._sequence, display_wall_time_s=self.clock())
            self._frame = frame

    def snapshot(self):
        with self._lock:
            if self._frame is None:
                raise RuntimeError("Publish the initial physics snapshot before opening the viewer.")
            return copy.deepcopy(self._frame)

    def wait_after_tick(self, tick_started_s, cancel):
        """Wait outside the world lock, with no catch-up burst after inference stalls."""
        remaining = max(0.0, tick_started_s + self.period_s - self.clock())
        if remaining:
            cancel.wait(remaining)
