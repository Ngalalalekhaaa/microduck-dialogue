"""Image-based skill selection for the local simulation experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from vision import TargetObservation


@dataclass(frozen=True)
class SkillCommand:
    """One bounded motion pulse, or a request to remain standing."""

    skill: str
    seconds: float = 0.0


def validate_command(command: SkillCommand) -> SkillCommand:
    """Reject unsupported skills and durations before calling an executor."""
    if not isinstance(command, SkillCommand):
        raise TypeError("Provide a SkillCommand with a skill name and duration in seconds.")
    allowed = {"stand", "move_forward", "move_backward", "move_left", "move_right"}
    if not isinstance(command.skill, str) or command.skill not in allowed:
        raise ValueError(f"Choose a supported skill: {', '.join(sorted(allowed))}.")
    seconds = command.seconds
    if isinstance(seconds, bool) or not isinstance(seconds, Real) or not math.isfinite(seconds):
        raise ValueError("Set seconds to a finite number.")
    if command.skill == "stand":
        if seconds != 0.0:
            raise ValueError("Use seconds=0 for stand; the executor maintains its standing control loop.")
    elif not 0.12 <= seconds <= 0.8:
        raise ValueError("Set a movement duration between 0.12 and 0.8 seconds.")
    elif not math.isclose(seconds / 0.02, round(seconds / 0.02), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Set a movement duration in multiples of 0.02 seconds.")
    return command


class Planner(Protocol):
    """Select a skill using observed target coordinates, without world target state.

    A future VLM adapter can implement this method. Its returned command must
    pass ``validate_command`` before execution; it cannot bypass motion bounds.
    The runner observes again after each pulse and settling interval.
    """

    def decide(
        self, observation: TargetObservation, *, rgb: np.ndarray | None = None, instruction: str | None = None
    ) -> SkillCommand: ...


class MarkerPlanner:
    """Correct the larger image-derived position error with a short motion pulse.

    Distances are in metres in the robot's horizontal frame: forward and left
    are positive. Pulse lengths are empirical starting points, not guaranteed
    displacements. The executor supplies fixed velocity commands and settling.
    """

    def __init__(self, standoff: float = 0.2, tolerance: float = 0.035):
        for name, value in (("standoff", standoff), ("tolerance", tolerance)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Set {name} to a positive, finite distance in metres.")
        self.standoff = standoff
        self.tolerance = tolerance

    def decide(
        self, observation: TargetObservation | None, *, rgb: np.ndarray | None = None, instruction: str | None = None
    ) -> SkillCommand:
        """Hold when detection is unavailable or already inside the goal box."""
        stand = SkillCommand("stand")
        if observation is None or not observation.found:
            return stand
        coordinates = (observation.forward_m, observation.left_m)
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in coordinates
        ):
            return stand
        forward_error = observation.forward_m - self.standoff
        left_error = observation.left_m
        forward_arrived = self.standoff - self.tolerance <= observation.forward_m <= self.standoff + self.tolerance
        if forward_arrived and abs(left_error) <= self.tolerance:
            return stand
        if abs(forward_error) / self.tolerance >= abs(left_error) / self.tolerance:
            if forward_error > 0:
                seconds = 0.6 if forward_error > 0.12 else 0.4 if forward_error > 0.065 else 0.32
                return SkillCommand("move_forward", seconds)
            seconds = 0.4 if -forward_error > 0.12 else 0.3 if -forward_error > 0.065 else 0.2
            return SkillCommand("move_backward", seconds)
        seconds = 0.32 if abs(left_error) > 0.09 else 0.16
        return SkillCommand("move_left" if left_error > 0 else "move_right", seconds)
