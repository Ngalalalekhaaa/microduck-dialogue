"""Command boundary and image-only planner checks, requiring no simulator."""

import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from planner import MarkerPlanner, SkillCommand, validate_command


def observed(forward=0.2, left=0.0, found=True):
    # Deliberately supply no simulator, world position, or scenario information.
    return SimpleNamespace(found=found, forward_m=forward, left_m=left)


@pytest.mark.parametrize(
    ("forward", "left", "skill"),
    [(0.4, 0, "move_forward"), (0.05, 0, "move_backward"), (0.2, 0.1, "move_left"), (0.2, -0.1, "move_right")],
)
def test_four_directions_use_observed_robot_frame(forward, left, skill):
    command = MarkerPlanner().decide(observed(forward, left))
    assert command.skill == skill
    validate_command(command)


@pytest.mark.parametrize(("forward", "left"), [(0.2, 0), (0.18, -0.03), (0.23, 0.03), (0.165, -0.035), (0.235, 0.035)])
def test_arrived_goal_box_holds(forward, left):
    assert MarkerPlanner().decide(observed(forward, left)) == SkillCommand("stand")


@pytest.mark.parametrize("coordinate", [None, math.nan, math.inf, -math.inf, "0.3", True])
@pytest.mark.parametrize("axis", ["forward", "left"])
def test_invalid_measurement_holds(coordinate, axis):
    assert MarkerPlanner().decide(observed(**{axis: coordinate})) == SkillCommand("stand")


def test_lost_detection_holds_despite_stale_position():
    assert MarkerPlanner().decide(observed(0.8, 0.4, found=False)) == SkillCommand("stand")
    assert MarkerPlanner().decide(None) == SkillCommand("stand")


def test_larger_normalized_error_is_corrected_first():
    planner = MarkerPlanner()
    assert planner.decide(observed(0.25, -0.15)).skill == "move_right"
    assert planner.decide(observed(0.45, 0.1)).skill == "move_forward"


def test_planner_needs_no_oracle_and_reacts_to_each_observation():
    planner = MarkerPlanner()
    assert planner.decide(observed(0.35)).skill == "move_forward"
    assert planner.decide(observed(0.1)).skill == "move_backward"
    assert planner.decide(observed(0.2)) == SkillCommand("stand")


def test_larger_errors_do_not_produce_shorter_pulses():
    planner = MarkerPlanner()
    for positions in [(0.24, 0.28, 0.4), (0.16, 0.12, 0.02)]:
        durations = [planner.decide(observed(position)).seconds for position in positions]
        assert durations == sorted(durations)


@pytest.mark.parametrize("skill", ["move_forward", "move_backward", "move_left", "move_right"])
@pytest.mark.parametrize("seconds", [0.12, 0.16, 0.3, 0.32, 0.6, 0.8])
def test_valid_pulses(skill, seconds):
    command = SkillCommand(skill, seconds)
    assert validate_command(command) is command


@pytest.mark.parametrize("seconds", [math.nan, math.inf, -math.inf, -0.2, 0, 0.1, 0.13, 0.81, "0.2", True])
def test_invalid_pulses_are_refused(seconds):
    with pytest.raises(ValueError):
        validate_command(SkillCommand("move_forward", seconds))


@pytest.mark.parametrize("skill", ["run", "turn_left", "", None, ["stand"]])
def test_unknown_skills_are_refused(skill):
    with pytest.raises(ValueError):
        validate_command(SkillCommand(skill, 0.2))


def test_stand_uses_zero_duration_and_commands_are_immutable():
    command = SkillCommand("stand")
    validate_command(command)
    with pytest.raises(ValueError):
        validate_command(SkillCommand("stand", 0.2))
    with pytest.raises(FrozenInstanceError):
        command.seconds = 0.2
    with pytest.raises(TypeError):
        validate_command({"skill": "stand", "seconds": 0})


@pytest.mark.parametrize("name", ["standoff", "tolerance"])
@pytest.mark.parametrize("value", [0, -0.1, math.nan, math.inf, "0.2", True])
def test_invalid_goal_configuration_is_refused(name, value):
    with pytest.raises(ValueError):
        MarkerPlanner(**{name: value})
