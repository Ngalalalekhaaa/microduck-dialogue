"""Motion finite checks, reference continuity, and observed-target navigation regressions."""

import math

import pytest

from .efficient_planner import EfficientKickPlanner
from .kick_region import reachable
from .motion import HeadingHold
from .test_tasks import ball, observed


def test_heading_reference_persists_across_contiguous_forward_chunks():
    motion = HeadingHold({"move_forward": [0.2, 0, -0.1]})
    motion.command("move_forward", (0.2, 0, 0), 0.7, 0)
    correction = motion.command("move_forward", (0.2, 0, 0), 0.9, 0)
    assert motion.reference == 0.7
    assert correction[2] < -0.1
    motion.command("stand", (0, 0, 0), 0.9, 0)
    motion.command("move_forward", (0.2, 0, 0), 0.9, 0)
    assert motion.reference == 0.9


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_invalid_motion_sensor_and_calibration_are_refused(bad):
    with pytest.raises(ValueError):
        HeadingHold({"move_left": [bad, 0.2, 0]})
    with pytest.raises(ValueError):
        HeadingHold().command("move_forward", (0.2, 0, 0), bad, 0)
    motion = HeadingHold()
    motion.visual_heading = bad
    with pytest.raises(ValueError):
        motion.command("move_forward", (0.2, 0, 0), 0, 0)


def test_heading_wrap_and_visual_target_do_not_read_world_position():
    motion = HeadingHold()
    motion.command("move_forward", (0.2, 0, 0), math.pi - 0.01, 0)
    assert abs(motion.command("move_forward", (0.2, 0, 0), -math.pi + 0.01, 0)[2]) < 0.1
    motion.visual_heading = -math.pi + 0.2
    assert motion.command("move_forward", (0.2, 0, 0), -math.pi + 0.01, 0)[2] > 0


def test_centered_reachable_ball_needs_no_lateral_alignment():
    planner = EfficientKickPlanner("red")
    obs = observed([ball(x=0.10, y=0)])
    assert planner.decide(obs).skill == "stand"
    assert planner.decide(obs).skill.startswith("kick_")


def test_center_hit_margin_and_distant_or_nonfinite_ball_are_refused():
    assert not reachable([0.12, 0, -0.08], "right")
    assert not reachable([0.15, 0.04, -0.08], "left")
    assert not reachable([math.nan, 0, -0.08], "right")
    assert reachable([0.10, 0, -0.08], "right")
    assert reachable([0.10, 0.03, -0.08], "left")


def test_lost_target_never_continues_forward_and_obstacle_is_avoided():
    planner = EfficientKickPlanner("red")
    assert planner.decide(observed([ball(x=0.4, y=0)])).skill == "move_forward"
    assert planner.decide(observed()).skill.startswith("turn_")
    decision = planner.decide(observed([ball(x=0.4), ball("blue", x=0.16, y=0)]))
    assert decision.phase == "avoid"


def test_far_visible_target_is_faced_then_tracked_in_camera_axes():
    planner = EfficientKickPlanner("red")
    assert planner.decide(observed([ball(x=0.4, y=0.2)])).skill == "turn_left"
    decision = planner.decide(observed([ball(x=0.4, y=0.02)], yaw=0.7))
    assert decision.skill == "move_forward"
    assert 0.7 < decision.heading < 0.8
