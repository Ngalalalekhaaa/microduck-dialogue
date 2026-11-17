"""Foot placement must change readiness, and broader reach must avoid retreat loops."""

import numpy as np
import pytest

from .rgb_tof_planner import SensorKickPlanner
from .rgb_tof_reach import choose_reachable_foot, foot_point


def _poses():
    return {
        side: dict(position_body_m=[0.005, sign * 0.042, -0.117], rotation_body_from_foot=np.eye(3).tolist())
        for side, sign in (("left", 1), ("right", -1))
    }


def _observation(point):
    return dict(
        time_s=1.0,
        sensor_age_s=0.0,
        yaw=0.0,
        tilt_deg=0.0,
        stationary=True,
        detections=[dict(color="red", confidence=1.0, center_body_m=point)],
        feet=_poses(),
    )


@pytest.mark.parametrize(
    "point,foot",
    [([0.106, 0.0147, -0.082], "left"), ([0.1072, -0.0509, -0.0819], "right"), ([0.076, 0.035, -0.08], "left")],
)
def test_old_failure_positions_and_closer_ball_trigger_settle_then_kick(point, foot):
    planner = SensorKickPlanner("red", foot_relative=True)
    obs = _observation(point)
    assert planner.decide(obs).skill == "stand"
    obs["time_s"] += 0.6
    assert planner.decide(obs).skill == f"kick_{foot}"


def test_identical_body_ball_position_is_not_kickable_with_retracted_feet():
    point, poses = [0.09, 0.035, -0.08], _poses()
    assert choose_reachable_foot(point, poses) == "left"
    for pose in poses.values():
        pose["position_body_m"][0] = -0.06
    assert choose_reachable_foot(point, poses) is None


def test_foot_transform_respects_nonidentity_foot_orientation():
    pose = dict(position_body_m=[0.02, 0.04, -0.1], rotation_body_from_foot=[[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(foot_point([0.01, 0.13, -0.06], pose), [0.09, 0.01, 0.04])


@pytest.mark.parametrize("age", [-0.01, 0.121, np.nan, np.inf, None])
def test_stale_or_invalid_encoder_frame_never_triggers_kick(age):
    planner = SensorKickPlanner("red", foot_relative=True)
    observation = _observation([0.09, 0.035, -0.08])
    observation["sensor_age_s"] = age
    assert planner.decide(observation) is None


@pytest.mark.parametrize(
    "bad_pose",
    [
        None,
        {},
        {"left": {}},
        {"left": dict(position_body_m=[float("nan"), 0, 0], rotation_body_from_foot=np.eye(3).tolist())},
    ],
)
def test_missing_or_invalid_encoder_pose_never_falls_back_to_body_rectangle(bad_pose):
    planner = SensorKickPlanner("red", foot_relative=True)
    observation = _observation([0.09, 0.035, -0.08])
    observation["feet"] = bad_pose
    assert planner.decide(observation) is None


def test_reachable_foot_precedes_legacy_alignment(monkeypatch):
    from .efficient_planner import EfficientKickPlanner

    def legacy_must_not_run(self, observation):
        pytest.fail("An encoder-confirmed kick position must not enter body alignment.")

    monkeypatch.setattr(EfficientKickPlanner, "decide", legacy_must_not_run)
    planner = SensorKickPlanner("red", foot_relative=True)
    observation = _observation([0.106, 0.0147, -0.082])
    decision = planner.decide(observation)
    assert (decision.phase, decision.skill) == ("settle_in_foot_reach", "stand")
    observation["time_s"] += 0.6
    assert planner.decide(observation).skill == "kick_left"


def test_selected_reachable_foot_survives_small_centerline_changes():
    planner = SensorKickPlanner("red", foot_relative=True)
    planner.foot = "right"
    observation = _observation([0.09, 0.001, -0.08])
    assert choose_reachable_foot(observation["detections"][0]["center_body_m"], observation["feet"]) == "left"
    assert planner.decide(observation).skill == "stand"
    observation["time_s"] += 0.6
    observation["detections"][0]["center_body_m"][1] = 0.002
    assert planner.decide(observation).skill == "kick_right"
    assert planner.foot == "right"


def test_changing_selected_foot_requires_another_settle():
    planner = SensorKickPlanner("red", foot_relative=True)
    planner.foot, planner.aligned = "left", True
    observation = _observation([0.09, 0.0, -0.08])
    observation["feet"]["left"]["position_body_m"][0] = -0.06
    decision = planner.decide(observation)
    assert (decision.phase, decision.skill, planner.foot) == ("settle_in_foot_reach", "stand", "right")
    observation["time_s"] += 0.6
    assert planner.decide(observation).skill == "kick_right"


@pytest.mark.parametrize("change,expected", [("position", "move_forward"), ("rotation", "move_backward")])
def test_current_encoder_pose_invalidates_previous_kick_readiness(change, expected):
    planner = SensorKickPlanner("red", foot_relative=True)
    observation = _observation([0.09, 0.035, -0.08])
    assert planner.decide(observation).skill == "stand"
    observation["time_s"] += 0.6
    for pose in observation["feet"].values():
        if change == "position":
            pose["position_body_m"][0] = -0.06
        else:
            pose["rotation_body_from_foot"] = np.diag([-1.0, -1.0, 1.0]).tolist()
    assert planner.decide(observation).skill == expected
    assert not planner.aligned
    observation["time_s"] += 0.6
    observation["feet"] = _poses()
    assert planner.decide(observation).skill == "stand"
    observation["time_s"] += 0.6
    assert planner.decide(observation).skill == "kick_left"


@pytest.mark.parametrize("stationary", [False, None, 0, np.nan, 1, "yes"])
@pytest.mark.parametrize("feet_reachable", [True, False])
def test_unverified_stationary_never_moves_or_kicks_near_ball(stationary, feet_reachable):
    planner = SensorKickPlanner("red", foot_relative=True)
    planner.foot, planner.aligned = "left", True
    observation = _observation([0.09, 0.035, -0.08])
    observation["stationary"] = stationary
    if not feet_reachable:
        for pose in observation["feet"].values():
            pose["position_body_m"][0] = -0.06
    decision = planner.decide(observation)
    assert decision is None or decision.skill == "stand"


@pytest.mark.parametrize("kind", ["nan_rotation", "reflection", "scaled_rotation", "short_position"])
def test_invalid_current_fk_revokes_already_aligned_kick(kind):
    planner = SensorKickPlanner("red", foot_relative=True)
    planner.foot, planner.aligned = "left", True
    observation = _observation([0.09, 0.035, -0.08])
    pose = observation["feet"]["left"]
    if kind == "short_position":
        pose["position_body_m"] = [0, 0]
    elif kind == "nan_rotation":
        pose["rotation_body_from_foot"][0][0] = np.nan
    elif kind == "reflection":
        pose["rotation_body_from_foot"] = np.diag([-1.0, 1.0, 1.0]).tolist()
    else:
        pose["rotation_body_from_foot"] = (2 * np.eye(3)).tolist()
    assert planner.decide(observation) is None
    assert planner.status == "invalid_foot_encoders"
