"""Sensor-only kick decisions, near-target loss and bounded settling budgets."""

import math
from types import SimpleNamespace

import pytest

from .rgb_tof_planner import SensorKickPlanner
from .rgb_tof_run import SensorServo


def _observation(point=(0.09, 0.035, -0.08), *, time_s=0.0, stationary=True, tilt=0.0):
    detections = [] if point is None else [{"color": "red", "center_body_m": list(point), "confidence": 0.9}]
    return {
        "time_s": time_s,
        "sensor_time_s": time_s,
        "yaw": 0.0,
        "tilt_deg": tilt,
        "stationary": stationary,
        "detections": detections,
    }


@pytest.mark.parametrize("key", ["time_s", "yaw", "tilt_deg"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_body_observation_never_produces_an_action(key, value):
    planner = SensorKickPlanner("red")
    observation = _observation()
    observation[key] = value
    assert planner.decide(observation) is None
    assert planner.status == "invalid_vision"


@pytest.mark.parametrize("point", [(math.nan, 0, 0), (0.1, math.inf, 0), (0.1, 0, -math.inf), (0.1, 0)])
def test_invalid_target_position_never_produces_an_action(point):
    planner = SensorKickPlanner("red")
    assert planner.decide(_observation(point)) is None
    assert planner.status == "invalid_vision"


@pytest.mark.parametrize("tilt", [-0.1, 25.0, 26.0, 90.0])
def test_tilt_at_or_beyond_upright_limit_is_refused(tilt):
    planner = SensorKickPlanner("red")
    assert planner.decide(_observation(tilt=tilt)) is None
    assert planner.status == "unstable_posture"


@pytest.mark.parametrize("point", [(0.05, 0.07, -0.08), (0.09, 0.035, -0.08), (0.159, -0.07, -0.08), (0.199, 0, -0.08)])
def test_near_unsettled_body_cannot_translate_turn_or_kick(point):
    planner = SensorKickPlanner("red")
    planner.aligned, planner.foot = True, "left"
    decision = planner.decide(_observation(point, stationary=False))
    assert decision is not None
    assert decision.skill == "stand"
    assert 0 < decision.seconds <= 0.6


@pytest.mark.parametrize("foot,side", [("left", 1), ("right", -1)])
def test_interior_foot_target_requires_alignment_and_stationary_confirmation(foot, side):
    planner = SensorKickPlanner("red")
    point = (0.09, side * 0.035, -0.08)
    unsettled = planner.decide(_observation(point, stationary=False))
    assert unsettled.skill == "stand"
    aligned = planner.decide(_observation(point, time_s=0.4))
    assert aligned.skill == "stand"
    assert planner.aligned and planner.foot == foot
    moving_again = planner.decide(_observation(point, time_s=1.0, stationary=False))
    assert moving_again.skill == "stand"
    ready = planner.decide(_observation(point, time_s=1.4))
    assert ready.phase == "kick"
    assert ready.skill == f"kick_{foot}"
    assert ready.seconds == pytest.approx(0.5)


@pytest.mark.parametrize("point", [None, (0.4, 0, -0.08), (0.4, 0.2, -0.08), (0.15, 0.07, -0.08), (0.06, 0.035, -0.08)])
def test_each_decision_has_finite_bounded_tick_aligned_duration(point):
    planner = SensorKickPlanner("red")
    decision = planner.decide(_observation(point))
    assert decision is not None
    assert decision.skill in {
        "stand",
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
        "kick_left",
        "kick_right",
    }
    assert math.isfinite(decision.seconds) and 0.02 <= decision.seconds <= 3.0
    assert decision.seconds / 0.02 == pytest.approx(round(decision.seconds / 0.02))
    heading = getattr(decision, "heading", None)
    assert heading is None or math.isfinite(heading)


@pytest.mark.parametrize("point,stationary", [((0.09, 0.035, -0.08), False), (None, True)])
def test_ninety_second_budget_also_applies_to_settling_and_search(point, stationary):
    planner = SensorKickPlanner("red")
    assert planner.decide(_observation(point, stationary=stationary)) is not None
    assert planner.decide(_observation(point, time_s=90.01, stationary=stationary)) is None
    assert planner.status == "timeout"


def test_three_hundred_step_budget_also_applies_to_early_settling():
    planner = SensorKickPlanner("red")
    for index in range(300):
        decision = planner.decide(_observation(time_s=index * 0.01, stationary=False))
        assert decision is not None, f"Settling exhausted its 300-step budget at step {index + 1}."
        assert decision.skill == "stand"
    assert planner.decide(_observation(time_s=3.0, stationary=False)) is None
    assert planner.status == "timeout"


class _Runtime:
    """Only sensor observations and the public execution method are available."""

    def __init__(self, points):
        self.points = iter(points)
        self.target_color = "red"
        self.policy = SimpleNamespace(motion=SimpleNamespace(visual_heading=None))
        self.diagnostics = {"rejected_rgb_candidates": []}
        self.time_s = 0.0
        self.executed = []

    def observe(self):
        return _observation(next(self.points), time_s=self.time_s), {}, {}

    def execute(self, skill, seconds):
        self.executed.append((skill, seconds))
        self.time_s += seconds
        return True


def test_lost_near_ball_stops_after_bounded_standing_without_search_turns():
    runtime = _Runtime([(0.15, 0.035, -0.08), *([None] * 10)])
    servo = SensorServo(runtime)
    assert servo.observe()["detections"]
    with pytest.raises(RuntimeError, match="stationary reacquisition"):
        servo.kick()
    assert runtime.executed
    assert all(skill == "stand" for skill, _ in runtime.executed)
    assert sum(seconds for _, seconds in runtime.executed) <= 2.0
    assert all(item["phase"] == "head_acquire" for item in servo.decisions)


def test_near_ball_can_reappear_without_any_blind_motion():
    point = (0.15, 0.035, -0.08)
    runtime = _Runtime([point, None, None, point])
    servo = SensorServo(runtime)
    servo.observe()
    reacquired = servo.observe()
    assert reacquired["detections"][0]["center_body_m"] == list(point)
    assert runtime.executed == [("stand", 0.2), ("stand", 0.2)]
