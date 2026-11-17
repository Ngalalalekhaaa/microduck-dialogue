"""Bound pursuit using measurements and pause only outside atomic kick recovery."""

import math

import pytest

from .efficient_planner import EfficientKickPlanner
from .kick_region import choose_foot
from .plan_protocol import VERSION, Plan
from .plan_tasks import PlanTaskSession
from .q1r_navigation_guard import NavigationGuard
from .test_plan_protocol import TARGET
from .test_plan_tasks import Runtime, action_plan, finish
from .test_skill_tasks import ScriptedBackend
from .test_tasks import MemoryRecorder, ball, observed


def test_search_wraps_measured_yaw_and_stops_after_one_complete_sweep():
    guard = NavigationGuard("red")
    for index in range(35):
        yaw = 3 + index * 0.2
        outcome = guard.check(observed(yaw=math.atan2(math.sin(yaw), math.cos(yaw)), time_s=index * 0.2))
        if outcome:
            break
    assert outcome == "target_not_found_after_full_search"
    assert 2 * math.pi + 0.25 <= guard.metrics["search_coverage_rad"] < 2 * math.pi + 0.5


def test_turn_oscillation_or_stalled_motion_times_out_without_claiming_full_sweep():
    guard = NavigationGuard("red")
    for index in range(51):
        outcome = guard.check(observed(yaw=0.1 if index % 2 else -0.1, time_s=index * 0.2))
    assert outcome == "target_search_timeout"
    assert guard.metrics["search_coverage_rad"] <= 0.21


def test_faster_target_cannot_be_chased_indefinitely():
    guard = NavigationGuard("red")
    assert guard.check(observed([ball(x=0.5)], time_s=0)) is None
    assert guard.check(observed([ball(x=0.7)], time_s=9.9)) is None
    assert guard.check(observed([ball(x=0.71)], time_s=10)) == "target_not_approaching"


def test_real_distance_progress_renews_pursuit_but_not_total_budget():
    guard = NavigationGuard("red")
    for index in range(10):
        assert guard.check(observed([ball(x=1.0 - index * 0.08)], time_s=index * 4.5)) is None
    assert guard.check(observed([ball(x=0.1)], time_s=45.1)) == "navigation_timeout"


def test_parent_pause_returns_current_goal_after_first_short_navigation_segment():
    runtime = Runtime()
    backend = ScriptedBackend(action_plan())
    backend.supports_continuation = False
    session = PlanTaskSession(runtime, MemoryRecorder(), backend, verbose=False)
    session.pause_boundary = lambda: (
        dict(success=False, paused=True, reason="safe_pause_boundary", metrics=dict(stable_observations=3))
        if runtime.calls
        else None
    )
    result = session.run("持续向目标移动")
    assert result["status"] == "paused"
    assert result["pause_metrics"]["stable_observations"] == 3
    assert runtime.calls == [("move_forward", 0.2)]


def test_pause_during_actual_kick_waits_through_recovery_and_does_not_repeat_kick():
    runtime = Runtime([ball(x=0.0895, y=0.0411)])
    kick = Plan(VERSION, "continue", TARGET, [{"skill": "kick_left", "seconds": 0.5}], "")
    backend = ScriptedBackend(kick, finish())
    backend.supports_continuation = False
    session = PlanTaskSession(runtime, MemoryRecorder(), backend, verbose=False)
    session.pause_boundary = lambda: (
        dict(success=False, paused=True, reason="safe_pause_boundary", metrics={}) if runtime.calls else None
    )
    result = session.run("踢红球")
    assert result["status"] == "model_reported"
    assert runtime.calls == [("kick_left", 0.5), ("stand", 3.0)]


@pytest.mark.parametrize("y", [-0.0024, 0.001, 0.008])
def test_center_line_ball_is_aligned_to_a_foot_before_advancing_or_kicking(y):
    point = [0.0976, y, -0.078]
    assert choose_foot(point) is None
    choice = EfficientKickPlanner("red").decide(observed([ball(x=point[0], y=y)]))
    assert choice.phase == "align"
    assert choice.skill == ("move_right" if y > 0.005 else "move_left")
    assert choice.seconds <= 0.2


def test_resume_evaluation_cannot_hide_contacts_recorded_before_pause(monkeypatch):
    from . import plan_tasks

    runtime = Runtime([ball(x=0.0895, y=0.0411)])
    runtime.contacts = [{"early_contact": True}]
    runtime.rows = ["pre_pause_row"]
    runtime.truth = lambda: {}
    checked = []

    def evaluate(color, foot, before, after, start, contacts, rows):
        checked.append((contacts, rows))
        return {"success": False}

    monkeypatch.setattr(plan_tasks, "evaluate_kick", evaluate)
    kick_plan = Plan(VERSION, "continue", TARGET, [{"skill": "kick_left", "seconds": 0.5}], "")
    backend = ScriptedBackend(kick_plan, finish())
    backend.supports_continuation = False
    session = PlanTaskSession(runtime, MemoryRecorder(), backend, expected_color="red", verbose=False)
    session.evaluation_contact_start = session.evaluation_row_start = 0
    assert session.run("踢红球")["status"] == "verification_failed"
    assert checked == [([{"early_contact": True}], ["pre_pause_row"])]
