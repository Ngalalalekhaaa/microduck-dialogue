"""Evaluation labels, metric denominators, and watchdog cancellation are independent."""

import copy
import threading

from .evaluate_toy_selector import summarize
from .test_toy_session import FakeRuntime
from .toy_fixtures import active, request
from .toy_selector_cases import cases
from .toy_selector_rollout import _watchdog
from .toy_teacher import ToyTeacher


def test_reviewed_cases_have_independent_expected_labels_and_valid_control_order():
    teacher = ToyTeacher()
    fixtures = cases()
    assert len(fixtures) == 14 and len({case["id"] for case in fixtures}) == 14
    assert max(len(step["expected"]["goals"]) for case in fixtures for step in case["steps"]) == 6
    for case in fixtures:
        execution = request("")["execution"]
        for step in case["steps"]:
            state = request(step["text"], planned=True)
            state["execution"] = copy.deepcopy(execution)
            response = teacher.decide(state)
            assert {key: response[key] for key in ("decision", "goals")} == step["expected"], case["id"]
            if response["decision"] == "replace":
                execution = active(response["goals"][0], pending=response["goals"][1:])
            elif response["decision"] in ("pause", "resume"):
                execution["state"] = "paused" if response["decision"] == "pause" else "running"
            elif response["decision"] == "cancel":
                execution = request("")["execution"]


def test_wrong_plan_can_succeed_physically_without_improving_semantic_metric():
    report = dict(
        turns=[
            dict(
                semantic_exact=False,
                raw_selector_exact=False,
                model_trace={"decision_source": "learned_selector", "seconds": 0.4},
            ),
            dict(
                semantic_exact=True,
                raw_selector_exact=None,
                model_trace={"decision_source": "host_stop", "seconds": 0.01},
            ),
        ],
        semantic_all_correct=False,
        expected_finite_plan_verified=False,
        goal_status={"succeeded": 1, "cancelled": 1},
        error=None,
        runtime_failure=None,
        timed_out=False,
    )
    summary = summarize([report])
    assert summary["learned_selector_turns"] == 1 and summary["learned_raw_plan_exact"] == 0
    assert summary["accepted_response_exact"] == 0.5 and summary["host_control_turns"] == 1
    assert summary["completed_physical_success_fraction"] == 1
    assert summary["expected_finite_plans_verified"] == 0
    assert summary["cancelled_attempts_excluded_from_completed_fraction"] == 1
    assert summary["model_response_seconds_mean"] == 0.4
    report["error"] = {"type": "TimeoutError", "message": "case_simulation_time_limit"}
    report["timed_out"] = True
    bounded = summarize([report])
    assert bounded["errors"] == bounded["timed_out"] == 1
    assert bounded["program_errors"] == 0
    report["error"] = {"type": "RuntimeError", "message": "invalid executor state"}
    assert summarize([report])["program_errors"] == 1


def test_watchdog_sets_cancel_even_when_slow_inference_holds_world_lock(tmp_path):
    runtime = FakeRuntime(tmp_path)
    done, expired = threading.Event(), []
    thread = threading.Thread(target=_watchdog, args=(runtime, 0.0, 0.01, 100, done, expired), daemon=True)
    try:
        with runtime.lock:
            thread.start()
            assert runtime.cancel.wait(1)
            assert expired == ["case_wall_time_limit"]
        thread.join(1)
        assert not thread.is_alive()
    finally:
        done.set()
        thread.join(1)
