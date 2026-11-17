"""A nested ball controller must never release its parent queue's stop latch."""

from types import SimpleNamespace

from .plan_tasks import PlanTaskSession
from .test_plan_tasks import Runtime, action_plan, finish
from .test_skill_tasks import ScriptedBackend
from .test_tasks import MemoryRecorder
from .toy_fixtures import kick
from .toy_goal_runner import ToyGoalRunner


def test_nested_plan_preserves_cancellation_and_never_calls_model_or_motion():
    runtime = Runtime()
    runtime.request_stop()
    backend = ScriptedBackend(action_plan(), finish())
    session = PlanTaskSession(runtime, MemoryRecorder(), backend, verbose=False)
    assert session.run("踢红球", reset_cancel=False)["status"] == "cancelled"
    assert runtime.cancel.is_set()
    assert not runtime.calls and not backend.requests


def test_standalone_new_task_can_still_release_previous_stop():
    runtime = Runtime()
    runtime.request_stop()
    session = PlanTaskSession(runtime, MemoryRecorder(), ScriptedBackend(action_plan(), finish()), verbose=False)
    assert session.run("往前走一点")["status"] == "model_reported"
    assert runtime.calls and not runtime.cancel.is_set()


def test_stop_during_nested_kick_setup_cannot_be_cleared(tmp_path, monkeypatch):
    from . import collect_plan_teacher

    runtime = Runtime()
    runtime.output = tmp_path
    session = SimpleNamespace(runtime=runtime, seed=17, attempt_seq=0)

    def stop_while_setting_up(folder, *args):
        folder.mkdir(parents=True)
        runtime.request_stop()
        return MemoryRecorder()

    monkeypatch.setattr(collect_plan_teacher, "RGBRecorder", stop_while_setting_up)
    result = ToyGoalRunner(session)._kick(kick("red"), "task_001", 0)
    assert not result["success"] and result["reason"] == "cancelled"
    assert runtime.cancel.is_set() and not runtime.calls


def test_cancelled_goal_does_not_enter_nested_controller(monkeypatch):
    runtime = Runtime()
    runtime.request_stop()
    runner = ToyGoalRunner(SimpleNamespace(runtime=runtime))

    def unexpected(*args):
        raise AssertionError("Cancelled queue entered the nested kick controller.")

    monkeypatch.setattr(runner, "_kick", unexpected)
    result = runner.execute(kick("red"), "task_001", 0)
    assert result == dict(success=False, reason="cancelled", metrics={})
