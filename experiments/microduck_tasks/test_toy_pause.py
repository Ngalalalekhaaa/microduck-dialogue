"""Safe pause finishes inference and complete goals, then retains only unfinished work."""

import threading
from types import SimpleNamespace

import pytest

from .test_toy_session import QueueBackend, _background
from .test_toy_session import session_factory as session_factory
from .test_toy_stop import BlockedBackend
from .toy_dispatch import ToyInputDispatcher
from .toy_fixtures import kick, move, perform, posture, turn
from .toy_goal_runner import ToyGoalRunner


def test_pause_signal_ignores_busy_locks_without_cancelling_motion_or_inference(session_factory):
    session = session_factory()
    epoch = session.cancel_epoch
    with session.user_lock, session.state_lock, session.dialogue_lock, session.runtime.lock:
        thread, done, errors = _background(session.request_pause)
        assert done.wait(0.5) and not errors
        assert session.pause_requested
        assert not session.runtime.cancel.is_set()
        assert session.cancel_epoch == epoch
    thread.join(1)


@pytest.mark.parametrize("lockstep", [False, True])
@pytest.mark.parametrize("after_stop", [False, True])
def test_pause_during_inference_preserves_generated_plan_without_starting_it(session_factory, lockstep, after_stop):
    session = session_factory()
    if after_stop:
        session.run("停")
    backend = BlockedBackend(lockstep=lockstep)
    session.backend = backend
    responses = []
    thread, done, errors = _background(lambda: responses.append(session.run("生成新计划", wait=False)))
    try:
        assert backend.entered.wait(1)
        epoch, cancelled = session.cancel_epoch, session.runtime.cancel.is_set()
        session.request_pause()
        assert session.pause_requested and not done.is_set()
        assert session.cancel_epoch == epoch and session.runtime.cancel.is_set() == cancelled
        backend.release.set()
        assert done.wait(1) and not errors
        session.wait(1)
        assert responses[0]["decision"] == "replace"
        assert session.execution["state"] == "paused"
        assert session.execution["active"] == perform("look_around")
        assert session.runner.calls == [] and session.runner.settle_calls == 1
        assert session.run("继续")["decision"] == "resume"
        assert [call[0] for call in session.runner.calls] == [perform("look_around")]
    finally:
        backend.release.set()
        thread.join(2)


@pytest.mark.parametrize("goal", [perform("roll"), kick("red"), turn(), move(), posture("sitting")])
def test_finite_goal_and_stability_recovery_complete_before_pause(session_factory, goal):
    session = session_factory(blocked=(0,))
    tail = posture("standing") if goal == posture("sitting") else perform("nod")
    session.backend = QueueBackend([goal, tail])
    session.run("执行长队列", wait=False)
    assert session.runner.started[0].wait(1)
    stop_count, epoch = session.runtime.stop_count, session.cancel_epoch
    assert session.run("暂停", wait=False)["decision"] == "pause"
    assert session.execution["state"] == "running" and not session.runtime.cancel.is_set()
    assert session.runtime.stop_count == stop_count and session.cancel_epoch == epoch
    assert session.runner.settle_calls == 0
    session.runner.release[0].set()
    session.wait(1)
    assert session.runner.settle_calls == 1
    assert session.execution["state"] == "paused" and session.execution["active"] == tail
    assert [row["status"] for row in session.results] == ["succeeded"]
    session.run("继续")
    assert [call[0] for call in session.runner.calls] == [goal, tail]
    assert [call[2] for call in session.runner.calls] == [0, 1]


def test_last_finite_goal_finishes_to_idle_instead_of_replaying_completed_work(session_factory):
    session = session_factory(blocked=(0,))
    session.run("前滚翻", wait=False)
    assert session.runner.started[0].wait(1)
    session.request_pause()
    session.runner.release[0].set()
    session.wait(1)
    assert session.execution["state"] == "idle" and not session.pause_requested
    assert session.run("继续")["decision"] == "clarify"
    assert len(session.runner.calls) == 1 and session.results[0]["status"] == "succeeded"


def test_pause_during_feedback_waits_for_inference_then_stands_before_next_goal(session_factory):
    session = session_factory(blocked=(0,))
    session.run("先点头，再左右看", wait=False)
    assert session.runner.started[0].wait(1)
    backend = BlockedBackend(feedback=True, lockstep=True)
    session.backend = backend
    session.runner.release[0].set()
    try:
        assert backend.entered.wait(1)
        session.request_pause()
        assert not session.runtime.cancel.is_set()
        backend.release.set()
        session.wait(1)
        assert session.execution["state"] == "paused"
        assert session.execution["active"] == perform("look_around")
        assert len(session.runner.calls) == 1 and session.runner.settle_calls == 1
        session.run("继续")
        assert [call[0] for call in session.runner.calls] == [perform("nod"), perform("look_around")]
    finally:
        backend.release.set()


def test_continue_withdraws_pending_pause_without_restarting_current_goal(session_factory):
    session = session_factory(blocked=(0,))
    session.run("先前滚翻，再点头", wait=False)
    assert session.runner.started[0].wait(1)
    session.run("暂停", wait=False)
    assert session.run("继续", wait=False)["decision"] == "respond"
    assert not session.pause_requested
    session.runner.release[0].set()
    session.wait(1)
    assert [call[0] for call in session.runner.calls] == [perform("roll"), perform("nod")]
    assert session.execution["state"] == "idle" and session.runner.settle_calls == 0


def test_continue_between_continuous_boundary_and_commit_is_not_lost(session_factory):
    session = session_factory()
    session.run("一直前进")
    assert session.runner.started[0].wait(1)
    with session.dialogue_lock:
        session.request_pause()
        assert session.runner.finished[0].wait(1)
        assert session.run("继续", wait=False)["decision"] == "respond"
        assert not session.pause_requested
    assert session.runner.started[1].wait(1)
    assert session.execution["state"] == "running"
    assert session.runner.calls[0][0] == session.runner.calls[1][0] == move(continuous=True)
    session.run("停")


def test_dispatch_does_not_reapply_old_pause_after_later_continue(session_factory):
    session = session_factory(blocked=(0,))
    inputs = ToyInputDispatcher(session, threading.Event(), threading.Event(), [])
    session.run("先前滚翻，再点头", wait=False)
    assert session.runner.started[0].wait(1)
    pause, _ = inputs.submit("暂停")
    resume, _ = inputs.submit("继续")
    assert not session.pause_requested
    inputs.close()
    inputs.dispatch()
    assert pause.is_set() and resume.is_set() and not inputs.errors
    assert not session.pause_requested
    session.runner.release[0].set()
    session.wait(1)
    assert session.execution["state"] == "idle" and session.runner.settle_calls == 0


def test_cancel_dominates_safe_pause_and_drops_every_remaining_goal(session_factory):
    session = session_factory(blocked=(0,))
    session.run("先前滚翻，再点头", wait=False)
    assert session.runner.started[0].wait(1)
    session.request_pause()
    assert session.run("停")["decision"] == "cancel"
    assert not session.pause_requested and session.runtime.cancel.is_set()
    assert session.execution["state"] == "idle" and session.execution["active"] is None
    assert len(session.runner.calls) == 1 and session.results[0]["status"] == "cancelled"


def test_old_resume_cannot_clear_new_pause_after_emergency_cancel(session_factory, monkeypatch):
    session = session_factory(blocked=(0, 1))
    session.run("前滚翻", wait=False)
    assert session.runner.started[0].wait(1)
    epoch = session.cancel_epoch
    entered, release = threading.Event(), threading.Event()
    resume = session.request_resume

    def delayed_resume(*, expected_cancel_epoch=None):
        entered.set()
        assert release.wait(3)
        return resume(expected_cancel_epoch=expected_cancel_epoch)

    monkeypatch.setattr(session, "request_resume", delayed_resume)
    thread, done, errors = _background(lambda: session.run("继续", wait=False, expected_cancel_epoch=epoch))
    try:
        assert entered.wait(1)
        session.run("停")
        session.run("前滚翻", wait=False)
        assert session.runner.started[1].wait(1)
        session.request_pause()
        release.set()
        assert done.wait(1) and not errors
        assert session.pause_requested, "An old request must not overwrite controls for the new task."
    finally:
        release.set()
        thread.join(2)


@pytest.mark.parametrize("unstable", [False, True])
def test_stability_confirmation_requires_consecutive_observations(unstable):
    observations = iter([False, True, False, True, True, True] if not unstable else [False] * 10)
    pulses = []
    runtime = SimpleNamespace(
        cancel=threading.Event(),
        execute=lambda skill, seconds: pulses.append((skill, seconds)) or True,
        observe=lambda: (dict(tilt_deg=3.0, stationary=next(observations)), None, None),
    )
    runner = ToyGoalRunner(SimpleNamespace(runtime=runtime))
    result = runner.settle_for_pause()
    assert result["success"] is not unstable
    assert len(pulses) == (10 if unstable else 6)
    assert all(pulse == ("stand", 0.2) for pulse in pulses)
    assert not runtime.cancel.is_set()


def test_cancel_during_stability_recovery_is_immediate():
    runtime = SimpleNamespace(cancel=threading.Event())

    def execute(skill, seconds):
        runtime.cancel.set()
        return False

    runtime.execute = execute
    runtime.observe = lambda: pytest.fail("Cancellation must not wait for another observation.")
    result = ToyGoalRunner(SimpleNamespace(runtime=runtime)).settle_for_pause()
    assert not result["success"] and result["reason"] == "cancelled"


@pytest.mark.parametrize("raises", [False, True])
def test_failed_initial_stability_does_not_latch_pause_for_a_later_new_task(session_factory, monkeypatch, raises):
    session = session_factory()
    settle = session.runner.settle_for_pause

    def failed_settle():
        if raises:
            raise RuntimeError("Sensor observation failed.")
        return dict(success=False, reason="pause_stability_unverified", metrics={})

    monkeypatch.setattr(session.runner, "settle_for_pause", failed_settle)
    session.request_pause()
    if raises:
        with pytest.raises(RuntimeError, match="Goal worker failed"):
            session.run("先前滚翻，再点头")
    else:
        session.run("先前滚翻，再点头")
    assert session.execution["state"] == "failed" and not session.pause_requested
    assert session.execution["active"] is None and session.execution["pending"] == []
    assert session.runner.calls == []
    monkeypatch.setattr(session.runner, "settle_for_pause", settle)
    session.run("点头")
    assert session.execution["state"] == "idle"
    assert [call[0] for call in session.runner.calls] == [perform("nod")]
