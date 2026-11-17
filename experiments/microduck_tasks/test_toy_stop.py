"""Host stop preempts blocked inference and invalidates every stale queue update."""

import copy
import threading

import pytest

from .test_toy_session import QueueBackend, _background, _long_queue
from .test_toy_session import session_factory as session_factory
from .toy_fixtures import move, perform
from .toy_protocol import VERSION
from .toy_teacher import ToyTeacher


class BlockedBackend(ToyTeacher):
    def __init__(self, *, decision="replace", lockstep=False, feedback=False):
        self.decision, self.requires_lockstep, self.feedback = decision, lockstep, feedback
        self.entered, self.release = threading.Event(), threading.Event()

    def decide(self, request):
        matching = request["event"]["kind"] == ("feedback" if self.feedback else "user")
        if matching:
            self.entered.set()
            if not self.release.wait(3):
                raise TimeoutError("Release the test inference gate.")
            return dict(
                schema_version=VERSION,
                reply="旧推理声称会执行后续目标。",
                decision="respond" if self.feedback else self.decision,
                goals=[perform("look_around")] if not self.feedback and self.decision in ("replace", "append") else [],
            )
        return super().decide(request)


def test_request_cancel_ignores_every_session_and_world_lock(session_factory):
    session = session_factory()
    before = session.cancel_epoch
    with session.user_lock, session.state_lock, session.dialogue_lock, session.runtime.lock:
        thread, done, errors = _background(session.request_cancel)
        assert done.wait(0.5), "Stop signalling must not acquire any main session or world lock."
        assert not errors
        assert session.runtime.cancel.is_set()
    thread.join(1)
    assert session.cancel_epoch == before + 1
    assert "正在" in session.last_reply


@pytest.mark.parametrize("text", ["停", "停下", "停一下", "先别继续了", "不用做了"])
def test_host_stop_clears_long_queue_and_new_explicit_task_can_start(session_factory, text):
    session = session_factory(blocked=(0,), call_count=21)
    session.backend = QueueBackend(_long_queue(20))
    session.run("执行长队列", wait=False)
    assert session.runner.started[0].wait(1)
    assert session.run(text)["decision"] == "cancel"
    assert len(session.runner.calls) == 1
    assert session.results[0]["status"] == "cancelled"
    assert session.execution["state"] == "idle"
    assert session.execution["active"] is None and session.execution["pending"] == []
    assert session.run("点头")["decision"] == "replace"
    assert [call[0] for call in session.runner.calls][1:] == [perform("nod")]
    assert session.results[-1]["status"] == "succeeded"


@pytest.mark.parametrize("misbehavior", ["raise", "replace"])
def test_stop_does_not_call_or_trust_backend(session_factory, misbehavior):
    session = session_factory(blocked=(0,))
    session.run("先点头，再左右看", wait=False)
    assert session.runner.started[0].wait(1)
    calls = []

    class HostileBackend:
        def decide(self, request):
            calls.append(request)
            if misbehavior == "raise":
                raise RuntimeError("Backend must not participate in a host stop.")
            return dict(schema_version=VERSION, reply="继续动作。", decision="replace", goals=[perform("nod")])

    session.backend = HostileBackend()
    assert session.run("停")["decision"] == "cancel"
    assert calls == []
    assert len(session.runner.calls) == 1
    assert session.execution["state"] == "idle" and session.execution["pending"] == []


@pytest.mark.parametrize("lockstep", [False, True])
@pytest.mark.parametrize("decision", ["replace", "append"])
def test_stop_latches_during_inference_and_discards_stale_plan(session_factory, lockstep, decision):
    session = session_factory(blocked=(0,), call_count=20)
    session.backend = QueueBackend(_long_queue(20))
    session.run("执行长队列", wait=False)
    assert session.runner.started[0].wait(1)
    backend = BlockedBackend(decision=decision, lockstep=lockstep)
    session.backend = backend
    responses = []
    inference, inferred, inference_errors = _background(lambda: responses.append(session.run("另一个任务", wait=False)))
    assert backend.entered.wait(1)
    stopping, stopped, stop_errors = _background(lambda: session.run("停", wait=False))
    try:
        assert session.runtime.cancel.wait(0.5), "Inference must never delay the cancellation latch."
        assert not inferred.is_set()
        assert session.last_reply == session._cancel_response()["reply"]
        backend.release.set()
        assert inferred.wait(1) and stopped.wait(1)
        assert not inference_errors and not stop_errors
        assert responses[0]["decision"] == "cancel"
        assert len(session.runner.calls) == 1
        assert session.results[0]["status"] == "cancelled"
        assert session.execution["state"] == "idle" and session.execution["pending"] == []
        assert session.last_reply == session._cancel_response()["reply"]
    finally:
        backend.release.set()
        inference.join(2)
        stopping.join(2)


def test_stale_resume_cannot_restart_paused_continuous_motion(session_factory):
    session = session_factory()
    session.run("一直前进")
    assert session.runner.started[0].wait(1)
    assert session.run("暂停")["decision"] == "pause"
    assert session.execution["active"] == move(continuous=True)
    backend = BlockedBackend(decision="resume", lockstep=True)
    session.backend = backend
    responses = []
    thread, done, errors = _background(lambda: responses.append(session.run("请接着执行剩余目标", wait=False)))
    try:
        assert backend.entered.wait(1)
        session.request_cancel()
        backend.release.set()
        assert done.wait(1) and not errors
        assert responses[0]["decision"] == "cancel"
        assert len(session.runner.calls) == 1
        assert session.execution["state"] == "idle" and session.execution["active"] is None
        assert session.runtime.cancel.is_set()
    finally:
        backend.release.set()
        thread.join(2)


def test_cancel_between_plan_acceptance_and_start_cannot_clear_latch(session_factory, monkeypatch):
    session = session_factory()
    entered, release = threading.Event(), threading.Event()
    start = session._start

    def gated_start(epoch):
        entered.set()
        assert release.wait(3)
        return start(epoch)

    monkeypatch.setattr(session, "_start", gated_start)
    responses = []
    thread, done, errors = _background(lambda: responses.append(session.run("点头", wait=False)))
    try:
        assert entered.wait(1)
        session.request_cancel()
        release.set()
        assert done.wait(1) and not errors
        assert responses[0]["decision"] == "cancel"
        assert session.runner.calls == []
        assert session.runtime.cancel.is_set()
        assert session.execution["state"] == "idle" and session.execution["pending"] == []
    finally:
        release.set()
        thread.join(2)


def test_cancel_during_completion_feedback_prevents_next_goal_and_stale_reply(session_factory):
    session = session_factory(blocked=(0,), call_count=20)
    session.backend = QueueBackend(_long_queue(20))
    session.run("执行长队列", wait=False)
    assert session.runner.started[0].wait(1)
    backend = BlockedBackend(feedback=True, lockstep=True)
    session.backend = backend
    session.runner.release[0].set()
    try:
        assert backend.entered.wait(1)
        signal, done, errors = _background(session.request_cancel)
        assert done.wait(0.5) and not errors
        signal.join(1)
        backend.release.set()
        session.wait(1)
        assert len(session.runner.calls) == 1
        assert session.execution["state"] == "idle" and session.execution["pending"] == []
        assert session.last_reply == session._cancel_response()["reply"]
        feedback = session.recorder.events[-1]
        assert feedback["event"] == "dialogue"
        assert feedback["response"]["reply"] == session.last_reply
    finally:
        backend.release.set()


def test_prequeued_old_request_is_discarded_without_replacing_new_reply(session_factory):
    session = session_factory()
    old_epoch = session.cancel_epoch
    current = session.request_cancel()
    assert session.run("停", expected_cancel_epoch=current)["decision"] == "cancel"
    assert session.cancel_epoch == current
    assert session.run("点头", expected_cancel_epoch=current)["decision"] == "replace"
    before = copy.deepcopy((session.execution, session.last_reply, session.runner.calls))
    assert session.run("前滚翻", expected_cancel_epoch=old_epoch)["decision"] == "cancel"
    assert (session.execution, session.last_reply, session.runner.calls) == before


def test_cancel_signal_alone_prevents_resuming_previously_paused_task(session_factory):
    session = session_factory()
    session.run("一直前进")
    assert session.runner.started[0].wait(1)
    assert session.run("暂停")["decision"] == "pause"
    session.request_cancel()
    assert session.run("继续")["decision"] == "clarify"
    assert len(session.runner.calls) == 1
    assert session.execution["state"] == "idle" and session.execution["active"] is None
    assert session.run("点头")["decision"] == "replace"
    assert len(session.runner.calls) == 2


def test_discarded_model_trace_is_retained_without_accepting_its_plan(session_factory):
    class TracedBackend(BlockedBackend):
        last_trace = None

        def decide(self, request):
            response = super().decide(request)
            self.last_trace = dict(model_called=True, raw_response=copy.deepcopy(response), raw_text="raw model text")
            return response

    session = session_factory()
    backend = TracedBackend(lockstep=True)
    session.backend = backend
    responses = []
    thread, done, errors = _background(lambda: responses.append(session.run("点头", wait=False)))
    try:
        assert backend.entered.wait(1)
        session.request_cancel()
        backend.release.set()
        assert done.wait(1) and not errors
        assert responses[0]["decision"] == "cancel"
        row = session.recorder.events[-1]
        assert row["response"]["decision"] == "cancel"
        trace = row["model_trace"]
        assert trace["decision_source"] == "host_cancel_superseded"
        assert trace["raw_prediction"] is None
        assert trace["superseded_model_trace"]["raw_response"]["decision"] == "replace"
        assert trace["superseded_model_trace"]["raw_text"] == "raw model text"
        assert session.runner.calls == []
    finally:
        backend.release.set()
        thread.join(2)


@pytest.mark.parametrize("epoch", [True, -1, "1", 1.5])
def test_expected_cancel_epoch_rejects_invalid_tokens(session_factory, epoch):
    session = session_factory()
    with pytest.raises(ValueError, match="expected_cancel_epoch"):
        session.run("点头", expected_cancel_epoch=epoch)
    assert session.runner.calls == []
