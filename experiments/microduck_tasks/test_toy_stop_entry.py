"""Stop admission bypasses queued inference, UI locks and the simulation world lock."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from .test_toy_session import FakeRuntime, GateRunner
from .toy_demo import _pump_without_waiting
from .toy_dispatch import ToyInputDispatcher
from .toy_fixtures import perform
from .toy_session import ToySession
from .toy_teacher import ToyTeacher


@contextmanager
def held_elsewhere(*locks):
    entered, release = threading.Event(), threading.Event()

    def hold():
        for lock in locks:
            lock.acquire()
        try:
            entered.set()
            assert release.wait(5)
        finally:
            for lock in reversed(locks):
                lock.release()

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert entered.wait(1)
    try:
        yield
    finally:
        release.set()
        thread.join(2)


@pytest.fixture
def entry(tmp_path):
    session = ToySession(FakeRuntime(tmp_path / "stop_entry"))
    session.runner = GateRunner(session.runtime)
    inputs = ToyInputDispatcher(session, threading.Event(), threading.Event(), [])
    thread = threading.Thread(target=inputs.dispatch, daemon=True)
    thread.start()
    yield session, inputs
    session.runner.release_all()
    inputs.close()
    thread.join(3)
    session.close()
    assert not thread.is_alive() and not inputs.errors


def test_stop_during_locked_inference_discards_old_inputs_but_keeps_later_request(entry):
    session, inputs = entry
    entered, release = threading.Event(), threading.Event()

    class SlowTeacher(ToyTeacher):
        requires_lockstep = True

        def decide(self, state):
            response = super().decide(state)
            if state["event"]["text"] == "先点头，再左右看":
                entered.set()
                assert release.wait(3)
            return response

    session.backend = SlowTeacher()
    first, _ = inputs.submit("先点头，再左右看")
    assert entered.wait(1)
    stale, stale_box = inputs.submit("前滚翻")
    stop_returned, stop_box = threading.Event(), []

    def stop():
        stop_box.append(inputs.submit("停"))
        stop_returned.set()

    stopping = threading.Thread(target=stop, daemon=True)
    stopping.start()
    try:
        assert stop_returned.wait(1), "Stop admission must not wait for inference/state/world locks."
        assert session.runtime.cancel.is_set() and inputs.stop_pending.is_set()
        later, later_box = inputs.submit("抬左脚")
        release.set()
        assert first.wait(2) and stale.wait(2) and stop_box[0][0].wait(2) and later.wait(2)
        session.wait(2)
        assert not stale_box
        assert stop_box[0][1][0]["decision"] == "cancel"
        assert later_box[0]["decision"] == "replace"
        assert [call[0] for call in session.runner.calls] == [perform("flamingo_left")]
    finally:
        release.set()
        stopping.join(2)


def test_epoch_rechecked_after_dispatch_before_session_run(entry):
    session, inputs = entry
    entered, release = threading.Event(), threading.Event()
    original = session.run

    def delayed(text, **options):
        if text == "点头":
            entered.set()
            assert release.wait(3)
        return original(text, **options)

    session.run = delayed
    old, _ = inputs.submit("点头")
    try:
        assert entered.wait(1)
        stopped, _ = inputs.submit("停止")
        later, _ = inputs.submit("左右看")
        release.set()
        assert old.wait(2) and stopped.wait(2) and later.wait(2)
        session.wait(2)
        assert [call[0] for call in session.runner.calls] == [perform("look_around")]
    finally:
        release.set()


@pytest.mark.parametrize("text", ["停", "停下", "立即停止", "取消所有任务", "别动"])
def test_every_direct_stop_phrase_latches_once_before_queued_acknowledgement(entry, text):
    session, inputs = entry
    epoch = session.cancel_epoch
    done, box = inputs.submit(text)
    assert session.runtime.cancel.is_set() and session.cancel_epoch == epoch + 1
    assert done.wait(2) and box[0]["decision"] == "cancel"
    assert session.cancel_epoch == epoch + 1 and not inputs.stop_pending.is_set()


def test_multiple_stops_discard_all_older_epochs_and_preserve_only_later_request(tmp_path):
    session = ToySession(FakeRuntime(tmp_path / "multiple_stops"))
    session.runner = GateRunner(session.runtime)
    inputs = ToyInputDispatcher(session, threading.Event(), threading.Event(), [])
    submitted = [inputs.submit(text) for text in ("点头", "停", "前滚翻", "别动", "左右看")]
    inputs.close()
    try:
        inputs.dispatch()
        session.wait(2)
        assert all(done.is_set() for done, _ in submitted)
        assert all(not box for _, box in submitted[:3])
        assert submitted[3][1][0]["decision"] == "cancel"
        assert submitted[4][1][0]["decision"] == "replace"
        assert [call[0] for call in session.runner.calls] == [perform("look_around")]
        assert not inputs.stop_pending.is_set() and not inputs.errors
    finally:
        session.close()


@pytest.mark.parametrize("lock_name", ["state", "world"])
def test_tk_stop_callback_runs_while_state_or_world_lock_is_busy(entry, lock_name):
    session, inputs = entry
    updated = []

    def update():
        updated.append(True)
        inputs.submit("停")

    view = SimpleNamespace(session=session, closed=False, root=SimpleNamespace(update=update))
    lock = session.state_lock if lock_name == "state" else session.runtime.lock
    with held_elsewhere(lock):
        assert _pump_without_waiting(view, lambda: pytest.fail("Busy UI must skip state/rendering."))
        assert updated == [True] and session.runtime.cancel.is_set()


def test_ui_paints_when_both_locks_available_and_handles_close_before_paint(entry):
    session, _ = entry
    calls = []
    view = SimpleNamespace(session=session, closed=False, root=SimpleNamespace(update=lambda: calls.append("tk")))
    assert _pump_without_waiting(view, lambda: calls.append("paint") or True)
    assert calls == ["tk", "paint"]
    view.root.update = lambda: setattr(view, "closed", True)
    assert not _pump_without_waiting(view, lambda: pytest.fail("Closed window must not render."))


@pytest.fixture
def runtime():
    from .bootstrap import configure

    try:
        configure()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    from .runtime import Runtime

    value = object.__new__(Runtime)
    value.lock = threading.RLock()
    value.cancel, value.shutdown, value.pulse_done = threading.Event(), threading.Event(), threading.Event()
    value.remaining, value.active_skill, value.worker, value.failure = 20, "forward", None, None
    value._requested_ticks, value._executed_ticks = 20, 0
    value.supports_skill = lambda skill: True
    yield value
    value.shutdown.set()
    if value.worker is not None:
        value.worker.join(2)


def test_runtime_stop_returns_before_busy_world_lock_is_released(runtime):
    returned = threading.Event()
    with held_elsewhere(runtime.lock):
        thread = threading.Thread(target=lambda: (runtime.request_stop(), returned.set()), daemon=True)
        thread.start()
        assert returned.wait(1), "Stopping must latch and return even while inference owns the world lock."
        assert runtime.cancel.is_set()
    thread.join(1)
    runtime.request_stop()
    assert runtime.remaining == 0 and runtime.active_skill == "stand" and runtime.pulse_done.is_set()


def test_realtime_next_tick_balances_after_stop_latched_behind_world_lock(runtime):
    ticks = []

    def tick(skill):
        ticks.append(skill)
        runtime.shutdown.set()

    runtime._tick = tick
    with held_elsewhere(runtime.lock):
        runtime.start_realtime()
        runtime.request_stop()
        assert runtime.cancel.is_set() and ticks == []
    runtime.worker.join(2)
    assert ticks == ["stand"]
    assert runtime.remaining == 0 and runtime.active_skill == "stand" and runtime.pulse_done.is_set()


def test_lockstep_execute_refuses_pending_motion_after_latched_stop(runtime):
    with held_elsewhere(runtime.lock):
        runtime.request_stop()
    runtime._tick = lambda skill: pytest.fail("Cancelled lockstep simulation must not advance a motion tick.")
    assert runtime.execute("forward", 0.04) is False
    assert runtime.active_skill == "stand" and runtime.remaining == 0 and runtime.pulse_done.is_set()
    assert not runtime.last_execution["started"] and not runtime.last_execution["completed"]
