"""Desktop state/IPC use real scheduling but fake physics and native windows."""

import json
import socket
import threading
from contextlib import contextmanager

import pytest

from .test_toy_session import FakeRuntime, GateRunner
from .toy_desktop_ipc import DesktopServer
from .toy_desktop_state import ToyDesktopState
from .toy_dispatch import ToyInputDispatcher
from .toy_fixtures import perform
from .toy_session import ToySession
from .toy_teacher import ToyTeacher


class DisplayRuntime(FakeRuntime):
    closed = False

    def display_snapshot(self):
        return dict(time_s=0.0, skill="stand", tilt_deg=0.0, stationary=True, display_frame_id=1)

    def save_trace(self):
        pass

    def close(self):
        self.closed = True


class Viewer:
    error = None

    def __init__(self):
        self.closed = False
        self.rect = (20, 30, 900, 700)

    def is_running(self):
        return not self.closed

    def geometry(self):
        return self.rect

    def place(self, *rect):
        self.rect = rect

    def close(self):
        self.closed = True


def attach_world(app, *, fail_index=None):
    app.episode += 1
    runtime = DisplayRuntime(app.output / f"episode_{app.episode}")
    session = ToySession(runtime, backend=app.backend)
    session.runner = GateRunner(runtime, blocked=(0,), fail_index=fail_index)
    app.session, app.viewer = session, Viewer()
    app.inputs = ToyInputDispatcher(session, threading.Event(), threading.Event(), app.errors)
    app.messages, app.event_index, app.busy = [], 0, 0
    app.execution, app.phase = dict(state="idle", active=None, pending=[]), "ready"
    app.dispatch_thread = app._thread(app.inputs.dispatch, "test-desktop-input")
    return session


@pytest.fixture
def app(tmp_path):
    value = ToyDesktopState(tmp_path, ToyTeacher)
    value.backend = ToyTeacher()
    attach_world(value)
    try:
        yield value
    finally:
        if value.session:
            value.session.runner.release_all()
        value.close()


@contextmanager
def held_elsewhere(*locks):
    held, release = threading.Event(), threading.Event()

    def hold():
        for lock in locks:
            lock.acquire()
        held.set()
        try:
            assert release.wait(5)
        finally:
            for lock in reversed(locks):
                lock.release()

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert held.wait(1)
    try:
        yield
    finally:
        release.set()
        worker.join(2)


@pytest.fixture
def endpoint(app, tmp_path):
    path = tmp_path / "chat.sock"
    server = DesktopServer(path, app)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    yield path
    server.shutdown()
    thread.join(2)
    server.server_close()


def request(path, value):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(1)
        client.connect(str(path))
        payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode() + b"\n"
        client.sendall(payload)
        with client.makefile("rb") as stream:
            return json.loads(stream.readline())


def test_state_ipc_never_waits_for_world_state_or_recorder_locks(app, endpoint):
    session = app.session
    with held_elsewhere(session.runtime.lock, session.state_lock, session.recorder.lock):
        state = request(endpoint, dict(command="state"))
    assert state["ready"] and state["sim"]["display_frame_id"] == 1
    assert state["execution"]["state"] == "idle" and state["queue_capacity"] == 15


@pytest.mark.parametrize("failed", [False, True])
def test_actual_goal_result_is_shown_only_after_execution_and_exactly_once(app, failed):
    session = app.session
    session.runner.fail_index = 0 if failed else None
    app.submit("点头")
    assert session.runner.started[0].wait(2)
    initial = app.snapshot()["messages"]
    assert any(row["role"] == "assistant" and "点头" in row["text"] for row in initial)
    assert not any("：完成。" in row["text"] or "：失败。" in row["text"] for row in initial)
    session.runner.release[0].set()
    session.wait(2)
    after = app.snapshot()["messages"]
    label = "失败" if failed else "完成"
    result = [row for row in after if f"：{label}。" in row["text"]]
    assert len(result) == 1 and result[0]["role"] == "execution" and label in result[0]["text"]
    assert app.snapshot()["messages"] == after
    assert session.results[0]["goal"] == perform("nod")


def test_stop_ipc_latches_dispatcher_while_inference_holds_world_lock(app, endpoint):
    entered, release = threading.Event(), threading.Event()

    class SlowTeacher(ToyTeacher):
        requires_lockstep = True

        def decide(self, state):
            if state["event"]["text"] == "点头":
                entered.set()
                assert release.wait(5)
            return super().decide(state)

    app.session.backend = SlowTeacher()
    assert request(endpoint, dict(command="message", text="点头")) == {"accepted": True}
    assert entered.wait(2)
    try:
        assert request(endpoint, dict(command="state"))["inference_pending"] == 1
        assert request(endpoint, dict(command="message", text="立即停止")) == {"accepted": True}
        assert app.session.runtime.cancel.is_set() and app.inputs.stop_pending.is_set()
    finally:
        release.set()


def test_reset_disposes_old_viewer_and_runtime_before_building_new_world(app):
    old_session, old_viewer = app.session, app.viewer
    app.submit("点头")
    assert old_session.runner.started[0].wait(2)

    def rebuild():
        assert old_viewer.closed and old_session.runtime.closed
        attach_world(app)

    app._new_world = rebuild
    app.reset()
    assert old_session.runtime.cancel.is_set()
    app.reset_thread.join(3)
    assert not app.reset_thread.is_alive() and app.phase == "ready" and app.episode == 2
    assert app.session is not old_session and app.viewer is not old_viewer
    assert not app.viewer.closed and not app.errors and app.messages == []


def test_trace_write_failure_still_disposes_independent_viewer_and_runtime(app):
    session, viewer = app.session, app.viewer

    def fail():
        raise OSError("fixture disk full")

    session.runtime.save_trace = fail
    try:
        with pytest.raises(OSError, match="disk full"):
            app._dispose_world()
        assert viewer.closed and session.runtime.closed and app.viewer is None and app.session is None
    finally:
        session.runtime.save_trace = lambda: None


def test_close_during_loading_does_not_construct_late_world(tmp_path):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def factory():
        entered.set()
        assert release.wait(2)
        return ToyTeacher()

    app = ToyDesktopState(tmp_path, factory)
    calls = []
    app._new_world = lambda: calls.append("late world")
    app.start()
    assert entered.wait(1)
    worker = threading.Thread(target=lambda: (app.close(), closed.set()), daemon=True)
    worker.start()
    assert app.shutdown.wait(1)
    release.set()
    assert closed.wait(2)
    worker.join(1)
    assert calls == [] and app.session is None


def test_ipc_socket_is_owner_only_and_preserves_chinese(app, endpoint):
    assert endpoint.stat().st_mode & 0o777 == 0o600
    assert request(endpoint, dict(command="message", text="你好小鸭")) == {"accepted": True}
    assert any(row == dict(role="user", text="你好小鸭") for row in app.snapshot()["messages"])


@pytest.mark.parametrize("value", [["state"], {"command": "unknown"}, {"command": "message", "text": 123}, b"x\n"])
def test_bad_ipc_request_returns_json_and_next_state_still_works(endpoint, value):
    assert "error" in request(endpoint, value)
    assert request(endpoint, dict(command="state"))["ready"]


def test_runtime_ipc_error_is_reported_as_json(app, endpoint, monkeypatch):
    def fail():
        raise RuntimeError("fixture snapshot unavailable")

    monkeypatch.setattr(app, "snapshot", fail)
    assert request(endpoint, dict(command="state")) == {"error": "fixture snapshot unavailable"}


def test_shutdown_ipc_latches_cancel_before_slow_process_cleanup(app, endpoint):
    assert request(endpoint, dict(command="shutdown")) == {"accepted": True}
    assert app.shutdown.is_set() and app.session.runtime.cancel.is_set()
