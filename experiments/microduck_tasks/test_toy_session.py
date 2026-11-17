"""Real ToySession scheduling/recording with deterministic simulated runner gates."""

import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from .toy_fixtures import kick, move, perform, posture, turn
from .toy_protocol import VERSION
from .toy_session import ToySession
from .toy_teacher import ToyTeacher


class FakeRuntime:
    """No physics claim: supply current frames and cooperative motor cancellation."""

    supports_sit_hold = True

    def __init__(self, output):
        self.output = output
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.data = SimpleNamespace(time=0.0, qpos=[0.0, 0.0, 0.12])
        self.qa, self.failure, self.frame, self.stop_count = 0, None, 0, 0

    def snapshot(self):
        with self.lock:
            return {"time_s": self.data.time}

    def observe(self):
        with self.lock:
            self.frame += 1
            observation = dict(
                frame_id=f"frame_{self.frame:05d}",
                time_s=self.data.time,
                tilt_deg=0.0,
                stationary=True,
                detections=[],
            )
        images = {camera: np.zeros((8, 8, 3), dtype=np.uint8) for camera in ("front", "near")}
        return observation, images, None

    def request_stop(self):
        self.stop_count += 1
        self.cancel.set()


class GateRunner:
    """Each goal can finish, fail, or wait until explicitly released/cancelled."""

    def __init__(self, runtime, *, blocked=(), fail_index=None, call_count=12, session=None):
        self.runtime, self.fail_index = runtime, fail_index
        self.session, self.settle_calls = session, 0
        self.calls = []
        self.started = [threading.Event() for _ in range(call_count)]
        self.finished = [threading.Event() for _ in range(call_count)]
        self.release = [threading.Event() for _ in range(call_count)]
        for index, event in enumerate(self.release):
            if index not in blocked:
                event.set()

    def execute(self, goal, task_id, goal_index):
        index = len(self.calls)
        self.calls.append((copy.deepcopy(goal), task_id, goal_index))
        self.started[index].set()
        continuous = goal["type"] == "move" and goal["limit"]["kind"] == "until_cancelled"
        while not self.runtime.cancel.is_set():
            if continuous and self.session is not None and self.session.pause_requested:
                self.finished[index].set()
                return dict(success=False, paused=True, reason="safe_pause_boundary", metrics={})
            if not continuous and self.release[index].wait(0.01):
                break
            if continuous:
                self.runtime.cancel.wait(0.01)
        if self.runtime.cancel.is_set():
            self.finished[index].set()
            return dict(success=False, reason="cancelled", metrics={})
        with self.runtime.lock:
            self.runtime.data.time += 0.1
            if goal["type"] == "posture":
                self.runtime.data.qpos[2] = 0.065 if goal["pose"] == "sitting" else 0.12
        success = index != self.fail_index
        self.finished[index].set()
        return dict(success=success, reason="fixture_verified" if success else "fixture_failed", metrics={})

    def release_all(self):
        for event in self.release:
            event.set()

    def settle_for_pause(self):
        self.settle_calls += 1
        return dict(success=not self.runtime.cancel.is_set(), reason="fixture_stable", metrics={})


@pytest.fixture
def session_factory(tmp_path, monkeypatch):
    from . import toy_live_inputs

    sessions = []
    live_capabilities = toy_live_inputs.capabilities

    def fixture_capabilities(runtime):
        capabilities = live_capabilities(runtime)
        if hasattr(runtime, "test_max_goals"):
            capabilities["max_goals"] = runtime.test_max_goals
        return capabilities

    monkeypatch.setattr(toy_live_inputs, "capabilities", fixture_capabilities)

    def build(*, blocked=(), fail_index=None, call_count=12, max_goals=None):
        runtime = FakeRuntime(tmp_path / f"episode_{len(sessions)}")
        runtime.test_max_goals = max_goals
        session = ToySession(runtime)
        session.runner = GateRunner(
            runtime, blocked=blocked, fail_index=fail_index, call_count=call_count, session=session
        )
        sessions.append(session)
        return session

    yield build
    for session in sessions:
        session.runner.release_all()
        session.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("先点头，再坐下，再站起来", [perform("nod"), posture("sitting"), posture("standing")]),
        ("先踢蓝球，再踢红球，再原地转一圈", [kick("blue"), kick("red"), turn()]),
    ],
)
def test_queue_executes_and_records_goal_and_feedback_in_order(session_factory, text, expected):
    session = session_factory()
    output = session.run(text)
    assert output["goals"] == expected
    assert [call[0] for call in session.runner.calls] == expected
    assert [call[2] for call in session.runner.calls] == [0, 1, 2]
    assert [row["goal"] for row in session.results] == expected
    assert all(row["status"] == "succeeded" for row in session.results)
    assert session.execution["state"] == "idle"
    assert session.execution["active"] is None and not session.execution["pending"]
    rows = session.recorder.events
    assert [row["event"] for row in rows] == ["dialogue", *["goal_start", "goal_result", "dialogue"] * 3]
    feedback = [row for row in rows if row["event"] == "dialogue" and row["request"]["event"]["kind"] == "feedback"]
    assert [row["result_ref"] for row in feedback] == [row["event_id"] for row in session.results]
    assert feedback[0]["request"]["execution"]["active"] == expected[1]
    assert feedback[-1]["request"]["execution"]["state"] == "idle"


def test_failure_discards_remaining_queue_and_reports_real_failure(session_factory):
    session = session_factory(fail_index=1)
    session.run("先点头，再抬左脚，再左右看")
    assert [call[0] for call in session.runner.calls] == [perform("nod"), perform("flamingo_left")]
    assert [row["status"] for row in session.results] == ["succeeded", "failed"]
    assert session.execution["state"] == "failed"
    assert session.execution["active"] is None and not session.execution["pending"]
    assert "未完成" in session.last_reply and "fixture_failed" in session.last_reply


def test_chat_and_visual_reply_do_not_interrupt_or_replace_running_goal(session_factory):
    session = session_factory(blocked=(0,))
    session.run("点头", wait=False)
    assert session.runner.started[0].wait(1)
    revision, stop_count = session.execution["revision"], session.runtime.stop_count
    assert session.run("你好，你叫什么")["decision"] == "respond"
    assert session.run("你现在能看到什么")["decision"] == "respond"
    assert session.execution["active"] == perform("nod")
    assert session.execution["revision"] == revision and session.runtime.stop_count == stop_count
    session.runner.release[0].set()
    session.wait(1)
    assert session.results[0]["status"] == "succeeded"


def _background(function):
    errors, done = [], threading.Event()

    def run():
        try:
            function()
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, done, errors


def test_stop_preempts_a_different_caller_blocked_in_run_wait_true(session_factory):
    session = session_factory(blocked=(0,))
    running, done, errors = _background(lambda: session.run("先点头，再左右看", wait=True))
    assert session.runner.started[0].wait(1)
    stopping, stopped, stop_errors = _background(lambda: session.run("停下"))
    try:
        assert stopped.wait(1), "A blocking run must release the user lock so stop can preempt it."
        assert done.wait(1)
        assert not errors and not stop_errors
        assert session.execution["state"] == "idle" and not session.execution["pending"]
        assert len(session.runner.calls) == 1
        assert session.results[0]["status"] == "cancelled"
    finally:
        session.runner.release_all()
        running.join(2)
        stopping.join(2)


def test_append_survives_completion_of_existing_queue(session_factory):
    session = session_factory(blocked=(0,))
    session.run("点头", wait=False)
    assert session.runner.started[0].wait(1)
    output = session.run("然后抬左脚，再左右看", wait=False)
    assert output["decision"] == "append"
    session.runner.release[0].set()
    session.wait(1)
    assert [call[0] for call in session.runner.calls] == [
        perform("nod"),
        perform("flamingo_left"),
        perform("look_around"),
    ]
    assert [row["status"] for row in session.results] == ["succeeded"] * 3
    assert session.execution["state"] == "idle" and not session.execution["pending"]


def test_append_is_atomic_with_goal_completion_during_teacher_decision(session_factory):
    session = session_factory(blocked=(0,))
    session.run("点头", wait=False)
    assert session.runner.started[0].wait(1)
    entered, release = threading.Event(), threading.Event()

    class PausingTeacher(ToyTeacher):
        def decide(self, request):
            response = super().decide(request)
            if response["decision"] == "append":
                entered.set()
                assert release.wait(2)
            return response

    session.backend = PausingTeacher()
    thread, done, errors = _background(lambda: session.run("然后抬左脚", wait=False))
    try:
        assert entered.wait(1)
        session.runner.release[0].set()
        assert session.runner.finished[0].wait(1)
        release.set()
        assert done.wait(1) and not errors
        session.wait(1)
        assert [call[0] for call in session.runner.calls] == [perform("nod"), perform("flamingo_left")]
        assert session.execution["state"] == "idle" and not session.execution["pending"]
    finally:
        release.set()
        thread.join(2)


def test_continuous_motion_pause_resume_cancel_keeps_distinct_control_semantics(session_factory):
    session = session_factory()
    assert session.run("一直往前走不要停")["goals"] == [move(continuous=True)]
    assert session.runner.started[0].wait(1)
    assert session.run("暂停")["decision"] == "pause"
    assert session.execution["state"] == "paused" and session.execution["active"] == move(continuous=True)
    assert session.run("继续")["decision"] == "resume"
    assert session.runner.started[1].wait(1)
    assert session.execution["state"] == "running"
    assert session.run("停下")["decision"] == "cancel"
    assert session.execution["state"] == "idle" and session.execution["active"] is None
    assert session.run("继续")["decision"] == "clarify"
    controls = [
        row["decision"]
        for row in session.recorder.events
        if row["event"] == "queue_control" and row.get("phase") != "safe_boundary_reached"
    ]
    assert controls == ["pause", "cancel"]
    assert [row["status"] for row in session.results] == ["blocked", "cancelled"]


def test_pause_of_bounded_action_finishes_current_goal_before_retaining_tail(session_factory):
    session = session_factory(blocked=(0,))
    session.run("先前滚翻，再点头", wait=False)
    assert session.runner.started[0].wait(1)
    revision, stop_count = session.execution["revision"], session.runtime.stop_count
    assert session.run("暂停", wait=False)["decision"] == "pause"
    assert session.execution["revision"] == revision and session.runtime.stop_count == stop_count
    assert session.execution["state"] == "running"
    session.runner.release[0].set()
    session.wait(1)
    assert session.results[0]["status"] == "succeeded"
    assert session.execution["state"] == "paused" and session.execution["active"] == perform("nod")
    assert session.run("继续")["decision"] == "resume"
    assert [call[0] for call in session.runner.calls] == [perform("roll"), perform("nod")]


class QueueBackend(ToyTeacher):
    """Preset semantic plans isolate queue scheduling from language parsing."""

    def __init__(self, initial, appended=()):
        self.plans = {"执行长队列": ("replace", initial), "追加长队列": ("append", appended)}

    def decide(self, request):
        event = request["event"]
        if event["kind"] == "user" and event["text"] in self.plans:
            decision, goals = self.plans[event["text"]]
            return dict(
                schema_version=VERSION, reply="好，我按顺序执行。", decision=decision, goals=copy.deepcopy(goals)
            )
        return super().decide(request)


def _long_queue(count):
    return [turn("left" if index % 2 else "right", angle=15 + index) for index in range(count)]


@pytest.mark.parametrize("count", [7, 20, 100])
def test_long_queue_executes_every_goal_and_records_feedback_without_truncation(session_factory, count):
    expected = _long_queue(count)
    session = session_factory(call_count=count)
    session.backend = QueueBackend(expected)
    assert session.run("执行长队列")["goals"] == expected
    assert [call[0] for call in session.runner.calls] == expected
    assert [call[2] for call in session.runner.calls] == list(range(count))
    assert [row["goal"] for row in session.results] == expected
    assert [row["status"] for row in session.results] == ["succeeded"] * count
    assert session.execution["state"] == "idle"
    assert session.execution["active"] is None and session.execution["pending"] == []
    feedback = [
        row
        for row in session.recorder.events
        if row["event"] == "dialogue" and row["request"]["event"]["kind"] == "feedback"
    ]
    assert len(feedback) == count
    assert [row["result_ref"] for row in feedback] == [row["event_id"] for row in session.results]
    assert all(row["request"]["capabilities"]["max_goals"] is None for row in feedback)
    assert feedback[0]["request"]["execution"]["pending"] == expected[2:]
    assert feedback[-1]["request"]["execution"]["state"] == "idle"


def test_append_to_six_existing_goals_keeps_active_worker_and_executes_all_twenty(session_factory):
    expected = _long_queue(20)
    session = session_factory(blocked=(0,), call_count=len(expected))
    session.backend = QueueBackend(expected[:6], expected[6:])
    session.run("执行长队列", wait=False)
    assert session.runner.started[0].wait(1)
    worker, revision, task_id = session.worker, session.execution["revision"], session.task_id
    stop_count = session.runtime.stop_count
    assert session.execution["pending"] == expected[1:6]
    assert session.run("追加长队列", wait=False)["goals"] == expected[6:]
    assert session.worker is worker and session.task_id == task_id
    assert session.execution["revision"] == revision and session.runtime.stop_count == stop_count
    assert session.execution["active"] == expected[0]
    assert session.execution["pending"] == expected[1:]
    assert len(session.runner.calls) == 1
    session.runner.release[0].set()
    session.wait(5)
    assert [call[0] for call in session.runner.calls] == expected
    assert [call[1] for call in session.runner.calls] == [task_id] * len(expected)
    assert [call[2] for call in session.runner.calls] == list(range(len(expected)))
    assert [row["status"] for row in session.results] == ["succeeded"] * len(expected)
    assert session.execution["state"] == "idle" and session.execution["pending"] == []


def test_cancel_clears_every_pending_goal_in_long_queue_without_starting_tail(session_factory):
    expected = _long_queue(20)
    session = session_factory(blocked=(0,), call_count=len(expected))
    session.backend = QueueBackend(expected)
    session.run("执行长队列", wait=False)
    assert session.runner.started[0].wait(1)
    assert session.execution["pending"] == expected[1:]
    assert session.run("停下")["decision"] == "cancel"
    session.wait(1)
    assert [call[0] for call in session.runner.calls] == expected[:1]
    assert [row["status"] for row in session.results] == ["cancelled"]
    assert session.execution["state"] == "idle"
    assert session.execution["active"] is None and session.execution["pending"] == []
    assert session.run("继续")["decision"] == "clarify"
    assert len(session.runner.calls) == 1


@pytest.mark.parametrize("fail_index", [0, 7, 18])
def test_failure_clears_long_queue_including_every_goal_after_failure(session_factory, fail_index):
    expected = _long_queue(20)
    session = session_factory(fail_index=fail_index, call_count=len(expected))
    session.backend = QueueBackend(expected)
    session.run("执行长队列")
    assert [call[0] for call in session.runner.calls] == expected[: fail_index + 1]
    assert [call[2] for call in session.runner.calls] == list(range(fail_index + 1))
    assert [row["status"] for row in session.results] == ["succeeded"] * fail_index + ["failed"]
    assert session.execution["state"] == "failed"
    assert session.execution["active"] is None and session.execution["pending"] == []
    assert session.execution["last_result"]["goal"] == expected[fail_index]
    assert "fixture_failed" in session.last_reply
