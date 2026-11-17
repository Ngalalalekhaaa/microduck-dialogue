"""Asynchronous continuation, cancellation and fresh-sensor plan invalidation."""

import threading
from types import SimpleNamespace

from .plan_protocol import VERSION, Plan
from .plan_tasks import PlanTaskSession
from .test_plan_protocol import TARGET
from .test_skill_tasks import ImageRuntime, ScriptedBackend
from .test_tasks import MemoryRecorder, ball


def action_plan(target=None):
    return Plan(
        VERSION,
        "continue",
        target,
        [{"skill": "move_forward", "seconds": 0.4}, {"skill": "move_forward", "seconds": 0.4}],
        "",
    )


def finish():
    return Plan(VERSION, "finish", None, [], "完成")


class Runtime(ImageRuntime):
    def __init__(self, detections=()):
        super().__init__(detections)
        self.lock = threading.RLock()
        self.policy = SimpleNamespace(motion=SimpleNamespace(visual_heading=None))
        self.progress = threading.Event()

    def execute(self, skill, seconds):
        result = super().execute(skill, seconds)
        self.progress.set()
        return result


def test_inference_runs_while_current_chunk_executes_without_fixed_standing():
    runtime = Runtime()

    class Backend(ScriptedBackend):
        def decide(self, request):
            if self.requests:
                assert request.context["committed_steps"]
                assert runtime.progress.wait(2), "Executor blocked on speculative inference"
            return super().decide(request)

    backend = Backend(action_plan(), finish())
    session = PlanTaskSession(runtime, MemoryRecorder(), backend, verbose=False)
    assert session.run("持续向目标移动")["status"] == "model_reported"
    assert runtime.calls == [("move_forward", 0.2)] * 4


def test_target_loss_discards_remaining_chunk_and_speculative_continuation():
    class LosingRuntime(Runtime):
        def execute(self, skill, seconds):
            result = super().execute(skill, seconds)
            self.detections = []
            return result

    runtime = LosingRuntime([ball(x=0.4)])
    backend = ScriptedBackend(action_plan(TARGET), action_plan(TARGET), finish())
    session = PlanTaskSession(runtime, MemoryRecorder(), backend, verbose=False)
    assert session.run("接近红球")["status"] == "model_reported"
    assert runtime.calls == [("move_forward", 0.2)]
    assert session.outcomes[-1]["outcome"] == "target_lost"


def test_cancelled_inference_cannot_execute_late_continuation():
    release = threading.Event()

    class CancellingRuntime(Runtime):
        def execute(self, skill, seconds):
            result = super().execute(skill, seconds)
            self.request_stop()
            release.set()
            return result

    runtime = CancellingRuntime()

    class Backend(ScriptedBackend):
        def decide(self, request):
            if self.requests:
                assert release.wait(2)
            return super().decide(request)

    backend = Backend(action_plan(), action_plan())
    session = PlanTaskSession(runtime, MemoryRecorder(), backend, verbose=False)
    assert session.run("持续向目标移动")["status"] == "cancelled"
    session._pending.join(2)
    assert runtime.calls == [("move_forward", 0.2)]


def test_kick_checks_current_reach_and_does_not_blindly_execute():
    runtime = Runtime([ball(x=0.4)])
    kick = Plan(VERSION, "continue", TARGET, [{"skill": "kick_left", "seconds": 0.5}], "")
    session = PlanTaskSession(runtime, MemoryRecorder(), ScriptedBackend(kick, finish()), verbose=False)
    assert session.run("踢红球")["status"] == "model_reported"
    assert not runtime.calls
    assert session.outcomes[-1]["outcome"] == "kick_target_outside_measured_region"


def test_single_operator_motion_cannot_be_repeated_by_successive_plans():
    runtime = Runtime()
    backend = ScriptedBackend(action_plan(), action_plan())
    session = PlanTaskSession(runtime, MemoryRecorder(), backend, verbose=False)
    assert session.run("往前走一点")["status"] == "command_budget_complete"
    assert runtime.calls == [("move_forward", 0.2), ("move_forward", 0.2)]
