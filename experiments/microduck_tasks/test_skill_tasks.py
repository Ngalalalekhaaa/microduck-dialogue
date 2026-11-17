"""Check VLM ownership of actions, observation provenance and late-response isolation."""

import copy
import json
import threading

import numpy as np
import pytest

from . import skill_tasks
from .skill_tasks import SkillTaskSession, guard_skill
from .skill_vlm import SkillProposal
from .test_tasks import FakeRuntime, MemoryRecorder, ball, observed


def motion(skill="move_forward", seconds=0.32, color=None):
    return SkillProposal("skill", skill, seconds, color, "")


def response(kind="finish", reply="完成。"):
    return SkillProposal(kind, None, 0, None, reply)


class ImageRuntime(FakeRuntime):
    """Every observation has a distinct image, independent of simulation time."""

    def __init__(self, detections=()):
        super().__init__(detections=detections)
        self.failure = None
        self.observation_count = 0

    def observe(self):
        obs, images, depths = super().observe()
        self.observation_count += 1
        obs["frame_id"] = f"frame_{self.observation_count:05d}"
        images["front"].fill(self.observation_count)
        return obs, images, depths


class ScriptedBackend:
    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.requests = []
        self.last_trace = {}

    def decide(self, request):
        self.requests.append(copy.deepcopy(request))
        self.last_trace = {
            "model": "test-pretrained-vlm",
            "raw_response": "scripted response",
            "input_state": copy.deepcopy(request.state),
            "input_context": copy.deepcopy(request.context),
        }
        proposal = self.proposals.pop(0)
        if isinstance(proposal, Exception):
            raise proposal
        return proposal


def session_for(backend, runtime=None, **kwargs):
    runtime = runtime or ImageRuntime()
    recorder = MemoryRecorder()
    return SkillTaskSession(runtime, recorder, backend, verbose=False, **kwargs), runtime, recorder


def test_model_chooses_legal_motion_even_when_rule_planner_would_choose_another(monkeypatch):
    def no_rule_planner(*_args):
        raise AssertionError("The rule planner must not select VLM actions.")

    monkeypatch.setattr("experiments.microduck_tasks.planner.KickPlanner.decide", no_rule_planner)
    backend = ScriptedBackend(motion("move_backward", color="red"), response())
    session, runtime, recorder = session_for(backend, ImageRuntime([ball("red", 0.4, 0)]))
    result = session.run("踢红球")
    assert runtime.calls == [("move_backward", 0.32), ("stand", 1.0)]
    assert result["status"] == "model_reported"
    executed = [event for event in recorder.events if event["execution"]["started"]]
    assert [event["controller_source"] for event in executed] == ["pretrained_vlm", "executor"]
    assert executed[0]["model_trace"]["model_proposal"]["skill"] == "move_backward"
    assert executed[0]["model_trace"]["guard_result"]["accepted"]


def test_finetuned_model_actions_remain_distinct_from_executor_recovery():
    class FinetunedBackend(ScriptedBackend):
        def decide(self, request):
            self.metadata = {"robot_finetuned": True}
            return super().decide(request)

    backend = FinetunedBackend(motion(), response())
    session, runtime, recorder = session_for(backend)
    result = session.run("往前走一点")
    assert result["controller_source"] == "finetuned_vlm"
    executed = [event for event in recorder.events if event["execution"]["started"]]
    assert [(event["controller_source"], event["phase"]) for event in executed] == [
        ("finetuned_vlm", "execute"),
        ("executor", "recover"),
    ]
    assert runtime.calls == [("move_forward", 0.32), ("stand", 1.0)]
    assert all(event["controller_source"] == "finetuned_vlm" for event in recorder.events if not event["decision"])


def test_each_model_step_gets_new_rgb_and_feedback_without_oracle_state():
    backend = ScriptedBackend(motion(), motion("turn_right", 0.6), response())
    session, runtime, _ = session_for(backend)
    result = session.run("先找找目标，再决定怎么走")
    assert result["status"] == "model_reported"
    assert len(backend.requests) == 3
    image_ids = [int(request.images["front"][0, 0, 0]) for request in backend.requests]
    assert image_ids == sorted(set(image_ids))
    assert all(
        np.all(request.images["front"] == value) for request, value in zip(backend.requests, image_ids, strict=True)
    )
    assert [request.state["sim_time_s"] for request in backend.requests] == pytest.approx([0.0, 1.32, 2.92])
    assert backend.requests[0].context["previous_action"] is None
    assert backend.requests[1].context["previous_action"]["skill"] == "move_forward"
    assert "executed_and_recovered" in backend.requests[1].context["previous_feedback"]
    assert backend.requests[2].context["previous_action"]["skill"] == "turn_right"
    for request in backend.requests:
        assert set(request.state) == {"yaw", "stationary", "tilt_deg", "sim_time_s", "detections_rgbd"}
        assert "truth" not in json.dumps(request.state)
        assert "task_id" not in request.context
    assert runtime.calls == [("move_forward", 0.32), ("stand", 1.0), ("turn_right", 0.6), ("stand", 1.0)]


def test_each_task_resets_model_clock_and_omits_sequence_id_but_keeps_raw_logs():
    backend = ScriptedBackend(motion(), response(), motion(), response())
    session, runtime, recorder = session_for(backend)
    session.run("向前走一点")
    runtime.time_s += 100.125
    session.run("向前走一点")
    assert [request.state["sim_time_s"] for request in backend.requests] == [0.0, 1.32, 0.0, 1.32]
    assert backend.requests[0].context == backend.requests[2].context
    assert backend.requests[1].context == backend.requests[3].context
    assert all("task_id" not in request.context for request in backend.requests)
    assert {event["task_id"] for event in recorder.events} == {"task_001", "task_002"}
    assert recorder.events[0]["observation"]["time_s"] == 2.0
    assert recorder.events[-1]["observation"]["time_s"] > 100.0


def test_rgb_only_request_omits_detector_measurements():
    backend = ScriptedBackend(response("answer", "我看到一个红球。"))
    session, runtime, _ = session_for(backend, ImageRuntime([ball()]), rgbd_assist=False)
    result = session.run("你能看到什么？")
    assert result["status"] == "answered"
    assert "detections_rgbd" not in backend.requests[0].state
    assert runtime.calls == []


def test_guard_only_rejects_and_never_supplies_alignment_or_substitute_motion():
    proposal = motion("kick_left", 0.5, "red")
    backend = ScriptedBackend(proposal, proposal, proposal)
    session, runtime, recorder = session_for(backend, ImageRuntime([ball("red", 0.4, 0)]))
    result = session.run("踢红球")
    assert result["status"] == "guard_rejected"
    assert runtime.calls == []
    rejected = [event for event in recorder.events if event["phase"] == "guard_rejected"]
    assert len(rejected) == 3
    assert all(event["decision"] is None and not event["execution"]["started"] for event in rejected)
    assert all(
        event["model_trace"]["guard_result"]["reason"] == "kick_target_outside_foot_region" for event in rejected
    )
    assert backend.requests[1].context["previous_feedback"] == "kick_target_outside_foot_region"
    assert len({int(request.images["front"][0, 0, 0]) for request in backend.requests}) == 3


@pytest.mark.parametrize(
    "observation,reason",
    [
        (observed([ball("red", 0.0895, 0.0411)], stationary=False), "kick_requires_stationary_body"),
        (observed([ball("blue", 0.0895, 0.0411)]), "kick_target_not_visible_in_rgbd"),
        (observed([ball("red", float("nan"), 0.0411)]), "invalid_rgbd_position"),
    ],
)
def test_kick_guard_rejects_unstable_wrong_target_and_invalid_position(observation, reason):
    assert guard_skill(motion("kick_left", 0.5, "red"), observation) == {"accepted": False, "reason": reason}


def test_guard_checks_new_measurement_after_inference_not_the_model_input_frame():
    class TargetMovedRuntime(ImageRuntime):
        def observe(self):
            if self.observation_count:
                self.detections = [ball("red", 0.4, 0)]
            return super().observe()

    backend = ScriptedBackend(motion("kick_left", 0.5, "red"), response())
    session, runtime, recorder = session_for(backend, TargetMovedRuntime([ball("red", 0.0895, 0.0411)]))
    session.run("踢红球")
    assert backend.requests[0].state["detections_rgbd"][0]["center_body_m"][0] == 0.0895
    assert runtime.calls == []
    rejected = next(event for event in recorder.events if event["phase"] == "guard_rejected")
    assert rejected["model_trace"]["guard_result"]["reason"] == "kick_target_outside_foot_region"
    assert rejected["observation"]["frame_id"] != rejected["model_trace"]["guard_observation_frame"]


@pytest.mark.parametrize(
    "proposal",
    [
        motion("unknown_skill"),
        motion(seconds=5),
        motion(seconds=float("nan")),
        motion("kick_left", 0.32, "red"),
        motion(color="green"),
        SkillProposal("answer", "move_forward", 0.32, None, "完成"),
    ],
)
def test_even_injected_invalid_proposal_fails_closed(proposal):
    backend = ScriptedBackend(proposal)
    session, runtime, recorder = session_for(backend)
    result = session.run("往前走")
    assert result["status"] == "error"
    assert runtime.calls == []
    assert runtime.stop_calls == 1
    assert any(event["phase"] == "model_error" for event in recorder.events)
    error = next(event for event in recorder.events if event["phase"] == "model_error")
    assert error["model_trace"]["raw_response"] == "scripted response"


@pytest.mark.parametrize("expected_color,status", [(None, "model_reported"), ("red", "verification_failed")])
def test_model_finish_is_not_physical_success(expected_color, status):
    backend = ScriptedBackend(response(reply="我已经把红球踢走了。"))
    session, runtime, _ = session_for(backend, expected_color=expected_color)
    result = session.run("踢红球")
    assert result["status"] == status
    assert result["physical_task_verified"] is False
    assert runtime.calls == []


def test_expected_color_and_hidden_positions_are_used_only_by_independent_evaluation(monkeypatch):
    class EvaluationRuntime(ImageRuntime):
        def __init__(self):
            super().__init__([ball("red", 0.0895, 0.0411)])
            self.truth_calls = 0

        def truth(self):
            self.truth_calls += 1
            return {"hidden_test_coordinate": self.truth_calls}

    evaluations = []

    def evaluate(*args):
        evaluations.append(args)
        return {"success": False}

    monkeypatch.setattr(skill_tasks, "evaluate_kick", evaluate)
    backend = ScriptedBackend(motion("kick_left", 0.5, "red"), response())
    session, runtime, _ = session_for(backend, EvaluationRuntime(), expected_color="blue")
    result = session.run("踢前面这个球")
    assert result["status"] == "verification_failed"
    assert len(evaluations) == 1
    assert evaluations[0][:4] == ("blue", "left", {"hidden_test_coordinate": 1}, {"hidden_test_coordinate": 2})
    assert runtime.truth_calls == 2
    for request in backend.requests:
        model_input = json.dumps(
            {"instruction": request.instruction, "state": request.state, "context": request.context}
        )
        assert "hidden_test_coordinate" not in model_input
        assert "expected_color" not in model_input
        assert "blue" not in model_input
    assert result["evaluation"]["expected_color"] == "blue"


def test_kick_without_independent_test_target_does_not_read_simulator_truth():
    backend = ScriptedBackend(motion("kick_left", 0.5, "red"), response())
    session, runtime, _ = session_for(backend, ImageRuntime([ball("red", 0.0895, 0.0411)]))
    assert session.run("踢红球")["status"] == "model_reported"
    assert runtime.calls == [("kick_left", 0.5), ("stand", 3.0)]


@pytest.mark.parametrize("kick_count,status", [(1, "success"), (2, "verification_failed")])
def test_single_kick_acceptance_does_not_accept_repeated_kicks(monkeypatch, kick_count, status):
    class EvaluationRuntime(ImageRuntime):
        def truth(self):
            return {}

    evaluated = []

    def evaluate(*args):
        evaluated.append(args)
        return {"success": True}

    monkeypatch.setattr(skill_tasks, "evaluate_kick", evaluate)
    backend = ScriptedBackend(*[motion("kick_left", 0.5, "red") for _ in range(kick_count)], response())
    session, _, _ = session_for(backend, EvaluationRuntime([ball("red", 0.0895, 0.0411)]), expected_color="red")
    result = session.run("踢红球")
    assert result["status"] == status
    assert len(evaluated) == kick_count
    assert result["physical_task_verified"] is (kick_count == 1)


def test_acceptance_is_rechecked_at_finish_after_later_motion(monkeypatch):
    class EvaluationRuntime(ImageRuntime):
        def truth(self):
            return {}

        def execute(self, skill, seconds):
            completed = super().execute(skill, seconds)
            if skill == "move_forward":
                self.rows.append("post_kick_motion")
                self.contacts.append("post_kick_contact")
            return completed

    evaluated = []

    def evaluate(*args):
        evaluated.append(args)
        return {"success": "post_kick_contact" not in args[5]}

    monkeypatch.setattr(skill_tasks, "evaluate_kick", evaluate)
    backend = ScriptedBackend(motion("kick_left", 0.5, "red"), motion(), response())
    session, _, _ = session_for(backend, EvaluationRuntime([ball("red", 0.0895, 0.0411)]), expected_color="red")
    result = session.run("踢红球")
    assert result["status"] == "verification_failed"
    assert len(evaluated) == 1
    assert evaluated[0][5:] == (["post_kick_contact"], ["post_kick_motion"])


class BlockingBackend(ScriptedBackend):
    def __init__(self):
        super().__init__(motion(), response("answer", "这是新任务的回复。"))
        self.entered, self.release = threading.Event(), threading.Event()

    def decide(self, request):
        if not self.requests:
            self.entered.set()
            if not self.release.wait(timeout=3):
                raise TimeoutError("Test failed to release its inference worker.")
        return super().decide(request)


def test_cancelled_inference_late_output_cannot_move_or_be_read_by_a_new_task():
    backend = BlockingBackend()
    session, runtime, _ = session_for(backend)
    results = []
    task = threading.Thread(target=lambda: results.append(session.run("往前走")))
    task.start()
    try:
        assert backend.entered.wait(timeout=1)
        runtime.request_stop()
        task.join(timeout=1)
        assert not task.is_alive()
        assert results[0]["status"] == "cancelled"
        assert runtime.calls == []
        while_pending = session.run("你现在看到什么？")
        assert while_pending["status"] == "error"
        assert "Previous inference" in while_pending["reply"]
        assert runtime.calls == []
    finally:
        backend.release.set()
        task.join(timeout=1)
        if session._pending is not None:
            session._pending.join(timeout=1)
    assert not session._pending.is_alive()
    latest = session.run("请回答新的问题")
    assert latest["status"] == "answered"
    assert latest["reply"] == "这是新任务的回复。"
    assert runtime.calls == []
    assert [request.instruction for request in backend.requests] == ["往前走", "请回答新的问题"]


def test_timed_out_inference_late_output_is_abandoned():
    backend = BlockingBackend()
    session, runtime, recorder = session_for(backend, timeout=0.02)
    try:
        result = session.run("往前走")
        assert result["status"] == "error"
        assert "deadline exceeded" in result["reply"]
        assert runtime.calls == []
        assert not any(event["execution"]["started"] for event in recorder.events)
    finally:
        backend.release.set()
        session._pending.join(timeout=1)
    assert not session._pending.is_alive()
    assert runtime.calls == []


def test_second_step_timeout_logs_current_request_without_prior_successful_proposal():
    class SecondStepBlockingBackend(ScriptedBackend):
        def __init__(self):
            super().__init__(motion(), motion("turn_left", 0.6))
            self.entered, self.release = threading.Event(), threading.Event()

        def decide(self, request):
            if len(self.requests) == 1:
                self.entered.set()
                if not self.release.wait(timeout=3):
                    raise TimeoutError("Test failed to release its second inference worker.")
            return super().decide(request)

    backend = SecondStepBlockingBackend()
    session, runtime, recorder = session_for(backend, timeout=0.1)
    try:
        result = session.run("先往前走，再观察")
        assert backend.entered.is_set()
        assert result["status"] == "error"
        assert "deadline exceeded" in result["reply"]
        assert runtime.calls == [("move_forward", 0.32), ("stand", 1.0)]
        successful = next(event for event in recorder.events if event["phase"] == "execute")
        assert successful["model_trace"]["model_proposal"]["skill"] == "move_forward"
        error = next(event for event in recorder.events if event["phase"] == "model_error")
        trace = error["model_trace"]
        assert trace["input_context"]["step"] == 1
        assert trace["input_state"]["sim_time_s"] == pytest.approx(1.32)
        assert trace["raw_response"] is None
        assert trace["request_status"] == "timeout"
        assert "model_proposal" not in trace
        assert "guard_result" not in trace
    finally:
        backend.release.set()
        session._pending.join(timeout=1)
    assert not session._pending.is_alive()
    assert runtime.calls == [("move_forward", 0.32), ("stand", 1.0)]
    assert error["model_trace"]["raw_response"] is None


def test_stop_command_bypasses_model_inference():
    backend = ScriptedBackend()
    session, runtime, _ = session_for(backend)
    result = session.run("马上停下")
    assert result["status"] == "cancelled"
    assert backend.requests == []
    assert runtime.stop_calls == 1
    assert runtime.calls == []


def test_cancel_after_prediction_during_fresh_guard_observation_prevents_execution():
    class CancelAtGuardRuntime(ImageRuntime):
        def observe(self):
            observation = super().observe()
            if self.observation_count == 2:
                self.request_stop()
            return observation

    backend = ScriptedBackend(motion())
    session, runtime, _ = session_for(backend, CancelAtGuardRuntime())
    assert session.run("往前走")["status"] == "cancelled"
    assert len(backend.requests) == 1
    assert runtime.calls == []
