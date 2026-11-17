"""Generated dialogue shares validated task plans while host controls remain prompt."""

import copy
import json
from types import SimpleNamespace

import pytest

from .q1r_backend import ToyQ1RBackend
from .q1r_prompt import validate_prediction
from .toy_fixtures import active, kick, perform, request


def backend(value):
    model = object.__new__(ToyQ1RBackend)
    calls = []

    def generate(state, images):
        calls.append(copy.deepcopy(state))
        if isinstance(value, Exception):
            raise value
        return json.dumps(value, ensure_ascii=False), dict(seconds=0.1)

    model.generator = SimpleNamespace(generate=generate)
    model.calls = calls
    model.last_trace = None
    return model


def test_greeting_while_running_preserves_tasks_and_returns_actual_generated_reply():
    state = request("继续忙你的，陪我聊聊", planned=True)
    state["execution"] = active(kick("blue"), pending=[perform("nod")])
    original = copy.deepcopy(state)
    model = backend(dict(decision="respond", goals=[], reply="可以呀，你今天过得怎么样？"))
    result = model.decide_with_images(state, {})
    assert result["reply"] == "可以呀，你今天过得怎么样？"
    assert result["decision"] == "respond" and not result["goals"]
    assert state == original and model.last_trace["reply_source"] == "model_generation"


def test_executable_command_has_generated_acknowledgement_and_plan():
    model = backend(dict(decision="replace", goals=[perform("nod")], reply="你好，我点个头回应你。"))
    result = model.decide_with_images(request("你好，点头打个招呼", planned=True), {})
    assert result["goals"] == [perform("nod")]
    assert result["reply"] == "你好，我点个头回应你。"


def test_unsupported_object_clarification_keeps_question_without_substitution():
    model = backend(dict(decision="clarify", goals=[], reply="目前不能踢方块，要改成踢红球吗？"))
    result = model.decide_with_images(request("踢方块", planned=True), {})
    assert result["decision"] == "clarify" and result["goals"] == []
    assert "方块" in result["reply"]


@pytest.mark.parametrize("text,decision", [("停", "cancel"), ("暂停", "pause"), ("继续", "resume")])
def test_host_controls_do_not_wait_for_model_inference(text, decision):
    model = backend(RuntimeError("Must not run inference"))
    state = request(text, planned=True)
    state["execution"] = active(kick("blue"), paused=decision == "resume")
    result = model.decide_with_images(state, {})
    assert result["decision"] == decision
    assert not model.calls and model.last_trace["reply_source"] == "host_control"


def test_feedback_cannot_launch_actions_even_if_structurally_valid():
    state = request("更新结果", planned=True)
    state["event"]["kind"] = "feedback"
    state["execution"]["last_result"] = dict(goal=kick("blue"), status="failed", reason="target_lost", verified=False)
    prediction = dict(decision="replace", goals=[kick("red")], reply="我再踢一次。")
    with pytest.raises(ValueError, match="feedback"):
        validate_prediction(prediction, state)
    model = backend(prediction)
    result = model.decide_with_images(state, {})
    assert result["decision"] == "respond" and result["goals"] == []
    assert model.last_trace["rejected"] and model.last_trace["reply_source"] == "host_failure_notice"


def test_invalid_prediction_keeps_raw_trace_and_is_not_repaired_by_teacher():
    value = dict(decision="replace", goals=[perform("fly")], reply="我来飞。")
    model = backend(value)
    result = model.decide_with_images(request("飞起来", planned=True), {})
    assert result["decision"] == "decline" and not result["goals"]
    assert model.last_trace["raw_prediction"] == value and model.last_trace["rejected"]
