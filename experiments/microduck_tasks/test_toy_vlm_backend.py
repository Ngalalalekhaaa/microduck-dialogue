"""Direct generation keeps raw language/plans auditable and respects executor context."""

import copy
import json
from types import SimpleNamespace

import pytest

from .test_toy_session import FakeRuntime, GateRunner
from .toy_fixtures import active, kick, move, perform, posture, request
from .toy_protocol import VERSION
from .toy_session import ToySession
from .toy_vlm_backend import ToyVLMBackend


def response(reply="这是模型自己生成的答复。", decision="respond", goals=()):
    return dict(schema_version=VERSION, reply=reply, decision=decision, goals=list(goals))


def backend(value):
    result = object.__new__(ToyVLMBackend)
    result.calls = []

    def generate(state, images):
        result.calls.append((copy.deepcopy(state), images))
        if isinstance(result.generated, Exception):
            raise result.generated
        raw = (
            result.generated if isinstance(result.generated, str) else json.dumps(result.generated, ensure_ascii=False)
        )
        return raw, dict(input_tokens=15, output_tokens=10, finish_reason="eos", seconds=0.25)

    result.generator = SimpleNamespace(generate=generate)
    result.generated = value
    result.metadata = dict(goal_source="vlm_generation", reply_source="vlm_generation")
    result.last_trace = None
    return result


def test_chat_preserves_generated_words_and_has_no_rule_teacher_dependency(monkeypatch):
    from .toy_teacher import ToyTeacher

    monkeypatch.setattr(ToyTeacher, "decide", lambda *_: pytest.fail("The model reply must not use the rule teacher."))
    expected = response("今天想和你聊一颗会说话的星星。")
    model = backend(expected)
    state = request("今天聊点什么？", planned=True)
    state["history"] = [dict(role="user", text="我喜欢星空"), dict(role="assistant", text="我记住了。")]
    images = dict(front="front-frame", near="near-frame")
    assert model.decide_with_images(state, images) == expected
    assert model.calls == [(state, images)] and model.calls[0][1] is images
    assert model.last_trace["raw_response"] == expected
    assert model.last_trace["raw_prediction"] == dict(decision="respond", goals=[])
    assert model.last_trace["decision_source"] == model.last_trace["reply_source"] == "vlm_generation"
    assert model.last_trace["model_called"] and "rejected" not in model.last_trace


def test_valid_but_wrong_semantic_plan_is_not_corrected_and_records_generated_reply(tmp_path):
    expected = response("我准备去踢红球。", "replace", [kick("red")])
    model = backend(expected)
    runtime = FakeRuntime(tmp_path / "episode")
    session = ToySession(runtime, backend=model)
    session.runner = GateRunner(runtime)
    try:
        assert session.run("踢蓝球") == expected
        assert [call[0] for call in session.runner.calls] == [kick("red")]
        user, feedback = [row for row in session.recorder.events if row["event"] == "dialogue"]
        assert user["response"] == user["model_trace"]["raw_response"] == expected
        assert user["model_trace"]["raw_prediction"] == dict(decision="replace", goals=[kick("red")])
        assert user["model_trace"]["reply_source"] == "vlm_generation"
        assert session.recorder.metadata["controller_source"] == model.name
        assert session.recorder.metadata["training_ready"] is False
        assert feedback["model_trace"]["decision_source"] == "trusted_execution_feedback"
        assert feedback["model_trace"]["reply_source"] == "template"
        assert not feedback["model_trace"]["model_called"] and len(model.calls) == 1
    finally:
        session.close()


@pytest.mark.parametrize(
    "invalid",
    [
        response("去做未知动作。", "replace", [perform("backflip")]),
        response("没有目标。", "replace"),
        response("聊聊天并偷偷走路。", "respond", [move()]),
        response("一直走后点头。", "replace", [move(continuous=True), perform("nod")]),
        {**response(), "unexpected": "field"},
    ],
)
def test_invalid_json_object_is_rejected_before_any_motion_and_preserves_raw(tmp_path, invalid):
    model = backend(invalid)
    session = ToySession(FakeRuntime(tmp_path / "episode"), backend=model)
    session.runner = GateRunner(session.runtime)
    try:
        output = session.run("点头")
        assert output["decision"] == "decline" and not output["goals"]
        assert not session.runner.calls and session.execution["state"] == "idle"
        assert model.last_trace["raw_response"] == invalid
        assert model.last_trace["raw_prediction"] == {key: invalid[key] for key in ("decision", "goals")}
        assert model.last_trace["rejected"]["reason"] and model.last_trace["reply_source"] == "template"
        assert model.last_trace["decision_source"] == "vlm_generation"
    finally:
        session.close()


@pytest.mark.parametrize("raw", ["```json\n{}\n```", '{"decision":"respond",', "[]", '{"goals":NaN}'])
def test_malformed_generation_is_not_repaired_and_is_still_auditable(raw):
    model = backend(raw)
    actual = model.decide_with_images(request("你好", planned=True), {})
    assert actual["decision"] == "decline" and not actual["goals"]
    assert model.last_trace["raw_text"] == raw and model.last_trace["raw_response"] is None
    assert model.last_trace["raw_prediction"] is None and model.last_trace["rejected"]
    json.dumps(model.last_trace, allow_nan=False)


@pytest.mark.parametrize(
    "change, decision, goals, reason",
    [
        (("observation", "environment", "motion_allowed"), "replace", [perform("nod")], "motion gate"),
        (("observation", "age_s"), "replace", [perform("nod")], "Refresh observations"),
        (("observation", "posture"), "replace", [perform("nod")], "standing transition"),
        (("capabilities", "self_actions"), "replace", [perform("nod")], "currently unavailable"),
        (("capabilities", "search_targets"), "replace", [kick("blue")], "supported visual search"),
        (("capabilities", "continuous_move"), "replace", [move(continuous=True)], "Continuous motion"),
        (("capabilities", "sit_hold"), "replace", [posture("sitting")], "sitting hold"),
        (("observation", "environment", "surface"), "replace", [move()], "edge guard"),
        (None, "append", [perform("nod")], "existing task"),
        (None, "resume", [], "paused tasks"),
    ],
)
def test_context_constraints_refuse_generated_motion(change, decision, goals, reason):
    state = request("请执行", planned=True)
    if change:
        values = dict(
            motion_allowed=False,
            age_s=1,
            posture="sitting",
            self_actions=[],
            search_targets=[],
            continuous_move=False,
            sit_hold=False,
            surface="desktop",
        )
        parent = state
        for key in change[:-1]:
            parent = parent[key]
        parent[change[-1]] = values[change[-1]]
    original = response("准备执行。", decision, goals)
    model = backend(original)
    actual = model.decide_with_images(state, {})
    assert actual["decision"] == "decline" and not actual["goals"]
    assert reason in model.last_trace["rejected"]["reason"]
    assert model.last_trace["raw_response"] == original and model.last_trace["model_called"]


def test_resume_only_accepts_actual_paused_continuous_task():
    model = backend(response("继续刚才的任务。", "resume"))
    state = request("接着来", planned=True)
    state["execution"] = active(perform("nod"), paused=True)
    assert model.decide_with_images(state, {})["decision"] == "decline"
    assert "continuous walking" in model.last_trace["rejected"]["reason"]
    state["execution"] = active(move(continuous=True), paused=True)
    assert model.decide_with_images(state, {}) == model.generated


@pytest.mark.parametrize("text, decision, source", [("停下", "cancel", "host_stop"), ("暂停", "pause", "host_pause")])
def test_explicit_controls_bypass_generator(text, decision, source):
    model = backend(RuntimeError("The generator must not run for host controls."))
    state = request(text, planned=True)
    state["execution"] = active(move(continuous=True))
    output = model.decide_with_images(state, {})
    assert output["decision"] == decision and not output["goals"] and not model.calls
    assert model.last_trace["decision_source"] == source and not model.last_trace["model_called"]
    assert model.last_trace["reply_source"] == "template" and model.last_trace["raw_response"] is None


@pytest.mark.parametrize("status, verified, phrase", [("succeeded", True, "通过执行验收"), ("failed", False, "未完成")])
def test_feedback_uses_verified_executor_result_without_model(status, verified, phrase):
    model = backend(RuntimeError("Do not infer completed results with the language model."))
    state = request("执行器结果", planned=True)
    state["event"]["kind"] = "feedback"
    state["execution"]["last_result"] = dict(
        goal=perform("nod"), status=status, reason="fixture_verification", verified=verified
    )
    output = model.decide_with_images(state, {})
    assert output["decision"] == "respond" and not output["goals"] and phrase in output["reply"]
    assert not model.calls and model.last_trace["decision_source"] == "trusted_execution_feedback"
    assert model.last_trace["reply_source"] == "template" and not model.last_trace["model_called"]


def test_generation_exception_is_logged_and_cannot_start_motion():
    model = backend(RuntimeError("fixture generation failure"))
    output = model.decide_with_images(request("前进", planned=True), {})
    assert output["decision"] == "decline" and not output["goals"]
    assert model.last_trace["model_called"] and model.last_trace["raw_response"] is None
    assert model.last_trace["rejected"] == dict(type="RuntimeError", reason="fixture generation failure")


def test_invalid_host_request_is_refused_before_generator():
    model = backend(response())
    state = request("你好", planned=True)
    state["observation"]["age_s"] = float("nan")
    with pytest.raises(ValueError, match="finite number"):
        model.decide_with_images(state, {})
    assert not model.calls
