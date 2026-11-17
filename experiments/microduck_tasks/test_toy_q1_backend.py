"""Online Q1 preserves model plans, exposes template replies, and encodes text only."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from .test_toy_session import FakeRuntime, GateRunner
from .toy_fixtures import active, kick, perform, request
from .toy_q1_backend import Q1PlanGenerator, ToyQ1Backend
from .toy_session import ToySession


def backend(value):
    model = object.__new__(ToyQ1Backend)
    model.calls = []

    def generate(state, images):
        model.calls.append((copy.deepcopy(state), images))
        if isinstance(value, Exception):
            raise value
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return raw, dict(seconds=0.2, input_tokens=20, output_tokens=12, finish_reason="eos")

    model.generator = SimpleNamespace(generate=generate)
    model.metadata = dict(goal_source="q1_lora_generation", reply_source="template")
    model.last_trace = None
    return model


def test_respond_is_explicitly_not_a_model_chat_reply_or_teacher_answer(monkeypatch):
    from .toy_teacher import ToyTeacher

    monkeypatch.setattr(ToyTeacher, "decide", lambda *_: pytest.fail("Q1 cannot be replaced by a rule teacher."))
    model = backend(dict(decision="respond", goals=[]))
    state = request("你叫什么名字？", planned=True)
    result = model.decide_with_images(state, object())
    assert result["decision"] == "respond" and result["goals"] == []
    assert "尚未训练自然语言回答" in result["reply"]
    assert model.last_trace["model_called"]
    assert model.last_trace["decision_source"] == "q1_lora_generation"
    assert model.last_trace["reply_source"] == "template"
    assert model.last_trace["raw_prediction"] == dict(decision="respond", goals=[])


def test_valid_wrong_plan_is_not_corrected_and_feedback_does_not_call_model(tmp_path):
    model = backend(dict(decision="replace", goals=[kick("red")]))
    session = ToySession(FakeRuntime(tmp_path / "episode"), backend=model)
    session.runner = GateRunner(session.runtime)
    try:
        output = session.run("踢蓝球")
        assert output["goals"] == [kick("red")]
        assert "红球" in output["reply"]
        assert [item[0] for item in session.runner.calls] == [kick("red")]
        rows = [row for row in session.recorder.events if row["event"] == "dialogue"]
        assert rows[0]["model_trace"]["raw_prediction"] == dict(decision="replace", goals=[kick("red")])
        assert rows[-1]["model_trace"]["decision_source"] == "trusted_execution_feedback"
        assert len(model.calls) == 1
    finally:
        session.close()


@pytest.mark.parametrize("count", [7, 15])
def test_live_capability_is_respected_without_a_hidden_training_length_limit(count):
    goals = [perform("nod")] * count
    model = backend(dict(decision="replace", goals=goals))
    state = request("按顺序点头", planned=True)
    state["capabilities"]["max_goals"] = 15
    assert model.decide_with_images(state, {})["goals"] == goals
    state["capabilities"]["max_goals"] = 6
    output = model.decide_with_images(state, {})
    assert output["decision"] == "decline" and output["goals"] == []
    assert model.last_trace["raw_prediction"]["goals"] == goals


@pytest.mark.parametrize(
    "value",
    [
        dict(decision="replace", goals=[]),
        dict(decision="respond", goals=[perform("nod")]),
        dict(decision="replace", goals=[perform("unavailable")]),
        dict(decision="respond", goals=[], reply="pretend this was a trained chat response"),
        '{"decision":"respond","decision":"replace","goals":[]}',
        '{"decision":"replace","goals":[NaN]}',
        "```json\n{}\n```",
        RuntimeError("Inference failed"),
    ],
)
def test_invalid_generation_is_rejected_without_rule_repair(value):
    model = backend(value)
    output = model.decide_with_images(request("点头", planned=True), {})
    assert output["decision"] == "decline" and output["goals"] == []
    assert model.last_trace["rejected"] and model.last_trace["model_called"]
    if isinstance(value, dict):
        assert model.last_trace["raw_prediction"] == value


@pytest.mark.parametrize("text", ["停", "停止", "停下", "取消当前任务"])
def test_stop_bypasses_q1_inference(text):
    model = backend(RuntimeError("Do not call the model for stop."))
    assert model.decide_with_images(request(text, planned=True), {})["decision"] == "cancel"
    assert not model.calls and not model.last_trace["model_called"]


@pytest.mark.parametrize("goal", [perform("nod"), kick("red")])
def test_safe_pause_and_resume_cover_finite_goals_without_model_inference(goal):
    model = backend(RuntimeError("Do not call Q1 for host controls."))
    state = request("暂停", planned=True)
    state["execution"] = active(goal)
    assert model.decide_with_images(state, {})["decision"] == "pause"
    state["event"]["text"] = "继续"
    state["execution"] = active(goal, paused=True)
    assert model.decide_with_images(state, {})["decision"] == "resume"
    assert not model.calls and not model.last_trace["model_called"]


def test_idle_resume_does_not_resurrect_a_cancelled_task():
    model = backend(RuntimeError("Do not call Q1 for host controls."))
    assert model.decide_with_images(request("继续", planned=True), {})["decision"] == "clarify"
    assert not model.calls


class Processor:
    def __init__(self, count=5, padding=0):
        self.count, self.padding = count, padding
        self.calls = []

    def apply_chat_template(self, messages, **options):
        self.messages, self.options = messages, options
        return "rendered P6 request"

    def __call__(self, **options):
        self.calls.append(options)
        mask = np.ones((1, self.count), dtype=np.int64)
        if self.padding:
            mask[0, -self.padding :] = 0
        return dict(input_ids=np.arange(self.count)[None, :], attention_mask=mask)


def generator(processor):
    result = object.__new__(Q1PlanGenerator)
    result.processor, result.model, result.max_input_tokens = processor, object(), 4096
    return result


def test_fresh_p6_encoding_never_passes_images_answers_or_padding_to_generation():
    processor = Processor()
    model = generator(processor)
    received = []
    model._generate = lambda *args: received.append(args) or ("generated plan", {})
    state = request("点头", planned=True)
    before = copy.deepcopy(state)
    assert model.generate(state, object()) == ("generated plan", {})
    assert state == before and received == [(model.model, processor, dict(prefix_ids=list(range(5))))]
    assert processor.calls == [dict(text=["rendered P6 request"], images=None, return_tensors="pt", padding=False)]
    assert processor.options["enable_thinking"] is False
    assert all(part["type"] == "text" for part in processor.messages[1]["content"])
    assert len(processor.messages) == 2 and processor.messages[-1]["role"] == "user"


@pytest.mark.parametrize("count,padding", [(4097, 0), (5, 1)])
def test_encoder_refuses_overlong_or_padded_input_without_truncation(count, padding):
    model = generator(Processor(count, padding))
    with pytest.raises(ValueError):
        model.encode_request(request("点头", planned=True))


@pytest.mark.parametrize("matching", [True, False])
def test_loader_selects_saved_adapter_and_checks_its_actual_weights(tmp_path, monkeypatch, matching):
    from . import q0_model

    checkpoint = tmp_path / "checkpoint_100"
    checkpoint.mkdir()
    (checkpoint / "identity.json").write_text(json.dumps(dict(step=100, weights_sha256="expected")))
    (checkpoint / "adapter_model.safetensors").write_bytes(b"test adapter")
    calls = []
    weight = object()
    model = SimpleNamespace(
        load_adapter=lambda *args, **kwargs: calls.append((args, kwargs)),
        set_adapter=lambda name: calls.append(name),
        named_parameters=lambda: [("layer.lora_A.online.weight", weight)],
        requires_grad_=lambda value: calls.append(("requires_grad", value)),
        eval=lambda: calls.append("eval"),
    )
    metadata = dict(trainable_names=["layer.lora_A.default.weight"])
    monkeypatch.setattr(q0_model, "load_model", lambda *args: (model, Processor(), metadata))

    def checksum(parameters):
        assert parameters == {"layer.lora_A.default.weight": weight}
        return "expected" if matching else "wrong"

    monkeypatch.setattr(q0_model, "weights_digest", checksum)
    if not matching:
        with pytest.raises(ValueError, match="differs from checkpoint identity"):
            Q1PlanGenerator(tmp_path, checkpoint)
        return
    loaded = Q1PlanGenerator(tmp_path, checkpoint)
    assert calls[0] == ((str(checkpoint),), dict(adapter_name="online", is_trainable=False))
    assert calls[1] == "online" and ("requires_grad", False) in calls
    assert loaded.metadata["checkpoint_identity_verified"]
    assert loaded.metadata["weights_sha256"] == "expected"
    assert loaded.metadata["images_passed_to_model"] is False
