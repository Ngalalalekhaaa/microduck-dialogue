"""Learned plans remain distinguishable from templates and cannot bypass task validation."""

import copy
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from .test_toy_session import FakeRuntime, GateRunner
from .toy_fixtures import active, move, perform, request
from .toy_selector_backend import ToySelectorBackend
from .toy_session import ToySession


def backend(prediction):
    result = object.__new__(ToySelectorBackend)
    result.calls = []

    def extract(state, images):
        result.calls.append((state, images))
        return np.zeros(2048, dtype=np.float32), {"seconds": 0.0, "tokens": 10}

    result.features = SimpleNamespace(extract=extract)
    result.selector = SimpleNamespace(predict=lambda vector: [copy.deepcopy(prediction)])
    result.metadata = {"feature_config": {"version": "test-only"}, "checkpoint_sha256": "test-only"}
    result.last_trace = None
    return result


def test_session_executes_predicted_goal_without_rule_correction_and_records_sources(tmp_path):
    class ObservedRuntime(FakeRuntime):
        def observe(self):
            observation, images, extra = super().observe()
            if self.frame == 1:
                self.first_images = images
            return observation, images, extra

    learned = backend({"decision": "replace", "goals": [perform("look_around")]})
    runtime = ObservedRuntime(tmp_path / "episode")
    session = ToySession(runtime, backend=learned)
    session.runner = GateRunner(runtime)
    try:
        output = session.run("点头")
        assert output["goals"] == [perform("look_around")]
        assert "左右看" in output["reply"] and "点头" not in output["reply"]
        assert [row[0] for row in session.runner.calls] == [perform("look_around")]
        assert len(learned.calls) == 1 and learned.calls[0][1] is runtime.first_images
        assert runtime.frame == 2, "One user capture and one feedback capture; the backend must not recapture."
        assert session.recorder.metadata["training_ready"] is False
        assert session.recorder.metadata["controller_source"] == learned.name
        assert all(row["controller_source"] == learned.name for row in session.recorder.events)
        user, feedback = [row for row in session.recorder.events if row["event"] == "dialogue"]
        assert user["model_trace"]["raw_prediction"]["goals"] == [perform("look_around")]
        assert user["model_trace"]["reply_source"] == "template"
        assert user["model_trace"]["goal_source"] == "learned_selector"
        assert feedback["model_trace"]["decision_source"] == "trusted_execution_feedback"
        assert feedback["response"]["decision"] == "respond" and not feedback["response"]["goals"]
        assert feedback["model_trace"]["model_called"] is False
        assert "通过执行验收" in feedback["response"]["reply"]
        from .toy_dataset import read_episode

        with pytest.raises(ValueError, match="relabeled demos cannot train"):
            read_episode(runtime.output)
    finally:
        session.close()


@pytest.mark.parametrize(
    "prediction",
    [
        {"decision": "replace", "goals": [perform("unsupported_action")]},
        {"decision": "resume", "goals": []},
        {"decision": "replace", "goals": []},
        {"decision": "replace", "goals": [perform("nod")], "unexpected": True},
    ],
)
def test_invalid_prediction_is_refused_and_original_plan_preserved(prediction):
    learned = backend(prediction)
    output = learned.decide_with_images(request("点头", planned=True), {})
    assert output["decision"] == "decline" and output["goals"] == []
    assert learned.last_trace["raw_prediction"] == prediction
    assert learned.last_trace["rejected"]["reason"]


def test_nonfinite_prediction_is_loggable_without_starting_a_goal():
    learned = backend({"decision": "replace", "goals": [move(seconds=float("nan"))]})
    output = learned.decide_with_images(request("前进", planned=True), {})
    assert output["decision"] == "decline" and output["goals"] == []
    assert "nan" in learned.last_trace["raw_prediction_repr"]
    json.dumps(learned.last_trace, allow_nan=False)


def test_direct_stop_skips_inference_and_cannot_restart_motion(tmp_path):
    learned = backend({"decision": "replace", "goals": [move(continuous=True)]})
    runtime = FakeRuntime(tmp_path / "episode")
    session = ToySession(runtime, backend=learned)
    session.runner = GateRunner(runtime)
    try:
        session.run("一直往前走不要停", wait=False)
        assert session.runner.started[0].wait(1)
        assert session.run("停下")["decision"] == "cancel"
        assert len(learned.calls) == 1 and len(session.runner.calls) == 1
        assert session.execution["state"] == "idle" and not session.execution["pending"]
        assert session.results[0]["status"] == "cancelled"
        assert learned.last_trace["decision_source"] == "host_stop"
        assert learned.last_trace["model_called"] is False
    finally:
        session.close()


@pytest.mark.parametrize("text", ["你好", "点头"])
def test_respond_template_never_turns_a_missed_plan_into_teacher_goals(text):
    learned = backend({"decision": "respond", "goals": []})
    output = learned.decide_with_images(request(text, planned=True), {})
    assert output["decision"] == "respond" and output["goals"] == []
    assert "小鸭" in output["reply"] if text == "你好" else "没有识别到" in output["reply"]


def test_finite_action_pause_prediction_is_declined():
    learned = backend({"decision": "pause", "goals": []})
    state = request("先等一下", planned=True)
    state["execution"] = active(perform("nod"))
    assert learned.decide_with_images(state, {})["decision"] == "decline"
    assert learned.last_trace["raw_prediction"]["decision"] == "pause"


def test_checkpoint_validation_runs_before_model_allocation(monkeypatch, tmp_path):
    from . import toy_head_features, toy_selector_model
    from .toy_dataset import contract_digest

    calls = []

    def load(path, *, feature_config, contract_sha256):
        assert path == tmp_path / "selector.pt"
        assert feature_config == toy_head_features.feature_config()
        assert contract_sha256 == contract_digest()
        calls.append("checkpoint")
        raise ValueError("incompatible feature version")

    def frozen(_path):
        pytest.fail("Do not allocate a VLM for an incompatible head.")

    monkeypatch.setattr(toy_selector_model, "load_selector", load)
    monkeypatch.setattr(toy_head_features, "ToyFrozenFeatures", frozen)
    with pytest.raises(ValueError, match="incompatible feature"):
        ToySelectorBackend(tmp_path / "selector.pt", "unused")
    assert calls == ["checkpoint"]


def test_selector_cli_requires_checkpoint_before_loading_any_model():
    from .toy_demo import main

    with pytest.raises(SystemExit) as exc:
        main(["--backend", "selector", "--no-viewer"])
    assert exc.value.code == 2


def test_lockstep_blocks_background_physics_through_capture_and_inference(tmp_path):
    learned = backend({"decision": "respond", "goals": []})
    runtime = FakeRuntime(tmp_path / "episode")
    session = ToySession(runtime, backend=learned)
    entered, release, attempted, ticked = (threading.Event() for _ in range(4))
    errors = []

    def extract(state, images):
        assert state["observation"]["age_s"] == 0
        entered.set()
        assert release.wait(2)
        assert runtime.data.time == 0, "A task worker cannot advance physics during inference."
        return np.zeros(2048, dtype=np.float32), {"seconds": 0.0}

    def infer():
        try:
            session.run("你好")
        except Exception as exc:
            errors.append(exc)

    def tick():
        attempted.set()
        with runtime.lock:
            runtime.data.time += 0.1
            ticked.set()

    learned.features.extract = extract
    inference = threading.Thread(target=infer, daemon=True)
    physics = threading.Thread(target=tick, daemon=True)
    try:
        inference.start()
        assert entered.wait(1)
        physics.start()
        assert attempted.wait(1)
        assert not ticked.wait(0.05)
        release.set()
        inference.join(2)
        physics.join(2)
        assert not errors and not inference.is_alive() and not physics.is_alive()
        assert ticked.is_set() and runtime.data.time == 0.1
        assert session.recorder.events[0]["observation_time_s"] == 0
    finally:
        release.set()
        inference.join(2)
        if physics.ident is not None:
            physics.join(2)
        session.close()
