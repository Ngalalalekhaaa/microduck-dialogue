"""The v2 client, training encoder and adapter server must use the same contract."""

import hashlib
import io
import json

import pytest

from .plan_backend import LocalPlanBackend
from .plan_protocol import CATALOG, SYSTEM, VERSION
from .serve_adapter import inspect_files
from .serve_adapter_protocol import decode_request
from .test_serve_adapter import sample
from .train_vlm import messages_for


def test_v2_wire_roundtrip_matches_training_and_preserves_camera_order():
    request = sample()
    request.state.clear()
    request.state.update(
        yaw=0.1,
        stationary=True,
        tilt_deg=2.0,
        entities=[],
        observation_id="frame_1",
        task_elapsed_s=0.0,
        available_skills=CATALOG,
        schema_version=VERSION,
    )
    request.context.clear()
    request.context.update(task_id="task_1", committed_steps=[], recent_outcomes=[])
    captured = []

    class Capture:
        def open(self, wire, timeout):
            captured.append(wire.data)
            result = {"schema_version": VERSION, "status": "answer", "target": None, "steps": [], "reply": "看到了。"}
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(result)}}]}).encode())

    backend = LocalPlanBackend("http://127.0.0.1:8085/v1/chat/completions", "microduck-plan-qlora")
    backend._opener = Capture()
    assert backend.decide(request).status == "answer"
    messages, images, tokens = decode_request(captured[0], backend.model, protocol="v2")
    row = dict(
        format_version="microduck-plan-sft-2",
        instruction=request.instruction,
        state=request.state,
        context=request.context,
    )
    assert messages == messages_for(row)
    assert tokens == 512
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert images[1].getpixel((0, 0)) == (0, 0, 255)
    with pytest.raises(ValueError):
        decode_request(captured[0], backend.model, protocol="v1")


def test_v2_adapter_identity_cannot_be_loaded_as_v1(tmp_path):
    model, adapter = tmp_path / "base", tmp_path / "run" / "final_adapter"
    model.mkdir()
    adapter.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"inert base")
    (adapter / "adapter_model.safetensors").write_bytes(b"inert adapter")
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM"}))
    (adapter.parent / "system_prompt.txt").write_text(SYSTEM)
    config = {
        "prompt_version": VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "model_sha256": hashlib.sha256(b"inert base").hexdigest(),
        "model_revision": "test-only-revision",
    }
    (adapter.parent / "training_config.json").write_text(json.dumps(config))
    _, metadata = inspect_files(model, adapter, protocol="v2")
    backend = LocalPlanBackend("http://127.0.0.1:8085/v1/chat/completions", "microduck-plan-qlora")
    assert backend._accept_model_metadata(metadata)["prompt_version"] == VERSION
    assert backend.metadata["robot_finetuned"] is True
    with pytest.raises(ValueError, match="prompt differs"):
        inspect_files(model, adapter)
