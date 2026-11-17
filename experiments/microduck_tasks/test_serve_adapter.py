"""Exercise the adapter HTTP boundary without loading Torch or a large model."""

import base64
import hashlib
import io
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from experiments.microduck_tasks.serve_adapter import handler_for, inspect_files
from experiments.microduck_tasks.serve_adapter_protocol import MAX_BODY_BYTES, decode_request
from experiments.microduck_tasks.skill_vlm import PROMPT_VERSION, SYSTEM_PROMPT, LocalVLMBackend, SkillInput
from experiments.microduck_tasks.train_vlm import messages_for

ALIAS = "microduck-qlora"
ANSWER = {"type": "answer", "skill": None, "seconds": 0, "target_color": None, "reply": "前面有红球。"}


def sample():
    return SkillInput(
        instruction="先看看红球在哪儿",
        images={
            "front": np.full((12, 16, 3), [255, 0, 0], dtype=np.uint8),
            "near": np.full((8, 10, 3), [0, 0, 255], dtype=np.uint8),
        },
        state={"yaw": 0.1, "stationary": True, "tilt_deg": 2.0, "sim_time_s": 3.2},
        context={"step": 0, "task_id": "log-only-task-id"},
    )


@pytest.fixture
def client_payload():
    """Capture the real client wire request instead of reproducing its encoder."""
    captured = {}

    class Capture:
        def open(self, request, timeout):
            captured.update(json.loads(request.data))
            value = {"choices": [{"message": {"content": json.dumps(ANSWER)}, "finish_reason": "stop"}]}
            return io.BytesIO(json.dumps(value).encode())

    backend = LocalVLMBackend(model=ALIAS)
    backend._opener = Capture()
    backend.decide(sample())
    return captured


def encoded_image(image, format="PNG"):
    stream = io.BytesIO()
    image.save(stream, format=format)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def exchange(server, payload=None, path="/v1/chat/completions", data=None, headers=None):
    if data is None and payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(request, timeout=3)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.loads(response.read())


def decode(payload):
    return decode_request(json.dumps(payload, ensure_ascii=False).encode("utf-8"), ALIAS)


class FakeEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []
        self.error = None
        self.metadata = {
            "model": ALIAS,
            "robot_finetuned": True,
            "adapter": "/test/qlora/final_adapter",
            "adapter_sha256": "a" * 64,
            "base": "/test/qwen/base",
            "prompt_version": PROMPT_VERSION,
            "generation_mode": "unconstrained_greedy_client_validated",
        }

    def generate(self, raw):
        self.calls.append(decode_request(raw, ALIAS))
        if self.error is not None:
            raise self.error
        return {
            "model": ALIAS,
            "choices": [{"message": {"role": "assistant", "content": json.dumps(ANSWER)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "microduck_model_metadata": self.metadata,
        }


@contextmanager
def serving(engine):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_client_request_reconstructs_training_messages_and_original_camera_pixels(client_payload):
    messages, images, max_tokens = decode(client_payload)
    original = sample()
    expected = messages_for({"instruction": original.instruction, "state": original.state, "context": {"step": 0}})
    assert messages == expected
    assert max_tokens == 256
    assert messages[1]["content"][-1]["text"] == client_payload["messages"][1]["content"][-1]["text"]
    assert len(images) == 2
    for image, camera in zip(images, ("front", "near"), strict=True):
        assert isinstance(image, Image.Image) and image.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(image), original.images[camera])


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/camera.png",
        "file:///etc/passwd",
        "/home/luckysir/private.png",
        "data:image/png;base64,%%%",
        "data:image/png;base64," + base64.b64encode(b"not a PNG").decode("ascii"),
    ],
)
def test_camera_payload_never_loads_urls_paths_or_invalid_png(client_payload, url):
    client_payload["messages"][1]["content"][1]["image_url"]["url"] = url
    with pytest.raises(ValueError):
        decode(client_payload)


@pytest.mark.parametrize("size,format", [((4097, 1), "PNG"), ((2049, 2048), "PNG"), ((8, 8), "JPEG")])
def test_reject_image_bombs_and_non_png_disguised_as_png(client_payload, size, format):
    client_payload["messages"][1]["content"][1]["image_url"]["url"] = encoded_image(Image.new("RGB", size), format)
    with pytest.raises(ValueError):
        decode(client_payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "some-other-model"),
        ("temperature", 0.2),
        ("stream", True),
        ("max_tokens", 257),
        ("max_tokens", True),
        ("response_format", {"type": "json_object"}),
        ("chat_template_kwargs", {"enable_thinking": True}),
    ],
)
def test_request_cannot_silently_change_model_or_decoding_contract(client_payload, field, value):
    client_payload[field] = value
    with pytest.raises(ValueError):
        decode(client_payload)


@pytest.mark.parametrize("mutation", ["system", "role", "third_message", "camera_order", "truth_state"])
def test_prompt_roles_camera_order_and_training_state_contract_are_enforced(client_payload, mutation):
    messages = client_payload["messages"]
    if mutation == "system":
        messages[0]["content"] = "A different policy prompt"
    elif mutation == "role":
        messages[1]["role"] = "assistant"
    elif mutation == "third_message":
        messages.append({"role": "assistant", "content": "pretend answer"})
    elif mutation == "camera_order":
        content = messages[1]["content"]
        messages[1]["content"] = content[2:4] + content[:2] + content[4:]
    else:
        text = messages[1]["content"][-1]
        payload = json.loads(text["text"])
        payload["state"]["ball_world_xyz"] = [0.1, 0, 0]
        text["text"] = json.dumps(payload)
    with pytest.raises(ValueError):
        decode(client_payload)


def test_loopback_health_and_completion_metadata_identify_loaded_adapter():
    engine = FakeEngine()
    with serving(engine) as server:
        backend = LocalVLMBackend(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", model=ALIAS)
        identity = backend.refresh_model_metadata()
        assert identity == engine.metadata
        assert backend.metadata["robot_finetuned"] is True
        assert backend.decide(sample()).to_dict() == ANSWER
    assert len(engine.calls) == 1
    assert backend.name == "local_finetuned_vlm"
    assert backend.last_trace["server_metadata"] == engine.metadata
    assert backend.last_trace["usage"]["total_tokens"] == 30


def test_oversized_http_body_is_rejected_before_reading_or_generating(client_payload):
    engine = FakeEngine()
    with serving(engine) as server:
        status, result = exchange(server, client_payload, headers={"Content-Length": str(MAX_BODY_BYTES + 1)})
    assert status == 413 and "error" in result
    assert engine.calls == []


@pytest.mark.parametrize("headers", [{"Host": "example.invalid"}, {"Origin": "https://example.invalid"}])
def test_wrong_host_and_browser_origin_do_not_reach_inference(client_payload, headers):
    engine = FakeEngine()
    with serving(engine) as server:
        status, result = exchange(server, client_payload, headers=headers)
    assert status == 403 and "error" in result
    assert engine.calls == []


def test_busy_server_returns_retryable_error_without_a_fabricated_action(client_payload):
    engine = FakeEngine()
    with serving(engine) as server, engine.lock:
        health_status, health = exchange(server, path="/health")
        status, result = exchange(server, client_payload)
    assert health_status == 200 and health["busy"] is True
    assert status == 503 and "error" in result and "choices" not in result
    assert engine.calls == []


@pytest.mark.parametrize(
    "error,status",
    [(ValueError("bad model input"), 400), (TimeoutError("deadline reached"), 504), (RuntimeError("CUDA error"), 500)],
)
def test_generation_error_never_becomes_success_and_releases_inference_lock(client_payload, error, status):
    engine = FakeEngine()
    engine.error = error
    with serving(engine) as server:
        actual, result = exchange(server, client_payload)
        assert actual == status and "error" in result and "choices" not in result
        assert not engine.lock.locked()
        engine.error = None
        recovered, result = exchange(server, client_payload)
    assert recovered == 200 and result["choices"][0]["message"]["content"] == json.dumps(ANSWER)


def test_invalid_json_is_reported_as_client_error_and_later_requests_work(client_payload):
    engine = FakeEngine()
    with serving(engine) as server:
        status, result = exchange(server, data=b"not JSON")
        assert status == 400 and "choices" not in result
        assert not engine.lock.locked()
        status, result = exchange(server, client_payload)
    assert status == 200 and "choices" in result


@pytest.fixture
def local_weights(tmp_path):
    """Tiny inert files exercise identity checks without constructing a model."""
    model = tmp_path / "base"
    adapter = tmp_path / "run" / "final_adapter"
    model.mkdir()
    adapter.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"inert base weights")
    (adapter / "adapter_model.safetensors").write_bytes(b"inert adapter weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM"}))
    config = {
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "prompt_version": PROMPT_VERSION,
        "model_sha256": hashlib.sha256(b"inert base weights").hexdigest(),
        "model_revision": "test-only-revision",
    }
    (adapter.parent / "training_config.json").write_text(json.dumps(config))
    (adapter.parent / "system_prompt.txt").write_text(SYSTEM_PROMPT)
    return model, adapter


def test_weight_identity_metadata_records_exact_adapter_and_matches_client_contract(local_weights):
    model, adapter = local_weights
    _, metadata = inspect_files(model, adapter)
    assert metadata["base"] == str(model.resolve())
    assert metadata["adapter"] == str(adapter.resolve())
    assert metadata["adapter_sha256"] == hashlib.sha256(b"inert adapter weights").hexdigest()
    assert metadata["schema_constrained_decoding"] is False
    engine = FakeEngine()
    engine.metadata = metadata
    with serving(engine) as server:
        backend = LocalVLMBackend(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", model=ALIAS)
        identity = backend.refresh_model_metadata()
    assert identity["adapter_sha256"] == metadata["adapter_sha256"]


@pytest.mark.parametrize("mutation", ["base", "prompt", "adapter_type"])
def test_wrong_base_prompt_or_adapter_type_are_rejected_before_model_loading(local_weights, mutation):
    model, adapter = local_weights
    if mutation == "base":
        (model / "model.safetensors").write_bytes(b"different weights")
    elif mutation == "prompt":
        (adapter.parent / "system_prompt.txt").write_text("different policy")
    else:
        (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "OTHER", "task_type": "CAUSAL_LM"}))
    with pytest.raises(ValueError):
        inspect_files(model, adapter)
