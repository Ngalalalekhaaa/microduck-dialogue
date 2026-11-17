"""Validate real local HTTP image transport and strict VLM skill proposals."""

import base64
import copy
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from experiments.microduck_tasks.skill_vlm import LocalVLMBackend, SkillInput, SkillProposal, parse_proposal


def proposal(**updates):
    value = {"type": "skill", "skill": "move_forward", "seconds": 0.32, "target_color": "red", "reply": "向前一点。"}
    value.update(updates)
    return value


def request(**updates):
    values = {
        "instruction": "把红色球踢走",
        "images": {
            "front": np.full((12, 16, 3), [255, 0, 0], dtype=np.uint8),
            "near": np.full((8, 10, 3), [0, 0, 255], dtype=np.uint8),
        },
        "state": {"yaw": 0.1, "stationary": True, "tilt_deg": 2.0, "sim_time_s": 3.2},
        "context": {"step": 0, "task_id": "task_001"},
    }
    values.update(updates)
    return SkillInput(**values)


@pytest.fixture
def server():
    state = {"requests": [], "reply": json.dumps(proposal()), "status": 200, "delay": 0, "finish_reason": "stop"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            state.setdefault("health_requests", []).append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(state.get("health_metadata", {})).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"payload": payload, "path": self.path, "headers": dict(self.headers)})
            time.sleep(state["delay"])
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            if state["status"] == 302:
                self.send_header("Location", "http://example.invalid/v1/chat/completions")
            self.end_headers()
            body = {"choices": [{"message": {"content": state["reply"]}, "finish_reason": state["finish_reason"]}]}
            body.update(state.get("response_metadata", {}))
            try:
                self.wfile.write(json.dumps(body).encode())
            except BrokenPipeError:
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    state["endpoint"] = f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize(
    ("skill", "seconds"),
    [
        ("stand", 0.02),
        ("stand", 1),
        ("move_forward", 0.6),
        ("move_backward", 0.26),
        ("move_left", 0.06),
        ("move_right", 0.1),
        ("turn_left", 1),
        ("turn_right", 0.02),
        ("kick_left", 0.5),
        ("kick_right", 0.5),
    ],
)
def test_supported_skills_preserve_selected_action(skill, seconds):
    value = proposal(skill=skill, seconds=seconds)
    parsed = parse_proposal(json.dumps(value))
    assert isinstance(parsed, SkillProposal)
    assert parsed.to_dict() == value


@pytest.mark.parametrize("kind", ["answer", "clarify", "finish"])
def test_nonmoving_proposals_require_explicit_null_skill(kind):
    value = proposal(type=kind, skill=None, seconds=0)
    assert parse_proposal(json.dumps(value)).type == kind
    with pytest.raises(ValueError, match="Non-skill"):
        parse_proposal(json.dumps(proposal(type=kind)))


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        "{}",
        "```json\n{}\n```",
        "{} {}",
        "not JSON",
        '{"type":"skill","type":"finish","skill":null,"seconds":0,"target_color":null,"reply":""}',
    ],
)
def test_rejects_nonobject_markdown_extra_json_and_duplicate_keys(raw):
    with pytest.raises(ValueError):
        parse_proposal(raw)


@pytest.mark.parametrize(
    "updates",
    [
        {"skill": "arbitrary_joint_command"},
        {"skill": None},
        {"skill": ["stand"]},
        {"seconds": True},
        {"seconds": "0.32"},
        {"seconds": float("nan")},
        {"seconds": float("inf")},
        {"seconds": -0.02},
        {"seconds": 0},
        {"seconds": 0.03},
        {"seconds": 0.62},
        {"skill": "turn_left", "seconds": 1.02},
        {"skill": "stand", "seconds": 3},
        {"skill": "kick_left", "seconds": 0.48},
        {"skill": "kick_right", "seconds": 0.52},
        {"target_color": "green"},
        {"target_color": []},
        {"type": "action"},
        {"type": []},
        {"reply": 1},
        {"reply": "x" * 1001},
        {"joints": [0] * 14},
    ],
)
def test_rejects_invalid_actions_without_repair(updates):
    with pytest.raises(ValueError):
        parse_proposal(json.dumps(proposal(**updates)))


def test_rejects_missing_fields():
    value = proposal()
    del value["target_color"]
    with pytest.raises(ValueError):
        parse_proposal(json.dumps(value))


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://192.168.1.2:8081/v1/chat/completions",
        "https://example.com/v1/chat/completions",
        "http://0.0.0.0:8081/v1/chat/completions",
        "http://127.0.0.1.example.com/v1/chat/completions",
        "ftp://127.0.0.1/v1/chat/completions",
        "http://user:password@127.0.0.1/v1/chat/completions",
        "http://127.0.0.1/v1/chat/completions?key=secret",
        "http://127.0.0.1/v1/chat/completions#fragment",
        "http://127.0.0.1/completions",
        "http://127.0.0.1:999999/v1/chat/completions",
    ],
)
def test_external_or_ambiguous_endpoints_are_rejected(endpoint):
    with pytest.raises(ValueError):
        LocalVLMBackend(endpoint=endpoint)


def test_localhost_is_canonicalized_to_numeric_loopback():
    assert LocalVLMBackend("http://localhost:8081/v1/chat/completions").endpoint.startswith("http://127.0.0.1:")
    assert LocalVLMBackend("http://[::1]:8081/v1/chat/completions").endpoint.startswith("http://[::1]:")


def test_actual_rgb_images_and_whitelisted_state_reach_local_server(server, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://external-proxy.invalid:8080")
    backend = LocalVLMBackend(server["endpoint"])
    sample = request()
    assert backend.decide(sample).skill == "move_forward"
    assert len(server["requests"]) == 1
    sent = server["requests"][0]
    assert sent["path"] == "/v1/chat/completions"
    payload = sent["payload"]
    assert payload["model"] == "microduck-vlm"
    assert payload["temperature"] == 0 and payload["max_tokens"] == 256 and payload["stream"] is False
    assert payload["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    content = payload["messages"][1]["content"]
    pictures = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
    for encoded, camera in zip(pictures, ("front", "near"), strict=True):
        assert encoded.startswith("data:image/png;base64,")
        image = Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1])))
        np.testing.assert_array_equal(np.array(image), sample.images[camera])
    text = json.loads(content[-1]["text"])
    assert text == {"instruction": sample.instruction, "state": sample.state, "context": {"step": 0}}
    assert sample.context["task_id"] == "task_001"
    assert backend.last_trace["raw_response"] == server["reply"]
    assert backend.last_trace["latency_s"] > 0
    assert backend.metadata["robot_finetuned"] is None
    assert backend.is_trained_vlm is True


def model_metadata(**updates):
    value = {
        "model": "microduck-qlora",
        "robot_finetuned": True,
        "adapter": "/models/final_adapter",
        "adapter_sha256": "a" * 64,
        "base": "/models/base",
        "prompt_version": "microduck-skills-5",
        "generation_mode": "greedy_unconstrained",
    }
    value.update(updates)
    return value


def test_only_configured_original_service_has_pretrained_default_identity():
    assert LocalVLMBackend().metadata["robot_finetuned"] is False
    backend = LocalVLMBackend("http://127.0.0.1:8082/v1/chat/completions", model="microduck-qlora")
    assert backend.metadata["robot_finetuned"] is None
    assert backend.name == "local_vlm"


@pytest.mark.parametrize("wrapped", [False, True])
def test_health_reports_finetuned_identity_before_any_inference(server, wrapped):
    identity = model_metadata()
    server["health_metadata"] = {"microduck_model_metadata": identity} if wrapped else identity
    backend = LocalVLMBackend(server["endpoint"], model="microduck-qlora")
    assert backend.refresh_model_metadata() == identity
    assert server["health_requests"] == ["/health"]
    assert server["requests"] == []
    assert backend.metadata["robot_finetuned"] is True
    assert backend.metadata["server_metadata"] == identity
    assert backend.name == "local_finetuned_vlm"


def test_completion_records_server_identity_and_updates_episode_metadata(server):
    identity = model_metadata()
    server["response_metadata"] = {"microduck_model_metadata": identity}
    backend = LocalVLMBackend(server["endpoint"], model="microduck-qlora")
    metadata_reference = backend.metadata
    backend.decide(request())
    assert backend.last_trace["server_metadata"] == identity
    assert metadata_reference["robot_finetuned"] is True
    assert metadata_reference["server_metadata"] == identity
    assert metadata_reference["backend"] == "local_finetuned_vlm"


@pytest.mark.parametrize("vision_provider", ["CUDAExecutionProvider", "CPUExecutionProvider"])
def test_onnx_identity_survives_health_and_action_trace(server, vision_provider):
    identity = model_metadata(
        model="microduck-onnx",
        inference_format="onnx",
        decoder="decoder_int4.onnx",
        decoder_quantization="int4_weight_only",
        execution_provider="CUDAExecutionProvider",
        vision_execution_provider=vision_provider,
        kv_cache_device="cuda",
    )
    server["health_metadata"] = identity
    server["response_metadata"] = {"microduck_model_metadata": identity}
    backend = LocalVLMBackend(server["endpoint"], model="microduck-onnx")
    assert backend.refresh_model_metadata() == identity
    backend.decide(request())
    assert backend.last_trace["server_metadata"] == identity


@pytest.mark.parametrize(
    "updates",
    [
        {"robot_finetuned": "true"},
        {"model": "another-model"},
        {"adapter": "relative/adapter"},
        {"base": 123},
        {"adapter_sha256": "invalid"},
        {"generation_mode": {}},
        {"prompt_version": ""},
    ],
)
def test_invalid_server_provenance_fails_before_returning_an_action(server, updates):
    server["response_metadata"] = {"microduck_model_metadata": model_metadata(**updates)}
    backend = LocalVLMBackend(server["endpoint"], model="microduck-qlora")
    with pytest.raises(ValueError):
        backend.decide(request())
    assert backend.metadata["robot_finetuned"] is None
    assert "error" in backend.last_trace


def test_task_identifier_changes_cannot_change_model_input(server):
    backend = LocalVLMBackend(server["endpoint"])
    first = request(context={"step": 0, "task_id": "task_002"})
    second = request(context={"step": 0, "task_id": "task_987"})
    backend.decide(first)
    backend.decide(second)
    assert server["requests"][0]["payload"] == server["requests"][1]["payload"]
    assert backend.last_trace["input_context"] == {"step": 0}
    assert first.context["task_id"] == "task_002"
    assert second.context["task_id"] == "task_987"


def test_task_identifier_is_validated_before_being_removed(server):
    with pytest.raises(ValueError, match="task_id must be a string"):
        LocalVLMBackend(server["endpoint"]).decide(request(context={"task_id": {"hidden": "state"}}))
    assert server["requests"] == []


@pytest.mark.parametrize("leak", ["qpos", "ball_xyz", "truth", "initial_layout_world_xy_m", "expected_target_color"])
def test_simulation_truth_fields_cannot_be_sent(server, leak):
    backend = LocalVLMBackend(server["endpoint"])
    sample = request()
    state = {**sample.state, leak: [0.2, 0.3, 0]}
    with pytest.raises(ValueError, match="state accepts only"):
        backend.decide(request(state=state))
    with pytest.raises(ValueError, match="context accepts only"):
        backend.decide(request(context={leak: [0.2, 0.3, 0]}))
    assert server["requests"] == []


def test_nested_detection_truth_and_history_fields_are_rejected(server):
    backend = LocalVLMBackend(server["endpoint"])
    state = request().state
    state["detections_rgbd"] = [{"color": "red", "center_body_m": [0.2, 0, 0], "world_xyz": [1, 2, 3]}]
    with pytest.raises(ValueError, match="RGB-D detection"):
        backend.decide(request(state=state))
    with pytest.raises(ValueError, match="history record"):
        backend.decide(request(context={"history": [{"action": None, "feedback": "", "oracle": {}}]}))
    assert server["requests"] == []


def test_rgbd_measurements_are_optional_and_explicit(server):
    sample = request()
    sample.state["detections_rgbd"] = [{"color": "red", "center_body_m": [0.2, 0.03, -0.1], "camera": "near"}]
    sample.context["previous_action"] = proposal()
    sample.context["previous_feedback"] = "Previous pulse completed."
    sample.context["history"] = [{"action": None, "feedback": "Initial observation"}]
    backend = LocalVLMBackend(server["endpoint"])
    backend.decide(sample)
    sent = server["requests"][0]["payload"]
    assert "auxiliary color/depth measurements" in sent["messages"][0]["content"]
    assert backend.last_trace["input_state"] == sample.state
    sample.state["detections_rgbd"][0]["color"] = "blue"
    assert backend.last_trace["input_state"]["detections_rgbd"][0]["color"] == "red"


@pytest.mark.parametrize(
    "images",
    [
        {},
        {"external": np.zeros((2, 2, 3), np.uint8)},
        {"front": np.zeros((2, 2, 3), np.float32)},
        {"front": np.zeros((2, 2), np.uint8)},
        {"front": np.zeros((0, 2, 3), np.uint8)},
    ],
)
def test_invalid_or_external_camera_images_are_not_sent(server, images):
    backend = LocalVLMBackend(server["endpoint"])
    with pytest.raises(ValueError):
        backend.decide(request(images=images))
    assert server["requests"] == []


def test_history_is_bounded(server):
    backend = LocalVLMBackend(server["endpoint"])
    with pytest.raises(ValueError, match="at most six"):
        backend.decide(request(context={"history": [{"action": None, "feedback": ""}] * 7}))
    assert not server["requests"]


def test_repeated_decisions_send_new_images_and_preserve_model_choice(server):
    backend = LocalVLMBackend(server["endpoint"])
    backend.decide(request())
    server["reply"] = json.dumps(proposal(skill="move_backward"))
    sample = request()
    sample.images["front"][:] = [0, 255, 0]
    sample.state["sim_time_s"] = 5
    result = backend.decide(sample)
    assert result.skill == "move_backward"
    messages = [item["payload"]["messages"][1]["content"] for item in server["requests"]]
    assert messages[0][1] != messages[1][1]
    assert backend.last_trace["input_state"]["sim_time_s"] == 5


def test_bad_model_json_is_logged_and_does_not_fall_back(server):
    server["reply"] = "not JSON"
    backend = LocalVLMBackend(server["endpoint"])
    with pytest.raises(ValueError):
        backend.decide(request())
    assert backend.last_trace["raw_response"] == "not JSON"
    assert "error" in backend.last_trace
    assert len(server["requests"]) == 1


def test_truncated_response_is_rejected_even_if_partial_content_parses(server):
    server["finish_reason"] = "length"
    backend = LocalVLMBackend(server["endpoint"])
    with pytest.raises(ValueError, match="truncated"):
        backend.decide(request())


def test_redirect_cannot_send_camera_images_to_another_host(server):
    server["status"] = 302
    backend = LocalVLMBackend(server["endpoint"])
    with pytest.raises(ValueError, match="redirects are refused"):
        backend.decide(request())
    assert len(server["requests"]) == 1


def test_http_failure_propagates_without_retry_or_fallback(server):
    server["status"] = 503
    backend = LocalVLMBackend(server["endpoint"])
    with pytest.raises(RuntimeError, match="Local VLM request failed"):
        backend.decide(request())
    assert len(server["requests"]) == 1


def test_request_timeout_propagates_and_records_latency(server):
    server["delay"] = 0.2
    backend = LocalVLMBackend(server["endpoint"], timeout=0.03)
    with pytest.raises(RuntimeError, match="Local VLM request failed"):
        backend.decide(request())
    assert 0.02 < backend.last_trace["latency_s"] < 1


def test_trace_is_reset_before_invalid_subsequent_request(server):
    backend = LocalVLMBackend(server["endpoint"])
    backend.decide(request())
    previous = copy.deepcopy(backend.last_trace)
    with pytest.raises(ValueError):
        backend.decide(request(state={"qpos": []}))
    assert backend.last_trace["raw_response"] is None
    assert previous["raw_response"] is not None


def test_server_model_and_usage_are_preserved_without_replacing_requested_alias(server):
    server["response_metadata"] = {
        "model": "Qwen-local-quantized",
        "usage": {"prompt_tokens": 1320, "completion_tokens": 31, "total_tokens": 1351},
    }
    backend = LocalVLMBackend(server["endpoint"])
    backend.decide(request())
    assert backend.last_trace["model"] == "microduck-vlm"
    assert backend.last_trace["server_model"] == "Qwen-local-quantized"
    assert backend.last_trace["usage"] == server["response_metadata"]["usage"]
    server["response_metadata"] = {}
    backend.decide(request())
    assert "usage" not in backend.last_trace and "server_model" not in backend.last_trace
