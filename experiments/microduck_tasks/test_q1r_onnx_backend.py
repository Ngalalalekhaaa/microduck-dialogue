"""CPU-only checks for native ONNX chat prompts, validation and control bypasses."""

import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import q1r_onnx_backend, q1r_onnx_runtime
from .q1r_onnx_benchmark import digest
from .toy_fixtures import active, kick, perform, request


def make_backend(monkeypatch, value):
    calls = []

    class Generator:
        def __init__(self, model_dir, tokenizer_dir):
            self.metadata = {"device_type": "CUDA"}
            self.last_generation = None

        def generate(self, state, images):
            calls.append(state)
            self.last_generation = {"generated_ids": [1, 248046], "sequence_validation": "exact"}
            if isinstance(value, Exception):
                raise value
            return json.dumps(value, ensure_ascii=False), {"seconds": 0.1}

    monkeypatch.setattr(q1r_onnx_backend, "Q1ROnnxGenerator", Generator)
    return q1r_onnx_backend.ToyQ1ROnnxBackend("unused"), calls


def test_backend_import_does_not_load_torch_or_transformers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from experiments.microduck_tasks.q1r_onnx_backend import ToyQ1ROnnxBackend; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
            "assert ToyQ1ROnnxBackend.requires_lockstep",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_codec_matches_all_ten_frozen_hf_prompts_and_tokens():
    pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    folder = Path(__file__).parent / "outputs/q1r_onnx_20261117"
    if not (folder / "encoded_inputs.jsonl").is_file():
        pytest.skip("The local frozen model benchmark is not present.")
    rows = {row["id"]: row for row in map(json.loads, (folder / "benchmark_inputs.jsonl").read_text().splitlines())}
    encoded = list(map(json.loads, (folder / "encoded_inputs.jsonl").read_text().splitlines()))
    codec = q1r_onnx_runtime.Q1RTextCodec(folder / "onnx_int8_fp16_head")
    assert Path(codec.metadata["directory"]) == (folder / "onnx_int8_fp16_head").resolve()
    assert len(encoded) == 10
    for row in encoded:
        actual = codec.encode(rows[row["id"]]["request"])
        assert actual["prompt"] == row["prompt"], row["id"]
        assert actual["prefix_ids"] == row["prefix_ids"], row["id"]
    assert codec.endings == [248046, 248044]


@pytest.mark.parametrize("text,decision", [("停", "cancel"), ("暂停", "pause"), ("继续", "resume")])
def test_host_controls_keep_bypassing_onnx(monkeypatch, text, decision):
    backend, calls = make_backend(monkeypatch, RuntimeError("Inference must not run"))
    state = request(text, planned=True)
    state["execution"] = active(kick("blue"), paused=decision == "resume")
    result = backend.decide_with_images(state, {})
    assert result["decision"] == decision
    assert not calls and backend.last_trace["reply_source"] == "host_control"


def test_model_plan_and_raw_trace_use_onnx_source(monkeypatch):
    prediction = dict(decision="replace", goals=[perform("nod")], reply="我点头回应你。")
    backend, calls = make_backend(monkeypatch, prediction)
    result = backend.decide_with_images(request("点头", planned=True), {})
    assert result["goals"] == [perform("nod")] and calls
    assert backend.last_trace["raw_prediction"] == prediction
    assert backend.last_trace["goal_source"] == "q1r_onnx_int8_generation"
    assert backend.last_trace["onnx_generation"]["generated_ids"] == [1, 248046]


def test_feedback_cannot_schedule_motion_through_onnx(monkeypatch):
    backend, _ = make_backend(monkeypatch, dict(decision="replace", goals=[kick("red")], reply="继续踢球。"))
    state = request("更新执行结果", planned=True)
    state["event"]["kind"] = "feedback"
    state["execution"]["last_result"] = dict(goal=kick("blue"), status="failed", reason="lost", verified=False)
    result = backend.decide_with_images(state, {})
    assert result["decision"] == "respond" and not result["goals"]
    assert backend.last_trace["rejected"] and backend.last_trace["reply_source"] == "host_failure_notice"


def test_excess_context_is_rejected_before_inference_and_without_truncation():
    generator = object.__new__(q1r_onnx_runtime.Q1ROnnxGenerator)
    tokens = [1] * 2561
    generator.codec = SimpleNamespace(encode=lambda state: dict(prefix_ids=tokens))
    generator.maximum_input_tokens = 2560
    generator.lock = threading.Lock()
    generator.last_generation = {"old": "must disappear"}
    with pytest.raises(ValueError, match="nothing was truncated"):
        generator.generate(request("你好", planned=True), {})
    assert len(tokens) == 2561 and generator.last_generation is None


def package_fixture(tmp_path, monkeypatch, *, context=4096, rope=4096):
    folder = tmp_path / "model"
    folder.mkdir()
    for name in ("model.onnx", "model.onnx.data", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (folder / name).write_text("test fixture")
    (folder / "genai_config.json").write_text(json.dumps(dict(model=dict(type="qwen3_5_text", context_length=context))))
    files = {path.name: dict(bytes=path.stat().st_size, sha256=digest(path)) for path in folder.iterdir()}
    (tmp_path / "model_manifest.json").write_text(json.dumps(dict(runtime_generation_pass=True, files=files)))
    initializers = [
        SimpleNamespace(name=name, dims=[rope, 32], external_data=[]) for name in ("cos_cache", "sin_cache")
    ]
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(
            load=lambda path, load_external_data: SimpleNamespace(graph=SimpleNamespace(initializer=initializers))
        ),
    )
    return folder


def test_changed_weights_cannot_use_old_evaluation_manifest(tmp_path, monkeypatch):
    folder = package_fixture(tmp_path, monkeypatch)
    (folder / "model.onnx.data").write_text("changed weights")
    with pytest.raises(ValueError, match="changed since evaluation"):
        q1r_onnx_runtime.verify_package(folder)


def test_json_only_context_extension_is_rejected(tmp_path, monkeypatch):
    folder = package_fixture(tmp_path, monkeypatch, context=8192, rope=4096)
    with pytest.raises(ValueError, match="RoPE cache is shorter"):
        q1r_onnx_runtime.verify_package(folder)


def test_validated_larger_context_package_is_accepted(tmp_path, monkeypatch):
    folder = package_fixture(tmp_path, monkeypatch, context=8192, rope=8192)
    config, metadata = q1r_onnx_runtime.verify_package(folder)
    assert config["model"]["context_length"] == 8192
    assert metadata["rope_cache_lengths"] == dict(cos_cache=8192, sin_cache=8192)
