"""Exercise feature alignment and real ORT cache execution without large weights."""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.microduck_tasks.onnx_vlm import (
    BoundDecoder,
    OnnxEngine,
    causal_mask,
    run_prefill,
    run_vision,
    scatter_images,
)
from experiments.microduck_tasks.serve_adapter import sha256
from experiments.microduck_tasks.serve_onnx import inspect_export
from experiments.microduck_tasks.skill_vlm import PROMPT_VERSION, SYSTEM_PROMPT


def test_causal_mask_includes_entire_past_but_not_future():
    mask = causal_mask(3, 2)
    assert mask.shape == (1, 1, 3, 5)
    assert mask.dtype == np.float16
    assert np.array_equal(mask[0, 0] == 0, [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    assert np.all(mask[mask != 0] == -65504)


def test_scatter_keeps_two_cameras_and_deepstack_in_token_order():
    ids = np.array([[0, 9, 9, 0, 9, 9]], dtype=np.int64)
    outputs = [[np.full((2, 2), 100 * image + stream, np.float16) for stream in range(4)] for image in range(2)]
    mask, streams = scatter_images(ids, 9, outputs, 2)
    assert mask.shape == (1, 6, 1)
    assert streams[2][0, :, 0].tolist() == [0, 2, 2, 0, 102, 102]
    with pytest.raises(ValueError, match="token count"):
        scatter_images(ids[:, :3], 9, outputs, 2)
    outputs[1][3][0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid features"):
        scatter_images(ids, 9, outputs, 2)


def tiny_session(path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    helper, tensor, numpy_helper = onnx.helper, onnx.TensorProto, onnx.numpy_helper
    inputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, dtype, shape in (
            ("input_ids", tensor.INT64, [1, "S"]),
            ("image_mask", tensor.BOOL, [1, "S", 1]),
            ("image_embeds", tensor.FLOAT16, [1, "S", 2]),
            ("position_ids", tensor.INT64, [3, 1, "S"]),
            ("attention_mask", tensor.FLOAT16, [1, 1, "S", "T"]),
        )
    ]
    inputs += [helper.make_tensor_value_info(f"deepstack_{i}", tensor.FLOAT16, [1, "S", 2]) for i in range(3)]
    inputs += [
        helper.make_tensor_value_info(f"past_{kind}_{i}", tensor.FLOAT16, [1, 1, "P", 2])
        for i in range(2)
        for kind in ("key", "value")
    ]
    initializers = [
        numpy_helper.from_array(np.array([1], np.int64), "axis"),
        numpy_helper.from_array(np.array([1, 2], np.int64), "sum_axes"),
    ]
    nodes = [
        helper.make_node("Unsqueeze", ["image_embeds", "axis"], ["new_cache"]),
    ]
    outputs = [helper.make_tensor_value_info("logits", tensor.FLOAT16, [1, 2])]
    for layer in range(2):
        for kind in ("key", "value"):
            name = f"present_{kind}_{layer}"
            nodes.append(helper.make_node("Concat", [f"past_{kind}_{layer}", "new_cache"], [name], axis=2))
            outputs.append(helper.make_tensor_value_info(name, tensor.FLOAT16, [1, 1, "T", 2]))
    nodes.append(helper.make_node("ReduceSum", ["present_key_0", "sum_axes"], ["logits"], keepdims=0))
    graph = helper.make_graph(nodes, "tiny_cached_decoder", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def test_real_ort_dynamic_prefill_then_cached_sequences(tmp_path):
    decoder = BoundDecoder(tiny_session(tmp_path / "decoder.onnx"), "cpu", 2, 1, 2)
    cache = decoder.empty_cache()
    for sequence, past in ((3, 0), (1, 3), (2, 4)):
        data = {
            "input_ids": np.zeros((1, sequence), np.int64),
            "image_mask": np.zeros((1, sequence, 1), bool),
            "image_embeds": np.tile(np.array([[[1, 2]]], np.float16), (1, sequence, 1)),
            "position_ids": np.zeros((3, 1, sequence), np.int64),
            "attention_mask": causal_mask(sequence, past),
            **{f"deepstack_{i}": np.zeros((1, sequence, 2), np.float16) for i in range(3)},
        }
        token, cache = decoder.run(data, cache)
        assert token == 1
        assert all(item.shape() == [1, 1, sequence + past, 2] for item in cache)
        assert all(np.all(item.numpy()[..., 0] == 1) and np.all(item.numpy()[..., 1] == 2) for item in cache)


@pytest.mark.parametrize("conservative", [False, True])
def test_chunked_prefill_matches_whole_prompt_and_slices_all_features(tmp_path, conservative):
    decoder = BoundDecoder(tiny_session(tmp_path / "decoder.onnx"), "cpu", 2, 1, 2)
    length = 7
    data = {
        "input_ids": np.arange(length, dtype=np.int64)[None],
        "image_mask": np.array([[[False], [True], [True], [False], [True], [True], [False]]]),
        "image_embeds": np.array([[[8, 0], [0, 1], [0, 2], [0, 1], [0, 2], [0, 3], [0, 2]]], np.float16),
        "position_ids": np.arange(3 * length, dtype=np.int64).reshape(3, 1, length),
        **{f"deepstack_{i}": np.full((1, length, 2), i, np.float16) for i in range(3)},
    }
    whole_token, whole_cache = run_prefill(decoder, data, decoder.empty_cache(), length, float("inf"))
    calls = []

    class Capture:
        def run(self, inputs, cache, *, shrink_arena=False):
            calls.append((inputs, shrink_arena))
            return decoder.run(inputs, cache, shrink_arena=shrink_arena)

    token, cache = run_prefill(
        Capture(), data, decoder.empty_cache(), 2, float("inf"), memory_conservative=conservative
    )
    assert token == whole_token == 1
    assert all(np.array_equal(left.numpy(), right.numpy()) for left, right in zip(cache, whole_cache, strict=True))
    assert [item[1] for item in calls] == ([True] * 4 if conservative else [False, False, False, True])
    assert [item[0]["attention_mask"].shape for item in calls] == [
        (1, 1, 2, 2),
        (1, 1, 2, 4),
        (1, 1, 2, 6),
        (1, 1, 1, 7),
    ]
    for name, value in data.items():
        axis = 2 if name == "position_ids" else 1
        assert np.array_equal(np.concatenate([item[0][name] for item in calls], axis=axis), value)


def test_vision_reclaims_workspace_after_each_camera():
    pytest.importorskip("onnxruntime")
    calls = []

    class Vision:
        def run(self, names, inputs, options):
            calls.append(
                (inputs["pixel_values"].copy(), options.get_run_config_entry("memory.enable_memory_arena_shrinkage"))
            )
            return inputs["pixel_values"].shape

    pixels = np.arange(640, dtype=np.float16)[:, None]
    assert run_vision(Vision(), ["image_embeds"], pixels, "cuda") == [(320, 1), (320, 1)]
    assert np.array_equal(np.concatenate([call[0] for call in calls]), pixels)
    assert [call[1] for call in calls] == ["gpu:0", "gpu:0"]


@pytest.fixture
def fake_engine(monkeypatch):
    engine = OnnxEngine.__new__(OnnxEngine)
    engine.timeout, engine.metadata, engine.hidden_size, engine.eos = 90, {"model": "microduck-onnx"}, 2, {99}
    engine.prefill_chunk_size = 128
    engine.memory_conservative = False
    engine.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda values, skip_special_tokens: str(values))
    )
    engine.prepare = lambda messages, images, max_tokens: ({"input_ids": np.array([[1, 2, 3]], dtype=np.int64)}, 3, -1)
    calls = []

    class Decoder:
        def empty_cache(self):
            return []

        def run(self, inputs, cache, *, shrink_arena=False):
            calls.append(inputs)
            return [10, 99][len(calls) - 1], [len(calls)]

    engine.decoder = Decoder()
    monkeypatch.setattr("experiments.microduck_tasks.onnx_vlm.decode_request", lambda raw, alias: ([], [], 3))
    return engine, calls


def test_generation_preserves_rope_delta_and_clears_image_features(fake_engine):
    engine, calls = fake_engine
    result = engine.generate(b"{}")
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert calls[1]["input_ids"].tolist() == [[10]]
    assert calls[1]["position_ids"].tolist() == [[[2]], [[2]], [[2]]]
    assert not calls[1]["image_mask"].any()
    assert all(not calls[1][f"deepstack_{index}"].any() for index in range(3))
    assert calls[1]["attention_mask"].shape == (1, 1, 1, 4)


def test_generation_deadline_never_returns_partial_action(fake_engine):
    engine, calls = fake_engine
    engine.timeout = -1
    with pytest.raises(TimeoutError, match="deadline"):
        engine.generate(b"{}")
    assert calls == []


def test_failed_prefill_releases_cache_even_when_exception_traceback_is_kept():
    live_cache = [object()]

    class Failing:
        count = 0

        def run(self, inputs, cache, *, shrink_arena=False):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("simulated OOM")
            return 10, live_cache

    with pytest.raises(RuntimeError, match="OOM") as failure:
        run_prefill(Failing(), {"input_ids": np.zeros((1, 4), np.int64)}, [], 2, float("inf"))
    assert failure.traceback is not None
    assert live_cache == []


@pytest.mark.parametrize("fail", [False, True])
def test_generation_releases_final_cache_on_return_and_failure(fake_engine, fail):
    engine, _ = fake_engine
    live_cache = [object()]

    class Decoder:
        count = 0

        def empty_cache(self):
            return []

        def run(self, inputs, cache, *, shrink_arena=False):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("simulated OOM")
            return (10 if fail else 99), live_cache

    engine.decoder = Decoder()
    if fail:
        with pytest.raises(RuntimeError, match="OOM"):
            engine.generate(b"{}")
    else:
        assert engine.generate(b"{}")["choices"][0]["finish_reason"] == "stop"
    assert live_cache == []


def test_decoder_failure_clears_partial_io_bindings():
    class Binding:
        inputs_cleared, outputs_cleared = False, False

        def bind_cpu_input(self, name, value):
            pass

        def bind_ortvalue_input(self, name, value):
            pass

        def bind_output(self, name, device, device_id=0):
            pass

        def clear_binding_inputs(self):
            self.inputs_cleared = True

        def clear_binding_outputs(self):
            self.outputs_cleared = True

    binding = Binding()

    def fail_run(binding, options):
        raise RuntimeError("simulated OOM")

    decoder = BoundDecoder.__new__(BoundDecoder)
    decoder.device = "cpu"
    decoder.cache_names, decoder.present_names = ["past_key_0"], ["present_key_0"]
    decoder.session = SimpleNamespace(io_binding=lambda: binding, run_with_iobinding=fail_run)
    with pytest.raises(RuntimeError, match="OOM"):
        decoder.run({"input_ids": np.array([[1]], np.int64)}, [object()])
    assert binding.inputs_cleared and binding.outputs_cleared


def model_bundle(tmp_path, decoder_names):
    onnx = pytest.importorskip("onnx")
    merged, exported = tmp_path / "merged", tmp_path / "onnx"
    merged.mkdir()
    exported.mkdir()
    (merged / "system_prompt.txt").write_text(SYSTEM_PROMPT)
    source = {
        "robot_finetuned": True,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }
    report = {
        "status": "complete",
        "remaining_lora_layers": 0,
        "source": source,
        "outputs": {"system_prompt.txt": sha256(merged / "system_prompt.txt")},
    }
    (merged / "merge_manifest.json").write_text(json.dumps(report))
    graph = onnx.helper.make_model(onnx.helper.make_graph([], "empty", [], []))
    for name in ("vision.onnx", *decoder_names):
        onnx.save(graph, exported / name)
    manifest = {
        "merged_model": str(merged),
        "merge_manifest_sha256": sha256(merged / "merge_manifest.json"),
        "source": source,
        "files": {path.name: sha256(path) for path in exported.iterdir()},
    }
    (exported / "manifest.json").write_text(json.dumps(manifest))
    return SimpleNamespace(merged_model=merged, onnx_model=exported, decoder="auto")


@pytest.mark.parametrize(
    "selection,expected,quantization",
    [
        ("auto", "decoder_int8.onnx", "int8_weight_only"),
        ("int8", "decoder_int8.onnx", "int8_weight_only"),
        ("int4", "decoder_int4.onnx", "int4_weight_only"),
        ("fp16", "decoder.onnx", "float16"),
    ],
)
def test_export_inspection_selects_requested_decoder_with_matching_provenance(
    tmp_path, selection, expected, quantization
):
    args = model_bundle(tmp_path, ("decoder_int8.onnx", "decoder_int4.onnx", "decoder.onnx"))
    args.decoder = selection
    metadata, path = inspect_export(args)
    assert path == args.onnx_model / expected
    assert metadata["decoder"] == expected
    assert metadata["decoder_quantization"] == quantization
    assert metadata["robot_finetuned"] is True


def test_int8_only_bundle_refuses_missing_explicit_decoder_and_modified_weights(tmp_path):
    args = model_bundle(tmp_path, ("decoder_int8.onnx",))
    assert inspect_export(args)[1].name == "decoder_int8.onnx"
    args.decoder = "fp16"
    with pytest.raises(ValueError, match="cached decoder"):
        inspect_export(args)
    args.decoder = "int8"
    (args.onnx_model / "decoder_int8.onnx").write_bytes(b"modified weights")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        inspect_export(args)


def test_service_default_cli_selects_compact_bundle_and_bounded_prefill(monkeypatch):
    from experiments.microduck_tasks import serve_onnx

    captured = {}

    class BeforeLoading(Exception):
        pass

    def inspect(args):
        captured.update(vars(args))
        raise BeforeLoading

    monkeypatch.setattr(sys, "argv", ["serve_onnx"])
    monkeypatch.setattr(serve_onnx, "inspect_export", inspect)
    for name, value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("TOKENIZERS_PARALLELISM", "false")):
        monkeypatch.setenv(name, value)
    with pytest.raises(BeforeLoading):
        serve_onnx.main()
    assert captured["onnx_model"] == serve_onnx.ROOT / ".cache/microduck_onnx/int8_compact"
    assert captured["prefill_chunk_size"] == 128
    assert captured["decoder"] == "auto"
    assert captured["memory_conservative"] is None
