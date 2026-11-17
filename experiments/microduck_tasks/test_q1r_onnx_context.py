"""CPU-only checks that cache extension preserves the original export."""

import json

import numpy as np
import pytest

from .q1r_onnx_context import DEFAULT_BUILDER, _cache_array, builder_caches, extend_context, file_hash

onnx = pytest.importorskip("onnx")
pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not DEFAULT_BUILDER.is_file(), reason="Isolated official builder is not installed")


@pytest.fixture
def export(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "text_config": {
            "head_dim": 256,
            "max_position_embeddings": 4,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    caches, _ = builder_caches(config, 4)
    initializers = [onnx.numpy_helper.from_array(array, name=name) for name, array in caches.items()]
    initializers.append(onnx.numpy_helper.from_array(np.eye(2, dtype=np.float16), name="unchanged_weight"))
    values = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [4, 32]) for name in caches]
    graph = onnx.helper.make_graph([], "fixture", [], [], initializer=initializers, value_info=values)
    model = onnx.helper.make_model(graph)
    onnx.save_model(
        model,
        source / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    genai = {"model": {"context_length": 4, "decoder": {"filename": "model.onnx"}}, "search": {"max_length": 4}}
    (source / "genai_config.json").write_text(json.dumps(genai))
    (source / "tokenizer.json").write_text("{}")
    _manifest(source)
    return source, tmp_path / "extended", config_path


def _manifest(source):
    files = {item.name: {"bytes": item.stat().st_size, "sha256": file_hash(item)} for item in source.iterdir()}
    source.with_name(source.name + "_manifest.json").write_text(json.dumps({"files": files}))


def test_extension_preserves_existing_bytes_weights_and_original_config(export):
    source, output, config = export
    original_hashes = {item.name: file_hash(item) for item in source.iterdir()}
    result = extend_context(source, output, config, length=8)
    assert result["prefix_bitwise_equal"] is True
    assert result["runtime_generation_pass"] is False
    assert result["context_accuracy_validated"] is False
    assert result["unmodified_initializer_count"] == 1
    assert result["rope_cache_shapes"] == {"cos_cache": [8, 32], "sin_cache": [8, 32]}
    assert {item.name: file_hash(item) for item in source.iterdir()} == original_hashes
    assert (source / "model.onnx.data").samefile(output / "model.onnx.data")
    assert (source / "tokenizer.json").samefile(output / "tokenizer.json")
    before = onnx.load(source / "model.onnx", load_external_data=False)
    after = onnx.load(output / "model.onnx", load_external_data=False)
    old_tensors = {tensor.name: _cache_array(tensor, source) for tensor in before.graph.initializer}
    for tensor in after.graph.initializer:
        array = _cache_array(tensor, output)
        assert array[:4].tobytes() == old_tensors[tensor.name].tobytes()
    assert all(value.type.tensor_type.shape.dim[0].dim_value == 8 for value in after.graph.value_info)
    assert json.loads((source / "genai_config.json").read_text())["model"]["context_length"] == 4
    assert json.loads((output / "genai_config.json").read_text())["search"]["max_length"] == 8
    for name, entry in result["files"].items():
        assert file_hash(output / name) == entry["sha256"]


def test_prefix_mismatch_refuses_before_creating_export(export):
    source, output, config = export
    settings = json.loads(config.read_text())
    settings["text_config"]["rope_parameters"]["rope_theta"] = 1000
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="prefix is not bitwise equal"):
        extend_context(source, output, config, length=8)
    assert not output.exists()


def test_nondefault_rope_refuses_before_creating_export(export):
    source, output, config = export
    settings = json.loads(config.read_text())
    settings["text_config"]["rope_parameters"]["rope_type"] = "linear"
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="Only default RoPE"):
        extend_context(source, output, config, length=8)
    assert not output.exists()


def test_changed_source_file_fails_checksum(export):
    source, output, config = export
    (source / "tokenizer.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        extend_context(source, output, config, length=8)
    assert not output.exists()


def test_existing_destination_is_never_overwritten(export):
    source, output, config = export
    output.mkdir()
    with pytest.raises(FileExistsError, match="new output directory"):
        extend_context(source, output, config, length=8)
