"""Extend the experimental default-RoPE ONNX cache without rewriting weights.

The installed official builder's isolated CPU cache method supplies the exact
formula. Existing cache bytes must match its prefix before any output is made.
This is a capacity change, not evidence of accuracy at longer contexts.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .bootstrap import REPO

DEFAULT_BUILDER = REPO / ".cache/microduck_head/onnx_deps/onnxruntime_genai/models/builders/base.py"
CACHE_NAMES = ("cos_cache", "sin_cache")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def builder_caches(config, length, builder=DEFAULT_BUILDER):
    """Run only the installed builder's RoPE method, using CPU tensors."""
    import torch

    text = config.get("text_config", config)
    rope = text.get("rope_parameters", {})
    if rope.get("rope_type") != "default" or text.get("rope_position_scale", 1) != 1:
        raise ValueError("Only default RoPE with unit position scale is supported.")
    head = text["head_dim"]
    partial = rope["partial_rotary_factor"]
    dim = int(head * partial)
    if dim <= 0 or dim % 2 or dim > head:
        raise ValueError("Invalid partial rotary dimension.")
    tree = ast.parse(Path(builder).read_text())
    model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_rotary_embedding_caches_from_scratch"
    )
    method.decorator_list = []
    scope = {"torch": torch}
    isolated = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(isolated, str(builder), "exec"), scope)
    state = SimpleNamespace(
        head_size=head,
        context_length=length,
        original_context_length=text["max_position_embeddings"],
        rope_attrs={
            "partial_rotary_factor": partial,
            "rescale_factors": 1,
            "theta": rope["rope_theta"],
            "position_scale": 1,
            "cache_length": length,
            "t_dtype": torch.int64,
            "position_start": 0,
            "mscale": 1,
        },
    )
    with torch.inference_mode(), torch.device("cpu"):
        caches = scope[method.name](state)
        # Qwen3.5's builder forces FLOAT caches, then slices the duplicated half.
        arrays = [tensor.float()[:, : dim // 2].contiguous().numpy() for tensor in caches]
    return dict(zip(CACHE_NAMES, arrays, strict=True)), hashlib.sha256(ast.unparse(method).encode()).hexdigest()


def _cache_array(tensor, source):
    import onnx

    copied = onnx.TensorProto()
    copied.CopyFrom(tensor)
    if onnx.external_data_helper.uses_external_data(copied):
        info = {entry.key: entry.value for entry in copied.external_data}
        location = info["location"]
        if Path(location).name != location:
            raise ValueError("External tensor data must be inside the checked export directory.")
        offset, length = int(info.get("offset", 0)), int(info["length"])
        if offset < 0 or length <= 0:
            raise ValueError("Invalid external cache range.")
        # The source manifest authenticates this file. ONNX's general-purpose
        # reader rejects hard links, which are intentional for these exports.
        with (source / location).open("rb") as stream:
            stream.seek(offset)
            copied.raw_data = stream.read(length)
        if len(copied.raw_data) != length:
            raise ValueError("External cache data is truncated.")
        copied.ClearField("external_data")
        copied.data_location = onnx.TensorProto.DEFAULT
    return onnx.numpy_helper.to_array(copied)


def _checked_manifest(source):
    path = source.with_name(source.name + "_manifest.json")
    manifest = json.loads(path.read_text())
    actual_files = {item.name for item in source.iterdir() if item.is_file()}
    if actual_files != set(manifest["files"]):
        raise ValueError("Source manifest must cover every export file exactly.")
    for name, expected in manifest["files"].items():
        path = source / name
        if Path(name).name != name or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Invalid source file entry: {name}")
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
    return manifest


def extend_context(source, output, config_path, length=8192, builder=DEFAULT_BUILDER):
    """Create a separate graph plus hard links; never mutate the source export."""
    import onnx

    source, output, config_path = Path(source).resolve(), Path(output).resolve(), Path(config_path).resolve()
    if output.exists() or output.with_name(output.name + "_manifest.json").exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    source_manifest = _checked_manifest(source)
    config = json.loads(config_path.read_text())
    genai = json.loads((source / "genai_config.json").read_text())
    old_length = genai["model"]["context_length"]
    if length <= old_length:
        raise ValueError("New context must be longer than the existing context.")
    graph_name = genai["model"]["decoder"]["filename"]
    if Path(graph_name).name != graph_name:
        raise ValueError("Decoder graph must be directly inside the export directory.")
    model = onnx.load(source / graph_name, load_external_data=False)
    caches, method_hash = builder_caches(config, length, builder)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    prefix_hashes = {}
    for name in CACHE_NAMES:
        original = _cache_array(initializers[name], source)
        extended = caches[name]
        if original.shape != (old_length, extended.shape[1]) or original.dtype != extended.dtype:
            raise ValueError(f"Unexpected original {name} shape or dtype.")
        if original.tobytes() != extended[:old_length].tobytes():
            raise ValueError(f"Original {name} prefix is not bitwise equal to builder output.")
        prefix_hashes[name] = hashlib.sha256(original.tobytes()).hexdigest()
    before = {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer}
    for name in CACHE_NAMES:
        initializers[name].CopyFrom(onnx.numpy_helper.from_array(caches[name], name=name))
    for value in model.graph.value_info:
        if value.name in CACHE_NAMES:
            value.type.tensor_type.shape.dim[0].dim_value = length
    for tensor in model.graph.initializer:
        if tensor.name not in CACHE_NAMES and tensor.SerializeToString() != before[tensor.name]:
            raise AssertionError(f"Unexpected initializer change: {tensor.name}")
    genai["model"]["context_length"] = length
    genai["search"]["max_length"] = length
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        for item in source.iterdir():
            if item.name not in (graph_name, "genai_config.json"):
                os.link(item, temporary / item.name)
        # Serialize directly: ONNX's external-data save helper could rewrite the
        # hard-linked weight file. Only the two new small caches are embedded.
        (temporary / graph_name).write_bytes(model.SerializeToString())
        (temporary / "genai_config.json").write_text(json.dumps(genai, indent=2) + "\n")
        files = {
            item.name: {"bytes": item.stat().st_size, "sha256": file_hash(item)} for item in sorted(temporary.iterdir())
        }
        manifest = {
            "precision": source_manifest.get("precision"),
            "runtime_generation_pass": False,
            "runtime_evidence": None,
            "context_length": length,
            "rope_cache_length": length,
            "rope_cache_shapes": {name: list(array.shape) for name, array in caches.items()},
            "source_context_length": old_length,
            "prefix_bitwise_equal": True,
            "prefix_sha256": prefix_hashes,
            "unmodified_initializer_count": len(before) - 2,
            "external_weights_hard_linked": True,
            "source_model": str(source),
            "source_config": str(config_path),
            "source_config_sha256": file_hash(config_path),
            "builder_source": str(Path(builder).resolve()),
            "builder_source_sha256": file_hash(builder),
            "builder_method_sha256": method_hash,
            "context_accuracy_validated": False,
            "generic_onnx_checker_pass": False,
            "generic_onnx_checker_note": source_manifest.get("generic_onnx_checker_note"),
            "files": files,
        }
        os.rename(temporary, output)
        output.with_name(output.name + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    args = parser.parse_args(argv)
    result = extend_context(args.source, args.output, args.config, args.context_length, args.builder)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
