"""Deduplicate the tied FP16 embedding/LM head without changing model weights."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx

from experiments.microduck_tasks.bundle_onnx import graph_files
from experiments.microduck_tasks.serve_adapter import sha256


def external_array(directory, tensor):
    entries = {entry.key: entry.value for entry in tensor.external_data}
    if tensor.data_type != onnx.TensorProto.FLOAT16 or "location" not in entries:
        raise ValueError("Expected the original external FP16 decoder weights.")
    path = directory / entries["location"]
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("External weights must be inside the source directory.")
    return np.memmap(path, dtype=np.uint16, mode="r", offset=int(entries.get("offset", 0)), shape=tuple(tensor.dims))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/microduck_onnx/model"))
    parser.add_argument("--output", type=Path, default=Path(".cache/microduck_onnx/fp16_compact"))
    parser.add_argument(
        "--decoder-name", choices=("decoder.onnx", "decoder_int4.onnx", "decoder_int8.onnx"), default="decoder.onnx"
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new compact directory.")
    model = onnx.load(str(source / args.decoder_name), load_external_data=False)
    heads = [node for node in model.graph.node if node.name == "/lm_head/MatMul" and node.op_type == "MatMul"]
    embeds = [node for node in model.graph.node if node.name == "/embed_tokens/Gather" and node.op_type == "Gather"]
    if len(heads) != 1 or len(embeds) != 1:
        raise ValueError("Expected the original named embedding and LM-head nodes.")
    head, embed = heads[0], embeds[0]
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    duplicate_name, shared_name = head.input[1], embed.input[0]
    duplicate, shared = initializers[duplicate_name], initializers[shared_name]
    if list(duplicate.dims) != list(reversed(shared.dims)):
        raise ValueError("LM-head and embedding shapes are not transposes.")
    if sum(name == duplicate_name for node in model.graph.node for name in node.input) != 1:
        raise ValueError("The duplicate LM-head tensor has another consumer; inspect the graph first.")
    head_array, embed_array = external_array(source, duplicate), external_array(source, shared)
    for row in range(0, embed_array.shape[0], 1024):
        if not np.array_equal(embed_array[row : row + 1024], head_array[:, row : row + 1024].T):
            raise ValueError(f"Embedding and transposed LM-head differ at rows starting {row}; refusing deduplication.")
    del head_array, embed_array
    removed_bytes = int(np.prod(duplicate.dims)) * 2
    head.op_type = "Gemm"
    head.name = "/lm_head/GemmSharedEmbedding"
    head.input[1] = shared_name
    head.ClearField("attribute")
    head.attribute.extend([onnx.helper.make_attribute("transB", 1)])
    retained = [tensor for tensor in model.graph.initializer if tensor.name != duplicate_name]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(retained)
    output.mkdir(parents=True)
    onnx.save_model(model, str(output / args.decoder_name))
    external_names = {
        entry.value for tensor in model.graph.initializer for entry in tensor.external_data if entry.key == "location"
    }
    for name in external_names:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        os.link(source / name, path)
    reference_graphs = ["vision.onnx"]
    if args.decoder_name != "decoder.onnx":
        reference_graphs.append("decoder.onnx")
    for name in graph_files(source, reference_graphs):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            os.link(source / name, path)
    report = {
        "status": "graph_complete_runtime_validation_pending",
        "source_graph": str(source / args.decoder_name),
        "source_graph_sha256": sha256(source / args.decoder_name),
        "output_graph_sha256": sha256(output / args.decoder_name),
        "embedding_initializer": shared_name,
        "removed_initializer": duplicate_name,
        "weight_transpose_bitwise_equal": True,
        "replacement": "Gemm(A, embedding_weight, transB=1)",
        "deduplicated_weight_bytes": removed_bytes,
        "gpu_weight_deduplication_measured": False,
        "weights_link_method": "hardlink",
    }
    (output / "compact_export.json").write_text(json.dumps(report, indent=2) + "\n")
    vision_report = json.loads((source / "vision.export.json").read_text())
    (output / "vision.export.json").write_text(json.dumps(vision_report, indent=2) + "\n")
    decoder_report = json.loads((source / "decoder.export.json").read_text())
    decoder_report["compact_transformation"] = report
    decoder_report["numerical_results_apply_to"] = (
        "original FP16 graph before quantization or tied-weight deduplication"
    )
    (output / "decoder.export.json").write_text(json.dumps(decoder_report, indent=2) + "\n")
    quant_report_name = Path(args.decoder_name).stem + "_export.json"
    if (source / quant_report_name).exists():
        quant_report = json.loads((source / quant_report_name).read_text())
        quant_report["compact_transformation"] = report
        (output / quant_report_name).write_text(json.dumps(quant_report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
