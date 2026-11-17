"""Export the merged Qwen3-VL language graph, including DeepStack and KV cache.

Only CPU export/verification is performed here. The graph uses batch size one;
sequence and past-cache lengths are dynamic. Image preprocessing, image-token
scatter, and mRoPE position indexing remain explicit runtime responsibilities.
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TensorCache:
    """The cache API needed by Qwen3VLTextAttention, with graph-visible concat."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.present = []

    def update(self, key, value, layer_idx, cache_kwargs=None):
        key = torch.cat((self.tensors[2 * layer_idx], key), dim=2)
        value = torch.cat((self.tensors[2 * layer_idx + 1], value), dim=2)
        self.present.extend((key, value))
        return key, value


class DecoderGraph(nn.Module):
    """Original HF decoder weights and operations, without HF generation state."""

    def __init__(self, text_model, lm_head):
        super().__init__()
        self.text = text_model
        self.lm_head = lm_head
        self.config = text_model.config
        self.config._attn_implementation = "eager"

    def forward(
        self,
        input_ids,
        image_mask,
        image_embeds,
        position_ids,
        attention_mask,
        deepstack_0,
        deepstack_1,
        deepstack_2,
        *past,
    ):
        hidden = torch.where(image_mask, image_embeds, self.text.embed_tokens(input_ids))
        position_embeddings = self.text.rotary_emb(hidden, position_ids)
        cache = TensorCache(past)
        deepstack = (deepstack_0, deepstack_1, deepstack_2)
        for index, layer in enumerate(self.text.layers):
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            if index < 3:
                hidden = hidden + deepstack[index]
        hidden = self.text.norm(hidden)
        logits = self.lm_head(hidden[:, -1, :])
        return (logits, *cache.present)


def io_names(config):
    inputs = [
        "input_ids",
        "image_mask",
        "image_embeds",
        "position_ids",
        "attention_mask",
        "deepstack_0",
        "deepstack_1",
        "deepstack_2",
    ]
    outputs = ["logits"]
    dynamic = {
        "input_ids": {1: "sequence"},
        "image_mask": {1: "sequence"},
        "image_embeds": {1: "sequence"},
        "position_ids": {2: "sequence"},
        "attention_mask": {2: "sequence", 3: "total_sequence"},
        "deepstack_0": {1: "sequence"},
        "deepstack_1": {1: "sequence"},
        "deepstack_2": {1: "sequence"},
    }
    for i in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            inputs.append(f"past_{kind}_{i}")
            outputs.append(f"present_{kind}_{i}")
            dynamic[inputs[-1]] = {2: "past_sequence"}
            dynamic[outputs[-1]] = {2: "total_sequence"}
    return inputs, outputs, dynamic


def sample_inputs(config, sequence, past_length, seed=27, dtype=torch.float16):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, config.vocab_size, (1, sequence), generator=generator)
    mask = torch.zeros(1, sequence, 1, dtype=torch.bool)
    if sequence > 1:
        mask[:, 1::2] = True
    image = torch.randn(1, sequence, config.hidden_size, generator=generator, dtype=dtype) * 0.02
    pos = torch.arange(past_length, past_length + sequence).reshape(1, 1, -1).expand(3, 1, -1).clone()
    # Exercise interleaved 3D RoPE rather than only identical text positions.
    if sequence > 1:
        pos[1, :, 1::2] += 2
        pos[2, :, 1::2] += 3
    causal = torch.arange(past_length + sequence).reshape(1, -1) > (past_length + torch.arange(sequence).reshape(-1, 1))
    attention = torch.zeros(sequence, past_length + sequence, dtype=dtype).masked_fill(causal, torch.finfo(dtype).min)[
        None, None
    ]
    deep = tuple(
        torch.randn(1, sequence, config.hidden_size, generator=generator, dtype=dtype) * 0.01 * mask for _ in range(3)
    )
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    cache = tuple(
        torch.randn(1, config.num_key_value_heads, past_length, head_dim, generator=generator, dtype=dtype) * 0.02
        for _ in range(config.num_hidden_layers * 2)
    )
    return (ids, mask, image, pos, attention, *deep, *cache)


def export_graph(model, output, example):
    inputs, outputs, dynamic = io_names(model.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            str(output),
            input_names=inputs,
            output_names=outputs,
            dynamic_axes=dynamic,
            opset_version=18,
            export_params=True,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )
    return {"export_seconds": time.monotonic() - started, "inputs": inputs, "outputs": outputs}


def error_metrics(actual, reference):
    actual, reference = actual.astype(np.float32), reference.astype(np.float32)
    diff = actual - reference
    norm = float(np.linalg.norm(reference.ravel()))
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "relative_l2_error": float(np.linalg.norm(diff.ravel()) / max(norm, 1e-20)),
        "cosine_similarity": float(
            np.clip(
                np.dot(actual.ravel(), reference.ravel()) / max(float(np.linalg.norm(actual.ravel())) * norm, 1e-20),
                -1,
                1,
            )
        ),
    }


def verify_graph(model, output, shapes, *, tiny=False, diagnose=False):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    names, output_names, _ = io_names(model.config)
    results = []
    comparisons = []
    all_passed = True
    for sequence, past_length in shapes:
        example = sample_inputs(model.config, sequence, past_length)
        with torch.inference_mode():
            reference = tuple(t.float().numpy() for t in model(*example))
        feed = {name: tensor.numpy() for name, tensor in zip(names, example, strict=True)}
        started = time.monotonic()
        actual = session.run(None, feed)
        seconds = time.monotonic() - started
        max_error = max(float(np.max(np.abs(a.astype(np.float32) - b))) for a, b in zip(actual, reference, strict=True))
        logits_error = float(np.max(np.abs(actual[0].astype(np.float32) - reference[0])))
        finite = all(np.isfinite(a).all() for a in actual)
        output_metrics = {}
        # Record all errors; a diagnostic run must never silently claim tolerance passes.
        for index, (a, b) in enumerate(zip(actual, reference, strict=True)):
            metric = error_metrics(a, b)
            close = np.isclose(a.astype(np.float32), b, atol=0.025 if tiny else 0.25, rtol=0.05)
            metric["outside_tolerance_elements"] = int((~close).sum())
            metric["total_elements"] = int(close.size)
            metric["tolerance_passed"] = bool(close.all())
            all_passed &= bool(close.all())
            output_metrics[output_names[index]] = metric
        result = {
            "sequence": sequence,
            "past_length": past_length,
            "finite": bool(finite),
            "max_abs_error": max_error,
            "logits_max_abs_error": logits_error,
            "pytorch_top_token": int(reference[0].argmax()),
            "onnx_top_token": int(actual[0].argmax()),
            "cpu_onnx_seconds": seconds,
            "outputs": output_metrics,
        }
        print(json.dumps({k: v for k, v in result.items() if k != "outputs"}), flush=True)
        results.append(result)
        if diagnose:
            comparisons.append((example, actual))
    del session
    gc.collect()
    if diagnose:
        # CPU ORT may promote FP16 operations. Compare independently against
        # FP32 execution of exactly the same half-rounded weights and inputs.
        model.float()
        for result, (example, actual) in zip(results, comparisons, strict=True):
            float_example = tuple(t.float() if t.is_floating_point() else t for t in example)
            with torch.inference_mode():
                fp32 = tuple(t.numpy() for t in model(*float_example))
            result["fp32_reference"] = {
                name: error_metrics(a, b) for name, a, b in zip(output_names, actual, fp32, strict=True)
            }
            result["fp32_reference_top_token"] = int(fp32[0].argmax())
    output.with_suffix(".verification.json").write_text(
        json.dumps({"cpu_tolerance_passed": all_passed, "results": results}, indent=2) + "\n"
    )
    if not all_passed and not diagnose:
        raise AssertionError(f"CPU numerical tolerance failed; see {output.with_suffix('.verification.json')}")
    return results


def verify_hf_equivalence(model):
    """Check the wrapper against HF's original DeepStack and cache implementation."""
    from transformers.cache_utils import DynamicCache

    results = []
    for sequence, past_length in [(5, 0), (1, 3), (3, 2)]:
        example = sample_inputs(model.config, sequence, past_length)
        ids, mask, image, pos, attention, *rest = example
        deep, old = rest[:3], rest[3:]
        original_cache = DynamicCache(config=model.config)
        for i in range(model.config.num_hidden_layers):
            original_cache.update(old[2 * i], old[2 * i + 1], i)
        with torch.inference_mode():
            embeds = torch.where(mask, image, model.text.embed_tokens(ids))
            reference = model.text(
                inputs_embeds=embeds,
                position_ids=pos,
                attention_mask=attention,
                past_key_values=original_cache,
                use_cache=True,
                cache_position=torch.arange(past_length, past_length + sequence),
                visual_pos_masks=mask[..., 0],
                deepstack_visual_embeds=[x[mask[..., 0]] for x in deep],
            )
            logits = model.lm_head(reference.last_hidden_state[:, -1, :])
            actual = model(*example)
        torch.testing.assert_close(actual[0], logits, rtol=0, atol=0)
        for index, layer in enumerate(original_cache.layers):
            torch.testing.assert_close(actual[1 + 2 * index], layer.keys, rtol=0, atol=0)
            torch.testing.assert_close(actual[2 + 2 * index], layer.values, rtol=0, atol=0)
        results.append({"sequence": sequence, "past_length": past_length, "exact_match": True})
    return results


def tiny_model():
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    config = Qwen3VLTextConfig(
        vocab_size=127,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
    )
    torch.manual_seed(42)
    text = Qwen3VLTextModel(config).half().eval()
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).half().eval()
    head.weight = text.embed_tokens.weight
    return DecoderGraph(text, head).eval().requires_grad_(False)


def load_merged(path):
    from transformers import Qwen3VLForConditionalGeneration

    whole = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.float16, device_map={"": "cpu"}, attn_implementation="eager"
        )
        .eval()
        .requires_grad_(False)
    )
    graph = DecoderGraph(whole.model.language_model, whole.lm_head).eval()
    del whole
    gc.collect()
    return graph


def load_quantization_source(source):
    """Load exported external weights after manifest hashes, including our hardlinks.

    ONNX 1.22 rejects any multiply-linked external file. This project deliberately
    hardlinks its own validated FP16 files into the compact package. For that case
    a complete source manifest provides the graph and weight hashes. Unmanifested
    inputs retain ONNX's normal loader and its protections.
    """
    import onnx

    from experiments.microduck_tasks.bundle_onnx import all_tensors, local_file, stable_hash

    manifest_path = source.parent / "manifest.json"
    if not manifest_path.is_file():
        return onnx.load(str(source), load_external_data=True)
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    if files.get(source.name) != stable_hash(source):
        raise ValueError("Source ONNX graph differs from its bundle manifest; restore the validated graph.")
    model = onnx.load(str(source), load_external_data=False)
    validated = set()
    for tensor in all_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        entries = {entry.key: entry.value for entry in tensor.external_data}
        name = entries.get("location")
        if not name or name not in files:
            raise ValueError("External weights are absent from the manifest; rebuild the complete source bundle.")
        path = local_file(source.parent, name)
        if name not in validated:
            if stable_hash(path) != files[name]:
                raise ValueError(f"External weights failed manifest SHA256: {name}.")
            validated.add(name)
        before = path.stat()
        offset = int(entries.get("offset", 0))
        length = int(entries.get("length", before.st_size - offset))
        if offset < 0 or length < 0 or offset + length > before.st_size:
            raise ValueError(f"External tensor exceeds verified file bounds: {name}.")
        with path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(length)
        after = path.stat()
        if len(data) != length or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"External weights changed while loading: {name}; stop writes before exporting.")
        tensor.raw_data = data
        tensor.ClearField("external_data")
        tensor.data_location = onnx.TensorProto.DEFAULT
    return model


def quantize_graph(source, output, *, bits=4, block_size=128, exclude_lm_head=False):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    if output.exists() or output.with_name(output.name + ".data").exists():
        raise FileExistsError(f"Quantized output already exists: {output}; select a new --quant-output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    graph = load_quantization_source(source)
    excluded = [node.name for node in graph.graph.node if node.op_type == "MatMul" and "lm_head" in node.name]
    if exclude_lm_head and len(excluded) != 1:
        raise ValueError("Expected one named lm_head MatMul to preserve; inspect the source ONNX graph.")
    excluded = excluded if exclude_lm_head else []
    quantizer = MatMulNBitsQuantizer(
        graph, bits=bits, block_size=block_size, is_symmetric=True, accuracy_level=4, nodes_to_exclude=excluded
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(output), use_external_data_format=True)
    return {
        "quantize_seconds": time.monotonic() - started,
        "block_size": block_size,
        "symmetric": True,
        "accuracy_level": 4,
        "format": f"MatMulNBits INT{bits}",
        "bits": bits,
        "excluded_nodes": excluded,
        "source_graph": str(source.resolve()),
        "output_graph": str(output.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=Path(".cache/microduck_onnx/merged_model"))
    parser.add_argument("--output", type=Path, default=Path(".cache/microduck_onnx/model/decoder.onnx"))
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--block-size", type=int, choices=(16, 32, 64, 128, 256), default=128)
    parser.add_argument("--exclude-lm-head", action="store_true")
    parser.add_argument("--quant-output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    report = {}
    if args.quantize_only:
        quant_output = args.quant_output or args.output.with_name(f"decoder_int{args.bits}.onnx")
        report["quantization"] = quantize_graph(
            args.output, quant_output, bits=args.bits, block_size=args.block_size, exclude_lm_head=args.exclude_lm_head
        )
        destination = quant_output.with_name(f"{quant_output.stem}_export.json")
    else:
        model = tiny_model() if args.tiny else load_merged(args.merged)
        inputs, outputs, _ = io_names(model.config)
        report.update(
            {
                "inputs": inputs,
                "outputs": outputs,
                "batch_size": 1,
                "dynamic_axes": ["sequence", "past_sequence"],
                "execution_dtype": "float16",
                "diagnostic_only": args.diagnose_only,
            }
        )
        report["original_hf_equivalence"] = verify_hf_equivalence(model)
        destination = args.output.with_suffix(".export.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if not args.verify_only and not args.diagnose_only:
            report.update(export_graph(model, args.output, sample_inputs(model.config, 3, 2)))
        report["graph_exported"] = args.output.is_file()
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if not args.skip_verify:
            report["verification"] = verify_graph(
                model, args.output, [(5, 0), (1, 3), (3, 2)], tiny=args.tiny, diagnose=args.diagnose_only
            )
            verification = json.loads(args.output.with_suffix(".verification.json").read_text())
            report["cpu_strict_tolerance_passed"] = verification["cpu_tolerance_passed"]
            report["all_outputs_finite"] = all(row["finite"] for row in report["verification"])
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed: {destination}", flush=True)


if __name__ == "__main__":
    main()
