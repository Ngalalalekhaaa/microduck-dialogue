"""Merge the verified local simulation LoRA into its original CPU fp16 base.

This creates ordinary Transformers weights, the input to a separate ONNX export.
It never dequantizes the NF4 training base and never touches an existing output.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.microduck_tasks.serve_adapter import inspect_files, sha256
from experiments.microduck_tasks.train_vlm import encode, read_records

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def compare_logits(before, after):
    import torch
    import torch.nn.functional as functional

    before, after = before.float(), after.float()
    if before.shape != after.shape or not torch.isfinite(after).all():
        raise RuntimeError("Merged model produced invalid logits; inspect the partial output before retrying.")
    difference = after - before
    # Millions of float32 entries can accumulate enough norm/dot error for a
    # cosine slightly greater than one. Measure this scalar in float64 instead.
    cosine = functional.cosine_similarity(before.flatten().double(), after.flatten().double(), dim=0)
    metrics = {
        "shape": list(before.shape),
        "max_absolute_error": float(difference.abs().max()),
        "mean_absolute_error": float(difference.abs().mean()),
        "root_mean_square_error": float(difference.square().mean().sqrt()),
        "cosine_similarity": float(cosine.clamp(-1.0, 1.0)),
        "cosine_method": "float64 dot/norm accumulation, clamped to [-1, 1] for floating-point roundoff",
        "argmax_agreement": float((before.argmax(-1) == after.argmax(-1)).float().mean()),
        "thresholds": {"max_absolute_error": 0.5, "mean_absolute_error": 0.05, "cosine_similarity": 0.9999},
    }
    metrics["passed"] = (
        metrics["max_absolute_error"] <= 0.5
        and metrics["mean_absolute_error"] <= 0.05
        and metrics["cosine_similarity"] >= 0.9999
    )
    if not metrics["passed"]:
        raise RuntimeError(f"Merge logits comparison exceeded fp16 tolerance: {json.dumps(metrics)}")
    return metrics


def merge_layers(model):
    import torch
    from peft.tuners.lora.layer import LoraLayer

    checks = []
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    if len(layers) != 28 * 7 or any(".language_model.layers." not in name for name, _ in layers):
        raise RuntimeError("Expected exactly 28 x 7 language LoRA layers; inspect the adapter configuration.")
    for number, (name, module) in enumerate(layers, 1):
        if list(module.lora_A) != ["default"] or module.fan_in_fan_out or module.lora_bias["default"]:
            raise RuntimeError(f"Unsupported adapter variant in {name}; use the pinned vanilla LoRA adapter.")
        layer = module.get_base_layer()
        dtype = layer.weight.dtype
        inputs = torch.randn(8, layer.in_features, generator=generator, dtype=torch.float32).to(dtype)
        reference = module(inputs).float()
        before_hash = tensor_digest(layer.weight)
        # PEFT's safe merge adds in the base dtype. Promote ONE layer at a time
        # so both BA and its addition use fp32, without doubling the whole base.
        layer.weight.data = layer.weight.data.float()
        expected = (layer.weight + module.get_delta_weight("default").float()).to(dtype)
        expected_hash = tensor_digest(expected)
        module.merge(safe_merge=True)
        layer.weight.data = layer.weight.data.to(dtype)
        if not torch.equal(layer.weight, expected):
            raise RuntimeError(f"Merged weight differs from W + scale*B@A in {name}.")
        actual = module(inputs).float()
        relative_rmse = float(
            (actual - reference).square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-8)
        )
        if not torch.isfinite(actual).all() or relative_rmse > 0.005:
            raise RuntimeError(f"Layer {name} exceeds numerical merge tolerance: relative RMSE={relative_rmse}")
        checks.append(
            {
                "module": name,
                "shape": list(layer.weight.shape),
                "rank": module.r["default"],
                "alpha": module.lora_alpha["default"],
                "scaling": module.scaling["default"],
                "output_dtype": str(dtype),
                "before_sha256": before_hash,
                "merged_sha256": tensor_digest(layer.weight),
                "expected_fp32_add_then_fp16_sha256": expected_hash,
                "weight_formula_exact": True,
                "random_input_relative_rmse": relative_rmse,
            }
        )
        del inputs, reference, expected, actual
        if number % 28 == 0:
            print(f"Merged and checked {number}/{len(layers)} layers", flush=True)
    # Layers already merged above; unload removes only adapter wrappers.
    merged = model.unload()
    if any(isinstance(module, LoraLayer) for module in merged.modules()):
        raise RuntimeError("LoRA wrappers remain after unloading; do not export this model.")
    if any("lora_" in name for name in merged.state_dict()):
        raise RuntimeError("LoRA tensors remain after unloading; do not export this model.")
    return merged, checks


def run(args, temporary):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    started = time.monotonic()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    config, source = inspect_files(args.model, args.adapter)
    print("Verified source base, adapter and training prompt hashes", flush=True)
    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
    if adapter_config.get("use_dora") or adapter_config.get("use_rslora") or adapter_config.get("modules_to_save"):
        raise ValueError("Use the pinned vanilla LoRA adapter; this exporter does not handle this adapter variant.")
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, min_pixels=65536, max_pixels=int(config["max_pixels"])
    )
    print("Loading the original non-quantized base on CPU as fp16", flush=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float16, device_map={"": "cpu"}, attn_implementation="sdpa"
    )
    if getattr(base, "is_quantized", False) or hasattr(base.config, "quantization_config"):
        raise RuntimeError("Base is quantized; choose the original full-precision training source.")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("Merge must remain entirely on CPU.")
    records = read_records(args.validation)
    row = records[args.sample_index]
    batch, boundary = encode(row, args.validation.parent, processor, int(config["max_length"]))
    positions = torch.arange(boundary - 1, batch["input_ids"].shape[1] - 1, device="cpu")
    for name, value in batch.items():
        if value.is_floating_point():
            batch[name] = value.to(torch.float16)
    print(
        f"Reference dual-camera forward: {batch['input_ids'].shape[1]} tokens, {len(positions)} supervised positions",
        flush=True,
    )
    with torch.inference_mode():
        reference = model(**batch, logits_to_keep=positions, use_cache=False).logits.float().clone()
        if not torch.isfinite(reference).all():
            raise RuntimeError("The unmerged fp16 reference is not finite; investigate before exporting.")
        merged, layers = merge_layers(model)
        merged.eval()
        print("Merged dual-camera forward for numerical equivalence", flush=True)
        actual = merged(**batch, logits_to_keep=positions, use_cache=False).logits.float()
        logits_check = compare_logits(reference, actual)
    print("Logits comparison: " + json.dumps(logits_check), flush=True)
    del reference, actual, batch, records
    merged.config.use_cache = True
    merged.config.text_config.use_cache = True
    merged.config.dtype = torch.float16
    merged.config.text_config.dtype = torch.float16
    merged.config.vision_config.dtype = torch.float16
    print("Saving one ordinary fp16 safetensors checkpoint and processor", flush=True)
    merged.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(temporary)
    # The prompt remains an application contract; it is not baked into weights.
    (temporary / "system_prompt.txt").write_text((args.adapter.parent / "system_prompt.txt").read_text())
    output_hashes = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(temporary.iterdir())
        if path.is_file()
    }
    _, final_source = inspect_files(args.model, args.adapter)
    if final_source != source:
        raise RuntimeError("Source weights or prompt changed while merging; discard this partial build.")
    report = {
        "format": "microduck-cpu-lora-merge-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "output": str(args.output),
        "source": source,
        "source_config_sha256": sha256(args.model / "config.json"),
        "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
        "adapter_config": adapter_config,
        "source_dtype": "bfloat16 original safetensors, loaded without quantization",
        "base_compute_dtype": "float16",
        "adapter_compute_dtype": "float32 (PEFT automatic promotion)",
        "merge_accumulation_dtype": "float32 per layer; one final float16 cast",
        "output_dtype": "float16",
        "device": "cpu",
        "onnx_exported": False,
        "remaining_lora_layers": 0,
        "quantization": None,
        "warning": (
            "Numerical check compares non-quantized fp16 base+LoRA to merged fp16, not the NF4 service. "
            "ONNX and task evaluation are separate steps."
        ),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors")
        },
        "merge_script_sha256": sha256(Path(__file__)),
        "input_validation": {
            "dataset": str(args.validation),
            "dataset_sha256": sha256(args.validation),
            "sample_index": args.sample_index,
            "sample_id": row["sample_id"],
            "instruction": row["instruction"],
            "images": {
                camera: {"path": str(args.validation.parent / path), "sha256": sha256(args.validation.parent / path)}
                for camera, path in row["images"].items()
            },
            "teacher_forced_assistant_positions": len(positions),
            "logits": logits_check,
        },
        "layer_checks": layers,
        "outputs": output_hashes,
    }
    write_json(temporary / "merge_manifest.json", report)
    temporary.rename(args.output)
    print(f"Merged model ready: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/microduck_train/model")
    parser.add_argument(
        "--adapter", type=Path, default=ROOT / "experiments/microduck_tasks/outputs/qlora_20261114/final_adapter"
    )
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/microduck_onnx/merged_model")
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "experiments/microduck_tasks/outputs/training_dataset_final_20261114/val.jsonl",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    for field in ("model", "adapter", "output", "validation"):
        setattr(args, field, getattr(args, field).resolve())
    if not 1 <= args.cpu_threads <= 16 or args.sample_index < 0:
        parser.error("Use cpu-threads 1..16 and a nonnegative sample-index.")
    temporary = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or temporary.exists():
        parser.error("Output or its .partial directory exists; inspect it and choose a new output path.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 7 * 1024**3:
        parser.error("Keep at least 7 GiB free before creating the merged checkpoint.")
    if not args.validation.is_file():
        parser.error("Validation JSONL is missing; supply --validation for the reviewed local dataset.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    temporary.mkdir()
    try:
        run(args, temporary)
    except BaseException as error:
        write_json(
            temporary / "merge_failure.json", {"status": "failed", "error": str(error), "type": type(error).__name__}
        )
        raise


if __name__ == "__main__":
    main()
