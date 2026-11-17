"""Freeze Q1R inputs, measure NF4 inference, and export its effective text weights."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    selected = json.loads((root / "experiments/microduck_tasks/outputs/q1r_20261116/selected_model.json").read_text())
    output = args.output.resolve()
    merged = output / "merged_effective_text"
    merged.mkdir(exist_ok=args.merge_only)
    if (merged / "model.safetensors").exists():
        raise FileExistsError("Preserve the existing merged checkpoint; choose a new output directory.")

    import bitsandbytes as bnb
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from .q1r_backend import Q1RGenerator
    from .q1r_prompt import messages_for
    from .setup_skill_head import WEIGHTS

    torch.set_num_threads(4)
    started = time.monotonic()
    generator = Q1RGenerator(selected["model_dir"], selected["checkpoint"])
    model = generator.model
    processor = generator.processor
    report = dict(
        selected=selected["checkpoint"],
        adapter_sha256=generator.metadata["weights_sha256"],
        source_precision="existing NF4 double-quantized base plus selected FP32 LoRA",
        export_precision="dequantize current NF4; FP32 add B@A * scale; cast linear weights to BF16",
        note="Floating merge and subsequent ONNX quantization can change outputs; compare against measured NF4.",
        load_seconds=time.monotonic() - started,
        visual_exported=False,
        mtp_exported=False,
        layers=[],
    )
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines()]
    encoded = []
    for row in rows:
        prompt = processor.apply_chat_template(
            messages_for(row["request"]), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = processor(text=[prompt], images=None, return_tensors="pt")["input_ids"][0].tolist()
        encoded.append(dict(id=row["id"], prefix_ids=ids, prompt=prompt))
    (output / "encoded_inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in encoded))
    if not args.merge_only:
        print("Baseline warmup", flush=True)
        generator._generate(model, processor, encoded[0])
        with (output / "baseline.jsonl").open("x") as stream:
            for repeat in range(args.repeats):
                for row in encoded:
                    raw, timing = generator._generate(model, processor, row)
                    record = dict(id=row["id"], repeat=repeat, backend="q1r_nf4_pytorch", raw_text=raw, timing=timing)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(json.dumps(dict(id=row["id"], repeat=repeat, **timing)), flush=True)

    base = model.get_base_model()
    checkpoint = Path(selected["model_dir"]) / WEIGHTS
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        names = [name for name in source.keys() if name.startswith("model.language_model.")]
    tensors = {}
    merge_start = time.monotonic()
    with torch.inference_mode():
        for i, name in enumerate(names):
            parent, leaf = name.rsplit(".", 1)
            module = base.get_submodule(parent)
            if leaf == "weight" and hasattr(module, "lora_A"):
                layer = module.get_base_layer()
                weight = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state).float().cpu()
                a = module.lora_A["default"].weight.float().cpu()
                b = module.lora_B["default"].weight.float().cpu()
                weight.addmm_(b, a, beta=1, alpha=float(module.scaling["default"]))
                tensors[name] = weight.to(torch.bfloat16).contiguous()
                report["layers"].append(dict(name=name, scale=float(module.scaling["default"])))
                del weight, a, b
            else:
                tensors[name] = getattr(module, leaf).detach().cpu().clone().contiguous()
            if i % 40 == 0:
                print(f"Merged {i + 1}/{len(names)} text tensors", flush=True)
    if len(report["layers"]) != 186:
        raise ValueError(f"Expected all 186 LoRA layers, got {len(report['layers'])}.")
    save_file(tensors, merged / "model.safetensors", metadata={"format": "pt"})
    for path in Path(selected["model_dir"]).iterdir():
        if path.is_file() and path.suffix in (".json", ".jinja", ".txt") and "index" not in path.name:
            shutil.copy2(path, merged / path.name)
    config = json.loads((merged / "config.json").read_text())
    config["text_config"]["max_position_embeddings"] = 4096
    save(merged / "config.json", config)
    report.update(
        tensors=len(tensors),
        merged_bytes=(merged / "model.safetensors").stat().st_size,
        merge_seconds=time.monotonic() - merge_start,
        max_position_embeddings=4096,
        max_position_note="Bound export cache to the existing Q1R 4096-input-token limit; no RoPE scaling change.",
    )
    with (merged / "model.safetensors").open("rb") as stream:
        report["merged_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    save(output / "merge_report.json", report)
    print("Baseline and effective-text merge complete", flush=True)


if __name__ == "__main__":
    main()
