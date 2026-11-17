"""Save a fixed dual-camera teacher-forced CPU reference for an ONNX export."""

import argparse
import json
import os
from pathlib import Path

from experiments.microduck_tasks.merge_adapter import ROOT, write_json
from experiments.microduck_tasks.serve_adapter import sha256
from experiments.microduck_tasks.train_vlm import encode, read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/microduck_onnx/merged_model")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/microduck_onnx/model_validation")
    args = parser.parse_args()
    args.model, args.output = args.model.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Output already exists; inspect it or select a new path.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest = json.loads((args.model / "merge_manifest.json").read_text())
    validation = manifest["input_validation"]
    dataset = Path(validation["dataset"])
    if sha256(dataset) != validation["dataset_sha256"]:
        raise ValueError("Validation dataset changed since merging; choose the original reviewed dataset.")
    source_config = json.loads((Path(manifest["source"]["adapter"]).parent / "training_config.json").read_text())
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, min_pixels=65536, max_pixels=int(source_config["max_pixels"])
    )
    row = read_records(dataset)[validation["sample_index"]]
    batch, boundary = encode(row, dataset.parent, processor, int(source_config["max_length"]))
    for name, value in batch.items():
        if value.is_floating_point():
            batch[name] = value.to(torch.float16)
    length = batch["input_ids"].shape[1]
    supervised_positions = torch.arange(boundary - 1, length - 1)
    positions = torch.arange(boundary - 1, length)
    print(f"Loading merged CPU fp16 model; reference length={length}, assistant boundary={boundary}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float16, device_map={"": "cpu"}, attn_implementation="sdpa"
    )
    model.requires_grad_(False)
    model.eval()
    with torch.inference_mode():
        logits = model(**batch, logits_to_keep=positions, use_cache=False).logits.float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("Merged reference logits are non-finite; do not use this ONNX comparison fixture.")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "inputs.npz", **{name: value.numpy() for name, value in batch.items()})
    np.savez_compressed(
        args.output / "reference_logits.npz",
        last_logits=logits[:, -1:, :].numpy(),
        supervised_logits=logits[:, :-1, :].numpy(),
        supervised_positions=supervised_positions.numpy(),
        assistant_target_token_ids=batch["input_ids"][:, boundary:].numpy(),
    )
    metadata = {
        "format": "microduck-merged-onnx-reference-v1",
        "model": str(args.model),
        "merge_manifest_sha256": sha256(args.model / "merge_manifest.json"),
        "script_sha256": sha256(Path(__file__)),
        "input_validation": validation,
        "contains_assistant_target": True,
        "purpose": "Teacher-forced numerical conversion validation, not task evaluation or generation scoring.",
        "prompt_length": boundary,
        "input_length": length,
        "last_logits_position": length - 1,
        "supervised_positions_description": "Positions boundary-1 through length-2 predict assistant target tokens.",
        "dtype": "CPU float16 model; reference logits float32",
        "inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in batch.items()},
        "files": {
            name: {"sha256": sha256(args.output / name), "bytes": (args.output / name).stat().st_size}
            for name in ("inputs.npz", "reference_logits.npz")
        },
    }
    write_json(args.output / "reference_manifest.json", metadata)
    print(f"Saved numerical reference: {args.output}", flush=True)


if __name__ == "__main__":
    main()
