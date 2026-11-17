"""Compare merged HF fp16 generation with saved NF4 and ONNX predictions."""

import argparse
import json
import os
import time
from pathlib import Path

from experiments.microduck_tasks.merge_adapter import ROOT, write_json
from experiments.microduck_tasks.serve_adapter import sha256
from experiments.microduck_tasks.skill_vlm import SYSTEM_PROMPT, parse_proposal
from experiments.microduck_tasks.train_vlm import encode, read_records, validation_indices


def read_predictions(path):
    return {row["index"]: row for line in path.read_text().splitlines() if line.strip() for row in [json.loads(line)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/microduck_onnx/merged_model")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/microduck_tasks/outputs/onnx_test_20261115")
    parser.add_argument("--indices", type=int, nargs="+", default=[991, 124, 1072, 1145])
    parser.add_argument("--all-24", action="store_true")
    args = parser.parse_args()
    args.model, args.output = args.model.resolve(), args.output.resolve()
    destination = args.output / "merged_generation.jsonl"
    if destination.exists():
        parser.error("merged_generation.jsonl exists; use a new --output directory to preserve previous results.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires the released GPU; coordinate before starting it.")
    if (args.model / "system_prompt.txt").read_text() != SYSTEM_PROMPT:
        raise ValueError("Merged checkpoint has another system prompt; use the matching checkout.")
    merge = json.loads((args.model / "merge_manifest.json").read_text())
    training = Path(merge["source"]["adapter"]).parent
    config = json.loads((training / "training_config.json").read_text())
    dataset = ROOT / "experiments/microduck_tasks/outputs/training_dataset_final_20261114/val.jsonl"
    records = read_records(dataset)
    indices = list(dict.fromkeys(args.indices + (validation_indices(records, 24) if args.all_24 else [])))
    previous_path = training / "predictions.jsonl"
    previous = read_predictions(previous_path)
    onnx_path = ROOT / "experiments/microduck_tasks/outputs/onnx_test_20261115/evaluation/predictions.jsonl"
    onnx = read_predictions(onnx_path) if onnx_path.is_file() else {}
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, min_pixels=65536, max_pixels=int(config["max_pixels"])
    )
    print("Loading the merged HF fp16 checkpoint on GPU (SDPA), no adapter/quantization", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa"
    )
    model.eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {key: 0 for key in ("valid", "type", "skill", "seconds", "target_color", "joint", "nf4_joint_match")}
    latencies = []
    with destination.open("x") as stream, torch.inference_mode():
        for index in indices:
            row = records[index]
            batch, boundary = encode(row, dataset.parent, processor, int(config["max_length"]))
            batch["input_ids"] = batch["input_ids"][:, :boundary]
            batch["attention_mask"] = batch["attention_mask"][:, :boundary]
            batch = {
                name: value.to(device="cuda", dtype=torch.float16 if value.is_floating_point() else value.dtype)
                for name, value in batch.items()
            }
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            generated = model.generate(**batch, max_new_tokens=192, do_sample=False, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            raw = processor.tokenizer.decode(generated[0, boundary:], skip_special_tokens=True)
            result = {
                "index": index,
                "id": row["sample_id"],
                "instruction": row["instruction"],
                "target": row["target"],
                "raw": raw,
                "latency_s": elapsed,
                "prompt_tokens": boundary,
                "completion_tokens": generated.shape[1] - boundary,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "nf4_prediction": previous.get(index, {}).get("prediction"),
                "nf4_raw": previous.get(index, {}).get("raw"),
                "onnx_int4_prediction": onnx.get(index, {}).get("prediction"),
                "onnx_int4_raw": onnx.get(index, {}).get("raw"),
                "onnx_int4_error": onnx.get(index, {}).get("error"),
            }
            try:
                prediction = parse_proposal(raw).to_dict()
            except (TypeError, ValueError) as error:
                result["error"] = str(error)
                result["valid_proposal"] = False
            else:
                counts["valid"] += 1
                result["valid_proposal"] = True
                result["prediction"] = prediction
                matches = {
                    key: prediction[key] == row["target"][key] for key in ("type", "skill", "seconds", "target_color")
                }
                result["matches"] = matches
                result["joint_correct"] = all(matches.values())
                for key, value in matches.items():
                    counts[key] += int(value)
                counts["joint"] += int(result["joint_correct"])
                baseline = previous.get(index, {}).get("prediction")
                result["nf4_joint_match"] = bool(baseline) and all(prediction[key] == baseline[key] for key in matches)
                counts["nf4_joint_match"] += int(result["nf4_joint_match"])
            latencies.append(elapsed)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {key: result.get(key) for key in ("index", "instruction", "raw", "valid_proposal", "latency_s")},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del batch, generated
            torch.cuda.empty_cache()
    summary = {
        "count": len(indices),
        "indices": indices,
        "method": "merged HF fp16 SDPA greedy, 192 max new tokens",
        "assistant_targets_excluded_from_input": True,
        "images": "original front and near RGB from each validation record",
        "model": str(args.model),
        "merge_manifest_sha256": sha256(args.model / "merge_manifest.json"),
        "dataset_sha256": sha256(dataset),
        "nf4_predictions_sha256": sha256(previous_path),
        "counts": counts,
        "rates": {key: value / len(indices) for key, value in counts.items()},
        "mean_latency_s": sum(latencies) / len(latencies),
        "max_latency_s": max(latencies),
        "predictions": str(destination),
        "script_sha256": sha256(Path(__file__)),
    }
    write_json(args.output / "merged_generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
