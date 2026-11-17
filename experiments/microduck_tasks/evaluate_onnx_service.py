"""Evaluate a running local VLM on the exact saved QLoRA generation subset.

This is an offline image/instruction test: no simulation or robot is moved.
Targets are used only after inference, never sent to the local model service.
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.microduck_tasks.serve_adapter import sha256
from experiments.microduck_tasks.skill_vlm import SYSTEM_PROMPT, LocalVLMBackend, SkillInput

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/microduck_tasks/outputs/qlora_20261114"
FIELDS = ("type", "skill", "target_color", "seconds")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def read_inputs(args):
    baseline = json.loads(args.reference.read_text())
    training = json.loads((args.reference.parent / "training_config.json").read_text())
    if sha256(args.validation) != training["val_sha256"]:
        raise ValueError("Validation dataset differs from the saved training checksum; select its original val.jsonl.")
    records = [json.loads(line) for line in args.validation.read_text().splitlines() if line.strip()]
    indices = baseline["indices"]
    if (
        not indices
        or len(indices) != len(set(indices))
        or any(type(i) is not int or not 0 <= i < len(records) for i in indices)
    ):
        raise ValueError("Reference indices must select distinct, existing validation records.")
    saved = {
        row["index"]: row
        for row in map(json.loads, (args.reference.parent / "predictions.jsonl").read_text().splitlines())
    }
    for index in indices:
        row, previous = records[index], saved[index]
        if row["sample_id"] != previous["id"] or row["target"] != previous["target"]:
            raise ValueError("Saved generation sample/target does not match the validation row; restore the dataset.")
        if row["messages"][0] != {"role": "system", "content": SYSTEM_PROMPT}:
            raise ValueError("Validation prompt differs from this checkout; use the matching trained prompt.")
    return records, indices[: args.limit] if args.limit else indices, baseline, saved


def skill_input(row, directory):
    images, paths = {}, {}
    for camera in ("front", "near"):
        path = (directory / row["images"][camera]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Dataset camera images must remain inside the validation dataset directory.")
        with Image.open(path) as image:
            images[camera] = np.asarray(image.convert("RGB")).copy()
        paths[camera] = {"path": str(path), "sha256": sha256(path)}
    return SkillInput(row["instruction"], images, row["state"], row["context"]), paths


def summarize(results):
    count = len(results)
    correct = {key: sum(record["matches"][key] for record in results) for key in FIELDS}
    valid = sum(record["valid_proposal"] for record in results)
    joint = sum(record["joint_correct"] for record in results)
    categories = {}
    for record in results:
        target = record["target"]
        category = target["skill"] if target["type"] == "skill" else target["type"]
        item = categories.setdefault(category, {"count": 0, "valid_json": 0, "joint_correct": 0})
        item["count"] += 1
        item["valid_json"] += int(record["valid_proposal"])
        item["joint_correct"] += int(record["joint_correct"])
    latencies = [record["latency_s"] for record in results]
    return {
        "count": count,
        "valid_json": valid,
        "valid_json_rate": valid / count if count else None,
        "correct": {**correct, "joint": joint},
        **{f"{key}_accuracy": value / count if count else None for key, value in {**correct, "joint": joint}.items()},
        "by_category": categories,
        "latency_seconds": {"median": statistics.median(latencies), "min": min(latencies), "max": max(latencies)}
        if latencies
        else {},
        "error_count": sum("error" in record for record in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8083/v1/chat/completions")
    parser.add_argument("--model", default="microduck-onnx")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--reference", type=Path, default=RUN / "generation_evaluation.json")
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "experiments/microduck_tasks/outputs/training_dataset_final_20261114/val.jsonl",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="0 selects every saved index; positive values are a debug subset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/microduck_tasks/outputs" / f"onnx_evaluation_{time.time_ns()}",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be zero or a positive sample count.")
    records, indices, baseline, saved = read_inputs(args)
    backend = LocalVLMBackend(endpoint=args.endpoint, model=args.model, timeout=args.timeout)
    backend.refresh_model_metadata()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "format": "microduck-service-generation-evaluation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "model": args.model,
        "indices": indices,
        "selection": "same_saved_qlora_generation_indices",
        "validation": str(args.validation.resolve()),
        "validation_sha256": sha256(args.validation),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "server": backend.metadata,
        "requested_max_tokens": 256,
        "baseline_max_tokens": 192,
        "comparison": (
            "Same held-out inputs; NF4+adapter versus merged ONNX does not isolate "
            "merge, precision or quantization effects."
        ),
        "metric": (
            "Exact type/skill/target_color/seconds agreement; free-text reply and physical task success are not scored."
        ),
        "baseline_summary": baseline,
    }
    write_json(args.output / "evaluation_config.json", metadata)
    results = []
    started = time.monotonic()
    with (args.output / "predictions.jsonl").open("x") as stream:
        for ordinal, index in enumerate(indices, 1):
            row = records[index]
            request, image_info = skill_input(row, args.validation.parent)
            result = {
                "index": index,
                "id": row["sample_id"],
                "instruction": row["instruction"],
                "images": image_info,
                "target": row["target"],
                "baseline_prediction": saved[index].get("prediction"),
                "baseline_raw": saved[index].get("raw"),
                "valid_proposal": False,
                "matches": dict.fromkeys(FIELDS, False),
                "joint_correct": False,
            }
            try:
                prediction = backend.decide(request).to_dict()
            except (RuntimeError, ValueError, TypeError) as error:
                result["error"] = f"{type(error).__name__}: {error}"
            else:
                matches = {key: prediction[key] == row["target"][key] for key in FIELDS}
                result.update(
                    prediction=prediction, valid_proposal=True, matches=matches, joint_correct=all(matches.values())
                )
            trace = backend.last_trace
            result.update(
                raw=trace.get("raw_response"),
                latency_s=trace.get("latency_s", 0),
                usage=trace.get("usage"),
                server_metadata=trace.get("server_metadata"),
                trace=trace,
            )
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            summary = {
                **metadata,
                **summarize(results),
                "status": "running",
                "elapsed_seconds": time.monotonic() - started,
            }
            write_json(args.output / "summary.json", summary)
            print(
                f"[{ordinal}/{len(indices)}] index={index} valid={result['valid_proposal']} "
                f"joint={result['joint_correct']} latency={result['latency_s']:.3f}s",
                flush=True,
            )
    summary.update(status="complete", elapsed_seconds=time.monotonic() - started)
    write_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {key: summary[key] for key in ("count", "correct", "latency_seconds", "error_count")},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved evaluation: {args.output}", flush=True)


if __name__ == "__main__":
    main()
