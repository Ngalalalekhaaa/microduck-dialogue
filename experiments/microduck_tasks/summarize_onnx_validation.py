"""Summarize saved VLM conversion evidence without inference or changing source files.

Run again after the final evaluation/simulation processes finish. Missing reports
remain pending; an evaluation loop finishing is not a claim of task success.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.microduck_tasks.skill_vlm import parse_proposal

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "experiments/microduck_tasks/outputs"
CACHE = ROOT / ".cache/microduck_onnx"
FIELDS = ("type", "skill", "target_color", "seconds")
ATTEMPTS = (
    ("int4_block128", "evaluation", "service.log"),
    ("int4_block32_fp16_head", "evaluation32", "candidate32_service.log"),
    ("fp16", "evaluation_fp16", "compact_service.log"),
    ("fp16_conservative", "evaluation_fp16_conservative", "compact_conservative_service.log"),
    ("int8_default_memory", "evaluation8", "int8_service.log"),
    ("int8_conservative", "evaluation8_conservative", "int8_conservative_service.log"),
)


def reference(path):
    path = path.resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Evidence changed while reading {path}; rerun after that process finishes.")
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": after.st_size}


def load(path):
    return json.loads(path.read_text())


def rows(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row["index"] for row in records}) != len(records):
        raise ValueError(f"Duplicate sample indices in {path}; inspect the original report.")
    return records


def latency(values):
    if not values:
        return None
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def generation(summary_path, predictions_path):
    if not summary_path.is_file() or not predictions_path.is_file():
        return {"status": "pending", "expected_summary": str(summary_path.resolve())}
    original, records = load(summary_path), rows(predictions_path)
    correct = Counter(dict.fromkeys((*FIELDS, "joint"), 0))
    failures, samples, all_times, valid_times = Counter(), [], [], []
    for row in records:
        prediction, error_kind, error = None, None, row.get("error")
        if error and not row.get("raw"):
            error_kind = "request_or_runtime_failure"
        else:
            try:
                prediction = parse_proposal(row.get("raw", "")).to_dict()
            except (ValueError, TypeError) as exc:
                error_kind, error = "invalid_proposal", str(exc)
        if prediction != row.get("prediction"):
            raise ValueError(
                f"Saved parsed prediction disagrees with current parser at {predictions_path}:{row['index']}."
            )
        if error_kind:
            failures[error_kind] += 1
        matches = {field: prediction is not None and prediction[field] == row["target"][field] for field in FIELDS}
        for field, value in matches.items():
            correct[field] += int(value)
        correct["joint"] += int(all(matches.values()))
        seconds = row.get("latency_s")
        if isinstance(seconds, int | float) and math.isfinite(seconds) and seconds >= 0:
            all_times.append(seconds)
            if prediction is not None:
                valid_times.append(seconds)
        samples.append(
            {
                "index": row["index"],
                "id": row["id"],
                "target": row["target"],
                "prediction": prediction,
                "valid_proposal": prediction is not None,
                "matches": matches,
                "joint_correct": all(matches.values()),
                "error_kind": error_kind,
                "error": error,
                "latency_s": seconds,
            }
        )
    valid = len(records) - sum(failures.values())
    stored_counts = original.get("correct", original.get("counts", {}))
    discrepancies = {
        key: {"saved": stored_counts[key], "recomputed": value}
        for key, value in correct.items()
        if key in stored_counts and stored_counts[key] != value
    }
    saved_valid = original.get("valid_json", stored_counts.get("valid"))
    if saved_valid is not None and saved_valid != valid:
        discrepancies["valid"] = {"saved": saved_valid, "recomputed": valid}
    if original.get("count") != len(records):
        discrepancies["count"] = {"saved": original.get("count"), "recomputed": len(records)}
    return {
        "status": "recorded",
        "summary": reference(summary_path),
        "predictions": reference(predictions_path),
        "saved_summary_status": original.get("status"),
        "count": len(records),
        "valid_proposals": valid,
        "failures": dict(failures),
        "correct": dict(correct),
        "joint_accuracy": correct["joint"] / len(records) if records else None,
        "latency_seconds_all_requests": latency(all_times),
        "latency_seconds_valid_proposals_only": latency(valid_times),
        "saved_summary_discrepancies": discrepancies,
        "metadata": {
            key: original[key]
            for key in (
                "method",
                "selection",
                "validation_sha256",
                "dataset_sha256",
                "reference_sha256",
                "nf4_predictions_sha256",
                "merge_manifest_sha256",
                "server",
                "requested_max_tokens",
                "baseline_max_tokens",
                "assistant_targets_excluded_from_input",
                "images",
            )
            if key in original
        },
        "samples": samples,
    }


def log_evidence(path):
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    matches = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if any(term in line.lower() for term in ("out of memory", "failed to allocate", "cuda failure")):
            matches.append({"line": number, "text": re.sub(r"\x1b\[[0-9;]*m", "", line)[:1200]})
    return {"source": reference(path), "allocation_error_lines": len(matches), "examples": matches[:2]}


def comparison(left, right):
    if left["status"] != "recorded" or right["status"] != "recorded":
        return {"status": "pending"}
    a = {item["index"]: item for item in left["samples"]}
    b = {item["index"]: item for item in right["samples"]}
    common = sorted(a.keys() & b.keys())
    changed = []
    for index in common:
        if a[index]["target"] != b[index]["target"] or a[index]["id"] != b[index]["id"]:
            raise ValueError(f"Different targets/sample IDs at index {index}; do not compare these evaluations.")
        pa, pb = a[index]["prediction"], b[index]["prediction"]
        if pa is None or pb is None or any(pa[field] != pb[field] for field in FIELDS):
            changed.append(index)
    return {
        "status": "recorded",
        "same_sample_index_set": a.keys() == b.keys(),
        "common_samples": len(common),
        "same_four_field_prediction": len(common) - len(changed),
        "different_or_invalid_indices": changed,
        "note": "Equal aggregate accuracy does not mean identical individual predictions. Reply text is excluded.",
    }


def simulation(directory):
    path = directory / "summary.json"
    if not path.is_file():
        return {"status": "pending", "expected_summary": str(path.resolve())}
    source = load(path)
    results = source.get("results", [])
    return {
        "status": "recorded",
        "source": reference(path),
        "events": event_evidence(directory),
        "exit_code": source.get("exit_code"),
        "runtime_failure": source.get("runtime_failure"),
        "errors": source.get("errors"),
        "metadata": source.get("metadata"),
        "result_status_counts": dict(Counter(item.get("status") for item in results)),
        "physical_task_verified_count": sum(item.get("physical_task_verified") is True for item in results),
        "results": results,
        "physics": source.get("physics"),
        "max_target_excess_rad": source.get("max_target_excess_rad"),
        "max_tilt_deg": source.get("max_tilt_deg"),
        "simulation_seconds": source.get("simulation_seconds"),
        "wall_seconds": source.get("wall_seconds"),
        "scope": "This saved scenario only; model-reported completion is not independent physical verification.",
    }


def event_evidence(directory):
    path = directory / "events.jsonl"
    if not path.is_file():
        return {"status": "pending", "expected_events": str(path)}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    traces = [record for record in records if (record.get("model_trace") or {}).get("latency_s") is not None]
    times = [record["model_trace"]["latency_s"] for record in traces]
    return {
        "source": reference(path),
        "decision_count": len(traces),
        "latency_seconds": latency(times),
        "decisions": [
            {
                "event_id": record["event_id"],
                "instruction": record["instruction"],
                "decision": record.get("decision"),
                "execution": record.get("execution"),
                "proposal": record["model_trace"].get("model_proposal"),
                "latency_s": record["model_trace"]["latency_s"],
                "manifest_sha256": record["model_trace"].get("server_metadata", {}).get("onnx_manifest_sha256"),
            }
            for record in traces
        ],
    }


def simulation_latency_comparison(closed_loop):
    baseline_root = OUTPUTS / "qlora_visual_test_20261115"
    baseline = {key: event_evidence(baseline_root / key) for key in ("basic", "aligned_kick", "gui")}
    nf4_times = [item["latency_s"] for value in baseline.values() for item in value.get("decisions", [])]
    onnx_times = [
        item["latency_s"] for value in closed_loop.values() for item in value.get("events", {}).get("decisions", [])
    ]
    return {
        "nf4_sources": baseline,
        "nf4_latency_seconds": latency(nf4_times),
        "final_onnx_latency_seconds": latency(onnx_times),
        "median_ratio_nf4_over_onnx": (
            statistics.median(nf4_times) / statistics.median(onnx_times) if nf4_times and onnx_times else None
        ),
        "scope": "Observed basic/aligned-kick/GUI decisions from separate simulation runs; "
        "not isolated equal-input benchmarking. Camera/state, output length and runtime settings may differ.",
    }


def merge_evidence():
    path = CACHE / "merged_model/merge_manifest.json"
    source = load(path)
    checks = source["layer_checks"]
    return {
        "manifest": reference(path),
        "source": source["source"],
        "status": source["status"],
        "source_dtype": source["source_dtype"],
        "merge_accumulation_dtype": source["merge_accumulation_dtype"],
        "output_dtype": source["output_dtype"],
        "remaining_lora_layers": source["remaining_lora_layers"],
        "checked_lora_layers": len(checks),
        "all_weight_formula_checks_exact": all(item["weight_formula_exact"] for item in checks),
        "maximum_random_input_relative_rmse": max(item["random_input_relative_rmse"] for item in checks),
        "teacher_forced_comparison": source["input_validation"],
        "report_correction": source.get("report_correction"),
        "checkpoint": source["outputs"]["model.safetensors"],
        "limitation": source["warning"],
        "hash_scope": "Manifest and evaluation files rehashed here; multi-GB checkpoint digest is from merge manifest.",
    }


def final_input_evidence(final):
    if final["status"] != "recorded":
        return {"status": "pending"}
    dataset_path = OUTPUTS / "training_dataset_final_20261114/val.jsonl"
    dataset = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    dataset_reference = reference(dataset_path)
    if dataset_reference["sha256"] != final["metadata"].get("validation_sha256"):
        raise ValueError("Final evaluation dataset hash has changed; restore the evaluated dataset.")
    checked = 0
    for row in rows(Path(final["predictions"]["path"])):
        expected = dataset[row["index"]]
        if row["id"] != expected["sample_id"] or row["target"] != expected["target"]:
            raise ValueError(f"Final sample {row['index']} differs from its recorded validation dataset.")
        for camera, image in row["images"].items():
            if Path(image["path"]).resolve() != (dataset_path.parent / expected["images"][camera]).resolve():
                raise ValueError(f"Final sample {row['index']} used another camera image.")
            if reference(Path(image["path"]))["sha256"] != image["sha256"]:
                raise ValueError(f"Final camera image changed: {image['path']}.")
            checked += 1
    return {"dataset": dataset_reference, "matched_image_hashes": checked, "sample_ids_and_targets_match": True}


def build(args):
    training = OUTPUTS / "qlora_20261114"
    nf4 = generation(training / "generation_evaluation.json", training / "predictions.jsonl")
    merged = generation(args.run / "merged_generation_summary.json", args.run / "merged_generation.jsonl")
    attempts = {}
    for name, directory, log in ATTEMPTS:
        attempt = generation(args.run / directory / "summary.json", args.run / directory / "predictions.jsonl")
        attempt["runtime_log"] = log_evidence(CACHE / log)
        attempts[name] = attempt
    final = generation(args.final_evaluation / "summary.json", args.final_evaluation / "predictions.jsonl")
    closed_loop = {"basic": simulation(args.basic), "aligned_kick": simulation(args.kick), "gui": simulation(args.gui)}
    manifest_path = args.final_model / "manifest.json"
    manifest = load(manifest_path) if manifest_path.is_file() else {}
    final_server = final.get("metadata", {}).get("server", {}).get("server_metadata", {})
    final_manifest = reference(manifest_path) if manifest else None
    manifest_matches = (
        final_server.get("onnx_manifest_sha256") == final_manifest["sha256"]
        if final_server and final_manifest
        else None
    )
    pending = [
        name for name, report in {"final_generation": final, **closed_loop}.items() if report["status"] == "pending"
    ]
    cleanup_path = args.run / "temporary_artifact_cleanup.json"
    cleanup = (
        {"source": reference(cleanup_path), "record": load(cleanup_path)}
        if cleanup_path.is_file()
        else {"status": "not_recorded"}
    )
    return {
        "format": "microduck-onnx-validation-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": reference(Path(__file__)),
        "status": "awaiting_evidence" if pending else "evidence_collected_review_results",
        "pending": pending,
        "metric": "Exact type, skill, target_color and seconds agreement against saved demonstration targets.",
        "limitations": [
            "The fixed 24 held-out examples are selected deterministically by skill/response category, not randomly.",
            "These accuracy values are not closed-loop task success rates or guarantees for other scenes/instructions.",
            "Same input examples do not imply identical predictions; per-sample differences are recorded below.",
            "NF4 plus adapter versus merged HF/ONNX changes numerical representation and runtime; "
            "this is not isolated quantization ablation.",
            "NF4/HF generation used 192 new tokens; service evaluations requested 256. "
            "Free-text reply correctness is not scored.",
            "Latency is full local request/model generation time; "
            "failed requests are separated because fast failures distort medians.",
            "No NF4 latency baseline is saved for the fixed 24 examples; "
            "separate simulation timing is shown only as a non-isolated practical comparison.",
            "Numeric conversion checks, generation accuracy and simulation task checks are separate evidence.",
            "The aligned kick is a prepared single scenario; it does not measure random search/approach/kick success.",
        ],
        "merge": merge_evidence(),
        "original_nf4": nf4,
        "merged_hf_fp16": merged,
        "historical_attempts": attempts,
        "temporary_artifact_cleanup": cleanup,
        "removed_artifact_note": "Failed INT4 directories listed in the cleanup record are diagnostic archives; "
        "their removed external weights mean those directories cannot now run as complete models.",
        "historical_initial_int4_basic": simulation(args.run / "basic"),
        "final_candidate": {
            "model_directory": str(args.final_model.resolve()),
            "manifest": final_manifest,
            "manifest_matches_evaluation": manifest_matches,
            "bundle_validation_status": manifest.get("validation_status"),
            "bundle_files": manifest.get("files"),
            "server_identity": (
                {
                    "source": reference(args.run / "server_identity.json"),
                    "record": load(args.run / "server_identity.json"),
                }
                if (args.run / "server_identity.json").is_file()
                else None
            ),
            "default_launch_health": (
                {
                    "source": reference(args.run / "default_service_health.json"),
                    "record": load(args.run / "default_service_health.json"),
                }
                if (args.run / "default_service_health.json").is_file()
                else None
            ),
            "gui_screenshot": (
                reference(args.run / "gui8_compact_window.png")
                if (args.run / "gui8_compact_window.png").is_file()
                else None
            ),
            "generation": final,
            "input_integrity": final_input_evidence(final),
            "simulation": closed_loop,
        },
        "simulation_timing_comparison": simulation_latency_comparison(closed_loop),
        "comparisons": {
            "nf4_vs_merged_hf": comparison(nf4, merged),
            "nf4_vs_final_onnx": comparison(nf4, final),
            "merged_hf_vs_final_onnx": comparison(merged, final),
        },
        "randomized_closed_loop_success_rate": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUTPUTS / "onnx_test_20261115")
    parser.add_argument("--final-model", type=Path, default=CACHE / "int8_compact")
    parser.add_argument("--final-evaluation", type=Path)
    parser.add_argument("--basic", type=Path)
    parser.add_argument("--kick", type=Path)
    parser.add_argument("--gui", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for key, default in {
        "final_evaluation": "evaluation8_compact",
        "basic": "basic8_compact",
        "kick": "aligned_kick8_compact",
        "gui": "gui8_compact",
        "output": "summary.json",
    }.items():
        if getattr(args, key) is None:
            setattr(args, key, args.run / default)
    report = build(args)
    if args.output.resolve().name != "summary.json" or args.output.resolve().parent != args.run.resolve():
        parser.error("Write only the top-level --run/summary.json; source reports must remain untouched.")
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "status": report["status"], "pending": report["pending"]}))


if __name__ == "__main__":
    main()
