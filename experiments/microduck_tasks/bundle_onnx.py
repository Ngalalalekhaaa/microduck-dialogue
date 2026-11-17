"""Validate local VLM export reports and checksum ONNX graphs/external weights.

The manifest references the merged checkpoint's processor and provenance; it
does not copy full model weights or turn export checks into task-success claims.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from experiments.microduck_tasks.serve_adapter import sha256

ROOT = Path(__file__).resolve().parents[2]
CASES = {(5, 0), (1, 3), (3, 2)}


def local_file(directory, name):
    relative = Path(name)
    path = directory / relative
    if relative.is_absolute() or ".." in relative.parts or not path.is_file():
        raise ValueError(f"Missing or invalid bundle file {name!r}; finish the local export first.")
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Bundle file {name!r} escapes the model directory; restore local export files.")
    return path


def read_report(directory, name):
    path = local_file(directory, name)
    return json.loads(path.read_text()), sha256(path)


def validate_vision(report, merged):
    expected = {
        "status": "complete",
        "grid_thw": [1, 16, 20],
        "camera_rgb_size": [320, 240],
        "input_shape": [320, 1536],
        "output_shapes": [[80, 2048]] * 4,
        "hf_wrapper_exact_match": True,
        "dtype": "float16",
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Vision export report does not validate the 320x240 camera and three DeepStack outputs.")
    if report.get("merged_model") != str(merged):
        raise ValueError("Vision export used a different merged checkpoint; export the intended model.")
    errors = report.get("onnx_max_abs_errors", [])
    if len(errors) != 4 or any(
        not isinstance(value, int | float) or not math.isfinite(value) or value < 0 for value in errors
    ):
        raise ValueError("Vision numerical comparison is incomplete; finish its ONNX validation.")


def decoder_names(layers):
    inputs = ["input_ids", "image_mask", "image_embeds", "position_ids", "attention_mask"]
    inputs += [f"deepstack_{index}" for index in range(3)]
    outputs = ["logits"]
    for index in range(layers):
        for kind in ("key", "value"):
            inputs.append(f"past_{kind}_{index}")
            outputs.append(f"present_{kind}_{index}")
    return inputs, outputs


def validate_decoder(report, layers=28, *, allow_pending=False):
    expected_inputs, expected_outputs = decoder_names(layers)
    if ("inputs" in report and report["inputs"] != expected_inputs) or (
        "outputs" in report and report["outputs"] != expected_outputs
    ):
        raise ValueError("Decoder report has another model/interface; export the full 28-layer cached decoder.")
    original = report.get("original_hf_equivalence", [])
    if {(item.get("sequence"), item.get("past_length")) for item in original} != CASES:
        raise ValueError("Original HF equivalence must cover prefill, decode and a nonempty past cache.")
    if any(item.get("exact_match") is not True for item in original):
        raise ValueError("Decoder wrapper differs from original HF behavior; fix it before bundling.")
    verification = report.get("verification", [])
    if {(item.get("sequence"), item.get("past_length")) for item in verification} != CASES:
        raise ValueError("Decoder ONNX checks are incomplete; run verification without --skip-verify.")
    for item in verification:
        if item.get("finite") is not True:
            raise ValueError("Decoder ONNX returned nonfinite values; fix the export before bundling.")
        for key in ("max_abs_error", "logits_max_abs_error"):
            value = item.get(key)
            if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
                raise ValueError("Decoder ONNX numerical error is missing or invalid; rerun verification.")
    outside_tolerance = [
        {"sequence": item["sequence"], "past_length": item["past_length"], "output": name}
        for item in verification
        for name, metrics in item.get("outputs", {}).items()
        if metrics.get("tolerance_passed") is False
    ]
    if outside_tolerance and not allow_pending:
        raise ValueError(
            "Some decoder CPU outputs exceed tolerance; use --allow-pending-decoder for integration testing."
        )
    return {
        "status": "partial_awaiting_full_generation" if outside_tolerance else "numerical_checks_passed",
        "strict_tolerance_passed": not outside_tolerance,
        "outside_tolerance_outputs": outside_tolerance,
        "all_cases_finite": True,
        "all_cases_top_token_match": all(
            item.get("onnx_top_token") == item.get("pytorch_top_token") for item in verification
        ),
    }


def all_tensors(message):
    """Walk initializer, attribute, sparse, function and subgraph tensor messages."""
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        yield message
        return
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        if field.is_repeated:
            for child in value:
                yield from all_tensors(child)
        else:
            yield from all_tensors(value)


def graph_files(directory, graph_names):
    import onnx

    files = set()
    for name in graph_names:
        path = local_file(directory, name)
        files.add(Path(name).as_posix())
        model = onnx.load(str(path), load_external_data=False)
        if name in ("decoder.onnx", "decoder_int4.onnx", "decoder_int8.onnx"):
            inputs, outputs = decoder_names(28)
            if [item.name for item in model.graph.input] != inputs or [
                item.name for item in model.graph.output
            ] != outputs:
                raise ValueError(f"Graph {name} is not the full 28-layer cached decoder; restore the correct export.")
        for tensor in all_tensors(model):
            entries = {entry.key: entry.value for entry in tensor.external_data}
            if tensor.data_location != onnx.TensorProto.EXTERNAL and not entries:
                continue
            location = entries.get("location")
            if not location:
                raise ValueError(f"External tensor in {name} has no location; regenerate the graph.")
            external_name = Path(name).parent / location
            external = local_file(directory, external_name)
            offset = int(entries.get("offset", 0))
            length = int(entries["length"]) if "length" in entries else external.stat().st_size - offset
            if offset < 0 or length < 0 or offset + length > external.stat().st_size:
                raise ValueError(f"External tensor range exceeds {external_name}; restore complete external weights.")
            files.add(external_name.as_posix())
        del model
    return sorted(files)


def stable_hash(path):
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while hashing: {path}; wait for export to finish and retry.")
    return digest


def build_manifest(model_dir, merged, *, allow_pending_decoder=False):
    merge, merge_hash = read_report(merged, "merge_manifest.json")
    if merge.get("status") != "complete" or merge.get("remaining_lora_layers") != 0:
        raise ValueError("The merged checkpoint is incomplete; finish the CPU merge before bundling.")
    source = merge.get("source", {})
    required = {
        "adapter",
        "adapter_sha256",
        "base",
        "base_model_sha256",
        "base_model_revision",
        "prompt_version",
        "system_prompt_sha256",
    }
    if required - source.keys() or source.get("robot_finetuned") is not True:
        raise ValueError("Merged checkpoint lacks the trained robot adapter provenance; inspect its merge manifest.")
    config = json.loads(local_file(merged, "config.json").read_text())
    if config.get("model_type") != "qwen3_vl" or config.get("text_config", {}).get("num_hidden_layers") != 28:
        raise ValueError("This bundle requires the validated Qwen3-VL-2B 28-layer export.")
    reports = {}
    vision, vision_hash = read_report(model_dir, "vision.export.json")
    validate_vision(vision, merged)
    decoder, decoder_hash = read_report(model_dir, "decoder.export.json")
    decoder_check = validate_decoder(decoder, allow_pending=allow_pending_decoder)
    reports["vision"] = {"report": "vision.export.json", "sha256": vision_hash, "validation": vision}
    reports["decoder_fp16"] = {
        "report": "decoder.export.json",
        "sha256": decoder_hash,
        "validation": decoder,
        "assessment": decoder_check,
    }
    names = ["vision.onnx", "decoder.onnx"]
    for bits in (4, 8):
        graph_name = f"decoder_int{bits}.onnx"
        if not (model_dir / graph_name).exists():
            continue
        report_name = f"decoder_int{bits}_export.json"
        quantized, quantized_hash = read_report(model_dir, report_name)
        quantization = quantized.get("quantization", {})
        if quantization.get("format") != f"MatMulNBits INT{bits}" or quantization.get("block_size") not in {
            32,
            64,
            128,
        }:
            raise ValueError("Quantization report is incomplete; finish the weight-only quantization first.")
        reports[f"decoder_int{bits}"] = {
            "report": report_name,
            "sha256": quantized_hash,
            "validation": quantized,
            "numerical_verification": False,
            "note": "Quantization metadata alone does not establish numerical or task accuracy; evaluate generation.",
        }
        names.append(graph_name)
    discovered = {path.relative_to(model_dir).as_posix() for path in model_dir.rglob("*.onnx")}
    if discovered != set(names):
        raise ValueError(
            f"Unexpected or missing ONNX graphs: {sorted(discovered ^ set(names))}; select a clean export directory."
        )
    files = {name: stable_hash(local_file(model_dir, name)) for name in graph_files(model_dir, names)}
    if sha256(merged / "merge_manifest.json") != merge_hash:
        raise ValueError("Merge provenance changed during bundling; retry with the completed export.")
    for report in reports.values():
        if sha256(model_dir / report["report"]) != report["sha256"]:
            raise ValueError("An export report changed during bundling; wait for verification to finish and retry.")
    return {
        "format": "microduck-onnx-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "merged_model": str(merged),
        "merge_manifest_sha256": merge_hash,
        "source": source,
        "files": files,
        "graphs": names,
        "exports": reports,
        "camera_rgb_size": [320, 240],
        "camera_grid_thw": [1, 16, 20],
        "batch_size": 1,
        "decoder_sequence_and_past_lengths": "dynamic",
        "task_success_validated": False,
        "validation_status": "partial_awaiting_full_generation",
        "bundle_script_sha256": sha256(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".cache/microduck_onnx/model")
    parser.add_argument("--merged", type=Path, default=ROOT / ".cache/microduck_onnx/merged_model")
    parser.add_argument("--replace-manifest", action="store_true")
    parser.add_argument("--allow-pending-decoder", action="store_true")
    args = parser.parse_args()
    args.model_dir, args.merged = args.model_dir.resolve(), args.merged.resolve()
    destination = args.model_dir / "manifest.json"
    if destination.exists() and not args.replace_manifest:
        parser.error("manifest.json already exists; use --replace-manifest only after reviewing the new export.")
    manifest = build_manifest(args.model_dir, args.merged, allow_pending_decoder=args.allow_pending_decoder)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(
        f"Checksummed bundle: {destination} ({len(manifest['files'])} model files; generation validation pending)",
        flush=True,
    )


if __name__ == "__main__":
    main()
