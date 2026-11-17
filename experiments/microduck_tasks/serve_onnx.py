"""Serve the merged, exported robot VLM through the existing simulation protocol."""

import argparse
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiments.microduck_tasks.serve_adapter import handler_for, sha256
from experiments.microduck_tasks.skill_vlm import PROMPT_VERSION, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[2]


def verified_file(directory, name, entry):
    """Manifest entries are relative files inside one local model directory."""
    relative = Path(name)
    path = directory / relative
    if relative.is_absolute() or ".." in relative.parts or not path.is_file():
        raise ValueError(f"Missing or invalid manifest file {name!r}; rebuild the complete ONNX model directory.")
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Manifest files must stay inside their model directory; restore the exported files.")
    expected = entry.get("sha256") if isinstance(entry, dict) else entry
    if not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch for {name}; restore or rebuild the validated local model.")
    return path


def inspect_export(args):
    manifest_path = args.onnx_model / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("ONNX manifest.json is missing; finish and validate the full vision/decoder export first.")
    manifest = json.loads(manifest_path.read_text())
    merge_path = args.merged_model / "merge_manifest.json"
    if manifest.get("merged_model") != str(args.merged_model) or not merge_path.is_file():
        raise ValueError("ONNX manifest references another merged checkpoint; select its matching --merged-model.")
    if manifest.get("merge_manifest_sha256") != sha256(merge_path):
        raise ValueError(
            "Merge provenance differs from the ONNX manifest; rebuild the export from the intended adapter."
        )
    merge = json.loads(merge_path.read_text())
    source = merge["source"]
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if merge.get("status") != "complete" or merge.get("remaining_lora_layers") != 0:
        raise ValueError("Merged checkpoint is incomplete; finish merging before serving ONNX.")
    if manifest.get("source") != source:
        raise ValueError("ONNX source metadata differs from its merged checkpoint; restore the matching manifest.")
    if source.get("prompt_version") != PROMPT_VERSION or source.get("system_prompt_sha256") != prompt_hash:
        raise ValueError("The robot prompt differs from the trained/exported prompt; use its matching checkout.")
    if source.get("robot_finetuned") is not True:
        raise ValueError("This service requires a merged robot adapter; select the validated Microduck export.")
    if (args.merged_model / "system_prompt.txt").read_text() != SYSTEM_PROMPT:
        raise ValueError("Saved system prompt differs from this checkout; restore the exported model assets.")
    # The runtime does not load merged weight tensors, only their processor and config.
    for name, entry in merge["outputs"].items():
        if name.endswith(".safetensors"):
            continue
        verified_file(args.merged_model, name, entry)
    files = manifest.get("files", {})
    selection = getattr(args, "decoder", "auto")
    selected = next(
        (name for name in ("decoder_int8.onnx", "decoder_int4.onnx", "decoder.onnx") if name in files), "decoder.onnx"
    )
    if selection != "auto":
        selected = {"int8": "decoder_int8.onnx", "int4": "decoder_int4.onnx", "fp16": "decoder.onnx"}[selection]
    if "vision.onnx" not in files or selected not in files:
        raise ValueError("Manifest must include vision.onnx and a cached decoder graph; finish the full export.")
    for name, entry in files.items():
        verified_file(args.onnx_model, name, entry)
    # Check that all external tensors loaded by ORT were included in the checksums.
    import onnx

    for name in ("vision.onnx", selected):
        graph = onnx.load(str(args.onnx_model / name), load_external_data=False)
        for tensor in graph.graph.initializer:
            for value in tensor.external_data:
                if value.key == "location" and str(Path(name).parent / value.value) not in files:
                    raise ValueError("An ONNX external weights file lacks a manifest checksum; complete the export.")
    metadata = {
        **source,
        "model": "microduck-onnx",
        "inference_format": "onnx",
        "onnx_model": str(args.onnx_model),
        "onnx_manifest_sha256": sha256(manifest_path),
        "decoder": selected,
        "decoder_quantization": {
            "decoder_int8.onnx": "int8_weight_only",
            "decoder_int4.onnx": "int4_weight_only",
            "decoder.onnx": "float16",
        }[selected],
        "generation_mode": "onnx_unconstrained_greedy_client_validated",
        "schema_constrained_decoding": False,
        "training": "merged robot simulation QLoRA adapter; ONNX frozen inference only",
    }
    return metadata, args.onnx_model / selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=ROOT / ".cache/microduck_onnx/int8_compact",
        help="Validated deployment directory (default: .cache/microduck_onnx/int8_compact).",
    )
    parser.add_argument("--merged-model", type=Path, default=ROOT / ".cache/microduck_onnx/merged_model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--vision-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Vision execution device; auto follows --device, cpu reserves GPU memory for decoding.",
    )
    parser.add_argument(
        "--memory-conservative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Shrink decoder arena after every run; defaults on for FP16/INT8, off for INT4.",
    )
    parser.add_argument("--decoder", choices=("auto", "int8", "int4", "fp16"), default="auto")
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=128,
        help="Maximum prompt tokens per cached prefill chunk (1..512; default: 128).",
    )
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--generation-timeout", type=float, default=90)
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.generation_timeout <= 300 or not 1 <= args.cpu_threads <= 32:
        parser.error("Use port 1..65535, generation-timeout 1..300 seconds and cpu-threads 1..32.")
    if not 1 <= args.prefill_chunk_size <= 512:
        parser.error("Use prefill-chunk-size 1..512 to bound the attention workspace.")
    args.onnx_model, args.merged_model = args.onnx_model.resolve(), args.merged_model.resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    metadata, decoder_path = inspect_export(args)
    print("Verified local ONNX export:\n" + json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    server.daemon_threads = True
    try:
        from experiments.microduck_tasks.onnx_vlm import OnnxEngine

        engine = OnnxEngine(args, metadata, decoder_path)
        server.RequestHandlerClass = handler_for(engine)
        print(f"Ready: http://127.0.0.1:{args.port}/health (Ctrl+C to stop)", flush=True)
        print("Full vision + cached decoder ONNX; greedy JSON proposals remain client-validated.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nONNX service stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
