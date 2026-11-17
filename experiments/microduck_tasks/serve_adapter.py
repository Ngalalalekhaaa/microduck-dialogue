"""Serve the trained local QLoRA adapter for HD1910 simulation, without training.

JSON schema is a requested output contract, NOT a constrained decoding grammar.
Generation is greedy and the existing client validates each proposal before use.
"""

import argparse
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiments.microduck_tasks.serve_adapter_protocol import MAX_BODY_BYTES, decode_request
from experiments.microduck_tasks.skill_vlm import PROMPT_VERSION, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_files(model_path, adapter_path, *, protocol="v1"):
    """Pin the local weights and reject a mismatched prompt or training base."""
    model_path, adapter_path = model_path.resolve(), adapter_path.resolve()
    run_path = adapter_path.parent
    if not (run_path / "training_config.json").is_file():
        run_path = run_path.parent
    config = json.loads((run_path / "training_config.json").read_text())
    system_prompt, prompt_version = SYSTEM_PROMPT, PROMPT_VERSION
    if protocol == "v2":
        from .plan_protocol import SYSTEM, VERSION

        system_prompt, prompt_version = SYSTEM, VERSION
    elif protocol != "v1":
        raise ValueError("Select protocol v1 or v2.")
    prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()
    if config.get("system_prompt_sha256") != prompt_hash or config.get("prompt_version") != prompt_version:
        raise ValueError("Adapter training prompt differs from current code; use its matching checkout.")
    if (run_path / "system_prompt.txt").read_text() != system_prompt:
        raise ValueError("Saved training system_prompt.txt differs from the current prompt.")
    base_hash = sha256(model_path / "model.safetensors")
    if base_hash != config.get("model_sha256"):
        raise ValueError("Base model SHA256 differs from training; select the original local model directory.")
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Select a local causal-language LoRA adapter produced by this training script.")
    metadata = {
        "model": "microduck-plan-qlora" if protocol == "v2" else "microduck-qlora",
        "robot_finetuned": True,
        "adapter": str(adapter_path),
        "adapter_sha256": sha256(adapter_path / "adapter_model.safetensors"),
        "base": str(model_path),
        "base_model_sha256": base_hash,
        "base_model_revision": config["model_revision"],
        "prompt_version": prompt_version,
        "system_prompt_sha256": prompt_hash,
        "generation_mode": "unconstrained_greedy_client_validated",
        "schema_constrained_decoding": False,
        "training": "robot simulation QLoRA adapter, frozen inference only",
    }
    return config, metadata


class AdapterEngine:
    def __init__(self, args, config, metadata):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this NF4 adapter service; activate the training environment.")
        torch.set_num_threads(6)
        self.lock = threading.Lock()
        self.timeout = args.generation_timeout
        self.max_length = int(config["max_length"])
        self.metadata = metadata
        self.processor = AutoProcessor.from_pretrained(
            args.model, local_files_only=True, min_pixels=65536, max_pixels=int(config["max_pixels"])
        )
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["visual", "lm_head"],
            ),
        )
        # Match training's fp32 norms, without installing any training hooks.
        for name, parameter in base.named_parameters():
            if "norm" in name and parameter.dtype == torch.bfloat16:
                parameter.data = parameter.data.float()
        self.model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.config.use_cache = True

    def generate(self, raw):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        started = time.monotonic()
        protocol = "v1" if self.metadata["prompt_version"] == PROMPT_VERSION else "v2"
        messages, images, max_tokens = decode_request(raw, self.metadata["model"], protocol=protocol)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=images, return_tensors="pt", padding=False)
        length = batch["input_ids"].shape[1]
        if length + max_tokens > self.max_length:
            raise ValueError(f"Input plus output exceeds {self.max_length} tokens; shorten history/instruction.")
        batch = {key: value.to("cuda") for key, value in batch.items()}
        deadline = started + self.timeout

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores):
                return time.monotonic() >= deadline

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = self.model.generate(
                **batch,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([Deadline()]),
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Generation deadline reached; no action returned. Increase timeout or shorten the input."
            )
        output = generated[0, length:]
        text = self.processor.tokenizer.decode(output, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        finished = len(output) > 0 and int(output[-1]) in eos
        return {
            "id": f"microduck-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.metadata["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop" if finished else "length",
                }
            ],
            "usage": {"prompt_tokens": length, "completion_tokens": len(output), "total_tokens": length + len(output)},
            "microduck_model_metadata": dict(self.metadata),
        }


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def reply(self, status, body):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(data)
            except (TimeoutError, BrokenPipeError, ConnectionResetError):
                pass

        def allowed(self):
            port = self.server.server_port
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self.reply(403, {"error": "Use the numeric loopback host and this service port."})
                return False
            if self.headers.get("Origin") is not None:
                self.reply(403, {"error": "Browser-origin calls are refused; use the local simulation CLI."})
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            if self.path != "/health":
                self.reply(404, {"error": "Use GET /health or POST /v1/chat/completions."})
                return
            self.reply(200, {"status": "ok", "busy": engine.lock.locked(), **engine.metadata})

        def do_POST(self):
            if not self.allowed():
                return
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": "Use POST /v1/chat/completions."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_BODY_BYTES:
                    self.reply(413, {"error": "Send a Content-Length body of at most 8 MiB; chunking is refused."})
                    return
                if self.headers.get_content_type() != "application/json":
                    self.reply(415, {"error": "Send Content-Type: application/json."})
                    return
                if not engine.lock.acquire(blocking=False):
                    self.reply(503, {"error": "GPU is busy; retry after the current simulation request completes."})
                    return
                try:
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("Incomplete request body; resend the JSON request.")
                    result = engine.generate(raw)
                    self.reply(200, result)
                finally:
                    engine.lock.release()
            except (ValueError, TypeError, UnicodeError, KeyError) as error:
                self.reply(400, {"error": str(error)})
            except TimeoutError as error:
                self.reply(504, {"error": str(error)})
            except Exception as error:
                print(f"Inference error: {type(error).__name__}: {error}", flush=True)
                self.reply(500, {"error": "Inference failed; inspect the service terminal before retrying."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/microduck_train/model")
    parser.add_argument(
        "--adapter", type=Path, default=ROOT / "experiments/microduck_tasks/outputs/qlora_20261114/final_adapter"
    )
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v1")
    parser.add_argument("--generation-timeout", type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.generation_timeout <= 300:
        parser.error("Use port 1..65535 and generation-timeout 1..300 seconds.")
    args.model, args.adapter = args.model.resolve(), args.adapter.resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    config, metadata = inspect_files(args.model, args.adapter, protocol=args.protocol)
    print("Verified local inference weights:\n" + json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    # Bind before loading CUDA so a conflicting port does not allocate another model.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    server.daemon_threads = True
    try:
        engine = AdapterEngine(args, config, metadata)
        server.RequestHandlerClass = handler_for(engine)
        print(f"Ready: http://127.0.0.1:{args.port}/health (Ctrl+C to stop)", flush=True)
        print(
            "Greedy generation; JSON grammar is NOT constrained. The robot client validates all proposals.", flush=True
        )
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAdapter service stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
