"""Local Qwen3-VL inference with ONNX vision, DeepStack and cached decoding.

The HF processor/tokenizer and position-index calculation run on CPU. All neural
network weights come from the ONNX graphs; CUDA KV caches stay on their device.
Only the exported 320 x 240 front/near camera configuration is accepted.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np

from experiments.microduck_tasks.serve_adapter_protocol import decode_request


def causal_mask(sequence_length, past_length):
    """Additive causal mask for one unpadded sequence, including the cached past."""
    query = np.arange(sequence_length)[:, None] + past_length
    keys = np.arange(sequence_length + past_length)[None, :]
    return np.where(keys <= query, np.float16(0), np.float16(-65504))[None, None]


def scatter_images(input_ids, image_token_id, vision_outputs, hidden_size):
    """Keep image order and all three DeepStack feature streams aligned."""
    positions = input_ids == image_token_id
    expected = sum(output[0].shape[0] for output in vision_outputs)
    if input_ids.shape[0] != 1 or int(positions.sum()) != expected:
        raise ValueError("Image token count differs from exported visual features; restore the matching processor.")
    arrays = []
    for stream in range(4):
        features = np.concatenate([output[stream] for output in vision_outputs], axis=0)
        if features.shape != (expected, hidden_size) or not np.isfinite(features).all():
            raise ValueError("Vision graph returned invalid features; regenerate and validate the ONNX export.")
        value = np.zeros((*input_ids.shape, hidden_size), dtype=np.float16)
        value[positions] = features
        arrays.append(value)
    return positions[..., None], arrays


def run_prefill(decoder, inputs, cache, chunk_size, deadline, *, memory_conservative=False):
    """Evaluate the whole prompt causally while bounding attention workspace."""
    length = inputs["input_ids"].shape[1]
    if chunk_size < 1 or length < 1:
        raise ValueError("Prefill needs a nonempty prompt and a positive chunk size.")
    try:
        for start in range(0, length, chunk_size):
            if time.monotonic() >= deadline:
                raise TimeoutError("Generation deadline reached during prefill; no action returned.")
            end = min(start + chunk_size, length)
            chunk = {
                name: value[:, :, start:end] if name == "position_ids" else value[:, start:end]
                for name, value in inputs.items()
                if name != "attention_mask"
            }
            chunk["attention_mask"] = causal_mask(end - start, start)
            # Shrink after prefill, or after each chunk in conservative mode.
            # Live cache OrtValues cannot be reclaimed by the arena.
            token, cache = decoder.run(chunk, cache, shrink_arena=memory_conservative or end == length)
    except BaseException:
        # A traceback can retain this frame after an OOM or timeout. Remove its
        # device tensor references before propagating the failure.
        cache.clear()
        raise
    return token, cache


def shrink_options(device):
    """Return unused arena allocations after this execution completes."""
    import onnxruntime as ort

    options = ort.RunOptions()
    options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0" if device == "cuda" else "cpu:0")
    return options


def run_vision(session, names, pixels, device):
    """Release each camera's temporary attention/convolution workspace promptly."""
    return [
        session.run(names, {"pixel_values": pixels[index : index + 320]}, shrink_options(device)) for index in (0, 320)
    ]


class BoundDecoder:
    """Bind cache outputs to the execution device without a host round trip."""

    def __init__(self, session, device, layers, heads, head_size):
        import onnxruntime as ort

        self.ort = ort
        self.session = session
        self.device = device
        self.cache_names = [f"past_{kind}_{layer}" for layer in range(layers) for kind in ("key", "value")]
        self.present_names = [name.replace("past_", "present_", 1) for name in self.cache_names]
        self.empty_shape = (1, heads, 0, head_size)
        self.expected_inputs = {
            "input_ids",
            "image_mask",
            "image_embeds",
            "position_ids",
            "attention_mask",
            "deepstack_0",
            "deepstack_1",
            "deepstack_2",
            *self.cache_names,
        }
        if {item.name for item in session.get_inputs()} != self.expected_inputs:
            raise ValueError("Decoder ONNX inputs differ from the Microduck cached-decoding contract; re-export it.")
        if {item.name for item in session.get_outputs()} != {"logits", *self.present_names}:
            raise ValueError("Decoder ONNX outputs differ from the Microduck cached-decoding contract; re-export it.")

    def empty_cache(self):
        # A zero-length tensor has no data to transfer. CUDA receives real cache
        # allocations after the first prefill execution.
        return [
            self.ort.OrtValue.ortvalue_from_numpy(np.empty(self.empty_shape, dtype=np.float16))
            for _ in self.cache_names
        ]

    def run(self, inputs, cache, *, shrink_arena=False):
        binding = self.session.io_binding()
        outputs = []
        try:
            for name, value in inputs.items():
                binding.bind_cpu_input(name, np.ascontiguousarray(value))
            for name, value in zip(self.cache_names, cache, strict=True):
                binding.bind_ortvalue_input(name, value)
            binding.bind_output("logits", "cpu")
            for name in self.present_names:
                binding.bind_output(name, self.device, 0)
            options = shrink_options(self.device) if shrink_arena else None
            self.session.run_with_iobinding(binding, options)
            binding.synchronize_outputs()
            outputs = binding.get_outputs()
            logits = outputs[0].numpy()
            if logits.ndim != 2 or logits.shape[0] != 1 or not np.isfinite(logits).all():
                raise ValueError("Decoder returned nonfinite or malformed logits; inspect the validated ONNX build.")
            return int(logits[0].argmax()), outputs[1:]
        except BaseException:
            # Successful output handles can alias ORT's binding storage. Clear
            # bindings only on failure, when no cache escapes to the caller.
            binding.clear_binding_inputs()
            binding.clear_binding_outputs()
            outputs.clear()
            value = None
            raise


class OnnxEngine:
    def __init__(self, args, metadata, decoder_path):
        import onnxruntime as ort
        import torch
        from transformers import AutoConfig, AutoProcessor, GenerationConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        torch.set_num_threads(args.cpu_threads)
        self.lock = threading.Lock()
        self.timeout = args.generation_timeout
        self.prefill_chunk_size = args.prefill_chunk_size
        self.memory_conservative = (
            metadata["decoder_quantization"] in ("float16", "int8_weight_only")
            if args.memory_conservative is None
            else args.memory_conservative
        )
        self.max_length = 4096
        self.metadata = metadata
        self.config = AutoConfig.from_pretrained(args.merged_model, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(
            args.merged_model, local_files_only=True, min_pixels=65536, max_pixels=98304
        )
        self.get_rope_index = Qwen3VLModel.get_rope_index
        self.model_stub = SimpleNamespace(config=self.config)
        self.hidden_size = self.config.text_config.hidden_size
        self.device = args.device
        self.vision_device = args.device if args.vision_device == "auto" else args.vision_device
        if "cuda" in (self.device, self.vision_device):
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA ONNX Runtime is unavailable; use the dedicated microduck_onnx environment.")
            cuda_providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "cudnn_conv_use_max_workspace": "0",
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                "CPUExecutionProvider",
            ]
        providers = cuda_providers if self.device == "cuda" else ["CPUExecutionProvider"]
        vision_providers = cuda_providers if self.vision_device == "cuda" else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.cpu_threads
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.vision = ort.InferenceSession(str(args.onnx_model / "vision.onnx"), options, providers=vision_providers)
        decoder = ort.InferenceSession(str(decoder_path), options, providers=providers)
        for session, device in ((self.vision, self.vision_device), (decoder, self.device)):
            if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    "CUDA initialization failed; inspect the service log. CPU fallback must be explicit."
                )
        if [item.name for item in self.vision.get_inputs()] != ["pixel_values"]:
            raise ValueError("Vision graph input differs from the fixed-camera export contract; re-export it.")
        self.vision_names = ["image_embeds", "deepstack_0", "deepstack_1", "deepstack_2"]
        if {item.name for item in self.vision.get_outputs()} != set(self.vision_names):
            raise ValueError("Vision graph must export merged image features plus three DeepStack feature streams.")
        text = self.config.text_config
        self.decoder = BoundDecoder(
            decoder,
            self.device,
            text.num_hidden_layers,
            text.num_key_value_heads,
            getattr(text, "head_dim", text.hidden_size // text.num_attention_heads),
        )
        eos = GenerationConfig.from_pretrained(args.merged_model, local_files_only=True).eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos or [])
        self.metadata.update(
            {
                "execution_provider": decoder.get_providers()[0],
                "vision_execution_provider": self.vision.get_providers()[0],
                "kv_cache_device": self.device,
                "runtime": f"onnxruntime-{ort.__version__}",
                "camera_grid_thw": [1, 16, 20],
                "prefill_chunk_size": self.prefill_chunk_size,
                "memory_conservative": self.memory_conservative,
                "arena_shrinkage": "every_decoder_run"
                if self.memory_conservative
                else "once_after_prefill_per_request",
                "vision_arena_shrinkage": "every_camera",
            }
        )

    def prepare(self, messages, images, max_tokens):
        if any(image.size != (320, 240) for image in images):
            raise ValueError("This ONNX build accepts exactly two 320x240 RGB camera images; keep simulation defaults.")
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=images, return_tensors="pt", padding=False)
        input_ids = batch["input_ids"].numpy().astype(np.int64, copy=False)
        length = input_ids.shape[1]
        if length + max_tokens > self.max_length:
            raise ValueError(f"Input plus output exceeds {self.max_length} tokens; shorten history/instruction.")
        if batch["image_grid_thw"].tolist() != [[1, 16, 20], [1, 16, 20]]:
            raise ValueError("Processor image grid differs from this fixed-camera ONNX build; restore its processor.")
        positions, delta = self.get_rope_index(
            self.model_stub,
            input_ids=batch["input_ids"],
            image_grid_thw=batch["image_grid_thw"],
            attention_mask=batch.get("attention_mask"),
        )
        pixels = batch["pixel_values"].numpy().astype(np.float16)
        if pixels.shape != (640, 1536) or not np.isfinite(pixels).all():
            raise ValueError("Processor image patches differ from this ONNX build; restore its processor.")
        vision = run_vision(self.vision, self.vision_names, pixels, self.vision_device)
        mask, features = scatter_images(input_ids, self.config.image_token_id, vision, self.hidden_size)
        inputs = {
            "input_ids": input_ids,
            "position_ids": positions.numpy().astype(np.int64),
            "image_mask": mask,
            "image_embeds": features[0],
            **{f"deepstack_{index}": feature for index, feature in enumerate(features[1:])},
        }
        return inputs, length, int(delta.item())

    def decode_tokens(self, inputs, rope_delta, max_tokens, started):
        cache = self.decoder.empty_cache()
        output = []
        past_length = 0
        zero_features = np.zeros((1, 1, self.hidden_size), dtype=np.float16)
        try:
            for _ in range(max_tokens):
                if time.monotonic() - started >= self.timeout:
                    raise TimeoutError(
                        "Generation deadline reached; no action returned. Increase timeout or shorten input."
                    )
                if not output:
                    token, cache = run_prefill(
                        self.decoder,
                        inputs,
                        cache,
                        self.prefill_chunk_size,
                        started + self.timeout,
                        memory_conservative=self.memory_conservative,
                    )
                else:
                    token, cache = self.decoder.run(inputs, cache, shrink_arena=self.memory_conservative)
                output.append(token)
                past_length += inputs["input_ids"].shape[1]
                if token in self.eos:
                    break
                inputs = {
                    "input_ids": np.array([[token]], dtype=np.int64),
                    "position_ids": np.full((3, 1, 1), past_length + rope_delta, dtype=np.int64),
                    "image_mask": np.zeros((1, 1, 1), dtype=bool),
                    "image_embeds": zero_features,
                    "attention_mask": causal_mask(1, past_length),
                    **{f"deepstack_{index}": zero_features for index in range(3)},
                }
            if time.monotonic() - started >= self.timeout:
                raise TimeoutError(
                    "Generation deadline reached; no action returned. Increase timeout or shorten input."
                )
            return output
        finally:
            cache.clear()

    def generate(self, raw):
        started = time.monotonic()
        messages, images, max_tokens = decode_request(raw, self.metadata["model"])
        inputs, length, rope_delta = self.prepare(messages, images, max_tokens)
        output = self.decode_tokens(inputs, rope_delta, max_tokens, started)
        text = self.processor.tokenizer.decode(output, skip_special_tokens=True)
        return {
            "id": f"microduck-onnx-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.metadata["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop" if output[-1] in self.eos else "length",
                }
            ],
            "usage": {"prompt_tokens": length, "completion_tokens": len(output), "total_tokens": length + len(output)},
            "microduck_model_metadata": dict(self.metadata),
            "microduck_timing": {"total_seconds": time.monotonic() - started},
        }
