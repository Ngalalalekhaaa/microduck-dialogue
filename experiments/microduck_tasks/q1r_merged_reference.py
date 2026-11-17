"""Measure the effective floating Q1R merge independently of NF4 and ONNX.

Load only language weights, preserving their saved BF16/FP32 dtypes. Frozen input
tokens are used directly; no vision model, prompt rewriting, or persistent cache.
"""

import argparse
import gc
import json
import struct
import time
from importlib.metadata import version
from pathlib import Path

from .q1r_onnx_benchmark import digest, load_inputs, save

BACKEND = "q1r_effective_merged_bf16_pytorch"
MAX_NEW_TOKENS = 1536
MAX_TIME_SECONDS = 180.0
EOS_TOKEN_IDS = [248046, 248044]
SOURCE_PREFIX = "model.language_model."


def read_header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))


def memory_requirement(header, config, maximum_tokens):
    """Count saved language tensors once, including the tied embedding/head."""
    weights = sum(
        item["data_offsets"][1] - item["data_offsets"][0]
        for name, item in header.items()
        if name.startswith(SOURCE_PREFIX)
    )
    full_layers = config["layer_types"].count("full_attention")
    kv = full_layers * 2 * config["num_key_value_heads"] * config["head_dim"] * maximum_tokens * 2
    return dict(
        weight_bytes=weights,
        maximum_kv_bytes=kv,
        workspace_reserve_bytes=2**30,
        required_free_bytes=weights + kv + 2**30,
    )


def mapped_state(source):
    names = list(source.keys())
    unexpected = [name for name in names if not name.startswith(SOURCE_PREFIX)]
    if unexpected:
        raise ValueError(f"Use the effective text-only merge; unexpected tensors: {unexpected[:5]}")
    state = {"model." + name.removeprefix(SOURCE_PREFIX): source.get_tensor(name) for name in names}
    if "model.embed_tokens.weight" not in state:
        raise ValueError("The merged text checkpoint is missing its tied word embedding.")
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    return state


def load_language(model_path, torch):
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    config_data = json.loads((model_path / "config.json").read_text())["text_config"]
    if config_data.get("tie_word_embeddings") is not True:
        raise ValueError("This reference requires the selected Q1R model's tied word embeddings.")
    config = Qwen3_5TextConfig.from_dict(config_data)
    config._attn_implementation = "sdpa"
    config.dtype = torch.bfloat16
    config.use_cache = True
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    # Nonpersistent RoPE buffers are absent from state_dict and cannot stay meta.
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device="cpu")
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as source:
        state = mapped_state(source)
        expected_dtypes = {name: tensor.dtype for name, tensor in state.items()}
        loaded = model.load_state_dict(state, strict=True, assign=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError(f"The floating language checkpoint did not load completely: {loaded}")
        model.tie_weights()
        del state
    if any(tensor.is_meta for tensor in [*model.parameters(), *model.buffers()]):
        raise ValueError("A language parameter or buffer remained on meta; do not benchmark an incomplete load.")
    model.requires_grad_(False)
    model.eval()
    model.to(device="cuda")
    model.tie_weights()
    actual = model.state_dict()
    if any(actual[name].dtype != dtype for name, dtype in expected_dtypes.items()):
        raise ValueError("Loading changed a saved BF16/FP32 dtype; preserve the floating reference precision.")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("The language output head must share the embedding weight on CUDA.")
    linear_layers = []
    for index, layer in enumerate(model.model.layers):
        if hasattr(layer, "linear_attn"):
            linear = layer.linear_attn
            if not linear.chunk_gated_delta_rule.__module__.startswith("fla."):
                raise RuntimeError("Use the existing FLA backend; refusing a silent pure-Torch fallback.")
            if not linear.recurrent_gated_delta_rule.__module__.startswith("fla."):
                raise RuntimeError("The FLA recurrent decode backend is required for this reference.")
            linear.causal_conv1d_fn = None
            linear_layers.append(index)
    gc.collect()
    torch.cuda.synchronize()
    return model, dict(
        strict_state_dict_load=True,
        assign=True,
        tied_embedding_and_head=True,
        saved_dtypes_preserved=True,
        linear_attention_layers=linear_layers,
        attention_implementation="sdpa",
        chunk_backend=model.model.layers[linear_layers[0]].linear_attn.chunk_gated_delta_rule.__module__,
        recurrent_backend=model.model.layers[linear_layers[0]].linear_attn.recurrent_gated_delta_rule.__module__,
        convolution_backend="torch",
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
    )


def generate(model, tokenizer, torch, row):
    prefix = row["prefix_ids"]
    ids = torch.tensor([prefix], dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            past_key_values=None,
            do_sample=False,
            use_cache=True,
            logits_to_keep=1,
            max_new_tokens=MAX_NEW_TOKENS,
            max_time=MAX_TIME_SECONDS,
            eos_token_id=EOS_TOKEN_IDS,
            pad_token_id=248044,
        )
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    if result.shape[0] != 1 or not torch.equal(result[0, : len(prefix)], ids[0]):
        raise RuntimeError("Generated sequence changed the frozen input tokens.")
    tokens = result[0, len(prefix) :].tolist()
    eos = bool(tokens and tokens[-1] in EOS_TOKEN_IDS)
    return dict(
        raw_text=tokenizer.decode(tokens, skip_special_tokens=True),
        generated_ids=tokens,
        timing=dict(
            seconds=seconds,
            input_tokens=len(prefix),
            output_tokens=len(tokens),
            finish_reason="eos" if eos else "limit",
            limit_reason=None if eos else "max_new_tokens" if len(tokens) >= MAX_NEW_TOKENS else "max_time",
        ),
    )


def benchmark(args):
    import torch
    from transformers import AutoTokenizer

    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    inputs, model_path, output = args.inputs.resolve(), args.model.resolve(), args.output.resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Choose a new output filename; existing reference evidence is never overwritten.")
    rows = load_inputs(inputs)
    config_path, weights_path = model_path / "config.json", model_path / "model.safetensors"
    config = json.loads(config_path.read_text())["text_config"]
    maximum = max(len(row["prefix_ids"]) for row in rows) + MAX_NEW_TOKENS
    if maximum > config["max_position_embeddings"]:
        raise ValueError("Frozen prompts plus 1536 output tokens exceed this merged checkpoint's context length.")
    if any(token >= config["vocab_size"] for row in rows for token in row["prefix_ids"]):
        raise ValueError("A frozen input token is outside the checkpoint vocabulary.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This floating reference requires a CUDA device with BF16 support.")
    free, total = torch.cuda.mem_get_info()
    memory = dict(free_bytes=free, total_bytes=total, **memory_requirement(read_header(weights_path), config, maximum))
    if free < memory["required_free_bytes"]:
        raise RuntimeError(
            f"Insufficient free CUDA memory: {free / 2**30:.2f} GiB available; "
            f"need at least {memory['required_free_bytes'] / 2**30:.2f} GiB for weights, KV and workspace. "
            "Finish other GPU jobs before rerunning; this script does not stop them."
        )
    if version("fla-core") != "0.4.1":
        raise ValueError("Use the current fla-core==0.4.1 reference environment.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if [tokenizer.eos_token_id, tokenizer.pad_token_id] != EOS_TOKEN_IDS:
        raise ValueError("Tokenizer EOS/pad IDs differ from the frozen Q1R generation contract.")
    torch.set_num_threads(4)
    metadata = dict(
        status="loading",
        backend=BACKEND,
        model=str(model_path),
        weights_sha256=digest(weights_path),
        config_sha256=digest(config_path),
        inputs=str(inputs),
        inputs_sha256=digest(inputs),
        tokenizer_sha256=digest(model_path / "tokenizer.json"),
        script_sha256=digest(Path(__file__)),
        source="Effective floating merge of NF4-dequantized base plus checkpoint513 LoRA; not NF4 inference",
        compute_dtype="BF16 autocast; preserve all saved FP32 parameters",
        quantized_runtime=False,
        visual_loaded=False,
        mtp_loaded=False,
        physical_executions=0,
        prefix_cache_shared=False,
        fresh_cache_per_request=True,
        eos_token_ids=EOS_TOKEN_IDS,
        pad_token_id=248044,
        max_new_tokens=MAX_NEW_TOKENS,
        max_time_seconds=MAX_TIME_SECONDS,
        do_sample=False,
        warmup_count=1,
        repeats=args.repeats,
        case_count=len(rows),
        memory_preflight=memory,
        versions={name: version(name) for name in ("torch", "transformers", "safetensors", "fla-core")},
        timing_scope="CUDA-synchronized model.generate; excludes loading, input tokenization and final text decoding",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save(metadata_path, metadata)
    try:
        started = time.monotonic()
        model, load_info = load_language(model_path, torch)
        metadata.update(status="warming_up", load_seconds=time.monotonic() - started, load_info=load_info)
        save(metadata_path, metadata)
        warmup = generate(model, tokenizer, torch, rows[0])
        metadata.update(status="running", warmup_case=rows[0]["id"], warmup_timing=warmup["timing"])
        save(metadata_path, metadata)
        completed = 0
        with output.open("x") as stream:
            for repeat in range(args.repeats):
                for row in rows:
                    prediction = generate(model, tokenizer, torch, row)
                    record = dict(id=row["id"], repeat=repeat, backend=BACKEND, **prediction)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    completed += 1
                    print(json.dumps(dict(id=row["id"], repeat=repeat, **prediction["timing"])), flush=True)
        metadata.update(status="complete", measured_records=completed, output_sha256=digest(output))
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(metadata_path, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL filename; metadata uses .metadata.json.")
    parser.add_argument("--repeats", type=int, default=1)
    benchmark(parser.parse_args())
