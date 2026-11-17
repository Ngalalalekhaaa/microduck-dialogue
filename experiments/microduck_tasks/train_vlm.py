"""Local, resumable QLoRA training on reviewed simulation demonstrations.

Only assistant JSON tokens contribute to loss. The entire image/instruction/state
context remains in attention; logits are evaluated only at supervised positions
to avoid allocating a full context-by-vocabulary matrix on a 6 GiB GPU.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

from experiments.microduck_tasks.skill_vlm import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    _input_context,
    _input_state,
    parse_proposal,
)

MODEL_REPO = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MODEL_SHA256 = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
MIN_FREE_BYTES = 8 * 1024**3
ACTIVE_OUTPUT = None
OUTPUT_LOCK = None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def enough_disk(path):
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"Only {free / 1024**3:.1f} GiB free; keep at least 8 GiB before continuing training.")


def read_records(path):
    path = Path(path).resolve()
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        v2 = row.get("format_version") == "microduck-plan-sft-2"
        if v2:
            from .plan_protocol import SYSTEM, parse_plan, validate_input

            expected_prompt, parse = SYSTEM, parse_plan
        else:
            expected_prompt, parse = SYSTEM_PROMPT, parse_proposal
        if "messages" in row and row["messages"][0] != {"role": "system", "content": expected_prompt}:
            raise ValueError(f"{path}:{number} was exported with a different system prompt; re-export consistently.")
        if v2 and row.get("system_prompt") != expected_prompt:
            raise ValueError("Re-export v2 samples with the current v2 prompt.")
        target = row.get("target", row.get("label"))
        parse(json.dumps(target, ensure_ascii=False, allow_nan=False))
        if v2:
            validate_input(row["state"], row["context"])
        else:
            row["state"] = _input_state(row["state"])
            row["context"] = _input_context(row["context"])
        if set(row["images"]) != {"front", "near"}:
            raise ValueError(f"{path}:{number} requires front and near RGB images.")
        if not isinstance(row["instruction"], str) or not row["instruction"].strip():
            raise ValueError(f"{path}:{number} requires the original instruction.")
        for image_path in row["images"].values():
            image_path = Path(image_path)
            resolved = (path.parent / image_path).resolve()
            if image_path.is_absolute() or not resolved.is_relative_to(path.parent) or not resolved.is_file():
                raise ValueError(f"{path}:{number} image must exist inside the dataset directory.")
        row["target"] = target
        records.append(row)
    if not records:
        raise ValueError(f"{path} contains no training examples.")
    return records


def messages_for(row):
    content = []
    for camera in ("front", "near"):
        content.extend([{"type": "text", "text": f"Current camera: {camera}"}, {"type": "image"}])
    payload = {"instruction": row["instruction"], "state": row["state"], "context": row["context"]}
    content.append({"type": "text", "text": json.dumps(payload, ensure_ascii=False, allow_nan=False)})
    prompt = SYSTEM_PROMPT
    if row.get("format_version") == "microduck-plan-sft-2":
        from .plan_protocol import SYSTEM

        prompt = SYSTEM
    return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]


def encode(row, root, processor, max_length):
    from PIL import Image

    images = []
    for camera in ("front", "near"):
        with Image.open(root / row["images"][camera]) as image:
            images.append(image.convert("RGB"))
    messages = messages_for(row)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": json.dumps(row["target"], ensure_ascii=False, allow_nan=False)})
    full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # No truncation: silently dropping a camera or history would change the task.
    batch = processor(text=[full], images=images, return_tensors="pt", padding=False)
    prompt_ids = processor(text=[prompt], images=images, return_tensors="pt", padding=False)["input_ids"]
    prompt_length = prompt_ids.shape[1]
    length = batch["input_ids"].shape[1]
    if length > max_length:
        raise ValueError(
            f"Sample {row.get('id', '?')} has {length} tokens, above max_length={max_length}; no truncation."
        )
    if not (batch["input_ids"][:, :prompt_length] == prompt_ids).all().item() or prompt_length >= length:
        raise ValueError("Assistant token boundary does not match the chat template.")
    return dict(batch), prompt_length


def load_model(model_dir, rank, resume=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    # PEFT casts every non-quantized weight to fp32. Frozen embeddings and vision
    # weights need no fp32 optimizer state, so keep them bf16 for this small GPU.
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and parameter.dtype == torch.float32 and "norm" not in name:
            parameter.data = parameter.data.to(torch.bfloat16)
    targets = [
        name
        for name, module in model.named_modules()
        if ".language_model.layers." in name
        and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ]
    if not targets:
        raise RuntimeError("No Qwen language layers matched; inspect the pinned model before training.")
    if resume:
        model = PeftModel.from_pretrained(model, Path(resume) / "adapter", is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            ),
        )
    if any(p.requires_grad for name, p in model.named_parameters() if ".visual." in name):
        raise RuntimeError("Visual encoder/projectors must remain frozen in this experiment.")
    return model


def assistant_loss(model, batch, prompt_length):
    import torch
    import torch.nn.functional as functional

    batch = {key: value.to("cuda") for key, value in batch.items()}
    labels = batch["input_ids"][:, prompt_length:]
    positions = torch.arange(prompt_length - 1, batch["input_ids"].shape[1] - 1, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**batch, logits_to_keep=positions, use_cache=False)
        logits = outputs.logits
    if logits.shape[:2] != labels.shape:
        raise RuntimeError("Selected logits and assistant labels are misaligned.")
    return functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))


def validation_indices(records, limit):
    """Deterministically rotate between skill/type classes, including rare kicks."""
    groups = {}
    for index, row in enumerate(records):
        target = row["target"]
        category = target["skill"] if target["type"] == "skill" else target["type"]
        groups.setdefault(category, []).append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: hashlib.sha256(str(records[index].get("sample_id", index)).encode()).digest())
    selected = []
    offset = 0
    count = min(len(records), limit) if limit else len(records)
    while len(selected) < count:
        for category in sorted(groups):
            indices = groups[category]
            if offset < len(indices) and len(selected) < count:
                selected.append(indices[offset])
        offset += 1
    return selected


def evaluate(model, records, root, processor, max_length, limit):
    import torch

    was_training = model.training
    model.eval()
    total = 0.0
    indices = validation_indices(records, limit)
    count = len(indices)
    with torch.no_grad():
        for index in indices:
            batch, boundary = encode(records[index], root, processor, max_length)
            total += float(assistant_loss(model, batch, boundary))
    model.train(was_training)
    return total / count


def predict_examples(model, records, root, processor, max_length, limit, destination):
    """Generate unforced JSON for a fixed validation subset, keeping every raw reply."""
    import torch

    was_training = model.training
    model.eval()
    indices = validation_indices(records, limit)
    count = len(indices)
    v2 = records[0].get("format_version") == "microduck-plan-sft-2"
    parse = parse_proposal
    if v2:
        from .plan_protocol import parse_plan

        parse = parse_plan
    correct = {
        key: 0
        for key in (
            ("status", "target", "steps", "joint") if v2 else ("type", "skill", "target_color", "seconds", "joint")
        )
    }
    valid = 0
    by_category = {}
    with destination.open("w") as stream, torch.no_grad():
        for index in indices:
            row = records[index]
            target = row["target"]
            category = target["status"] if v2 else target["skill"] if target["type"] == "skill" else target["type"]
            category_result = by_category.setdefault(category, {"count": 0, "valid_json": 0, "joint_correct": 0})
            category_result["count"] += 1
            batch, boundary = encode(row, root, processor, max_length)
            batch["input_ids"] = batch["input_ids"][:, :boundary]
            batch["attention_mask"] = batch["attention_mask"][:, :boundary]
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**batch, max_new_tokens=512 if v2 else 192, do_sample=False, use_cache=True)
            raw = processor.tokenizer.decode(generated[0, boundary:], skip_special_tokens=True)
            record = {"index": index, "id": row.get("sample_id", row.get("id")), "target": row["target"], "raw": raw}
            try:
                prediction = parse(raw).to_dict()
            except (ValueError, TypeError) as error:
                record["error"] = str(error)
            else:
                valid += 1
                category_result["valid_json"] += 1
                record["prediction"] = prediction
                matches = {key: prediction[key] == row["target"][key] for key in correct if key != "joint"}
                for key, match in matches.items():
                    correct[key] += int(match)
                correct["joint"] += int(all(matches.values()))
                category_result["joint_correct"] += int(all(matches.values()))
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    model.train(was_training)
    return {
        "count": count,
        "selection": "deterministic_round_robin_skill_or_response_type",
        "indices": indices,
        "by_category": by_category,
        "valid_json_rate": valid / count,
        **{f"{key}_accuracy": value / count for key, value in correct.items()},
    }


def save_checkpoint(out, model, optimizer, scheduler, state, keep):
    import torch

    enough_disk(out)
    destination = out / f"checkpoint-{state['step']:06d}"
    if destination.exists():
        raise FileExistsError("This checkpoint already exists; resume the latest checkpoint instead of overwriting it.")
    temporary = Path(tempfile.mkdtemp(prefix=f".checkpoint-{state['step']:06d}-", suffix=".tmp", dir=out))
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    torch.save(
        {
            **state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
        },
        temporary / "training_state.pt",
    )
    write_json(temporary / "progress.json", state)
    temporary.replace(destination)
    write_json(out / "latest_checkpoint.json", {"path": str(destination.resolve()), **state})
    owned = []
    for path in out.glob("checkpoint-*"):
        progress = path / "progress.json"
        if path.name.removeprefix("checkpoint-").isdigit() and progress.is_file():
            if json.loads(progress.read_text()).get("run_config_digest") == state["run_config_digest"]:
                owned.append(path)
    older = sorted(owned)[:-keep]
    for path in older:
        shutil.rmtree(path)
    return destination


def main():
    global ACTIVE_OUTPUT, OUTPUT_LOCK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_train/model"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=98304)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--predict-limit", type=int, default=0, help="Save unforced generation on fixed validation examples."
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--resource-smoke", action="store_true", help="Mark synthetic/repeated resource-test data explicitly."
    )
    args = parser.parse_args()
    if (
        min(
            args.epochs,
            args.max_steps,
            args.grad_accum,
            args.rank,
            args.save_every,
            args.eval_every,
            args.keep_checkpoints,
        )
        < 1
    ):
        parser.error("Epochs, steps, accumulation, rank and checkpoint intervals must be positive.")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup

    torch.set_num_threads(6)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Linux advisory lock prevents two invocations sharing one run directory.
    import fcntl

    OUTPUT_LOCK = (args.output / ".training.lock").open("a")
    try:
        fcntl.flock(OUTPUT_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(
            "This output directory has an active run; wait for it or choose a different output."
        ) from None
    ACTIVE_OUTPUT = args.output
    write_json(args.output / "run_status.json", {"status": "running", "phase": "preflight", "pid": os.getpid()})
    enough_disk(args.output)
    train_path, val_path = args.dataset / "train.jsonl", args.dataset / "val.jsonl"
    train_records, val_records = read_records(train_path), read_records(val_path)
    protocols = {r.get("format_version") == "microduck-plan-sft-2" for r in train_records + val_records}
    if len(protocols) != 1:
        raise ValueError("Do not mix v1 and v2 examples in one training run.")
    system_prompt, prompt_version = SYSTEM_PROMPT, PROMPT_VERSION
    if True in protocols:
        from .plan_protocol import SYSTEM, VERSION

        system_prompt, prompt_version = SYSTEM, VERSION
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, min_pixels=65536, max_pixels=args.max_pixels
    )
    lengths = []
    maximum = (0, None)
    maximum_train = (0, None)
    for index, row in enumerate(train_records + val_records):
        batch, boundary = encode(row, args.dataset, processor, args.max_length)
        length = batch["input_ids"].shape[1]
        lengths.append(length)
        if length > maximum[0]:
            maximum = (length, index)
        if index < len(train_records) and length > maximum_train[0]:
            maximum_train = (length, index)
    lengths.sort()
    preflight = {
        "train_examples": len(train_records),
        "validation_examples": len(val_records),
        "min_tokens": lengths[0],
        "median_tokens": lengths[len(lengths) // 2],
        "max_tokens": maximum[0],
        "longest_combined_index": maximum[1],
        "longest_train_index": maximum_train[1],
        "system_tokens": len(processor.tokenizer(system_prompt).input_ids),
        "truncation": False,
    }
    write_json(args.output / "preflight.json", preflight)
    print(json.dumps({"preflight": preflight}), flush=True)
    if args.preflight_only:
        write_json(args.output / "run_status.json", {"status": "completed", "phase": "preflight", "pid": os.getpid()})
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This QLoRA configuration needs a CUDA GPU supporting bf16.")
    if digest(args.model / "model.safetensors") != MODEL_SHA256:
        raise RuntimeError("Official base model SHA256 mismatch; restore the pinned model before training.")
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "prompt_version": prompt_version,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "train_sha256": digest(train_path),
        "val_sha256": digest(val_path),
        "dataset_manifest_sha256": digest(args.dataset / "manifest.json")
        if (args.dataset / "manifest.json").exists()
        else None,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "loss": "assistant_json_only_full_context_selected_logits",
        "vision_encoder_and_projectors_frozen": True,
        "base_quantization": "nf4_double_quant_bf16_compute_language_only",
    }
    existing_config = args.output / "training_config.json"
    if args.resume:
        args.resume = args.resume.resolve()
        if args.resume.parent != args.output or not args.resume.name.startswith("checkpoint-"):
            raise ValueError("Resume a checkpoint belonging to this output directory, not another training run.")
        previous = json.loads(existing_config.read_text())
        for key in (
            "model_sha256",
            "train_sha256",
            "val_sha256",
            "dataset_manifest_sha256",
            "system_prompt_sha256",
            "rank",
            "grad_accum",
            "seed",
            "epochs",
            "max_steps",
            "learning_rate",
            "max_length",
            "max_pixels",
            "resource_smoke",
            "torch_version",
        ):
            if previous[key] != config[key]:
                raise ValueError(f"Resume configuration changed {key}; use the original data/configuration.")
        write_json(args.output / f"resume_invocation_{time.time_ns()}.json", config)
    elif existing_config.exists():
        raise FileExistsError("Output already contains training_config.json; use --resume or a new output directory.")
    else:
        write_json(existing_config, config)
    (args.output / "system_prompt.txt").write_text(system_prompt)
    write_json(args.output / "run_status.json", {"status": "running", "phase": "model_loading", "pid": os.getpid()})
    model = load_model(args.model, args.rank, args.resume)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable = sum(parameter.numel() for parameter in parameters)
    print(
        json.dumps({"trainable_parameters": trainable, "allocated_mib": torch.cuda.memory_allocated() / 1024**2}),
        flush=True,
    )
    write_json(
        args.output / "model_summary.json",
        {
            "trainable_parameters": trainable,
            "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
            "trainable_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        },
    )
    total_micro = args.epochs * len(train_records)
    total_steps = min(args.max_steps, math.ceil(total_micro / args.grad_accum))
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(total_steps * 0.05)), total_steps)
    state = {
        "step": 0,
        "micro_cursor": 0,
        "best_val_loss": None,
        "total_steps": total_steps,
        "run_config_digest": digest(existing_config),
    }
    if args.resume:
        restored = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        if restored.get("run_config_digest") != state["run_config_digest"]:
            raise ValueError("Checkpoint configuration fingerprint does not match this training run.")
        if (
            restored["total_steps"] != total_steps
            or not 0 <= restored["step"] <= total_steps
            or restored["micro_cursor"] != min(restored["step"] * args.grad_accum, total_micro)
        ):
            raise ValueError("Checkpoint step/cursor does not match the deterministic data order.")
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        torch.set_rng_state(restored["torch_rng"])
        torch.cuda.set_rng_state_all(restored["cuda_rng"])
        random.setstate(restored["python_rng"])
        state.update({key: restored[key] for key in state})
    if args.eval_only:
        loss = evaluate(model, val_records, args.dataset, processor, args.max_length, args.eval_limit)
        evaluation = {"validation_loss": loss, "checkpoint": str(args.resume)}
        if args.predict_limit > 0:
            evaluation["generation"] = predict_examples(
                model,
                val_records,
                args.dataset,
                processor,
                args.max_length,
                args.predict_limit,
                args.output / "predictions.jsonl",
            )
        write_json(args.output / "evaluation.json", evaluation)
        print(json.dumps(evaluation), flush=True)
        write_json(args.output / "run_status.json", {"status": "completed", "phase": "evaluation", "pid": os.getpid()})
        return
    if not args.resume:
        write_json(args.output / "run_status.json", {"status": "running", "phase": "baseline", "pid": os.getpid()})
        baseline = {
            "validation_loss": evaluate(model, val_records, args.dataset, processor, args.max_length, args.eval_limit),
            "adapter_updates": 0,
        }
        if args.predict_limit > 0:
            baseline["generation"] = predict_examples(
                model,
                val_records,
                args.dataset,
                processor,
                args.max_length,
                args.predict_limit,
                args.output / "baseline_predictions.jsonl",
            )
        write_json(args.output / "baseline_evaluation.json", baseline)
        print(json.dumps({"baseline": baseline}), flush=True)
    # Deterministic per-epoch permutation makes the cursor exactly resumable.
    order = []
    for epoch in range(args.epochs):
        shuffled = list(range(len(train_records)))
        random.Random(args.seed + epoch).shuffle(shuffled)
        order.extend(shuffled)
    if args.resource_smoke:
        # Explicitly exercise the longest real context before declaring GPU fit.
        position = order.index(maximum_train[1])
        order[0], order[position] = order[position], order[0]
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    log = (args.output / "metrics.jsonl").open("a", buffering=1)
    interrupted = False
    failed = None
    try:
        while state["step"] < total_steps and state["micro_cursor"] < len(order):
            enough_disk(args.output)
            remaining = len(order) - state["micro_cursor"]
            accumulation = min(args.grad_accum, remaining)
            losses = []
            sequence_lengths = []
            for _ in range(accumulation):
                row = train_records[order[state["micro_cursor"]]]
                batch, boundary = encode(row, args.dataset, processor, args.max_length)
                sequence_lengths.append(batch["input_ids"].shape[1])
                loss = assistant_loss(model, batch, boundary)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss became non-finite; inspect data and the last checkpoint.")
                losses.append(float(loss.detach()))
                (loss / accumulation).backward()
                state["micro_cursor"] += 1
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            metric = {
                "step": state["step"],
                "micro_cursor": state["micro_cursor"],
                "train_loss": sum(losses) / len(losses),
                "sequence_tokens_max": max(sequence_lengths),
                "grad_norm": norm,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_s": time.monotonic() - start,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
            if state["step"] % args.eval_every == 0 or state["step"] == total_steps:
                metric["validation_loss"] = evaluate(
                    model, val_records, args.dataset, processor, args.max_length, args.eval_limit
                )
                if state["best_val_loss"] is None or metric["validation_loss"] < state["best_val_loss"]:
                    state["best_val_loss"] = metric["validation_loss"]
                    enough_disk(args.output)
                    model.save_pretrained(args.output / "best_adapter", safe_serialization=True)
                    write_json(args.output / "best_adapter" / "selection.json", metric)
            log.write(json.dumps(metric, allow_nan=False) + "\n")
            print(json.dumps(metric, allow_nan=False), flush=True)
            if state["step"] == 10 or state["step"] % args.save_every == 0 or state["step"] == total_steps:
                save_checkpoint(args.output, model, optimizer, scheduler, state, args.keep_checkpoints)
            write_json(
                args.output / "run_status.json",
                {
                    **state,
                    "status": "running",
                    "phase": "training",
                    "pid": os.getpid(),
                    "elapsed_s": time.monotonic() - start,
                    "last_metric": metric,
                },
            )
    except KeyboardInterrupt:
        # Never save partially accumulated gradients as a completed update.
        interrupted = True
        print("Interrupted; resume the last complete checkpoint listed in latest_checkpoint.json.", flush=True)
    except Exception as error:
        failed = f"{type(error).__name__}: {error}"
        raise
    finally:
        log.close()
        write_json(
            args.output / "run_status.json",
            {
                **state,
                "interrupted": interrupted,
                "elapsed_s": time.monotonic() - start,
                "resource_smoke": args.resource_smoke,
                "status": "failed" if failed else "interrupted" if interrupted else "finalizing",
                "phase": "training" if failed or interrupted else "saving_and_generation_evaluation",
                "error": failed,
                "pid": os.getpid(),
            },
        )
    if not interrupted:
        model.save_pretrained(args.output / "final_adapter", safe_serialization=True)
        processor.save_pretrained(args.output / "processor")
        if args.predict_limit > 0:
            generation = predict_examples(
                model,
                val_records,
                args.dataset,
                processor,
                args.max_length,
                args.predict_limit,
                args.output / "predictions.jsonl",
            )
            write_json(args.output / "generation_evaluation.json", generation)
        write_json(
            args.output / "run_status.json",
            {
                **state,
                "status": "completed",
                "phase": "complete",
                "pid": os.getpid(),
                "elapsed_s": time.monotonic() - start,
                "resource_smoke": args.resource_smoke,
            },
        )


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        if ACTIVE_OUTPUT is not None:
            path = ACTIVE_OUTPUT / "run_status.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            write_json(
                path,
                {
                    **status,
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "pid": os.getpid(),
                },
            )
        raise
