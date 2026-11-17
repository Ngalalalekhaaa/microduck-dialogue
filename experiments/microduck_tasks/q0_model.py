"""Pinned language-only LoRA and complete answer loss without prompt logits."""

import hashlib
import time
from importlib.metadata import version

import torch

from .head_features import FrozenFeatures
from .run_ablation import save


def memory():
    torch.cuda.synchronize()
    return dict(
        allocated_mib=torch.cuda.memory_allocated() / 2**20,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        reserved_mib=torch.cuda.memory_reserved() / 2**20,
    )


def load_model(model_dir, output):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if version("peft") != "0.19.1":
        raise ValueError("Use the isolated peft==0.19.1 overlay.")
    torch.manual_seed(17)
    loaded = FrozenFeatures(model_dir)
    stages = dict(after_base_load=memory())
    base = prepare_model_for_kbit_training(
        loaded.model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Keep the original frozen large embedding/visual weights in BF16; norms stay FP32.
    # Quantized weights remain NF4. Only the new LoRA matrices will be trainable.
    for parameter in base.parameters():
        if parameter.dtype == torch.float32 and parameter.ndim >= 2 and not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    targets = [
        name
        for name, module in base.named_modules()
        if name.startswith("model.language_model.layers.") and isinstance(module, torch.nn.Linear)
    ]
    if not targets or not any("linear_attn" in n for n in targets) or not any("self_attn" in n for n in targets):
        raise ValueError("LoRA must cover both kinds of language attention plus language MLPs.")
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n or ".language_model.layers." not in n for n in trainable):
        raise ValueError("Only language-layer LoRA weights may update.")
    model.config.use_cache = False
    stages["after_lora_preparation"] = memory()
    metadata = dict(
        peft=version("peft"),
        rank=8,
        alpha=16,
        dropout=0.05,
        targets=targets,
        trainable_parameters=sum(p.numel() for p in trainable.values()),
        trainable_names=list(trainable),
        base_frozen=True,
        visual_frozen=True,
        embedding_and_lm_head_frozen=True,
        precision="NF4 double base; frozen large floating weights BF16; norms and LoRA FP32; BF16 autocast",
        gradient_checkpointing=dict(enabled=True, use_reentrant=False),
        memory=stages,
        loader_seconds=loaded.load_seconds,
        references=["https://huggingface.co/docs/peft/v0.19.0/developer_guides/quantization"],
        implementation="Local audited PEFT loop; reference ms-swift script is not executed.",
    )
    save(output / "model_preflight.json", metadata)
    return model, loaded.processor, metadata


def batch(row, pad_id):
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
    pad = row["padded_tokens"] - ids.shape[1]
    mask = torch.nn.functional.pad(torch.ones_like(ids), (0, pad))
    ids = torch.nn.functional.pad(ids, (0, pad), value=pad_id)
    positions = torch.arange(row["input_tokens"] - 1, row["total_tokens"] - 1, device="cuda")
    targets = torch.tensor(row["answer_ids"], device="cuda")
    return dict(input_ids=ids, attention_mask=mask, mm_token_type_ids=torch.zeros_like(ids)), positions, targets


def loss(model, row, pad_id):
    inputs, positions, targets = batch(row, pad_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**inputs, logits_to_keep=positions, use_cache=False, return_dict=True)
        if output.logits.shape[1] != len(targets):
            raise ValueError("Every answer token, including EOS, needs its own prediction logit.")
        value = torch.nn.functional.cross_entropy(output.logits[0].float(), targets)
    if not torch.isfinite(value):
        raise ValueError("Nonfinite supervised answer loss.")
    return value


def weights_digest(parameters):
    h = hashlib.sha256()
    for name, parameter in parameters.items():
        h.update(name.encode())
        h.update(parameter.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def smoke(model, processor, rows, output):
    """Two longest microbatches and one optimizer update; restore adapters afterwards."""
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    before = weights_digest(trainable)
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4, weight_decay=0.01)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    selected = sorted(rows, key=lambda r: r["total_tokens"], reverse=True)[:2]
    report = dict(stage="Q0_smoke", microbatches=2, optimizer_steps=1, rows=[], blind_access=False)
    save(output / "smoke.json", report)
    for row in selected:
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        value = loss(model, row, processor.tokenizer.pad_token_id)
        after_forward = memory()
        (value / len(selected)).backward()
        report["rows"].append(
            dict(
                id=row["id"],
                total_tokens=row["total_tokens"],
                padded_tokens=row["padded_tokens"],
                supervised_tokens=row["supervised_tokens"],
                loss=float(value.detach()),
                after_forward=after_forward,
                after_backward=memory(),
                seconds=time.monotonic() - started,
            )
        )
        save(output / "smoke.json", report)
    grads = [p.grad for p in trainable.values() if p.grad is not None]
    if (
        not grads
        or not all(torch.isfinite(g).all() for g in grads)
        or not any(torch.count_nonzero(g) > 0 for g in grads)
    ):
        raise ValueError("LoRA gradients must be present, finite and nonzero.")
    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
    optimizer.step()
    after = weights_digest(trainable)
    if before == after:
        raise ValueError("LoRA weights did not change after optimizer.step.")
    report.update(
        finite_nonzero_gradients=True,
        gradient_norm=float(norm),
        weights_before=before,
        weights_after=after,
        optimizer_memory=memory(),
        passed=True,
        restored_before_formal_q0=True,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(initial[name])
    assert weights_digest(trainable) == before
    del optimizer, initial, grads
    torch.cuda.empty_cache()
    save(output / "smoke.json", report)
    return report
