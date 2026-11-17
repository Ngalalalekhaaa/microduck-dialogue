"""Two-sample generative SFT probe with a frozen backbone and output adapter.

Every assistant token, including reply and all ordered goals, contributes to CE.
Only a rank-four residual on the frozen vocabulary projection receives gradients.
This establishes training feasibility, not task accuracy or deployment readiness.
"""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from .toy_dataset import contract_digest, digest, dumps
from .toy_schema import SYSTEM
from .toy_train_probe import encode, read_records


def _select(rows):
    candidates = [row for row in rows if len(row["target"]["goals"]) >= 2]
    if len(candidates) < 2:
        raise ValueError("Provide at least two real multi-goal training rows; holdout data is never optimized.")
    first = candidates[0]
    second = next((row for row in candidates[1:] if row["episode_id"] != first["episode_id"]), candidates[1])
    return [first, second]


def _teacher_hidden(backbone, batch, boundary):
    """Causal teacher-forcing positions predict each actual next assistant token."""
    import torch

    length = batch["input_ids"].shape[1]
    labels = batch["labels"][:, boundary:].clone().flatten()
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    padding = ((length + 1023) // 1024) * 1024 - length
    for key, pad_value in (
        ("input_ids", backbone.processor.tokenizer.pad_token_id),
        ("attention_mask", 0),
        ("mm_token_type_ids", 0),
        ("token_type_ids", 0),
    ):
        if key in inputs:
            inputs[key] = torch.nn.functional.pad(inputs[key], (0, padding), value=pad_value)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.no_grad():
        result = backbone.model.model(**inputs, use_cache=False, return_dict=True)
        hidden = result.last_hidden_state[0, boundary - 1 : length - 1].detach().cpu().clone()
    if hidden.shape != (len(labels), 2048) or (labels < 0).any() or not torch.isfinite(hidden).all():
        raise ValueError("Teacher-forced hidden states and assistant-only labels must align exactly.")
    return hidden, labels


def _epoch(head, adapter, examples, *, backward, chunk_tokens):
    import torch

    count = sum(len(labels) for _, labels in examples)
    total = 0.0
    for hidden, labels in examples:
        for offset in range(0, len(labels), chunk_tokens):
            features = hidden[offset : offset + chunk_tokens].to("cuda")
            targets = labels[offset : offset + chunk_tokens].to("cuda")
            with torch.no_grad():
                base_logits = head(features.to(head.weight.dtype)).float()
            with torch.set_grad_enabled(backward):
                logits = base_logits + adapter(features.float())
                summed_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
                if not torch.isfinite(summed_loss):
                    raise ValueError("Non-finite complete-JSON supervision loss.")
                if backward:
                    (summed_loss / count).backward()
            total += float(summed_loss.detach())
    return total / count


def probe(dataset, model_dir, output, *, steps=3, chunk_tokens=16):
    import torch

    from .head_features import FrozenFeatures
    from .setup_skill_head import MODEL_REPO, MODEL_REVISION, WEIGHTS_SHA256

    if not 1 <= steps <= 3 or not 1 <= chunk_tokens <= 32:
        raise ValueError("The pilot allows at most three optimizer updates and 32 vocabulary rows at once.")
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory for the generative probe.")
    rows = _select(read_records(dataset / "train.jsonl"))
    output.mkdir(parents=True)
    torch.manual_seed(31415)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    backbone = FrozenFeatures(model_dir)
    head = backbone.model.get_output_embeddings()
    if head is None or head.weight.shape[1] != 2048 or any(p.requires_grad for p in backbone.model.parameters()):
        raise ValueError("Require the pinned frozen 2048-dimensional Qwen3.5 vocabulary projection.")
    examples, tokenization = [], []
    for row in rows:
        batch, boundary = encode(row, dataset, backbone.processor)
        examples.append(_teacher_hidden(backbone, batch, boundary))
        tokenization.append(
            {
                "id": row["id"],
                "episode_id": row["episode_id"],
                "goal_count": len(row["target"]["goals"]),
                "goal_types": [goal["type"] for goal in row["target"]["goals"]],
                "target_sha256": hashlib.sha256(dumps(row["target"]).encode()).hexdigest(),
                "input_tokens": boundary,
                "assistant_tokens": len(examples[-1][1]),
                "prompt_masked": bool((batch["labels"][:, :boundary] == -100).all()),
                "camera_count": int(batch["image_grid_thw"].shape[0]),
            }
        )
        print(dumps({"encoded_multi_goal": tokenization[-1]}), flush=True)
    adapter = torch.nn.Sequential(
        torch.nn.Linear(2048, 4, bias=False),
        torch.nn.Linear(4, head.weight.shape[0], bias=False),
    ).to("cuda")
    torch.nn.init.normal_(adapter[0].weight, std=0.02)
    torch.nn.init.zeros_(adapter[1].weight)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in adapter.named_parameters()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.002, weight_decay=0.0)
    losses, norms = [], []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _epoch(head, adapter, examples, backward=True, chunk_tokens=chunk_tokens)
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss)
        norms.append(float(norm))
        print(dumps({"optimizer_step": step + 1, "complete_json_loss": loss, "gradient_norm": float(norm)}), flush=True)
    final_loss = _epoch(head, adapter, examples, backward=False, chunk_tokens=chunk_tokens)
    changed = {
        name: not torch.equal(initial[name], parameter.detach().cpu()) for name, parameter in adapter.named_parameters()
    }
    if not any(changed.values()) or any(parameter.grad is not None for parameter in backbone.model.parameters()):
        raise ValueError("Expected nonzero adapter updates and no gradients in the frozen base model.")
    checkpoint = output / "probe_output_adapter.pt"
    torch.save(
        {
            "kind": "rank4_lm_head_residual",
            "state_dict": {name: parameter.detach().cpu() for name, parameter in adapter.named_parameters()},
            "rank": 4,
            "input_dim": 2048,
            "vocab_size": head.weight.shape[0],
            "deployment_ready": False,
            "contract_sha256": contract_digest(),
            "model_revision": MODEL_REVISION,
            "weights_sha256": WEIGHTS_SHA256,
        },
        checkpoint,
    )
    report = {
        "kind": "real_multigoal_complete_json_sft_feasibility",
        "implementation_sha256": digest(__file__),
        "training_rows_only": True,
        "generative_backward_tested": True,
        "backbone_frozen": True,
        "base_vocabulary_projection_frozen": True,
        "input_prompt_supervised": False,
        "supervised_content": "all assistant JSON tokens: reply, decision, every ordered goal and target",
        "deployment_ready": False,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "contract_sha256": contract_digest(),
        "rows": tokenization,
        "rank": 4,
        "trainable_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "optimizer_steps": steps,
        "chunk_tokens": chunk_tokens,
        "complete_json_loss_before": losses[0],
        "complete_json_loss_after": final_loss,
        "losses": losses,
        "gradient_norms": norms,
        "parameter_changed": changed,
        "adapter_sha256": digest(checkpoint),
        "adapter_bytes": checkpoint.stat().st_size,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "elapsed_s": time.monotonic() - started,
        "limits": [
            "Two training examples and three or fewer updates only establish format and backward feasibility.",
            "No held-out quality, autonomous generation, scene generalization or robot task competence is claimed.",
            "This output adapter does not train the separate low-level skill policies or the existing 11-class head.",
        ],
    }
    (output / "probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        report = probe(args.dataset, args.model, args.output, steps=args.steps, chunk_tokens=args.chunk_tokens)
    except Exception as error:
        if not existed:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "failure.json").write_text(
                json.dumps({"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}, indent=2)
                + "\n"
            )
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
