"""Tokenize real dialogue SFT and run a tiny frozen-VLM routing-head smoke test.

This is a feasibility probe, not a trained dialogue generator or skill controller.
No held-out row is optimized and no training-set accuracy is presented as quality.
"""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from .toy_dataset import FORMAT_VERSION, SOURCE, _asset, contract_digest, digest, dumps
from .toy_inputs import model_payload, validate_input
from .toy_protocol import DECISIONS, validate_output
from .toy_schema import SYSTEM


def messages_for(request):
    """Shared online/offline multimodal serialization; no answer tokens enter input."""
    content = []
    for camera in ("front", "near"):
        content.extend([{"type": "text", "text": f"Current camera: {camera}"}, {"type": "image"}])
    content.append({"type": "text", "text": dumps(model_payload(request))})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def read_records(path):
    path = Path(path).resolve()
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest["files_sha256"].get(path.stem) != digest(path):
        raise ValueError("Dataset split changed since export; regenerate its provenance manifest.")
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if (
            row.get("format_version") != FORMAT_VERSION
            or row.get("controller_source") != SOURCE
            or row.get("training_ready") is not True
            or row.get("contract_sha256") != contract_digest()
            or row.get("system_prompt") != SYSTEM
        ):
            raise ValueError("Only current exported runtime dialogue samples may enter this probe.")
        validate_input(row["request"])
        validate_output(row["target"], row["request"])
        if set(row["images"]) != {"front", "near"}:
            raise ValueError("Require the two actual recorded camera views.")
        for camera, relative in row["images"].items():
            if digest(_asset(path.parent, relative)) != row["image_sha256"][camera]:
                raise ValueError("An exported camera image changed.")
        rows.append(row)
    if not rows:
        raise ValueError("The requested split contains no examples.")
    return rows


def encode(row, root, processor, max_length=8192):
    """Return exact assistant-only labels without silently truncating any input."""
    from PIL import Image

    images = []
    for camera in ("front", "near"):
        with Image.open(Path(root) / row["images"][camera]) as image:
            images.append(image.convert("RGB"))
    messages = messages_for(row["request"])
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    messages.append({"role": "assistant", "content": dumps(row["target"])})
    full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    batch = dict(processor(text=[full], images=images, return_tensors="pt", padding=False))
    prefix = processor(text=[prompt], images=images, return_tensors="pt", padding=False)["input_ids"]
    boundary, length = prefix.shape[1], batch["input_ids"].shape[1]
    if length > max_length:
        raise ValueError(f"Sample {row['id']} needs {length} tokens; truncation is disabled.")
    if boundary >= length or not (batch["input_ids"][:, :boundary] == prefix).all().item():
        raise ValueError("Assistant label boundary does not match the current model chat template.")
    labels = batch["input_ids"].clone()
    labels[:, :boundary] = -100
    batch["labels"] = labels
    return batch, boundary


def _selection(rows, limit):
    groups = defaultdict(list)
    for row in rows:
        groups[row["target"]["decision"]].append(row)
    selected = []
    while len(selected) < min(limit, len(rows)):
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    return selected


def _probe_rows(dataset, *, processor_only, limit):
    """Read every declared split for validation, but only train for optimization."""
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    names = sorted(manifest["files_sha256"], key=lambda name: (name != "train", name)) if processor_only else ["train"]
    rows, split_by_id = [], {}
    for split in names:
        if split not in ("train", "val", "test"):
            raise ValueError("Use the declared train, val, and test split names.")
        path = dataset / f"{split}.jsonl"
        if digest(path) != manifest["files_sha256"].get(split):
            raise ValueError("Dataset split changed since export; regenerate its provenance manifest.")
        if not path.stat().st_size:
            continue
        current = read_records(path)
        for row in current:
            if row["id"] in split_by_id:
                raise ValueError("Repeated sample identifiers across dataset splits.")
            split_by_id[row["id"]] = split
        rows.extend(current)
    if not rows:
        raise ValueError("No real dialogue rows are available for this probe.")
    if not processor_only:
        rows = _selection(rows, limit)
    return rows, {row["id"]: split_by_id[row["id"]] for row in rows}


def _feature(backbone, batch, boundary):
    """Use only input tokens, never teacher answers, for routing features."""
    import torch

    prefix = {key: value for key, value in batch.items() if key != "labels"}
    pad = ((boundary + 1023) // 1024) * 1024 - boundary
    for key, pad_value in (
        ("input_ids", backbone.processor.tokenizer.pad_token_id),
        ("attention_mask", 0),
        ("mm_token_type_ids", 0),
        ("token_type_ids", 0),
    ):
        if key in prefix:
            prefix[key] = torch.nn.functional.pad(prefix[key][:, :boundary], (0, pad), value=pad_value)
    prefix = {key: value.to("cuda") for key, value in prefix.items()}
    with torch.inference_mode():
        result = backbone.model.model(**prefix, use_cache=False, return_dict=True)
        feature = result.last_hidden_state[0, boundary - 1].float().cpu().clone()
    if feature.shape != (2048,) or not torch.isfinite(feature).all():
        raise ValueError("Expected a finite, input-only 2048-dimensional backbone feature.")
    return feature


def probe(dataset, model_dir, output, *, limit=8, steps=12, processor_only=False):
    import torch
    from transformers import AutoProcessor

    if not processor_only and (not 1 <= limit <= 16 or not 1 <= steps <= 32):
        raise ValueError("This probe is limited to 16 real rows and 32 small-head updates.")
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError("Use a new probe output directory.")
    if processor_only:
        torch.set_num_threads(2)
    rows, split_by_id = _probe_rows(dataset, processor_only=processor_only, limit=limit)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, min_pixels=65536, max_pixels=98304)
    output.mkdir(parents=True)
    tokenized = []
    tokenization_started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        batch, boundary = encode(row, dataset, processor)
        labels = batch["labels"]
        tokenized.append(
            {
                "id": row["id"],
                "split": split_by_id[row["id"]],
                "input_tokens": boundary,
                "assistant_tokens": int((labels != -100).sum()),
                "prompt_masked": bool((labels[:, :boundary] == -100).all()),
                "camera_count": int(batch["image_grid_thw"].shape[0]),
                "goal_count": len(row["target"]["goals"]),
                "target_sha256": hashlib.sha256(dumps(row["target"]).encode()).hexdigest(),
            }
        )
        if index % 200 == 0 or index == len(rows):
            print(
                dumps(
                    {
                        "tokenized": index,
                        "total": len(rows),
                        "elapsed_s": round(time.monotonic() - tokenization_started, 2),
                        "processor_only": processor_only,
                    }
                ),
                flush=True,
            )
    report = {
        "kind": (
            "real_dialogue_processor_validation"
            if processor_only
            else "real_dialogue_format_and_frozen_route_head_feasibility"
        ),
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "contract_sha256": contract_digest(),
        "training_rows_only": set(split_by_id.values()) == {"train"},
        "all_splits_tokenized_read_only": processor_only,
        "tokenized_split_counts": dict(Counter(split_by_id.values())),
        "tokenization_elapsed_s": time.monotonic() - tokenization_started,
        "max_input_tokens": max(item["input_tokens"] for item in tokenized),
        "max_total_tokens": max(item["input_tokens"] + item["assistant_tokens"] for item in tokenized),
        "optimizer_steps": 0,
        "generative_labels_validated": True,
        "generative_backward_tested": False,
        "skill_execution_trained": False,
        "deployment_ready": False,
        "rows": tokenized,
        "selected_decisions": [row["target"]["decision"] for row in rows],
        "limits": (
            "Read-only processor validation proves image/token/label compatibility; no model was trained or evaluated."
            if processor_only
            else "A small routing-head update proves differentiability only, not dialogue or multi-task competence."
        ),
    }
    if not processor_only:
        from .head_features import FrozenFeatures
        from .setup_skill_head import MODEL_REPO, MODEL_REVISION, WEIGHTS_SHA256

        started = time.monotonic()
        torch.manual_seed(2718)
        backbone = FrozenFeatures(model_dir)
        features = []
        for index, row in enumerate(rows):
            batch, boundary = encode(row, dataset, backbone.processor)
            features.append(_feature(backbone, batch, boundary))
            print(dumps({"feature": index + 1, "total": len(rows), "sample": row["id"]}), flush=True)
        x = torch.nn.functional.normalize(torch.stack(features).clone(), dim=-1)
        y = torch.tensor([DECISIONS.index(row["target"]["decision"]) for row in rows])
        head = torch.nn.Linear(2048, len(DECISIONS))
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=0.01)
        losses, norms = [], []
        before = head.weight.detach().clone()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(x), y)
            if not torch.isfinite(loss):
                raise ValueError("The routing probe loss is non-finite.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        final_loss = float(torch.nn.functional.cross_entropy(head(x), y).detach())
        if torch.equal(before, head.weight) or any(
            parameter.grad is not None for parameter in backbone.model.parameters()
        ):
            raise ValueError("Expected a changed routing head and an entirely frozen backbone.")
        torch.save(
            {"state_dict": head.state_dict(), "classes": DECISIONS, "deployment_ready": False},
            output / "probe_router.pt",
        )
        report.update(
            model_repo=MODEL_REPO,
            model_revision=MODEL_REVISION,
            weights_sha256=WEIGHTS_SHA256,
            trainable_parameters=sum(parameter.numel() for parameter in head.parameters()),
            optimizer_steps=steps,
            train_loss_before=losses[0],
            train_loss_after=final_loss,
            losses=losses,
            gradient_norms=norms,
            backbone_frozen=True,
            feature_pooling="last_valid_input_token_before_assistant_answer",
            elapsed_s=time.monotonic() - started,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        )
    (output / "probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--processor-only", action="store_true")
    args = parser.parse_args()
    report = probe(
        args.dataset,
        args.model,
        args.output,
        limit=args.limit,
        steps=args.steps,
        processor_only=args.processor_only,
    )
    summary = {key: value for key, value in report.items() if key not in ("rows", "selected_decisions")}
    summary.update(report=str(args.output / "probe_report.json"), rows=len(report["rows"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
