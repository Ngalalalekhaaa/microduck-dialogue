"""CPU fitting and cache provenance checks for the factorized toy selector."""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .extract_toy_head_features import read_splits
from .toy_dataset import contract_digest, digest, dumps
from .toy_protocol import DECISIONS
from .toy_selector_labels import FIELDS, MOTION, TOKENS, canonical_plan, check_support, encode_targets
from .toy_selector_metrics import evaluate
from .toy_selector_model import SelectorModel, decode_outputs, training_loss


def read_training_data(features, dataset):
    features, dataset = Path(features), Path(dataset)
    manifest = json.loads((features / "manifest.json").read_text())
    if not manifest.get("complete") or set(manifest["splits"]) != {"train", "val", "test"}:
        raise ValueError("Finish feature extraction for all train, val and test rows before training.")
    if manifest["contract_sha256"] != contract_digest():
        raise ValueError("Feature cache uses a different dialogue contract.")
    if digest(dataset / "manifest.json") != manifest["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from feature extraction.")
    if hashlib.sha256(dumps(manifest["config"]).encode()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Feature configuration checksum mismatch.")
    rows = read_splits(dataset)
    arrays, ordered_rows = {}, {}
    for split in ("train", "val", "test"):
        if digest(dataset / f"{split}.jsonl") != manifest["source_hashes"][split]:
            raise ValueError(f"Source {split} changed after feature extraction.")
        path, record_path = features / f"{split}.npz", features / f"{split}_records.json"
        if digest(path) != manifest["splits"][split]["cache_sha256"]:
            raise ValueError(f"Feature cache checksum mismatch for {split}.")
        if digest(record_path) != manifest["splits"][split]["records_sha256"]:
            raise ValueError(f"Feature provenance checksum mismatch for {split}.")
        with np.load(path, allow_pickle=False) as data:
            x, ids = data["x"].astype(np.float32), data["ids"].tolist()
        sources = {row["id"]: row for row in rows[split]}
        if len(ids) != len(set(ids)) or set(ids) != set(sources) or len(ids) != manifest["expected_rows"][split]:
            raise ValueError(f"Feature IDs do not match source {split} exactly.")
        if x.shape != (len(ids), 2048) or not len(ids) or not np.isfinite(x).all():
            raise ValueError("Require finite 2048-dimensional features and nonempty held-out splits.")
        ordered_rows[split] = [sources[identifier] for identifier in ids]
        records = json.loads(record_path.read_text())
        if len(records) != len(ids):
            raise ValueError("Feature provenance record count differs from cached vectors.")
        for row, record in zip(ordered_rows[split], records, strict=True):
            expected = dict(
                id=row["id"], split=split, source_row_sha256=hashlib.sha256(dumps(row).encode()).hexdigest()
            )
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError("Feature rows and semantic labels are misaligned.")
        arrays[split] = torch.from_numpy(x)
    labels = {split: encode_targets(items) for split, items in ordered_rows.items()}
    support = check_support(labels)
    tensors = {split: {key: torch.from_numpy(value) for key, value in items.items()} for split, items in labels.items()}
    return manifest, arrays, ordered_rows, tensors, support


def infer(model, features, batch_size=128):
    model.eval()
    predictions, raw_tokens = [], []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            output = model(features[start : start + batch_size])
            predictions.extend(decode_outputs(output))
            raw_tokens.extend(output["token"].argmax(-1).tolist())
    return predictions, raw_tokens


def selection_key(metrics):
    decision = metrics["decision_user"]["macro_f1_present"] or 0.0
    plan = metrics["action_plan_exact"]["accuracy"] or 0.0
    token = metrics["action_fields"].get("semantic_token", {}).get("accuracy") or 0.0
    return (0.4 * decision + 0.6 * plan, token, decision)


def loss_weights(labels):
    user_weights = torch.where(labels["user"], 1.0, 0.15)
    counts = torch.zeros(len(DECISIONS)).scatter_add_(0, labels["decision"], user_weights)
    decision_weights = (counts.max() / counts.clamp_min(1)).pow(0.75).clamp(max=15)
    decision_weights /= decision_weights.mean()
    motion = (labels["decision"] == DECISIONS.index("replace")) | (labels["decision"] == DECISIONS.index("append"))
    valid = (labels["token"] >= 0) & motion[:, None]
    token_counts = torch.bincount(labels["token"][valid], minlength=len(TOKENS)).float()
    token_weights = (token_counts.max() / token_counts.clamp_min(1)).sqrt().clamp(max=4)
    return decision_weights, token_weights


def weight_audit(labels):
    decision, token = loss_weights(labels)
    examples = torch.where(labels["user"], 1.0, 0.15)
    effective = examples * decision[labels["decision"]]
    return dict(
        feedback_sample_weight=0.15,
        feedback_share_after_sample_weight=float(examples[~labels["user"]].sum() / examples.sum()),
        feedback_share_after_decision_class_weight=float(effective[~labels["user"]].sum() / effective.sum()),
        decision_class_weights=dict(zip(DECISIONS, decision.tolist(), strict=True)),
        action_token_weights=dict(zip(TOKENS, token.tolist(), strict=True)),
        action_token_weight_counts="replace/append training rows only; feedback EOS does not affect these weights",
        action_token_and_parameter_loss="replace/append rows only, with padding and inapplicable fields masked",
    )


def fit(arrays, labels, rows, config, log, *, epochs=150, batch_size=128, patience=30):
    """This function cannot access test features or labels through its arguments."""
    if set(arrays) != {"train", "val"} or set(labels) != {"train", "val"} or set(rows) != {"train", "val"}:
        raise ValueError("Model selection accepts train and validation splits only.")
    torch.manual_seed(config["seed"])
    model = SelectorModel(arrays["train"].shape[1], config["hidden"])
    with torch.no_grad():
        model.mean.copy_(arrays["train"].mean(0))
        model.scale.copy_(arrays["train"].std(0, unbiased=False).clamp_min(0.1))
    decision_weights, token_weights = loss_weights(labels["train"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    train_plans = {canonical_plan(row["target"]["goals"]) for row in rows["train"] if row["target"]["goals"]}
    best_state, best_key, best_epoch, history = None, (-1, -1, -1), 0, []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(arrays["train"]))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_labels = {name: values[indices] for name, values in labels["train"].items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(arrays["train"][indices], teacher=batch_labels)
            loss = training_loss(output, batch_labels, decision_weights, token_weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; inspect the input feature cache.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        if epoch % 5 and epoch != epochs:
            continue
        predictions, raw_tokens = infer(model, arrays["val"])
        metrics = evaluate(rows["val"], predictions, train_plans, raw_tokens=raw_tokens)
        key = selection_key(metrics)
        if key > best_key:
            best_state, best_key, best_epoch = copy.deepcopy(model.state_dict()), key, epoch
        record = dict(
            config=config,
            epoch=epoch,
            train_loss=total_loss / len(order),
            val_user_decision_macro_f1=metrics["decision_user"]["macro_f1_present"],
            val_action_plan_exact=metrics["action_plan_exact"],
            val_goal_token_accuracy=metrics["action_fields"].get("semantic_token"),
            selection_score=key[0],
            best_epoch=best_epoch,
            elapsed_s=time.monotonic() - started,
        )
        history.append(record)
        log(record)
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(config=config, best_epoch=best_epoch, key=list(best_key), history=history)


def prediction_records(rows, predictions):
    return [
        dict(
            id=row["id"],
            episode_id=row["episode_id"],
            split_group=row["split_group"],
            event=row["request"]["event"],
            expected={name: row["target"][name] for name in ("decision", "goals")},
            predicted=prediction,
        )
        for row, prediction in zip(rows, predictions, strict=True)
    ]


def support_report(rows):
    return {
        split: dict(
            rows=len(items),
            user_rows=sum(row["request"]["event"]["kind"] == "user" for row in items),
            motion_rows=sum(row["target"]["decision"] in MOTION for row in items),
            unique_plans=len({canonical_plan(row["target"]["goals"]) for row in items if row["target"]["goals"]}),
            field_values={
                name: sorted({int(value) for value in encode_targets(items)[name].ravel() if value != -100})
                for name in FIELDS
            },
        )
        for split, items in rows.items()
    }
