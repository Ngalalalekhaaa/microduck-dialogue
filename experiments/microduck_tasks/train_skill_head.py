"""Train small CPU classifiers on frozen features; select on validation, test once."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .setup_skill_head import digest
from .train_vlm import write_json


def make_head(dim, classes, hidden):
    if hidden == 0:
        return torch.nn.Linear(dim, classes)
    return torch.nn.Sequential(
        torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden, classes)
    )


def metrics(truth, prediction, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for y, p in zip(truth, prediction, strict=True):
        matrix[int(y), int(p)] += 1
    tp = matrix.diagonal()
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(classes)), where=support + predicted != 0)
    return dict(
        accuracy=float(tp.sum() / max(1, matrix.sum())),
        macro_f1=float(f1.mean()),
        macro_f1_present=float(f1[support > 0].mean()),
        per_class={
            name: dict(support=int(support[i]), recall=float(recall[i]), f1=float(f1[i]))
            for i, name in enumerate(classes)
        },
        confusion_matrix=matrix.tolist(),
        classes=classes,
    )


def fit(train_x, train_y, val_x, val_y, classes, config):
    torch.manual_seed(17)
    model = make_head(train_x.shape[1], len(classes), config["hidden"])
    counts = torch.bincount(train_y, minlength=len(classes)).float()
    weights = (counts.max() / counts.clamp_min(1)).sqrt().clamp(max=4)
    weights /= weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best, history, best_key, best_epoch = None, [], (-1.0, -float("inf")), 0
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 5:
            continue
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            result = metrics(val_y.numpy(), logits.argmax(-1).numpy(), classes)
            nll = float(torch.nn.functional.cross_entropy(logits, val_y))
        key = (result["macro_f1"], -nll)
        history.append(dict(epoch=epoch, train_loss=float(loss.detach()), val_macro_f1=key[0], val_nll=nll))
        if key > best_key:
            best_key, best_epoch, best = key, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 50:
            break
    model.load_state_dict(best)
    model.eval()
    return model, dict(
        config=config, best_epoch=best_epoch, val_macro_f1=best_key[0], val_nll=-best_key[1], history=history
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.features / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Finish feature extraction before training.")
    data = {}
    for split in ("train", "val", "test"):
        path = args.features / f"{split}.npz"
        if digest(path) != manifest["splits"][split]["cache_sha256"]:
            raise ValueError("Feature cache checksum mismatch.")
        with np.load(path, allow_pickle=False) as values:
            data[split] = {k: values[k] for k in values.files}
    classes = sorted(set(data["train"]["labels"].tolist()))
    for split in data:
        if set(data[split]["labels"]) - set(classes):
            raise ValueError(f"{split} contains a class with no training support.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    mean = torch.from_numpy(data["train"]["x"].astype(np.float32)).mean(0)
    scale = torch.from_numpy(data["train"]["x"].astype(np.float32)).std(0).clamp_min(0.1)
    x = {s: (torch.from_numpy(v["x"].astype(np.float32)) - mean) / scale for s, v in data.items()}
    y = {s: torch.tensor([classes.index(v) for v in d["labels"]]) for s, d in data.items()}
    candidates = [
        dict(hidden=0, lr=0.001, weight_decay=0.1),
        dict(hidden=0, lr=0.003, weight_decay=0.1),
        dict(hidden=32, lr=0.001, weight_decay=0.1),
    ]
    trials, best, key, winner = [], None, (-1.0, -float("inf")), None
    for config in candidates:
        model, trial = fit(x["train"], y["train"], x["val"], y["val"], classes, config)
        trials.append(trial)
        candidate_key = (trial["val_macro_f1"], -trial["val_nll"])
        if candidate_key > key:
            best, key, winner = model, candidate_key, trial
    checkpoint = dict(
        state_dict=best.state_dict(),
        mean=mean,
        scale=scale,
        classes=classes,
        dim=len(mean),
        hidden=winner["config"]["hidden"],
        feature_config=manifest["config"],
        feature_manifest_sha256=digest(args.features / "manifest.json"),
        input_contract=manifest.get("input_contract", "microduck-primitive-relabel-1"),
        input_contract_sha256=manifest.get("input_contract_sha256"),
        seed=17,
        target_binding="existing_instruction_parser_and_visible_entities",
        duration="fixed_per_skill",
        continuation=False,
    )
    torch.save(checkpoint, args.output / "head.pt")
    summary = dict(
        training_seconds=time.monotonic() - started,
        feature_extraction_seconds=manifest["extraction_seconds"],
        selected=winner,
        candidates=trials,
        classes=classes,
        class_counts=manifest["class_counts"],
        trainable_parameters=sum(p.numel() for p in best.parameters()),
        feature_config=manifest["config"],
        checkpoint_sha256=digest(args.output / "head.pt"),
        input_contract=checkpoint["input_contract"],
        input_contract_sha256=checkpoint["input_contract_sha256"],
        evaluation={},
        majority_baseline={},
    )
    majority = int(torch.bincount(y["train"]).argmax())
    with torch.no_grad():
        for split in ("train", "val", "test"):
            probabilities = best(x[split]).softmax(-1)
            predictions = probabilities.argmax(-1)
            summary["evaluation"][split] = metrics(y[split].numpy(), predictions.numpy(), classes)
            summary["majority_baseline"][split] = metrics(y[split].numpy(), np.full(len(y[split]), majority), classes)
            write_json(
                args.output / f"{split}_predictions.json",
                [
                    dict(
                        id=str(data[split]["ids"][i]),
                        expected=classes[int(y[split][i])],
                        predicted=classes[int(predictions[i])],
                        probabilities={name: float(probabilities[i, j]) for j, name in enumerate(classes)},
                    )
                    for i in range(len(predictions))
                ],
            )
    write_json(args.output / "results.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("candidates", "evaluation", "majority_baseline")},
            ensure_ascii=False,
            indent=2,
        )
    )
    for split, result in summary["evaluation"].items():
        print(split, "accuracy", result["accuracy"], "macro_f1", result["macro_f1"])


if __name__ == "__main__":
    main()
