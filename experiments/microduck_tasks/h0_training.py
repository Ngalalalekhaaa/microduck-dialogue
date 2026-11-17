"""Train the original GRU at H0/H1 width using only train/common-development data."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .ablation_contract import checksum, dumps
from .ablation_score import score, summarize
from .h0_features import read_data, record
from .run_ablation import save
from .toy_dataset import contract_digest, digest
from .toy_protocol import DECISIONS
from .toy_selector_labels import FIELDS, TOKENS, canonical_plan, encode_targets
from .toy_selector_metrics import evaluate
from .toy_selector_model import MODEL_VERSION, decode_outputs
from .toy_selector_training import fit, infer, weight_audit

SEEDS = (17, 43, 71)


def load_data(dataset, features):
    _, rows = read_data(dataset)
    manifest = json.loads((features / "manifest.json").read_text())
    if not manifest["complete"] or set(manifest["splits"]) != {"train", "val"}:
        raise ValueError("Finish train and validation features; blind extraction is forbidden.")
    if (
        manifest["dataset_manifest_sha256"] != digest(dataset / "manifest.json")
        or manifest["contract_sha256"] != contract_digest()
    ):
        raise ValueError("Dataset or dialogue contract differs from extraction.")
    if checksum(manifest["config"]) != manifest["config_sha256"]:
        raise ValueError("Feature configuration checksum mismatch.")
    arrays = {}
    for split, items in rows.items():
        path = features / f"{split}.npz"
        recpath = features / f"{split}_records.json"
        expected = manifest["splits"][split]
        if digest(path) != expected["cache_sha256"] or digest(recpath) != expected["records_sha256"]:
            raise ValueError("Feature bytes or provenance were modified.")
        with np.load(path, allow_pickle=False) as data:
            ids, values = data["ids"].tolist(), data["x"].astype(np.float32)
        if ids != [r["id"] for r in items] or values.shape != (len(items), 2048) or not np.isfinite(values).all():
            raise ValueError("Invalid or misaligned feature matrix.")
        records = json.loads(recpath.read_text())
        for row, rec in zip(items, records, strict=True):
            if any(rec.get(k) != v for k, v in record(row, split).items()):
                raise ValueError("Input features and labels are misaligned.")
        arrays[split] = torch.from_numpy(values)
    return manifest, rows, arrays


def prediction_report(rows, predictions):
    records = [
        dict(
            row,
            raw_text=dumps(prediction),
            score=score(dumps(prediction), row, plan_only=True),
            timing={},
            generation_error=None,
            source="learned_head_decoded",
            host_control=None,
        )
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    return dict(summary=summarize(records, expected_count=len(rows)), predictions=records)


def oracle_decision(model, values, rows):
    """Diagnostic only: replace decision logits after normal autoregressive decoding."""
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), 128):
            output = model(values[start : start + 128])
            logits = torch.full_like(output["decision"], -torch.inf)
            for index, row in enumerate(rows[start : start + 128]):
                logits[index, DECISIONS.index(row["expected"]["decision"])] = 0
            output["decision"] = logits
            predictions.extend(decode_outputs(output))
    return predictions


def train(dataset, features, output, *, epochs=150, hidden=128):
    if hidden not in (128, 256):
        raise ValueError("Use hidden=128 for H0 or hidden=256 for H1.")
    torch.set_num_threads(2)
    manifest, rows, arrays = load_data(dataset, features)
    selected_indices = [i for i, row in enumerate(rows["val"]) if row["domain"] == "common"]
    selection_rows = dict(train=rows["train"], val=[rows["val"][i] for i in selected_indices])
    selection_arrays = dict(train=arrays["train"], val=arrays["val"][selected_indices])
    labels = {
        s: {k: torch.from_numpy(v) for k, v in encode_targets(items).items()} for s, items in selection_rows.items()
    }
    for name in ("decision", "token", *FIELDS):
        train_labels = set(labels["train"][name].ravel().tolist()) - {-100}
        if set(labels["val"][name].ravel().tolist()) - {-100} - train_labels:
            raise ValueError(f"Unsupported validation class for {name}.")
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(
        hidden=hidden,
        seeds=list(SEEDS),
        lr=0.001,
        weight_decay=0.01,
        epochs=epochs,
        batch_size=128,
        patience_epochs=30,
        evaluate_every_epochs=5,
        input_view="P6",
        selection=(
            "0.4*common-val decision macro-F1 + 0.6*common-val action exact; ties token, decision; seed tie first"
        ),
        selection_samples=len(selected_indices),
        total_validation_samples=len(rows["val"]),
        blind_access=False,
        human_reviewed=False,
        offline_diagnostic=True,
        normalisation="Train mean/std only; original std floor 0.1",
        source_sha256={
            n: digest(Path(__file__).with_name(n))
            for n in ("h0_training.py", "toy_selector_model.py", "toy_selector_training.py", "toy_selector_labels.py")
        },
    )
    save(output / "training_config.json", settings)
    save(output / "loss_weight_audit.json", weight_audit(labels["train"]))
    plans = {canonical_plan(row["target"]["goals"]) for row in rows["train"] if row["target"]["goals"]}
    trials, started = [], time.monotonic()
    for seed in SEEDS:
        run = output / f"seed_{seed}"
        run.mkdir()

        def log(value, run=run, seed=seed):
            with (run / "history.jsonl").open("a") as handle:
                handle.write(dumps(value) + "\n")
            save(output / "progress.json", dict(stage="training", seed=seed, **value))
            print(dumps(dict(seed=seed, **value)), flush=True)

        cfg = dict(hidden=hidden, lr=0.001, weight_decay=0.01, seed=seed)
        model, trial = fit(
            selection_arrays, labels, selection_rows, cfg, log, epochs=epochs, batch_size=128, patience=30
        )
        checkpoint = dict(
            version=MODEL_VERSION,
            state_dict=model.state_dict(),
            dim=model.dim,
            hidden=model.hidden,
            decisions=list(DECISIONS),
            tokens=list(TOKENS),
            fields={k: list(v) for k, v in FIELDS.items()},
            feature_config=manifest["config"],
            contract_sha256=manifest["contract_sha256"],
            selected={k: v for k, v in trial.items() if k != "history"},
            training_config=settings,
            deployment_ready=False,
            blind_access=False,
        )
        torch.save(checkpoint, run / "selector.pt")
        predictions, raw = infer(model, arrays["val"])
        reports = {
            "all240": prediction_report(rows["val"], predictions),
            "common236": prediction_report(selection_rows["val"], [predictions[i] for i in selected_indices]),
            "oracle_decision_diagnostic": prediction_report(
                rows["val"], oracle_decision(model, arrays["val"], rows["val"])
            ),
        }
        for name, report in reports.items():
            save(run / f"{name}.json", report)
        train_predictions, train_raw = infer(model, arrays["train"])
        summary = dict(
            seed=seed,
            trial={k: v for k, v in trial.items() if k != "history"},
            parameters=sum(p.numel() for p in model.parameters()),
            checkpoint_sha256=digest(run / "selector.pt"),
            train=evaluate(rows["train"], train_predictions, plans, raw_tokens=train_raw),
            common_validation=evaluate(
                selection_rows["val"],
                [predictions[i] for i in selected_indices],
                plans,
                raw_tokens=[raw[i] for i in selected_indices],
            ),
            validation=reports["all240"]["summary"],
            oracle_decision=reports["oracle_decision_diagnostic"]["summary"],
            blind_access=False,
            deployment_ready=False,
        )
        save(run / "summary.json", summary)
        trials.append(summary)
    winner = max(trials, key=lambda r: tuple(r["trial"]["key"]))
    result = dict(
        complete=True,
        seeds=trials,
        selected_seed=winner["seed"],
        prototype_gate_pass=winner["validation"]["prototype_gate_pass"],
        elapsed_s=time.monotonic() - started,
        blind_access=False,
        physical_executions=0,
        action_exact_mean=float(np.mean([t["validation"]["metrics"]["action_plan_exact"]["rate"] for t in trials])),
        action_exact_std=float(np.std([t["validation"]["metrics"]["action_plan_exact"]["rate"] for t in trials])),
    )
    save(output / "results.json", result)
    save(
        output / "progress.json",
        dict(stage="complete", complete=True, selected_seed=winner["seed"], blind_access=False),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden", type=int, choices=(128, 256), default=128)
    args = parser.parse_args()
    train(args.dataset, args.features, args.output, epochs=args.epochs, hidden=args.hidden)


if __name__ == "__main__":
    main()
