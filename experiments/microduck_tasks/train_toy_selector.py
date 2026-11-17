"""Train a factorized CPU skill selector on frozen Qwen features, selecting on val only."""

import argparse
import json
import time
from pathlib import Path

import torch

from .toy_dataset import digest
from .toy_protocol import DECISIONS
from .toy_selector_labels import FIELDS, LIMITATIONS, TOKENS, canonical_plan
from .toy_selector_metrics import evaluate, majority_user_baseline
from .toy_selector_model import MODEL_VERSION
from .toy_selector_training import fit, infer, prediction_records, read_training_data, support_report, weight_audit


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def train(features, dataset, output, *, hidden=(64, 128), epochs=150, batch_size=128, patience=30, threads=2):
    if not 1 <= len(hidden) <= 3 or any(size not in (32, 64, 128) for size in hidden):
        raise ValueError("Choose one through three candidate hidden sizes from 32,64,128.")
    if epochs < 1 or batch_size < 1 or patience < 1 or not 1 <= threads <= 8:
        raise ValueError("Epochs, batch size and patience must be positive; CPU threads must be 1..8.")
    torch.set_num_threads(threads)
    features, dataset, output = Path(features), Path(dataset), Path(output)
    manifest, arrays, rows, labels, label_support = read_training_data(features, dataset)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    configurations = [dict(hidden=size, lr=0.001, weight_decay=0.01, seed=17) for size in hidden]
    settings = dict(
        features=str(features.resolve()),
        dataset=str(dataset.resolve()),
        configs=configurations,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        cpu_threads=threads,
        diagnostic_only=bool(manifest["config"].get("diagnostic_random_features")),
        feedback_weight=0.15,
        model_selection=(
            "0.4 * val user decision macro-F1 + 0.6 * val action-plan exact-match; then goal-token accuracy"
        ),
        test_policy=("Audit held-out class coverage before fitting; evaluate test predictions only after selection."),
        normalization="Train feature mean/std only, std floor 0.1; shared by all splits and online inference.",
        numerical_output="Observed-value categorical fields; no rounding or parameter extraction by a rule teacher.",
    )
    write_json(output / "training_config.json", settings)

    def log(record):
        with (output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(output / "progress.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    best, winner, trials = None, None, []
    for config in configurations:
        model, trial = fit(
            {key: arrays[key] for key in ("train", "val")},
            {key: labels[key] for key in ("train", "val")},
            {key: rows[key] for key in ("train", "val")},
            config,
            log,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        trials.append(trial)
        if winner is None or tuple(trial["key"]) > tuple(winner["key"]):
            best, winner = model, trial
    checkpoint = dict(
        version=MODEL_VERSION,
        state_dict=best.state_dict(),
        dim=best.dim,
        hidden=best.hidden,
        decisions=list(DECISIONS),
        tokens=list(TOKENS),
        fields={name: list(values) for name, values in FIELDS.items()},
        feature_config=manifest["config"],
        contract_sha256=manifest["contract_sha256"],
        feature_manifest_sha256=digest(features / "manifest.json"),
        dataset_manifest_sha256=manifest["dataset_manifest_sha256"],
        source_hashes=manifest["source_hashes"],
        training_config=settings,
        selected={key: value for key, value in winner.items() if key != "history"},
        limitations=LIMITATIONS,
        implementations={
            name: digest(Path(__file__).with_name(name))
            for name in (
                "toy_selector_model.py",
                "toy_selector_labels.py",
                "toy_selector_training.py",
                "toy_selector_metrics.py",
                "train_toy_selector.py",
            )
        },
    )
    torch.save(checkpoint, output / "selector.pt")
    train_plans = {canonical_plan(row["target"]["goals"]) for row in rows["train"] if row["target"]["goals"]}
    result = dict(
        selected=checkpoint["selected"],
        candidates=trials,
        trainable_parameters=sum(parameter.numel() for parameter in best.parameters()),
        checkpoint_sha256=digest(output / "selector.pt"),
        feature_config=manifest["config"],
        contract_sha256=manifest["contract_sha256"],
        data_support=support_report(rows),
        loss_weighting=weight_audit(labels["train"]),
        label_support=label_support,
        label_vocabulary=dict(decisions=list(DECISIONS), tokens=list(TOKENS), fields=checkpoint["fields"]),
        evaluation={},
        limitations=LIMITATIONS,
        test_evaluations=0,
        deployment_ready=False,
        trained_free_text_reply=False,
        diagnostic_only=settings["diagnostic_only"],
        test_evaluation_scope=(
            "test_evaluations counts the selected trained checkpoint only; fixed train-derived baseline statistics "
            "are reported separately and never participate in model selection"
        ),
    )
    baselines = dict(
        train_user_majority=majority_user_baseline(rows["train"]),
        train_motion_majority=majority_user_baseline(
            [row for row in rows["train"] if row["target"]["decision"] in ("replace", "append")]
        ),
    )
    result["baselines"] = {name: dict(**baseline, evaluation={}) for name, baseline in baselines.items()}
    for split in ("train", "val", "test"):
        predictions, raw_tokens = infer(best, arrays[split])
        result["evaluation"][split] = evaluate(rows[split], predictions, train_plans, raw_tokens=raw_tokens)
        for name, baseline in baselines.items():
            result["baselines"][name]["evaluation"][split] = evaluate(
                rows[split], [baseline["prediction"]] * len(rows[split]), train_plans
            )
        write_json(output / f"{split}_predictions.json", prediction_records(rows[split], predictions))
        if split == "test":
            result["test_evaluations"] += 1
    result["training_seconds"] = time.monotonic() - started
    write_json(output / "results.json", result)
    brief = dict(
        complete=True,
        training_seconds=result["training_seconds"],
        parameters=result["trainable_parameters"],
        selected=result["selected"],
        test_user_decision=result["evaluation"]["test"]["decision_user"],
        test_action_plan_exact=result["evaluation"]["test"]["action_plan_exact"],
    )
    write_json(output / "progress.json", brief)
    print(json.dumps(brief, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    train(
        args.features,
        args.dataset,
        args.output,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
