"""Fit H2 with the H0/H1 labels, loss, hyperparameters and checkpoint rule."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .ablation_contract import dumps
from .h0_training import SEEDS, prediction_report
from .h2_data import MemoryData, infer
from .h2_model import MODEL_VERSION, AttentionSelector
from .run_ablation import save
from .toy_dataset import digest
from .toy_protocol import DECISIONS
from .toy_selector_labels import FIELDS, TOKENS, canonical_plan, encode_targets
from .toy_selector_metrics import evaluate
from .toy_selector_model import training_loss
from .toy_selector_training import loss_weights, selection_key, weight_audit


def fit(data, labels, selection, seed, log, *, epochs=150, device="cuda"):
    torch.manual_seed(seed)
    model = AttentionSelector()
    data.normalise(model)
    model.to(device)
    weights = [value.to(device) for value in loss_weights(labels)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    plans = {canonical_plan(r["target"]["goals"]) for r in data.rows["train"] if r["target"]["goals"]}
    best_state, best_key, best_epoch = None, (-1, -1, -1), 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(data.rows["train"]))
        total_loss = 0.0
        for start in range(0, len(order), 128):
            indices = order[start : start + 128]
            batch_labels = {k: v[indices].to(device) for k, v in labels.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(*data.batch("train", indices.numpy(), device), teacher=batch_labels)
            loss = training_loss(output, batch_labels, *weights)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite H2 loss; investigate memory and training.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        if epoch % 5 and epoch != epochs:
            continue
        predictions, raw = infer(model, data, "val", selection)
        metrics = evaluate([data.rows["val"][i] for i in selection], predictions, plans, raw_tokens=raw)
        key = selection_key(metrics)
        if key > best_key:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_key, best_epoch = key, epoch
        log(
            dict(
                seed=seed,
                epoch=epoch,
                train_loss=total_loss / len(order),
                val_user_decision_macro_f1=metrics["decision_user"]["macro_f1_present"],
                val_action_plan_exact=metrics["action_plan_exact"],
                val_goal_token_accuracy=metrics["action_fields"].get("semantic_token"),
                selection_score=key[0],
                best_epoch=best_epoch,
                elapsed_s=time.monotonic() - started,
            )
        )
        if epoch - best_epoch >= 30:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(
        config=dict(hidden=128, lr=0.001, weight_decay=0.01, seed=seed), best_epoch=best_epoch, key=list(best_key)
    )


def train(dataset, features, memory, output, *, epochs=150, device="cuda"):
    torch.set_num_threads(2)
    data = MemoryData(dataset, features, memory)
    rows = data.rows
    selection = [i for i, row in enumerate(rows["val"]) if row["domain"] == "common"]
    labels = {k: torch.from_numpy(v) for k, v in encode_targets(rows["train"]).items()}
    val_labels = encode_targets([rows["val"][i] for i in selection])
    for name in ("decision", "token", *FIELDS):
        if set(val_labels[name].ravel().tolist()) - {-100} - set(labels[name].ravel().tolist()):
            raise ValueError(f"Unsupported validation class for {name}.")
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(
        hidden=128,
        seeds=list(SEEDS),
        lr=0.001,
        weight_decay=0.01,
        epochs=epochs,
        batch_size=128,
        patience_epochs=30,
        evaluate_every_epochs=5,
        device=device,
        input_view="P6",
        decision_branch="unchanged last-input-token global projection",
        selection="same H0/H1 selection_key; common validation only; seed tie first",
        selection_samples=len(selection),
        total_validation_samples=len(rows["val"]),
        blind_access=False,
        human_reviewed=False,
        offline_diagnostic=True,
        normalisation="global H0 train moments; memory train valid slots only; std floor 0.1",
        memory_manifest_sha256=digest(memory / "manifest.json"),
        global_manifest_sha256=digest(features / "manifest.json"),
        dataset_manifest_sha256=digest(dataset / "manifest.json"),
    )
    snapshot = output / "implementation"
    snapshot.mkdir()
    names = [
        "h2_training.py",
        "h2_data.py",
        "h2_model.py",
        "h0_training.py",
        "toy_selector_model.py",
        "toy_selector_training.py",
        "toy_selector_labels.py",
        "ablation_score.py",
    ]
    for name in names:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    settings["source_sha256"] = {name: digest(snapshot / name) for name in names}
    save(output / "training_config.json", settings)
    save(output / "loss_weight_audit.json", weight_audit(labels))
    plans = {canonical_plan(r["target"]["goals"]) for r in rows["train"] if r["target"]["goals"]}
    trials, started = [], time.monotonic()
    for seed in SEEDS:
        run = output / f"seed_{seed}"
        run.mkdir()

        def log(record, run=run):
            with (run / "history.jsonl").open("a") as handle:
                handle.write(dumps(record) + "\n")
            save(output / "progress.json", dict(stage="H2_training", **record))
            print(dumps(record), flush=True)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model, trial = fit(data, labels, selection, seed, log, epochs=epochs, device=device)
        checkpoint = dict(
            version=MODEL_VERSION,
            state_dict={k: v.cpu() for k, v in model.state_dict().items()},
            dim=model.dim,
            hidden=model.hidden,
            decisions=list(DECISIONS),
            tokens=list(TOKENS),
            fields={k: list(v) for k, v in FIELDS.items()},
            feature_config=data.global_manifest["config"],
            memory_config=data.manifest["config"],
            training_config=settings,
            selected=trial,
            contract_sha256=data.manifest["contract_sha256"],
            deployment_ready=False,
            blind_access=False,
        )
        torch.save(checkpoint, run / "selector.pt")
        predictions, raw = infer(model, data, "val", list(range(len(rows["val"]))))
        common_rows = [rows["val"][i] for i in selection]
        reports = dict(
            all240=prediction_report(rows["val"], predictions),
            common236=prediction_report(common_rows, [predictions[i] for i in selection]),
            oracle_decision_diagnostic=prediction_report(
                rows["val"], infer(model, data, "val", list(range(len(rows["val"]))), oracle=True)[0]
            ),
        )
        for name, report in reports.items():
            save(run / f"{name}.json", report)
        train_predictions, train_raw = infer(model, data, "train", list(range(len(rows["train"]))))
        summary = dict(
            seed=seed,
            trial=trial,
            parameters=sum(p.numel() for p in model.parameters()),
            checkpoint_sha256=digest(run / "selector.pt"),
            train=evaluate(rows["train"], train_predictions, plans, raw_tokens=train_raw),
            common_validation=evaluate(
                common_rows, [predictions[i] for i in selection], plans, raw_tokens=[raw[i] for i in selection]
            ),
            validation=reports["all240"]["summary"],
            oracle_decision=reports["oracle_decision_diagnostic"]["summary"],
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
            blind_access=False,
            deployment_ready=False,
        )
        save(run / "summary.json", summary)
        trials.append(summary)
        del model
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "features", "memory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    train(args.dataset, args.features, args.memory, args.output, epochs=args.epochs, device=args.device)
