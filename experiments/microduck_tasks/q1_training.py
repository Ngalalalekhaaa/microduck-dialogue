"""Train a fresh P6 language LoRA on all 1200 rows for exactly two epochs."""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .ablation_contract import dumps
from .q0_model import load_model, loss, memory, weights_digest
from .q1_data import load
from .q1_evaluate import SELECTION, evaluate, selection_key
from .q1_schedule import batches, learning_rate, require_q0
from .run_ablation import save
from .toy_dataset import digest


def configuration(root):
    return dict(
        seed=17,
        samples=1200,
        epochs=2,
        batch_size=1,
        gradient_accumulation=16,
        total_microbatches=2400,
        optimizer_steps=150,
        validation_steps=[50, 100, 150],
        lr=1e-4,
        warmup_updates=8,
        scheduler="cosine",
        weight_decay=0.01,
        gradient_clip=1.0,
        input_view="P6",
        rank=8,
        alpha=16,
        lora_dropout=0.05,
        maximum_sequence_length=4096,
        padding_multiple=1024,
        initialization="fresh pinned base and fresh seed17 adapter; no Q0 trained weights",
        loss="mean answer JSON+EOS token CE per sample, then mean over 16 samples",
        selection=SELECTION,
        generation="greedy nonthinking P6; max_new_tokens768; max_time90s; no grammar or repairs",
        reload_examples="first development single-group row, first six-goal row, first boundary-group row",
        data_manifest_sha256=digest(root / "q1/data/manifest.json"),
        blind_access=False,
        human_reviewed=False,
        physical_executions=0,
    )


def run(root, model_dir):
    work = root / "q1"
    q0 = json.loads((root / "q0/run100/results.json").read_text())
    require_q0(q0)
    manifest, rows = load(work / "data", root / "h0/data")
    settings = configuration(root)
    if json.loads((work / "planned_config.json").read_text()) != settings:
        raise ValueError("Q1 configuration changed after queueing.")
    output = work / "train1200"
    output.mkdir(exist_ok=False)
    save(output / "training_config.json", settings)
    status = dict(started_at=datetime.now(UTC).isoformat(), pid=os.getpid(), complete=False, blind_access=False)

    def notify(stage, **fields):
        status.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **fields)
        save(work / "status.json", status)
        path = root / "CURRENT_STATUS.json"
        current = json.loads(path.read_text())
        current.update(
            state=stage,
            Q1_status="q1/status.json",
            updated_at=status["updated_at"],
            next_stage="Finish Q1 two-epoch development curve and selected adapter reload; keep blind sealed.",
        )
        save(path, current)
        print(dumps(status), flush=True)

    try:
        notify("Q1_loading")
        model, processor, _ = load_model(model_dir, output)
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        initial = weights_digest(params)
        if initial != q0["weights_before"]:
            raise ValueError("Q1 fresh adapter initialization differs from the audited Q0 initialization.")
        save(output / "initialization.json", dict(fresh_adapter=True, weights_sha256=initial, q0_weights_reused=False))
        optimizer = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=0.01)
        torch.manual_seed(17)
        started = time.monotonic()
        training_seconds, evaluation_seconds = 0.0, 0.0
        curve = []
        for step, (epoch, indices) in enumerate(batches(), 1):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            for index in indices:
                value = loss(model, rows["train"][index], processor.tokenizer.pad_token_id)
                (value / 16).backward()
                losses.append(float(value.detach()))
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite Q1 gradient norm.")
            optimizer.step()
            duration = time.monotonic() - tick
            training_seconds += duration
            average = training_seconds / (step * 16)
            eval_average = evaluation_seconds / len(curve) if curve else 240 * 6.4
            remaining = average * (2400 - step * 16) + eval_average * (3 - len(curve)) + 120
            record = dict(
                step=step,
                epoch=epoch,
                microbatches=step * 16,
                loss=float(np.mean(losses)),
                gradient_norm=float(norm),
                lr=optimizer.param_groups[0]["lr"],
                step_seconds=duration,
                training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds,
                elapsed_s=time.monotonic() - started,
                estimated_remaining_s=remaining,
                **memory(),
            )
            with (output / "history.jsonl").open("a") as handle:
                handle.write(dumps(record) + "\n")
            notify("Q1_training", **record)
            if step in settings["validation_steps"]:
                optimizer.zero_grad(set_to_none=True)
                checkpoint = output / f"checkpoint_{step}"
                model.save_pretrained(checkpoint, safe_serialization=True, save_embedding_layers=False)
                save(checkpoint / "identity.json", dict(step=step, weights_sha256=weights_digest(params)))
                # Preserve dropout RNG across evaluation, including any generation internals.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    tick = time.monotonic()
                    evaluation = evaluate(model, processor, rows["val"], output, step, notify)
                    evaluation_seconds += time.monotonic() - tick
                curve.append(evaluation)
                winner = max(curve, key=selection_key)
                save(output / "learning_curve.json", dict(checkpoints=curve, selected_step=winner["step"]))
                notify("Q1_checkpoint_evaluated", step=step, selected_step=winner["step"], validation_completed=240)
        after = weights_digest(params)
        if after == initial:
            raise ValueError("Q1 LoRA weights did not change.")
        winner = max(curve, key=selection_key)
        save(
            output / "results.json",
            dict(
                complete=False,
                training_complete=True,
                reload_pending=True,
                selected_step=winner["step"],
                selected_checkpoint=f"checkpoint_{winner['step']}",
                selected_validation=winner,
                prototype_gate_pass=winner["all240"]["prototype_gate_pass"],
                checkpoints=curve,
                training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds,
                total_seconds=time.monotonic() - started,
                weights_before=initial,
                weights_after=after,
                data_sha256={s: info["sha256"] for s, info in manifest["splits"].items()},
                blind_access=False,
                physical_executions=0,
            ),
        )
        notify("Q1_training_complete_reload_pending", training_complete=True, selected_step=winner["step"])
    except Exception as error:
        notify("Q1_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    run(args.root.resolve(), args.model.resolve())
