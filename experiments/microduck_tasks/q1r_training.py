"""Continue the exact selected Q1 adapter with grounded dialogue and planning."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .ablation_contract import dumps
from .q0_model import loss, memory, smoke, weights_digest
from .q1r_encoding import load
from .q1r_evaluate import evaluate, screen, selection_key
from .q1r_prompt import metadata
from .q1r_training_state import Status, batches, checkpoint, learning_rate, load_continuation, restore
from .run_ablation import save
from .toy_dataset import digest


def configuration(root, parent, samples):
    steps_per_epoch = math.ceil(samples / 16)
    return dict(
        seed=1611,
        samples=samples,
        epochs=2,
        batch_size=1,
        gradient_accumulation=16,
        optimizer_steps=steps_per_epoch * 2,
        total_microbatches=samples * 2,
        full_validation_steps=[steps_per_epoch, steps_per_epoch * 2],
        screen_step=50,
        lr=5e-5,
        warmup_updates=min(20, steps_per_epoch),
        weight_decay=0.01,
        gradient_clip=1.0,
        scheduler="linear warmup then cosine to 5% of peak",
        checkpoint_interval=50,
        maximum_sequence_length=4096,
        padding_multiple=1024,
        parent_checkpoint=str(parent.resolve()),
        data_manifest_sha256=digest(root / "encoded/manifest.json"),
        prompt=metadata(),
        initialization="exact selected Q1 checkpoint_100 weights; new optimizer",
        loss="all answer JSON+EOS token CE averaged per row, then over actual accumulation size",
        selection="full held-out: false starts, validity, semantic plan, premature claims, reply presence, early step",
        screen_does_not_select=True,
        images_passed_to_model=False,
        blind_access=False,
        human_reviewed=False,
    )


def run(root, parent, model_dir, resume=None):
    manifest, rows = load(root / "encoded")
    settings = configuration(root, parent, len(rows["train"]))
    output = root / "train"
    if resume is None:
        output.mkdir(exist_ok=False)
        save(output / "training_config.json", settings)
    elif json.loads((output / "training_config.json").read_text()) != settings:
        raise ValueError("Continuation configuration changed; preserve the old run and inspect it.")
    notify = Status(root)
    attempt = time.time_ns()
    try:
        notify(
            "loading",
            total_steps=settings["optimizer_steps"],
            train_samples=len(rows["train"]),
            validation_samples=len(rows["val"]),
        )
        model, processor, params = load_continuation(model_dir, parent, output)
        initial = weights_digest(params)
        if resume is None:
            notify("smoke", detail="longest padded input and longest answer; real update, then exact restore")
            all_rows = rows["train"] + rows["val"]
            extremes = [
                max(all_rows, key=lambda r: (r["padded_tokens"], r["total_tokens"])),
                max(all_rows, key=lambda r: r["supervised_tokens"]),
            ]
            report = smoke(model, processor, extremes, output)
            report.update(stage="Q1R_continuation_smoke", restored_before_formal_q1r=True)
            report["extreme_selection"] = ["largest padded input then total", "largest supervised answer"]
            report.pop("restored_before_formal_q0", None)
            save(output / "smoke.json", report)
        optimizer = torch.optim.AdamW(params.values(), lr=settings["lr"], weight_decay=settings["weight_decay"])
        torch.manual_seed(settings["seed"])
        np.random.seed(settings["seed"])
        random.seed(settings["seed"])
        counters = dict(microbatches=0, training_seconds=0.0, evaluation_seconds=0.0)
        completed = 0
        if resume:
            completed, counters = restore(resume, model, optimizer, params, settings)
        curve_path = output / "learning_curve.json"
        curve = json.loads(curve_path.read_text())["checkpoints"] if curve_path.exists() else []
        if curve and any(item["step"] > completed for item in curve):
            raise ValueError("Resume must not move behind already evaluated checkpoints.")

        def validate_step(step):
            full = step in settings["full_validation_steps"]
            if not full and step != settings["screen_step"]:
                return
            directory = output / f"validation_{step}_{'full' if full else 'screen'}"
            report_path = directory / "report.json"
            if report_path.exists():
                evaluation = json.loads(report_path.read_text())
            else:
                if directory.exists():
                    directory.rename(output / f"{directory.name}_interrupted_{time.time_ns()}")
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    tick = time.monotonic()
                    selected = rows["val"] if full else screen(rows["val"])
                    evaluation = evaluate(model, processor, selected, output, step, notify, full=full)
                    counters["evaluation_seconds"] += time.monotonic() - tick
            if full and not any(item["step"] == step for item in curve):
                curve.append(evaluation)
                best = max(curve, key=selection_key)
                save(curve_path, dict(checkpoints=curve, selected_step=best["step"]))

        if completed:
            validate_step(completed)
        for step, (epoch, indices) in enumerate(
            batches(settings["samples"], settings["epochs"], settings["gradient_accumulation"], settings["seed"]), 1
        ):
            if step <= completed:
                continue
            rate = learning_rate(step, settings["optimizer_steps"], settings["lr"], settings["warmup_updates"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            tick, losses = time.monotonic(), []
            for offset, index in enumerate(indices, 1):
                value = loss(model, rows["train"][index], processor.tokenizer.pad_token_id)
                (value / len(indices)).backward()
                losses.append(float(value.detach()))
                if offset % 8 == 0:
                    notify(
                        "training",
                        step=step - 1,
                        optimizer_step_in_progress=step,
                        epoch=epoch,
                        microbatch_in_step=offset,
                        microbatches=counters["microbatches"] + offset,
                    )
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), settings["gradient_clip"])
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite Q1R gradient norm; no optimizer update was applied.")
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.monotonic() - tick
            counters["training_seconds"] += duration
            counters["microbatches"] += len(indices)
            if step == 1 and weights_digest(params) == initial:
                raise ValueError("First continuation optimizer update did not change Q1 weights.")
            per_row = counters["training_seconds"] / counters["microbatches"]
            remaining_train = per_row * (settings["total_microbatches"] - counters["microbatches"])
            remaining_eval = sum(s >= step for s in settings["full_validation_steps"]) * len(rows["val"]) * 5
            record = dict(
                attempt_id=attempt,
                step=step,
                epoch=epoch,
                loss=float(np.mean(losses)),
                gradient_norm=float(norm),
                lr=rate,
                step_seconds=duration,
                **counters,
                estimated_remaining_s=remaining_train + remaining_eval,
                eta_basis="measured training mean plus 5 seconds per remaining validation row",
                **memory(),
            )
            with (output / "history.jsonl").open("a") as handle:
                handle.write(dumps(record) + "\n")
            notify("training", **record)
            if step % settings["checkpoint_interval"] == 0 or step in settings["full_validation_steps"]:
                optimizer.zero_grad(set_to_none=True)
                path = checkpoint(model, optimizer, params, output, step, settings, counters)
                notify("checkpoint_saved", checkpoint=str(path), step=step)
                validate_step(step)
        after = weights_digest(params)
        if after == initial:
            raise ValueError("Continuation ended with unchanged parent weights.")
        best = max(curve, key=selection_key)
        save(
            output / "results.json",
            dict(
                complete=False,
                training_complete=True,
                reload_pending=True,
                selected_step=best["step"],
                selected_checkpoint=f"checkpoint_{best['step']:06d}",
                selected_validation=best,
                checkpoints=curve,
                **counters,
                weights_before=initial,
                weights_after=after,
                data_sha256={split: info["sha256"] for split, info in manifest["splits"].items()},
                blind_access=False,
                physical_executions=0,
                deployment_ready=False,
            ),
        )
        notify("training_complete_reload_pending", training_complete=True, selected_step=best["step"])
    except Exception as error:
        notify("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.parent.resolve(), args.model.resolve(), args.resume)
