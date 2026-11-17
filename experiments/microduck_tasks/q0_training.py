"""Run the predeclared 100-update Q0 overfit check after a passed full-step probe."""

import argparse
import gc
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .ablation_contract import dumps
from .q0_data import load
from .q0_evaluate import evaluate, generate
from .q0_model import load_model, loss, memory, weights_digest
from .run_ablation import save
from .toy_dataset import digest


def run(root, model_dir):
    work = root / "q0"
    probe = json.loads((work / "probe/status.json").read_text())
    if not probe["complete"] or not probe["passed"]:
        raise ValueError("Pass the longest-sample forward/backward/update probe first.")
    manifest, rows = load(work / "data")
    output = work / "run100"
    output.mkdir(exist_ok=False)
    settings = dict(
        seed=17,
        optimizer_steps=100,
        gradient_accumulation=8,
        total_microbatches=800,
        lr=1e-4,
        warmup_updates=5,
        scheduler="cosine",
        weight_decay=0.01,
        gradient_clip=1.0,
        input_view="P6",
        rank=8,
        alpha=16,
        lora_dropout=0.05,
        samples=32,
        maximum_sequence_length=4096,
        padding_multiple=1024,
        batch_size=1,
        selection="Use final step100; no development or blind selection",
        overfit_check="all-plan exact>=90%, action exact>=90%, schema/context100%, no nonaction false starts",
        generation="same greedy P6, nonthinking prefix, max_new_tokens768, max_time90s, no grammar or repair",
        data_manifest_sha256=digest(work / "data/manifest.json"),
        blind_access=False,
        generalisation_scored=False,
        human_reviewed=False,
    )
    snapshot = output / "implementation"
    snapshot.mkdir()
    names = ["q0_training.py", "q0_model.py", "q0_data.py", "q0_evaluate.py", "head_features.py", "ablation_score.py"]
    for name in names:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    settings["source_sha256"] = {n: digest(snapshot / n) for n in names}
    save(output / "training_config.json", settings)
    status = dict(
        started_at=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        complete=False,
        blind_access=False,
        physical_executions=0,
    )

    def notify(stage, **fields):
        status.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **fields)
        save(work / "status.json", status)
        current_path = root / "CURRENT_STATUS.json"
        current = json.loads(current_path.read_text())
        current.update(
            state=stage,
            Q0_status="q0/status.json",
            updated_at=status["updated_at"],
            next_stage="Finish Q0 training-subset generation and adapter reload check; keep blind sealed.",
        )
        save(current_path, current)
        print(dumps(status), flush=True)

    try:
        notify("Q0_loading")
        model, processor, _ = load_model(model_dir, output)
        before = evaluate(model, processor, rows, output, "Q0_before_training", notify)
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        initial = weights_digest(params)
        optimizer = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=0.01)
        torch.manual_seed(17)
        rng = np.random.default_rng(17)
        order = []
        started = time.monotonic()
        for step in range(1, 101):
            if step <= 5:
                scale = step / 5
            else:
                scale = 0.5 * (1 + math.cos(math.pi * (step - 6) / 95))
            for group in optimizer.param_groups:
                group["lr"] = 1e-4 * scale
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            for _ in range(8):
                if not order:
                    order = rng.permutation(len(rows)).tolist()
                row = rows[order.pop()]
                value = loss(model, row, processor.tokenizer.pad_token_id)
                (value / 8).backward()
                losses.append(float(value.detach()))
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite LoRA gradient norm.")
            optimizer.step()
            record = dict(
                step=step,
                microbatches=step * 8,
                loss=float(np.mean(losses)),
                gradient_norm=float(norm),
                lr=optimizer.param_groups[0]["lr"],
                step_seconds=time.monotonic() - tick,
                elapsed_s=time.monotonic() - started,
                **memory(),
            )
            with (output / "history.jsonl").open("a") as handle:
                handle.write(dumps(record) + "\n")
            notify("Q0_training", **record)
            if step in (50, 100):
                model.save_pretrained(
                    output / f"checkpoint_{step}", safe_serialization=True, save_embedding_layers=False
                )
        optimizer.zero_grad(set_to_none=True)
        del value, norm
        after_weights = weights_digest(params)
        if after_weights == initial:
            raise ValueError("Q0 adapter did not change.")
        after = evaluate(model, processor, rows, output, "Q0_after_training", notify)
        selected = [next(r for r in rows if r["row"]["group"] == group) for group in ("single", "multi6", "boundary")]
        expected = {r["id"]: next(p["raw_text"] for p in after["predictions"] if p["id"] == r["id"]) for r in selected}
        del model, params, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        released = memory()
        save(output / "before_reload_memory.json", released)
        if released["allocated_mib"] > 512:
            raise RuntimeError("Old Q0 GPU tensors remain alive; release them before reloading the base model.")
        notify("Q0_reloading")
        reload_folder = output / "reload"
        reload_folder.mkdir()
        restored, processor, _ = load_model(model_dir, reload_folder)
        restored.load_adapter(str(output / "checkpoint_100"), adapter_name="verified", is_trainable=False)
        restored.set_adapter("verified")
        checks = []
        for row in selected:
            raw, timing = generate(restored, processor, row)
            checks.append(
                dict(id=row["id"], raw_text=raw, exactly_same_output=raw == expected[row["id"]], timing=timing)
            )
        reload_pass = all(r["exactly_same_output"] for r in checks)
        save(output / "reload_check.json", dict(passed=reload_pass, checks=checks))
        metrics = after["summary"]["metrics"]
        fitted = (
            metrics["accepted_exact"]["rate"] >= 0.9
            and metrics["action_plan_exact"]["rate"] >= 0.9
            and metrics["schema_valid"]["rate"] == 1
            and metrics["context_valid"]["rate"] == 1
            and metrics["nonaction_attempted_start"]["correct"] == 0
        )
        result = dict(
            complete=True,
            before=before["summary"],
            after=after["summary"],
            overfit_check_pass=bool(fitted and reload_pass),
            reload_pass=reload_pass,
            weights_before=initial,
            weights_after=after_weights,
            blind_access=False,
            generalisation_scored=False,
            optimizer_steps=100,
            total_microbatches=800,
            data_sha256=manifest["data_sha256"],
            physical_executions=0,
        )
        save(output / "results.json", result)
        notify(
            "Q0_complete",
            complete=True,
            overfit_check_pass=result["overfit_check_pass"],
            result="q0/run100/results.json",
        )
    except Exception as error:
        notify("Q0_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    run(args.root.resolve(), args.model.resolve())
