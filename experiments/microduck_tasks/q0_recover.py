"""Complete an interrupted Q0 reload in a fresh process without retraining or rescoring replacements."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from safetensors.torch import load_file

from .ablation_contract import checksum
from .ablation_score import score, summarize
from .q0_data import load
from .q0_evaluate import generate
from .q0_model import load_model, weights_digest
from .q1_schedule import require_q0
from .run_ablation import save
from .toy_dataset import digest


def audit_report(path, rows):
    report = json.loads(path.read_text())
    records = report["predictions"]
    if [r["id"] for r in records] != [r["id"] for r in rows]:
        raise ValueError("Saved Q0 generations do not cover the frozen subset exactly in order.")
    for record, row in zip(records, rows, strict=True):
        if (
            any(record[k] != v for k, v in row["row"].items())
            or record["prefix_sha256"] != row["prefix_sha256"]
            or record["score"] != score(record["raw_text"], row["row"], plan_only=True)
            or record["generation_error"] is not None
        ):
            raise ValueError("Saved Q0 generation input, source or original score changed.")
    if report["summary"] != summarize(records, expected_count=32):
        raise ValueError("Saved Q0 report summary differs from its original raw predictions.")
    return report


def checkpoint_parameters(checkpoint, names):
    saved = load_file(checkpoint / "adapter_model.safetensors", device="cpu")
    keys = [n.replace(".default.", ".") for n in names]
    if not names or len(set(keys)) != len(names) or set(keys) != set(saved):
        raise ValueError("Require every saved LoRA tensor exactly once, with no extra or missing names.")
    return {name: saved[key] for name, key in zip(names, keys, strict=True)}


def run(root, model_dir):
    work = root / "q0"
    output = work / "run100"
    recovery = output / "reload_recovery1"
    recovery.mkdir(exist_ok=False)
    prior = json.loads((work / "status.json").read_text())
    save(recovery / "prior_status.json", prior)
    status = dict(
        started_at=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        complete=False,
        recovery_from="same-process reload OOM after all 100 updates and all 32 final generations",
        training_complete=True,
        optimizer_steps=100,
        blind_access=False,
        physical_executions=0,
    )

    def notify(stage, **fields):
        status.update(stage=stage, updated_at=datetime.now(UTC).isoformat(), **fields)
        save(work / "status.json", status)
        current = root / "CURRENT_STATUS.json"
        old = json.loads(current.read_text())
        old.update(state=stage, updated_at=status["updated_at"], Q0_status="q0/status.json")
        save(current, old)

    try:
        notify("Q0_recovering_reload")
        manifest, rows = load(work / "data")
        history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
        if [r["step"] for r in history] != list(range(1, 101)) or history[-1]["microbatches"] != 800:
            raise ValueError("Q0 requires its full completed training history before reload recovery.")
        for name in ("q0_data.py", "q0_model.py", "q0_evaluate.py", "ablation_score.py"):
            if digest(Path(__file__).with_name(name)) != digest(output / "implementation" / name):
                raise ValueError(
                    "Recovery must keep the original Q0 model, data and generation/scoring implementation."
                )
        before = audit_report(output / "Q0_before_training/report.json", rows)
        after = audit_report(output / "Q0_after_training/report.json", rows)
        checkpoint = output / "checkpoint_100"
        evidence = dict(
            original_before_report_sha256=digest(output / "Q0_before_training/report.json"),
            original_after_report_sha256=digest(output / "Q0_after_training/report.json"),
            checkpoint_sha256=digest(checkpoint / "adapter_model.safetensors"),
            original_history_sha256=digest(output / "history.jsonl"),
            recovery_implementation_sha256=digest(__file__),
            rerun_training_updates=0,
            replacement_predictions=0,
            fresh_process=True,
        )
        save(recovery / "evidence.json", evidence)
        model, processor, metadata = load_model(model_dir, recovery)
        original = json.loads((output / "model_preflight.json").read_text())
        names = metadata["trainable_names"]
        if names != original["trainable_names"]:
            raise ValueError("Recovery adapter modules differ from the trained adapter.")
        initial = weights_digest({n: p for n, p in model.named_parameters() if p.requires_grad})
        smoke = json.loads((work / "probe/smoke.json").read_text())
        if initial != smoke["weights_before"]:
            raise ValueError("Reconstructed seed17 initialization differs from the saved preflight initialization.")
        saved_hash = weights_digest(checkpoint_parameters(checkpoint, names))
        model.load_adapter(str(checkpoint), adapter_name="verified", is_trainable=False)
        model.set_adapter("verified")
        parameters = dict(model.named_parameters())
        loaded_hash = weights_digest({n: parameters[n.replace(".default.", ".verified.")] for n in names})
        if saved_hash != loaded_hash or initial == loaded_hash:
            raise ValueError("Reloaded adapter must equal all saved weights and differ from initial weights.")
        checks = []
        for group in ("single", "multi6", "boundary"):
            row = next(r for r in rows if r["row"]["group"] == group)
            expected = next(r["raw_text"] for r in after["predictions"] if r["id"] == row["id"])
            raw, timing = generate(model, processor, row)
            checks.append(dict(id=row["id"], raw_text=raw, exactly_same_output=raw == expected, timing=timing))
            notify("Q0_recovering_reload", reload_completed=len(checks), reload_total=3)
        passed = all(r["exactly_same_output"] for r in checks)
        check = dict(passed=passed, checks=checks, weights_sha256=loaded_hash, fresh_process=True)
        save(recovery / "reload_check.json", check)
        if not passed:
            raise ValueError("Reloaded Q0 does not reproduce all three predeclared raw outputs.")
        result = dict(
            complete=True,
            before=before["summary"],
            after=after["summary"],
            overfit_check_pass=True,
            reload_pass=True,
            weights_before=initial,
            weights_after=loaded_hash,
            blind_access=False,
            generalisation_scored=False,
            optimizer_steps=100,
            total_microbatches=800,
            data_sha256=manifest["data_sha256"],
            physical_executions=0,
            recovery=dict(
                **evidence,
                original_failure="reload_recovery1/prior_status.json",
                initialization_hash_source="reconstructed fresh seed17 equals saved probe initial hash",
            ),
        )
        require_q0(result)
        if checksum(after["summary"]) != checksum(
            json.loads((output / "Q0_after_training/report.json").read_text())["summary"]
        ):
            raise ValueError("Original report changed during recovery.")
        save(output / "reload_check.json", check)
        save(output / "results.json", result)
        notify("Q0_complete", complete=True, overfit_check_pass=True, reload_pass=True, result="q0/run100/results.json")
    except Exception as error:
        notify("Q0_recovery_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    run(args.root.resolve(), args.model.resolve())
