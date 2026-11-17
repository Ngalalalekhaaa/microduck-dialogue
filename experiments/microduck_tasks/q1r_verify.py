"""Fresh-process adapter reload and development-only post-training replay."""

import argparse
import json
import time
from pathlib import Path

from .q0_model import load_model, weights_digest
from .q1r_compare import compare
from .q1r_encoding import encode, load
from .q1r_evaluate import evaluate, generate
from .q1r_training_state import Status
from .run_ablation import save


def verify(root, model_dir):
    work = root / "train"
    result_path = work / "results.json"
    result = json.loads(result_path.read_text())
    if not result["training_complete"] or result["complete"]:
        raise ValueError("Require finished training with the first reload check still pending.")
    _, splits = load(root / "encoded")
    checkpoint = work / result["selected_checkpoint"]
    directory = work / "reload_selected"
    if directory.exists():
        directory.rename(work / f"reload_interrupted_{time.time_ns()}")
    directory.mkdir(exist_ok=False)
    notify = Status(root)
    notify("reload_check", selected_step=result["selected_step"])
    try:
        model, processor, _ = load_model(model_dir, directory)
        loaded = model.load_adapter(str(checkpoint), adapter_name="default", is_trainable=True, local_files_only=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError("Selected adapter reload has missing or unexpected weights.")
        model.set_adapter("default")
        params = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        actual = weights_digest(params)
        if actual != json.loads((checkpoint / "identity.json").read_text())["weights_sha256"]:
            raise ValueError("Selected adapter weight identity differs after fresh-process reload.")
        model.requires_grad_(False)
        model.eval()
        predictions = {
            item["id"]: item
            for line in (work / f"validation_{result['selected_step']}_full/predictions.jsonl").read_text().splitlines()
            for item in [json.loads(line)]
        }
        rows = splits["val"]
        selected = [rows[0], rows[len(rows) // 2], rows[-1]]
        checks = []
        for row in selected:
            raw, timing = generate(model, processor, row)
            checks.append(
                dict(
                    id=row["id"],
                    raw_text=raw,
                    timing=timing,
                    exactly_same_output=raw == predictions[row["id"]]["raw_text"],
                )
            )
        passed = all(row["exactly_same_output"] for row in checks)
        save(directory / "check.json", dict(passed=passed, weights_sha256=actual, checks=checks))
        if not passed:
            raise ValueError("Reload changed a predeclared validation output; inspect before using this adapter.")
        result.update(reload_pending=False, reload_pass=True)
        save(result_path, result)
        suites = [root / "suite.jsonl"]
        probes = {}
        for path in suites:
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["id"] in probes and probes[row["id"]]["request"] != row["request"]:
                    raise ValueError("Development replay contains conflicting IDs.")
                probes[row["id"]] = row
        if not probes:
            raise ValueError("The original development stress suite must be saved for post-training replay.")
        encoded = [encode(processor, row) for row in probes.values()]
        comparison = root / "post_training"
        if comparison.exists():
            comparison.rename(root / f"post_training_interrupted_{time.time_ns()}")
        comparison.mkdir(exist_ok=False)
        replay = evaluate(model, processor, encoded, comparison, result["selected_step"], notify, full=True)
        replay.update(
            role="development replay of observed failure families; not a blind generalization test",
            checkpoint_selection_used=False,
            human_reviewed=False,
        )
        save(comparison / "report.json", replay)
        compare(root, comparison / f"validation_{result['selected_step']}_full/predictions.jsonl")
        result.update(complete=True, development_replay="../post_training/report.json")
        save(result_path, result)
        save(
            root / "selected_model.json",
            dict(
                checkpoint=str(checkpoint),
                model_dir=str(model_dir),
                weights_sha256=actual,
                reply_source="model_generation",
                validated_reload=True,
                simulation_only=True,
                deployment_ready=False,
                full_validation=result["selected_validation"],
                development_replay=replay,
            ),
        )
        notify(
            "complete",
            complete=True,
            training_complete=True,
            reload_pass=True,
            selected_step=result["selected_step"],
            selected_checkpoint=str(checkpoint),
        )
    except Exception as error:
        notify("verification_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    verify(args.root.resolve(), args.model.resolve())
