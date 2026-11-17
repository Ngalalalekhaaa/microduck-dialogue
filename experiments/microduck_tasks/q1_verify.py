"""Reload the selected adapter in a fresh GPU process and verify fixed examples."""

import argparse
import json
from pathlib import Path

from .q0_evaluate import generate
from .q0_model import load_model, weights_digest
from .q1_data import load
from .run_ablation import save


def reload_rows(rows):
    return [
        next(r for r in rows if r["row"]["group"] == "single"),
        next(r for r in rows if len(r["row"]["expected"]["goals"]) == 6),
        next(r for r in rows if r["row"]["group"] == "boundary"),
    ]


def verify(root, model_dir):
    work = root / "q1/train1200"
    result_path = work / "results.json"
    result = json.loads(result_path.read_text())
    if not result["training_complete"] or result["complete"]:
        raise ValueError("Require completed training with its first reload verification still pending.")
    _, rows = load(root / "q1/data", root / "h0/data")
    checkpoint = work / result["selected_checkpoint"]
    expected_weights = json.loads((checkpoint / "identity.json").read_text())["weights_sha256"]
    saved = json.loads((work / f"validation_{result['selected_step']}/report.json").read_text())
    predictions = {r["id"]: r for r in saved["predictions"]}
    directory = work / "reload_selected"
    directory.mkdir(exist_ok=False)
    model, processor, metadata = load_model(model_dir, directory)
    model.load_adapter(str(checkpoint), adapter_name="verified", is_trainable=False)
    model.set_adapter("verified")
    parameters = dict(model.named_parameters())
    verified = {n: parameters[n.replace(".default.", ".verified.")] for n in metadata["trainable_names"]}
    actual_weights = weights_digest(verified)
    if actual_weights != expected_weights:
        raise ValueError("Reloaded selected adapter weights differ from the saved checkpoint identity.")
    checks = []
    # Fixed input-based selection, independent of correctness or reload output.
    for row in reload_rows(rows["val"]):
        raw, timing = generate(model, processor, row)
        checks.append(
            dict(
                id=row["id"], raw_text=raw, exactly_same_output=raw == predictions[row["id"]]["raw_text"], timing=timing
            )
        )
    passed = all(r["exactly_same_output"] for r in checks)
    save(directory / "check.json", dict(passed=passed, weights_sha256=actual_weights, checks=checks))
    if not passed:
        raise ValueError("Selected adapter reload changed a predeclared development output.")
    result.update(complete=True, reload_pending=False, reload_pass=True)
    save(result_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    verify(args.root.resolve(), args.model.resolve())
