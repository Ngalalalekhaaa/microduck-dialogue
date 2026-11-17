"""Compare fixed H-stage predictions with paired semantic-family resampling."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .run_ablation import save
from .toy_dataset import digest

RUNS = dict(H0="h0/train128", H1="h1/train256", H2="h2/train128_attention")
METRICS = ("context_valid", "action_plan_exact", "three_six_exact", "nonaction_attempted_start")


def paired(older, newer, *, repetitions=5000):
    grouped = defaultdict(list)
    if len(older) != len(newer):
        raise ValueError("Paired comparison requires equal example counts.")
    for a, b in zip(older, newer, strict=True):
        if a["id"] != b["id"] or a["expected"] != b["expected"]:
            raise ValueError("Paired IDs or labels differ.")
        if a["domain"] == "common" and a["expected"]["decision"] in ("replace", "append"):
            grouped[a["family_id"]].append(int(b["score"]["accepted_exact"]) - int(a["score"]["accepted_exact"]))
    values = np.array([(sum(v), len(v)) for v in grouped.values()])
    rng = np.random.default_rng(17043)
    choices = rng.integers(len(values), size=(repetitions, len(values)))
    sums = values[choices].sum(1)
    delta = sums[:, 0] / sums[:, 1]
    return dict(
        common_action_rows=int(values[:, 1].sum()),
        semantic_families=len(values),
        paired_rate_difference=float(values[:, 0].sum() / values[:, 1].sum()),
        family_bootstrap_95_percentile=np.quantile(delta, [0.025, 0.975]).tolist(),
        improved=sum(v == 1 for values in grouped.values() for v in values),
        regressed=sum(v == -1 for values in grouped.values() for v in values),
        repetitions=repetitions,
        bootstrap_seed=17043,
        limitation="Exploratory interval over synthetic development families; not an independent-test guarantee.",
    )


def compare(root):
    results, predictions, hashes = {}, {}, {}
    for stage, path in RUNS.items():
        folder = root / path
        result = json.loads((folder / "results.json").read_text())
        if not result["complete"]:
            raise ValueError("Finish all three H stages before comparing.")
        hashes[stage] = digest(folder / "results.json")
        results[stage] = dict(
            selected_seed=result["selected_seed"],
            action_mean=result["action_exact_mean"],
            action_std=result["action_exact_std"],
            elapsed_training_and_reporting_s=result["elapsed_s"],
            seeds=[],
        )
        predictions[stage] = {}
        for trial in result["seeds"]:
            seed = trial["seed"]
            predictions[stage][seed] = json.loads((folder / f"seed_{seed}/all240.json").read_text())["predictions"]
            results[stage]["seeds"].append(
                dict(
                    seed=seed,
                    parameters=trial["parameters"],
                    best_epoch=trial["trial"]["best_epoch"],
                    train_action_exact=trial["train"]["action_plan_exact"],
                    validation={k: trial["validation"]["metrics"][k] for k in METRICS},
                    novelty=trial["common_validation"]["action_by_plan_novelty"],
                    oracle_action=trial["oracle_decision"]["metrics"]["action_plan_exact"],
                    prototype_gate_pass=trial["validation"]["prototype_gate_pass"],
                )
            )
    paired_results = {
        f"{a}_to_{b}": {str(seed): paired(predictions[a][seed], predictions[b][seed]) for seed in (17, 43, 71)}
        for a, b in (("H0", "H1"), ("H0", "H2"), ("H1", "H2"))
    }
    costs = {}
    for stage, path in (("H0", "h0/features_p6"), ("H2", "h2/memory")):
        folder = root / path
        records = sum([json.loads((folder / f"{s}_records.json").read_text()) for s in ("train", "val")], [])
        times = [r["seconds"] for r in records]
        costs[stage] = dict(
            samples=len(records),
            encode_and_pool_seconds_p50_p95=np.quantile(times, [0.5, 0.95]).tolist(),
            final_cache_bytes=sum(f.stat().st_size for f in folder.glob("*.np*")),
            scope="Input encoding and pooling only; no head decoding, cold model load or file writing.",
        )
    report = dict(
        results=results,
        paired=paired_results,
        feature_costs=costs,
        source_sha256=hashes,
        blind_test_used=False,
        physical_executions=0,
        development_set_previously_used_for_prompt_tuning=True,
    )
    save(root / "head_comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    compare(parser.parse_args().root)
