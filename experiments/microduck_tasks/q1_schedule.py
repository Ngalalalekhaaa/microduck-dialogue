"""Fixed two-epoch Q1 sampling and Q0 continuation conditions."""

import math

import numpy as np


def batches(count=1200, epochs=2, accumulation=16, seed=17):
    if count <= 0 or epochs <= 0 or accumulation <= 0 or count % accumulation:
        raise ValueError("Each complete epoch must divide into full accumulation groups.")
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(count).tolist()
        for offset in range(0, count, accumulation):
            yield epoch, order[offset : offset + accumulation]


def learning_rate(step, total=150, warmup=8, maximum=1e-4):
    if not 1 <= step <= total or not 1 <= warmup < total - 1:
        raise ValueError("Optimizer step or warmup outside the fixed Q1 schedule.")
    scale = (
        step / warmup if step <= warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup - 1) / (total - warmup - 1)))
    )
    return maximum * scale


def require_q0(result):
    if not all(result.get(k) for k in ("complete", "overfit_check_pass", "reload_pass")):
        raise ValueError("Q0 did not pass its fit and adapter reload checks; diagnose Q0 before scaling.")
    metrics = result["after"]["metrics"]
    if (
        result["after"]["completed"] != 32
        or result["after"]["expected"] != 32
        or result["after"]["errors"] != 0
        or result["optimizer_steps"] != 100
        or result["total_microbatches"] != 800
        or result["weights_before"] == result["weights_after"]
        or metrics["accepted_exact"]["rate"] < 0.9
        or metrics["action_plan_exact"]["rate"] < 0.9
        or metrics["schema_valid"]["rate"] != 1
        or metrics["context_valid"]["rate"] != 1
        or metrics["nonaction_attempted_start"]["correct"] != 0
    ):
        raise ValueError("Q0 completion flags disagree with its predeclared fit criteria.")
