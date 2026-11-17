"""First-generation planning scores and limited, explicitly separate reply checks."""

import copy
import re
import time
from collections import Counter

import torch

from .ablation_contract import dumps
from .q1r_prompt import validate_prediction
from .run_ablation import save
from .toy_vlm_json import parse_json_object


def generate(model, processor, row):
    ids = torch.tensor([row["prefix_ids"]], dtype=torch.long, device="cuda")
    tokenizer = processor.tokenizer
    endings = list(dict.fromkeys([tokenizer.eos_token_id, tokenizer.pad_token_id]))
    torch.cuda.synchronize()
    started = time.monotonic()
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=torch.zeros_like(ids),
            do_sample=False,
            use_cache=True,
            logits_to_keep=1,
            max_new_tokens=1536,
            max_time=180.0,
            eos_token_id=endings,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    tokens = result[0, len(row["prefix_ids"]) :].tolist()
    return tokenizer.decode(tokens, skip_special_tokens=True), dict(
        seconds=time.monotonic() - started,
        input_tokens=len(row["prefix_ids"]),
        output_tokens=len(tokens),
        finish_reason="eos" if tokens and tokens[-1] in endings else "limit",
    )


def canonical(goals):
    goals = copy.deepcopy(goals)
    for goal in goals:
        target = goal.get("target")
        if target and target.get("attributes"):
            target["attributes"].sort(key=lambda item: (item["name"], item["value"]))
            target["entity_id"] = None
    return goals


def assess(raw, row):
    expected = row["expected"]
    result = dict(
        json_valid=False,
        context_valid=False,
        decision_correct=False,
        goals_correct=False,
        plan_exact=False,
        semantic_plan_exact=False,
        attempted_false_start=False,
        accepted_false_start=False,
        reply_present=False,
        reply_reference_exact=False,
        premature_completion=False,
        error=None,
    )
    try:
        parsed = parse_json_object(raw)
        result["json_valid"] = True
        result["decision_correct"] = parsed.get("decision") == expected["decision"]
        result["goals_correct"] = parsed.get("goals") == expected["goals"]
        nonmotion = expected["decision"] not in ("replace", "append", "resume")
        result["attempted_false_start"] = nonmotion and (
            parsed.get("decision") in ("replace", "append", "resume") or bool(parsed.get("goals"))
        )
        reply = parsed.get("reply", "")
        result["reply_present"] = isinstance(reply, str) and bool(re.search("[\u4e00-\u9fff]", reply))
        result["reply_reference_exact"] = reply == expected["reply"]
        if row["request"]["event"]["kind"] == "user" and parsed.get("decision") in ("replace", "append"):
            result["premature_completion"] = bool(re.search(r"已经完成|已完成|已经踢中|已经做完|已经执行完", reply))
        validate_prediction(parsed, row["request"])
        result["context_valid"] = True
        result["accepted_false_start"] = result["attempted_false_start"]
        result["plan_exact"] = result["decision_correct"] and result["goals_correct"]
        result["semantic_plan_exact"] = result["decision_correct"] and canonical(parsed["goals"]) == canonical(
            expected["goals"]
        )
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def summarize(records):
    metrics = {
        key: sum(bool(row["score"][key]) for row in records)
        for key in (
            "json_valid",
            "context_valid",
            "plan_exact",
            "semantic_plan_exact",
            "attempted_false_start",
            "accepted_false_start",
            "reply_present",
            "premature_completion",
            "reply_reference_exact",
        )
    }
    groups = {}
    for group in sorted({row["group"] for row in records}):
        selected = [row for row in records if row["group"] == group]
        groups[group] = dict(
            total=len(selected), semantic_plan_exact=sum(r["score"]["semantic_plan_exact"] for r in selected)
        )
    return dict(
        total=len(records),
        counts=metrics,
        groups=groups,
        generation_seconds=sum(row["timing"]["seconds"] for row in records),
        reply_quality="Only presence and obvious premature-completion checks; no comprehensive chat-quality claim.",
        source="independent_synthetic_validation",
        physical_executions=0,
        blind_access=False,
    )


def screen(rows, limit=48):
    counts = Counter()
    selected = []
    remaining = sorted(rows, key=lambda row: (row["row"]["group"], row["id"]))
    while remaining and len(selected) < limit:
        row = min(remaining, key=lambda r: (counts[r["row"]["group"]], r["id"]))
        selected.append(row)
        counts[row["row"]["group"]] += 1
        remaining.remove(row)
    return selected


def evaluate(model, processor, rows, output, step, notify, *, full):
    directory = output / f"validation_{step}_{'full' if full else 'screen'}"
    directory.mkdir(exist_ok=False)
    records = []
    for encoded in rows:
        raw, timing = generate(model, processor, encoded)
        row = encoded["row"]
        record = dict(
            id=row["id"],
            group=row["group"],
            request=row["request"],
            expected=row["expected"],
            raw_text=raw,
            timing=timing,
            score=assess(raw, row),
        )
        records.append(record)
        with (directory / "predictions.jsonl").open("a") as stream:
            stream.write(dumps(record) + "\n")
        notify(
            "validating", step=step, validation_completed=len(records), validation_total=len(rows), validation_full=full
        )
    report = dict(step=step, full=full, **summarize(records))
    save(directory / "report.json", report)
    return report


def selection_key(report):
    if not report["full"]:
        raise ValueError("Only complete held-out validation can select a checkpoint.")
    c = report["counts"]
    return (
        -c["accepted_false_start"],
        -c["attempted_false_start"],
        c["context_valid"],
        c["semantic_plan_exact"],
        -c["premature_completion"],
        c["reply_present"],
        -report["step"],
    )
