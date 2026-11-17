"""CPU-only analysis of paired Q1R backend records; never runs model inference.

Records: {id, raw_text, timing:{seconds,input_tokens,output_tokens}, backend,...}.
Optional repeat/repetition identifies repeated trials; warmup=true excludes a trial from
latency summaries. Historical desktop timings are never used as the speed baseline.
"""

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from .q1r_onnx_inputs import checksum
from .toy_protocol import VERSION, keys, validate_output
from .toy_vlm_json import parse_json_object


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def canonical(goals):
    """Match the existing Q1R colour-target comparison, after context validation."""
    result = copy.deepcopy(goals)
    for goal in result:
        target = goal.get("target")
        if target and target.get("attributes"):
            target["attributes"].sort(key=lambda item: (item["name"], item["value"]))
            target["entity_id"] = None
    return result


def assess(raw, row):
    """Use the live contract for CPU plan scoring; no automatic chat-quality claim."""
    score = dict(json_valid=False, context_valid=False, error=None)
    if row.get("expected") is not None:
        score.update(
            decision_correct=False,
            goals_correct=False,
            plan_exact=False,
            semantic_plan_exact=False,
            reply_reference_exact=False,
        )
    try:
        parsed = parse_json_object(raw)
        score["json_valid"] = True
        keys(parsed, "decision goals reply", "Q1R benchmark response")
        if row["request"]["event"]["kind"] == "feedback" and parsed["decision"] != "respond":
            raise ValueError("Feedback must not create or alter tasks.")
        validate_output(dict(schema_version=VERSION, **parsed), row["request"])
        score["context_valid"] = True
        score["reply_present"] = bool(parsed["reply"].strip())
        if row.get("expected") is not None:
            expected = row["expected"]
            score["decision_correct"] = parsed["decision"] == expected["decision"]
            score["goals_correct"] = parsed["goals"] == expected["goals"]
            score["plan_exact"] = score["decision_correct"] and score["goals_correct"]
            score["semantic_plan_exact"] = score["decision_correct"] and canonical(parsed["goals"]) == canonical(
                expected["goals"]
            )
            score["reply_reference_exact"] = parsed["reply"] == expected.get("reply")
        return parsed, score
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        score["error"] = f"{type(exc).__name__}: {exc}"
        return None, score


def prepare(records, inputs):
    counts, seen, result = Counter(), set(), []
    for original in records:
        record = copy.deepcopy(original)
        identifier = record["id"]
        if identifier not in inputs:
            raise ValueError(f"Unknown benchmark input: {identifier}")
        if record.get("warmup", False):
            continue
        if record.get("timing_role") == "historical_reference_only" or record.get("fair_benchmark") is False:
            raise ValueError("Historical timing cannot be used as a paired speed baseline.")
        if "repeat" in record and "repetition" in record and record["repeat"] != record["repetition"]:
            raise ValueError("repeat and repetition disagree for the same trial.")
        trial = record.get("repetition", record.get("repeat", counts[identifier]))
        if type(trial) is not int or trial < 0:
            raise ValueError("repetition must be a nonnegative integer.")
        counts[identifier] += 1
        key = (identifier, trial)
        if key in seen:
            raise ValueError(f"Duplicate benchmark trial: {key}")
        seen.add(key)
        timing = record["timing"]
        seconds = timing["seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(f"Invalid measured latency for {identifier}: {seconds}")
        for field in ("input_tokens", "output_tokens"):
            if type(timing[field]) is not int or timing[field] < 0:
                raise ValueError(f"{field} must be a nonnegative integer.")
        if not isinstance(record.get("backend"), str) or not record["backend"]:
            raise ValueError("Each new timing record must identify its backend.")
        frozen = inputs[identifier]
        wanted = checksum(frozen["request"])
        if frozen.get("request_sha256", wanted) != wanted:
            raise ValueError(f"Frozen input checksum changed: {identifier}")
        if "request_sha256" in record and record["request_sha256"] != wanted:
            raise ValueError(f"Backend measured a different request: {identifier}")
        parsed, score = assess(record.get("raw_text"), frozen)
        record.update(
            repetition=trial,
            parsed=parsed,
            assessment=score,
            request_hash_verified=record.get("request_sha256") == wanted,
        )
        result.append(record)
    return result


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def summary(records):
    if not records:
        return dict(measured_trials=0)
    seconds = [r["timing"]["seconds"] for r in records]
    output_tokens = sum(r["timing"]["output_tokens"] for r in records)
    fields = ("json_valid", "context_valid", "plan_exact", "semantic_plan_exact", "reply_reference_exact")
    return dict(
        measured_trials=len(records),
        unique_inputs=len({r["id"] for r in records}),
        backends=sorted({r["backend"] for r in records}),
        total_seconds=sum(seconds),
        latency_seconds=dict(
            mean=statistics.mean(seconds),
            median=statistics.median(seconds),
            p95=percentile(seconds, 0.95),
            minimum=min(seconds),
            maximum=max(seconds),
        ),
        total_output_tokens=output_tokens,
        end_to_end_output_tokens_per_second=output_tokens / sum(seconds),
        assessments={field: sum(bool(r["assessment"].get(field)) for r in records) for field in fields},
        reply_quality="Exact reply equality and nonempty replies do not measure chat correctness.",
        throughput_scope="Output tokens divided by whole generation latency including prefill; not pure decode speed.",
    )


def conditions(left, right):
    fields = ("device", "precision", "prompt_sha256", "checkpoint_sha256", "max_new_tokens", "generation_config_sha256")
    unknown, differences = [], {}
    for field in fields:
        a, b = left.get(field), right.get(field)
        if a is None or b is None:
            unknown.append(field)
        elif a != b:
            differences[field] = dict(baseline=a, onnx=b)
    if not left["request_hash_verified"] or not right["request_hash_verified"]:
        unknown.append("request_sha256")
    input_equal = left["timing"]["input_tokens"] == right["timing"]["input_tokens"]
    output_equal = left["timing"]["output_tokens"] == right["timing"]["output_tokens"]
    return dict(
        metadata_missing=unknown,
        reported_differences=differences,
        input_token_count_equal=input_equal,
        output_token_count_equal=output_equal,
        backend_only_speed_claim_supported=not unknown and not differences and input_equal and output_equal,
    )


def compare(inputs, baseline, onnx):
    frozen = {r["id"]: r for r in inputs}
    if len(frozen) != len(inputs):
        raise ValueError("Benchmark input ids must be unique.")
    left, right = prepare(baseline, frozen), prepare(onnx, frozen)
    indexed = [{(r["id"], r["repetition"]): r for r in records} for records in (left, right)]
    pairs = []
    for key in sorted(set(indexed[0]) & set(indexed[1])):
        a, b = (index[key] for index in indexed)
        pa, pb = a["parsed"], b["parsed"]
        valid = pa is not None and pb is not None
        reference = frozen[key[0]].get("frozen_reference", {})
        old_raw = reference.get("raw_text")
        pairs.append(
            dict(
                id=key[0],
                repetition=key[1],
                baseline=a,
                onnx=b,
                observed_latency_ratio_baseline_over_onnx=a["timing"]["seconds"] / b["timing"]["seconds"],
                conditions=conditions(a, b),
                raw_text_equal=a.get("raw_text") == b.get("raw_text"),
                decision_equal=valid and pa["decision"] == pb["decision"],
                goals_exact_equal=valid and pa["goals"] == pb["goals"],
                semantic_plan_equal=valid
                and pa["decision"] == pb["decision"]
                and canonical(pa["goals"]) == canonical(pb["goals"]),
                reply_equal=valid and pa["reply"] == pb["reply"],
                baseline_matches_historical_raw=a.get("raw_text") == old_raw,
                onnx_matches_historical_raw=b.get("raw_text") == old_raw,
                historical_reference_is_gold=False,
                input_semantic_note=frozen[key[0]].get("semantic_note"),
            )
        )
    paired_left = [p["baseline"] for p in pairs]
    paired_right = [p["onnx"] for p in pairs]
    ratios = [p["observed_latency_ratio_baseline_over_onnx"] for p in pairs]
    total_ratio = (
        (sum(r["timing"]["seconds"] for r in paired_left) / sum(r["timing"]["seconds"] for r in paired_right))
        if pairs
        else None
    )
    unmatched = {
        name: [dict(id=k[0], repetition=k[1]) for k in sorted(set(indexed[i]) - set(indexed[1 - i]))]
        for i, name in enumerate(("baseline_only", "onnx_only"))
    }
    return dict(
        version="q1r-onnx-cpu-comparison-1",
        benchmark_inputs=len(inputs),
        paired_trials=len(pairs),
        unmatched=unmatched,
        missing_input_ids={
            name: sorted(set(frozen) - {r["id"] for r in records})
            for name, records in (("baseline", left), ("onnx", right))
        },
        baseline=summary(left),
        onnx=summary(right),
        paired_baseline=summary(paired_left),
        paired_onnx=summary(paired_right),
        paired_total_latency_ratio=total_ratio,
        median_paired_latency_ratio=statistics.median(ratios) if ratios else None,
        output_comparison={
            field: sum(bool(p[field]) for p in pairs)
            for field in ("raw_text_equal", "decision_equal", "goals_exact_equal", "semantic_plan_equal", "reply_equal")
        },
        matched_input_token_counts=sum(p["conditions"]["input_token_count_equal"] for p in pairs),
        matched_output_token_counts=sum(p["conditions"]["output_token_count_equal"] for p in pairs),
        historical_latency_used_for_speedup=False,
        model_inference_performed=False,
        gpu_used=False,
        limitations=[
            "Paired latency ratios depend on device, precision and output length, beyond the backend alone.",
            "Warmup rows are excluded. Loading/export time is excluded unless the runner explicitly includes it.",
            "Original model errors remain in frozen_reference; matching them does not demonstrate correctness.",
            "Expected replies are agent guides. Reply equality does not measure quality; review changed replies.",
        ],
        pairs=pairs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-note", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Comparison already exists; use a new output filename.")
    report = compare(read_rows(args.inputs), read_rows(args.baseline), read_rows(args.onnx))
    report["comparison_notes"] = args.comparison_note
    report["sources"] = {
        name: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name, path in (("inputs", args.inputs), ("baseline", args.baseline), ("onnx", args.onnx))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, ensure_ascii=False, indent=2))
