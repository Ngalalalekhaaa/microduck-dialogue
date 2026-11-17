"""Score raw Q1 development generations and select by strict common-domain plans."""

from .ablation_score import fraction, score, summarize
from .q0_evaluate import generate
from .run_ablation import save

SELECTION = (
    "common236 gate-pass first, then action_plan_exact, three_six_exact, accepted_exact; "
    "then fewer nonaction invalid-or-attempted-start outputs, higher context_valid, earliest step. "
    "Counts are compared lexicographically; all 240 rows are reported. "
    "The four full-only cases never participate in selection. "
    "Diagnostic selection does not imply the prototype gate passed."
)


def report(records, expected_count):
    summary = summarize(records, expected_count=expected_count)
    # Separate 84 respond/clarify/decline cases from five cancel cases for H comparisons.
    boundary = [r for r in records if r["expected"]["decision"] in ("respond", "clarify", "decline")]
    cancels = [r for r in records if r["expected"]["decision"] == "cancel"]
    unknown = sum(not r["score"]["json_parsed"] for r in boundary)
    summary["nonaction_answer_cases"] = dict(
        count=len(boundary),
        attempted_start=fraction(boundary, "attempted_false_start"),
        accepted_start=fraction(boundary, "accepted_false_start"),
        unparseable=unknown,
        invalid_or_attempted_start=sum(
            not r["score"]["context_valid"] or r["score"]["attempted_false_start"] for r in boundary
        ),
    )
    summary["cancel_cases"] = dict(count=len(cancels), accepted_exact=fraction(cancels, "accepted_exact"))
    summary["all_nonaction_invalid_or_attempted_start"] = sum(
        not r["score"]["context_valid"] or r["score"]["attempted_false_start"] for r in boundary + cancels
    )
    summary["prototype_gate_pass"] = (
        summary["prototype_gate_pass"] and summary["all_nonaction_invalid_or_attempted_start"] == 0
    )
    return summary


def common_gate(summary):
    m = summary["metrics"]
    return (
        summary["completed"] == 236
        and m["action_plan_exact"]["total"] == 147
        and m["three_six_exact"]["total"] == 60
        and m["context_valid"]["rate"] >= 0.99
        and m["action_plan_exact"]["rate"] >= 0.95
        and m["three_six_exact"]["rate"] >= 0.90
        and summary["all_nonaction_invalid_or_attempted_start"] == 0
    )


def selection_key(result):
    common = result["common236"]
    metrics = common["metrics"]
    return (
        int(common_gate(common)),
        *(metrics[k]["correct"] for k in ("action_plan_exact", "three_six_exact", "accepted_exact")),
        -common["all_nonaction_invalid_or_attempted_start"],
        metrics["context_valid"]["correct"],
        -result["step"],
    )


def evaluate(model, processor, rows, output, step, notify):
    directory = output / f"validation_{step}"
    directory.mkdir(exist_ok=False)
    records = []
    for row in rows:
        raw, timing = generate(model, processor, row)
        record = dict(row["row"])
        record.update(
            raw_text=raw,
            timing=timing,
            score=score(raw, row["row"], plan_only=True),
            generation_error=None,
            source="q1_development",
            prefix_sha256=row["prefix_sha256"],
            host_control=None,
        )
        save(directory / f"{row['id']}.json", record)
        records.append(record)
        notify("Q1_validating", step=step, validation_completed=len(records), validation_total=len(rows))
    common = [r for r in records if r["domain"] == "common"]
    full_only = [r for r in records if r["domain"] != "common"]
    result = dict(
        step=step,
        all240=report(records, 240),
        common236=report(common, 236),
        full_only4=report(full_only, 4),
        selection_rule=SELECTION,
        selection_key=None,
        total_generation_seconds=sum(r["timing"]["seconds"] for r in records),
        training_subset=False,
        development_reused_for_selection=True,
        blind_access=False,
    )
    result["common_gate_pass"] = common_gate(result["common236"])
    result["selection_key"] = selection_key(result)
    save(directory / "report.json", dict(**result, predictions=records))
    return result
