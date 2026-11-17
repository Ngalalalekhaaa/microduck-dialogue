"""Compare first model generations without counting host controls as model skill."""

import json

from .q1r_evaluate import canonical
from .run_ablation import save


def compare(root, new_predictions):
    path = root / "baseline_reviewed.jsonl"
    if not path.exists():
        raise ValueError("The reviewed original 320-case baseline is required for comparison.")
    old = {row["id"]: row for line in path.read_text().splitlines() for row in [json.loads(line)]}
    new = {row["id"]: row for line in new_predictions.read_text().splitlines() for row in [json.loads(line)]}
    if set(old) != set(new):
        raise ValueError("Original and continuation development replay case IDs differ.")
    records = []
    for id_, before in old.items():
        after = new[id_]
        trace = before["model_trace"]
        model_called = trace.get("model_called", False)
        plan = trace.get("raw_prediction") or {}
        expected = before["expected"]
        valid = model_called and not trace.get("rejected") and set(plan) == {"decision", "goals"}
        correct = (
            valid
            and plan["decision"] == expected["decision"]
            and canonical(plan["goals"]) == canonical(expected["goals"])
        )
        records.append(
            dict(
                id=id_,
                group=before["group"],
                old_model_called=model_called,
                old_raw_plan_valid=valid,
                old_semantic_plan_exact=bool(correct),
                new_semantic_plan_exact=after["score"]["semantic_plan_exact"],
                request=before["request"]["event"]["text"],
                old_raw=before.get("raw"),
                new_raw=after["raw_text"],
                new_score=after["score"],
            )
        )
    called = [r for r in records if r["old_model_called"]]
    report = dict(
        total=len(records),
        comparable_model_called_cases=len(called),
        host_only_cases_excluded=len(records) - len(called),
        old_raw_semantic_plan_exact=sum(r["old_semantic_plan_exact"] for r in called),
        new_raw_semantic_plan_exact=sum(r["new_semantic_plan_exact"] for r in called),
        improved_ids=[r["id"] for r in called if not r["old_semantic_plan_exact"] and r["new_semantic_plan_exact"]],
        regressed_ids=[r["id"] for r in called if r["old_semantic_plan_exact"] and not r["new_semantic_plan_exact"]],
        physical_success_rate_claimed=False,
        blind_access=False,
        caveats=[
            "Development cases informed failure categories; this is not a blind generalization result.",
            "Continuation changes both prompt and output contract; the difference is the whole intervention effect.",
            "Unsupported requests now clarify instead of the old decline policy, so some gains reflect policy changes.",
            "Natural-language replies require separate qualitative review; plan scores do not measure chat quality.",
        ],
    )
    save(root / "post_training/comparison.json", report)
    with (root / "post_training/comparison_cases.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return report
