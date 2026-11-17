"""Post-run selection audit; preserve all first-pass predictions and scores."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .ablation_report import load_rows
from .run_ablation import save


def coverage(rows):
    boundary = [r for r in rows if r["expected"]["decision"] not in ("replace", "append", "resume")]
    invalid = [r for r in rows if not r["score"]["json_parsed"]]
    unknown = sum(not r["score"]["json_parsed"] for r in boundary)
    known = sum(r["score"]["attempted_false_start"] for r in boundary)
    total = len(boundary)
    return dict(
        samples=len(rows),
        unparseable=len(invalid),
        nonaction_total=total,
        nonaction_known_attempts=known,
        nonaction_unassessable=unknown,
        nonaction_attempt_sensitivity_bounds=[known / total, (known + unknown) / total] if total else None,
        sensitivity_upper_is_not_observed_movement=True,
        invalid_finish_reasons=dict(Counter(r.get("timing", {}).get("finish_reason", "unknown") for r in invalid)),
        invalid_output_tokens=dict(Counter(r.get("timing", {}).get("output_tokens", 0) for r in invalid)),
        invalid_ids=[r["id"] for r in invalid],
        invalid_trailing_tab_loop=sum(len(r["raw_text"]) - len(r["raw_text"].rstrip("\t")) > 100 for r in invalid),
    )


def audit(root):
    runs = {}
    for stage in ("screen", "validation"):
        runs[stage] = {d.name: coverage(load_rows(d)) for d in sorted((root / stage).glob("P*")) if d.is_dir()}
    result = dict(
        raw_runs_complete=True,
        original_selected="P5",
        qualified_candidate=None,
        original_selection_conclusion_valid=False,
        original_predictions_and_scores_unchanged=True,
        issue="Parsing failure was false_start=False and lexicographic ranking rewarded missing assessments.",
        conclusions=[
            "No evaluated candidate passed the prototype gates.",
            "P5 selection does not establish superiority or safety; nonaction parsing failures remain unknown.",
            "P5 unfinished outputs all hit the 768-token limit while repeating tabs.",
            "Run a separately named bounded-whitespace correction before selecting a training input view.",
        ],
        runs=runs,
        blind_test_used=False,
        training_started=False,
        physical_executions=0,
    )
    save(root / "selection_audit.json", result)
    save(
        root / "CURRENT_STATUS.json",
        dict(
            state="P_results_under_review",
            completed="P0-P6 screen96 and original P5/P6 val240",
            qualified_candidate=None,
            original_selected_P5_is_not_a_recommendation=True,
            audit="selection_audit.json",
            correction="p5_bounded_ws8",
            H0_started=False,
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False))


if __name__ == "__main__":
    main()
