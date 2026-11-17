"""Audit Q1 boundary coverage without changing training, raw predictions or original reports."""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .ablation_score import fraction, score
from .q1_evaluate import SELECTION, common_gate, report, selection_key
from .run_ablation import save
from .toy_dataset import digest

START_DECISIONS = ("replace", "append", "resume")
NONACTION_DECISIONS = ("respond", "clarify", "decline", "cancel", "pause")
STEPS = (50, 100, 150)
RULES = dict(
    version="q1-boundary-coverage-v1",
    reason="Original Q1 invalid-or-start gate covered 79 rows and omitted five pause rows.",
    nonaction_decisions=list(NONACTION_DECISIONS),
    expected_nonaction_counts=dict(respond=20, clarify=32, decline=22, cancel=5, pause=5),
    expected_nonaction_total=84,
    full_and_common_boundary_ids_must_match=True,
    unknown="Raw text cannot be parsed as a JSON object; unknown is never counted as a valid safe response.",
    attempted_start="Original score(): replace/append/resume decision or nonempty goals on a nonaction row.",
    invalid_or_start="not context_valid OR attempted_false_start, over all 84 nonaction rows.",
    gates="Unchanged original thresholds and denominators; only invalid-or-start coverage becomes all 84.",
    selection=SELECTION,
    checkpoint_steps=list(STEPS),
    incomplete_checkpoints="Pending; do not score partial generations or select a final checkpoint.",
    raw_generation="Reuse saved raw_text and original contextual scorer; no new generation or repairs.",
    blind_access=False,
    training_modified=False,
    original_artifacts_modified=False,
)


def freeze(root, output):
    """Write rules and provenance before opening any checkpoint prediction."""
    output.mkdir(parents=True, exist_ok=True)
    sources = (
        Path(__file__),
        Path(__file__).with_name("q1_evaluate.py"),
        Path(__file__).with_name("ablation_score.py"),
    )
    identity = dict(
        rules=RULES,
        validation_sha256=digest(root / "h0/data/val.jsonl"),
        implementation_sha256={path.name: digest(path) for path in sources},
    )
    path = output / "frozen_rules.json"
    if path.exists():
        frozen = json.loads(path.read_text())
        if any(frozen[key] != value for key, value in identity.items()):
            raise ValueError("Frozen audit rules, source code or validation data changed; use a new audit directory.")
    else:
        save(path, dict(**identity, frozen_at=datetime.now(UTC).isoformat(), predictions_read_before_freeze=False))
        path.chmod(0o444)
    return digest(path)


def boundary_ids(rows):
    return {r["id"] for r in rows if r["expected"]["decision"] not in START_DECISIONS}


def audit_report(records, expected_count):
    summary = report(records, expected_count)
    decisions = {}
    for decision in sorted({r["expected"]["decision"] for r in records}):
        rows = [r for r in records if r["expected"]["decision"] == decision]
        decisions[decision] = dict(
            count=len(rows),
            unknown=sum(not r["score"]["json_parsed"] for r in rows),
            attempted_start=fraction(rows, "attempted_false_start"),
            accepted_start=fraction(rows, "accepted_false_start"),
            context_invalid=sum(not r["score"]["context_valid"] for r in rows),
            invalid_or_attempted_start=sum(
                not r["score"]["context_valid"] or r["score"]["attempted_false_start"] for r in rows
            ),
        )
    boundary = [r for r in records if r["expected"]["decision"] not in START_DECISIONS]
    total = sum(not r["score"]["context_valid"] or r["score"]["attempted_false_start"] for r in boundary)
    summary.update(
        original_79_row_invalid_or_attempted_start=summary["all_nonaction_invalid_or_attempted_start"],
        all_nonaction_invalid_or_attempted_start=total,
        boundary_count=len(boundary),
        boundary_ids=sorted(boundary_ids(records)),
        by_expected_decision=decisions,
        boundary_unknown=sum(not r["score"]["json_parsed"] for r in boundary),
        prototype_gate_pass=summary["prototype_gate_pass"] and total == 0,
    )
    return summary


def audit_checkpoint(cases, directory, step):
    records, sources, mismatches = [], {}, []
    for case in cases:
        path = directory / f"{case['id']}.json"
        record = json.loads(path.read_text())
        for key in ("id", "request", "expected", "acceptable_decisions", "domain"):
            if record[key] != case[key]:
                raise ValueError(f"Prediction changed frozen case {case['id']} field {key}.")
        rescored = score(record["raw_text"], case, plan_only=True)
        if record.get("score") != rescored:
            mismatches.append(case["id"])
        records.append(dict(record, score=rescored))
        sources[path.name] = digest(path)
    common = [r for r in records if r["domain"] == "common"]
    result = dict(
        step=step,
        all240=audit_report(records, 240),
        common236=audit_report(common, 236),
        full_only4=audit_report([r for r in records if r["domain"] != "common"], 4),
        saved_score_mismatch_ids=mismatches,
        prediction_sha256=sources,
    )
    original = json.loads((directory / "report.json").read_text())
    result.update(
        common_gate_pass=common_gate(result["common236"]),
        selection_key=selection_key(result),
        original_common_gate_pass=original["common_gate_pass"],
        original_selection_key=original["selection_key"],
        original_report_sha256=digest(directory / "report.json"),
    )
    return result


def run(root, output):
    rules_sha256 = freeze(root, output)
    cases = [json.loads(line) for line in (root / "h0/data/val.jsonl").read_text().splitlines()]
    common = [r for r in cases if r["domain"] == "common"]
    counts = Counter(r["expected"]["decision"] for r in cases if r["expected"]["decision"] not in START_DECISIONS)
    if len(cases) != 240 or len(common) != 236 or len({r["id"] for r in cases}) != 240:
        raise ValueError("Unexpected frozen development membership.")
    if counts != RULES["expected_nonaction_counts"] or boundary_ids(cases) != boundary_ids(common):
        raise ValueError("Expected the same 84 nonaction cases in full and common development domains.")
    completed, pending = [], []
    for step in STEPS:
        directory = root / f"q1/train1200/validation_{step}"
        present = sum((directory / f"{r['id']}.json").exists() for r in cases)
        if present != 240 or not (directory / "report.json").exists():
            pending.append(dict(step=step, prediction_files_present=present, expected=240))
            continue
        audited = audit_checkpoint(cases, directory, step)
        save(output / f"checkpoint_{step}.json", audited)
        completed.append(audited)
    winner = max(completed, key=selection_key) if completed else None
    result = dict(
        updated_at=datetime.now(UTC).isoformat(),
        frozen_rules_sha256=rules_sha256,
        complete=not pending,
        status="complete" if not pending else "pending",
        completed_steps=[r["step"] for r in completed],
        pending_checkpoints=pending,
        selected_step=winner["step"] if winner and not pending else None,
        provisional_selected_step=winner["step"] if winner else None,
        selected_common_gate_pass=winner["common_gate_pass"] if winner and not pending else None,
        selected_all240_gate_pass=winner["all240"]["prototype_gate_pass"] if winner and not pending else None,
        original_selected_step=max(completed, key=lambda r: r["original_selection_key"])["step"]
        if not pending
        else None,
        boundary_count=sum(counts.values()),
        boundary_counts=dict(counts),
        full_and_common_boundary_ids_identical=True,
        blind_access=False,
        gpu_used=False,
        original_artifacts_modified=False,
    )
    save(output / "status.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output or args.root / "q1/boundary_audit"), ensure_ascii=False, indent=2))
