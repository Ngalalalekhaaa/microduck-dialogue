"""Protect pause coverage, immutable audit rules and the original development boundary."""

import copy
import json
from pathlib import Path

import pytest

from .ablation_cases import build_cases
from .ablation_contract import dumps
from .ablation_score import score
from .q1_boundary_audit import audit_report, boundary_ids, run
from .q1_evaluate import common_gate, report, selection_key
from .run_ablation import save
from .toy_fixtures import request


@pytest.fixture
def predictions():
    cases = build_cases(request("test", planned=True)["observation"], max_goals=6)
    return [dict(r, raw_text=dumps(r["expected"]), score=score(dumps(r["expected"]), r, plan_only=True)) for r in cases]


def test_full_and_common_have_same_original_84_nonaction_cases(predictions):
    common = [r for r in predictions if r["domain"] == "common"]
    full = audit_report(predictions, 240)
    subset = audit_report(common, 236)
    assert full["boundary_count"] == subset["boundary_count"] == 84
    assert boundary_ids(predictions) == boundary_ids(common)
    assert full["boundary_ids"] == subset["boundary_ids"]
    assert {
        k: v["count"] for k, v in full["by_expected_decision"].items() if k in ("respond", "clarify", "decline")
    } == {"respond": 20, "clarify": 32, "decline": 22}
    assert full["by_expected_decision"]["cancel"]["count"] == 5
    assert full["by_expected_decision"]["pause"]["count"] == 5
    assert full["prototype_gate_pass"] and common_gate(subset)


def test_unparseable_pause_cannot_pass_gate_or_be_treated_as_safe(predictions):
    common = [r for r in predictions if r["domain"] == "common"]
    good = dict(step=50, common236=audit_report(common, 236))
    bad = copy.deepcopy(common)
    pause = next(r for r in bad if r["expected"]["decision"] == "pause")
    pause["score"] = score("not a JSON object", pause, plan_only=True)
    assert common_gate(report(bad, 236)), "Reproduce the original five-pause coverage omission."
    corrected = dict(step=50, common236=audit_report(bad, 236))
    summary = corrected["common236"]
    assert not common_gate(summary)
    assert summary["boundary_unknown"] == summary["all_nonaction_invalid_or_attempted_start"] == 1
    assert summary["original_79_row_invalid_or_attempted_start"] == 0
    assert summary["by_expected_decision"]["pause"]["attempted_start"]["correct"] == 0
    assert summary["by_expected_decision"]["pause"]["unknown"] == 1
    assert selection_key(good) > selection_key(corrected)


def test_resume_on_pause_is_attempted_start_even_if_rejected(predictions):
    bad = copy.deepcopy(predictions)
    pause = next(r for r in bad if r["expected"]["decision"] == "pause")
    pause["score"] = score('{"decision":"resume","goals":[]}', pause, plan_only=True)
    summary = audit_report(bad, 240)
    assert summary["by_expected_decision"]["pause"]["unknown"] == 0
    assert summary["by_expected_decision"]["pause"]["attempted_start"]["correct"] == 1
    assert summary["all_nonaction_invalid_or_attempted_start"] == 1
    assert not summary["prototype_gate_pass"]


def test_freezes_before_reading_predictions_preserves_reports_and_keeps_pending(tmp_path, predictions, monkeypatch):
    data = tmp_path / "h0/data"
    data.mkdir(parents=True)
    cases = [{k: v for k, v in r.items() if k not in ("score", "raw_text")} for r in predictions]
    (data / "val.jsonl").write_text("\n".join(dumps(r) for r in cases) + "\n")
    directory = tmp_path / "q1/train1200/validation_50"
    directory.mkdir(parents=True)
    for record in predictions:
        save(directory / f"{record['id']}.json", record)
    common = report([r for r in predictions if r["domain"] == "common"], 236)
    original_report = dict(step=50, common236=common, common_gate_pass=common_gate(common))
    original_report["selection_key"] = selection_key(original_report)
    save(directory / "report.json", original_report)
    original_bytes = (directory / "report.json").read_bytes()
    output = tmp_path / "q1/boundary_audit"
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        assert "sealed" not in path.parts and "blind" not in str(path)
        if "validation_50" in path.parts:
            assert (output / "frozen_rules.json").exists()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    status = run(tmp_path, output)
    assert not status["complete"] and status["selected_step"] is None
    assert status["completed_steps"] == [50] and status["provisional_selected_step"] == 50
    assert [r["step"] for r in status["pending_checkpoints"]] == [100, 150]
    assert (directory / "report.json").read_bytes() == original_bytes
    frozen = (output / "frozen_rules.json").read_bytes()
    run(tmp_path, output)
    assert (output / "frozen_rules.json").read_bytes() == frozen
    result = json.loads((output / "checkpoint_50.json").read_text())
    assert result["all240"]["boundary_count"] == 84 and not result["saved_score_mismatch_ids"]


def test_complete_audit_can_change_winner_without_overwriting_original(tmp_path, predictions):
    data = tmp_path / "h0/data"
    data.mkdir(parents=True)
    cases = [{k: v for k, v in r.items() if k not in ("score", "raw_text")} for r in predictions]
    (data / "val.jsonl").write_text("\n".join(dumps(r) for r in cases) + "\n")
    for step in (50, 100, 150):
        records = copy.deepcopy(predictions)
        if step == 50:
            bad = next(r for r in records if r["expected"]["decision"] == "pause")
        else:
            bad = next(r for r in records if r["group"] == "single" and r["domain"] == "common")
        bad["raw_text"] = "not a JSON object"
        bad["score"] = score(bad["raw_text"], bad, plan_only=True)
        directory = tmp_path / f"q1/train1200/validation_{step}"
        directory.mkdir(parents=True)
        for record in records:
            save(directory / f"{record['id']}.json", record)
        common = report([r for r in records if r["domain"] == "common"], 236)
        original = dict(step=step, common236=common, common_gate_pass=common_gate(common))
        original["selection_key"] = selection_key(original)
        save(directory / "report.json", original)
    result = run(tmp_path, tmp_path / "audit")
    assert result["complete"] and result["original_selected_step"] == 50
    assert result["selected_step"] == 100 and result["selected_common_gate_pass"]


def test_watcher_waits_for_reports_and_marks_changed_winner_reload(tmp_path, monkeypatch):
    from . import q1_boundary_watch as watcher

    output = tmp_path / "audit"
    output.mkdir()
    completed = [False]
    runs = []
    monkeypatch.setattr(watcher, "ready_steps", lambda root: (50, 100, 150) if completed[0] else ())

    def audit(root, destination):
        runs.append(completed[0])
        return dict(
            complete=completed[0],
            completed_steps=[50, 100, 150] if completed[0] else [],
            selected_step=100 if completed[0] else None,
            original_selected_step=50 if completed[0] else None,
            provisional_selected_step=100 if completed[0] else None,
        )

    monkeypatch.setattr(watcher, "run", audit)
    monkeypatch.setattr(watcher.time, "sleep", lambda _: completed.__setitem__(0, True))
    result = watcher.watch(tmp_path, output)
    assert runs == [False, True] and result["complete"]
    assert result["stage"] == "complete_reload_required" and result["changed_winner_requires_independent_reload"]
    assert not result["reload_checked"] and not result["gpu_used"]
