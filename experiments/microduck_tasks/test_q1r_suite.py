"""Semantic probes use independent labels and keep host/model/reply failures separate."""

import copy
import json

import pytest

from .q1r_baseline import assess, canonical, review, summary
from .q1r_suite import cases, extra_cases, write_suite
from .toy_catalog import SELF_ACTIONS
from .toy_protocol import VERSION, validate_output


def test_suite_is_valid_independent_and_covers_live_capabilities(tmp_path):
    rows = cases()
    assert len(rows) >= 300
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["split"] == "development_only" and not row["training_allowed"] for row in rows)
    assert all(not row["physical_executed"] for row in rows)
    actions = {goal["action"] for row in rows for goal in row["expected"]["goals"] if goal["type"] == "perform"}
    assert actions == set(SELF_ACTIONS)
    assert max(len(row["request"]["history"]) for row in rows) == 12
    assert max(len(row["expected"]["goals"]) for row in rows) == 15
    for row in rows:
        validate_output(dict(schema_version=VERSION, **row["expected"]), row["request"])
    meta = write_suite(tmp_path)
    assert meta["cases"] == len(rows) and not meta["old_blind_access"]
    assert [json.loads(line) for line in (tmp_path / "suite.jsonl").read_text().splitlines()] == rows


def test_user_requested_clarification_for_unsupported_skills_is_explicit_policy_delta():
    row = next(r for r in cases() if r["group"] == "unsupported")
    response = dict(schema_version=VERSION, reply="当前不支持。", decision="decline", goals=[])
    score = assess(row, response, dict(model_called=True, decision_source="q1_lora_generation"))
    assert not score["plan_exact"]
    assert [cause["category"] for cause in score["causes"]] == ["policy_contract_update"]


def test_correct_chat_decision_does_not_count_template_as_a_real_answer():
    row = next(r for r in cases() if r["group"] == "chat")
    response = dict(schema_version=VERSION, reply="识别为聊天。", decision="respond", goals=[])
    score = assess(
        row,
        response,
        dict(model_called=True, reply_source="template", raw_prediction=dict(decision="respond", goals=[])),
    )
    assert score["plan_exact"] and score["reply_capability_gap"] and score["unexpected"]
    assert not score["semantic_reply_scored"] and not score["reply_exact_match_used"]


def test_host_controls_are_not_misreported_as_model_predictions():
    row = next(r for r in cases() if r["group"] == "control" and r["expected"]["decision"] == "pause")
    response = dict(schema_version=VERSION, reply="继续。", decision="respond", goals=[])
    score = assess(row, response, dict(model_called=False, decision_source="host_pause"))
    assert not score["model_called"]
    assert score["causes"][0]["category"] == "host_control_mismatch"
    report = summary([dict(group=row["group"], assessment=score)], 1, 1)
    assert report["host_control_cases"] == 1 and report["model_calls"] == 0


def test_canonical_bound_target_is_equivalent_without_mutation():
    goals = [
        dict(
            type="kick",
            target=dict(category="ball", entity_id="ball:red", attributes=[dict(name="color", value="red")]),
        )
    ]
    before = copy.deepcopy(goals)
    assert canonical(goals)[0]["target"]["entity_id"] is None
    assert goals == before


def test_sitting_supported_motion_inserts_standing_without_refusal():
    rows = [r for r in cases() if r["family_id"] in ("dev/sit_direct_move", "dev/sit_direct_kick")]
    assert len(rows) == 2
    for row in rows:
        assert row["request"]["observation"]["posture"] == "sitting"
        assert row["expected"]["decision"] == "replace"
        assert row["expected"]["goals"][0] == dict(type="posture", pose="standing")


def test_two_sides_does_not_mean_two_repeats_and_noisy_probes_remain_separate(tmp_path):
    row = next(row for row in cases() if row["id"] == "dev_0005")
    assert row["expected"]["goals"][0]["repeat"] == 1
    extras = extra_cases()
    assert len(extras) == 20 and not ({row["id"] for row in extras} & {row["id"] for row in cases()})
    assert write_suite(tmp_path, extra=True)["cases"] == 20
    for row in extras:
        validate_output(dict(schema_version=VERSION, **row["expected"]), row["request"])


def test_review_preserves_original_prediction_and_requires_logged_gold_correction(tmp_path):
    current = next(row for row in cases() if row["id"] == "dev_0005")
    original = copy.deepcopy(current)
    original["expected"]["goals"][0]["repeat"] = 2
    response = dict(schema_version=VERSION, **current["expected"])
    trace = dict(
        model_called=True,
        decision_source="q1_lora_generation",
        raw_prediction={key: response[key] for key in ("decision", "goals")},
    )
    record = dict(
        **original,
        response=response,
        raw="actual raw bytes",
        model_trace=trace,
        assessment=assess(original, response, trace),
    )
    predictions = json.dumps(record, ensure_ascii=False) + "\n"
    (tmp_path / "predictions.jsonl").write_text(predictions)
    (tmp_path / "suite.jsonl").write_text(json.dumps(original, ensure_ascii=False) + "\n")
    (tmp_path / "status.json").write_text(json.dumps(dict(state="complete", seconds=1)))
    with pytest.raises(KeyError):
        review(tmp_path)
    (tmp_path / "gold_corrections.json").write_text(
        json.dumps(
            {
                current["id"]: dict(
                    reason="Two sides is not two repetitions.", corrected_expected_goals=current["expected"]["goals"]
                )
            }
        )
    )
    assert review(tmp_path)["plan_exact"] == 1
    assert (tmp_path / "predictions.jsonl").read_text() == predictions
    corrected = json.loads((tmp_path / "reviewed_predictions.jsonl").read_text())
    assert corrected["raw"] == "actual raw bytes"
    assert corrected["original_expected"]["goals"][0]["repeat"] == 2


def test_host_rejection_matching_gold_is_not_raw_model_success():
    row = next(row for row in cases() if row["family_id"] == "dev/motion_gate_0")
    response = dict(schema_version=VERSION, reply="当前拒绝运动。", decision="decline", goals=[])
    trace = dict(
        model_called=True,
        raw_prediction=dict(decision="replace", goals=[dict(type="perform", action="nod", repeat=1)]),
        rejected=dict(reason="The local motion gate is closed."),
        decision_source="q1_lora_generation",
    )
    result = assess(row, response, trace)
    assert result["host_response_plan_exact"]
    assert result["raw_plan_exact"] is False and result["plan_exact"] is False
    assert result["causes"][0]["category"] == "state_contract_rejected"
