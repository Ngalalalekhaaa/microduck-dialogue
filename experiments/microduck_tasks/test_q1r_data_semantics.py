"""Independent semantic invariants for synthetic continuation labels."""

import json

import pytest

from .q1r_data import audit, build, generate
from .toy_catalog import SELF_ACTIONS


@pytest.fixture(scope="module")
def rows():
    return build().rows


def examples(rows, prefix):
    return [row for row in rows if row["family_id"].startswith(prefix)]


def test_complete_contract_split_and_coverage(rows):
    result = audit(rows)
    assert result["train_val_family_overlap"] == result["train_val_exact_utterance_overlap"] == 0
    assert 6000 <= result["splits"]["train"]["count"] <= 10000
    assert 300 <= result["splits"]["val"]["count"] <= 450
    assert result["splits"]["train"]["families"] >= 500
    for split in ("train", "val"):
        stats = result["splits"][split]
        assert set(stats["self_actions"]) == set(SELF_ACTIONS)
        assert max(stats["goal_lengths"]) == 15
        assert max(stats["history_lengths"]) == 12


def test_chat_and_feedback_never_mutate_queue(rows):
    for row in rows:
        if row["group"].startswith("chat_") or row["request"]["event"]["kind"] == "feedback":
            assert row["expected"]["decision"] == "respond"
            assert row["expected"]["goals"] == []


def test_supporting_foot_is_opposite_raised_foot(rows):
    for row in rows:
        text = row["request"]["event"]["text"]
        if "用右脚单脚站立" in text or "支撑身体的是右脚" in text:
            assert row["expected"]["goals"][-1]["action"] == "flamingo_left"
        if "用左脚单脚站立" in text or "另一只右脚抬起来" in text:
            assert row["expected"]["goals"][-1]["action"] == "flamingo_right"


def test_sitting_prerequisite_and_persistent_posture(rows):
    matched = []
    for row in rows:
        if row["expected"]["decision"] != "replace":
            continue
        goals = row["expected"]["goals"]
        if row["request"]["observation"]["posture"] == "sitting":
            assert goals[0]["type"] == "posture"
            matched.append(row)
        if "站起来，别做坐下再起身那一套" == row["request"]["event"]["text"]:
            assert goals == [{"type": "posture", "pose": "standing"}]
    assert len(matched) > 200


def test_partial_negation_keeps_explicit_navigate(rows):
    found = []
    for row in rows:
        text = row["request"]["event"]["text"]
        if text in ("只靠近蓝球，这次别踢它", "到黄球边上就停住，省掉踢球"):
            assert row["expected"]["goals"][-1]["type"] == "navigate"
            assert all(goal["type"] != "kick" for goal in row["expected"]["goals"])
            found.append(row)
    assert len(found) == 20


def test_latest_instruction_does_not_import_old_purple_ball(rows):
    for row in examples(rows, "latest_correction_"):
        targets = [goal["target"] for goal in row["expected"]["goals"] if "target" in goal]
        assert all(attribute["value"] != "purple" for target in targets for attribute in target["attributes"])
    for row in examples(rows, "latest_correction_0"):
        assert row["expected"]["goals"] == [{"type": "perform", "action": "nod", "repeat": 1}]


def test_queue_overflow_is_not_silently_truncated(rows):
    for row in examples(rows, "queue_overflow_"):
        assert row["expected"]["decision"] == "clarify"
        assert not row["expected"]["goals"]
    for row in examples(rows, "append_"):
        request = row["request"]
        current = 1 + len(request["execution"]["pending"])
        if current == 15:
            assert row["expected"]["decision"] == "clarify"


def test_stop_cannot_resume_empty_queue(rows):
    for row in examples(rows, "control_resume_"):
        if row["request"]["execution"]["state"] in ("idle", "failed"):
            assert row["expected"]["decision"] == "clarify"
            assert row["expected"]["goals"] == []


def test_failed_kick_and_pause_feedback_are_honest(rows):
    for row in examples(rows, "failure_detail_"):
        result = row["request"]["execution"]["last_result"]
        assert "尚未完成" in row["expected"]["reply"]
        assert not result["verified"]
        if result["reason"] == "safe_pause_boundary":
            assert result["status"] == "blocked"
            assert row["request"]["execution"]["state"] == "paused"
            assert "暂停" in row["expected"]["reply"]
        if result["reason"] == "verification_failed":
            assert "不能说踢中了" in row["expected"]["reply"]


def test_synthetic_result_and_execution_state_agree(rows):
    for row in rows:
        if row["group"] not in ("chat_completed_status", "executor_feedback"):
            continue
        execution = row["request"]["execution"]
        result = execution["last_result"]
        if result is None or result["status"] in ("succeeded", "cancelled"):
            assert execution["state"] == "idle"
            assert execution["active"] is None
            assert not execution["pending"]
        elif result["status"] == "failed":
            assert execution["state"] == "failed"
            assert execution["active"] is None
        else:
            assert execution["state"] == "paused"
            assert execution["active"] == result["goal"]
            assert row["request"]["observation"]["stationary"] is True


def test_frozen_output_cannot_be_replaced(tmp_path):
    manifest, report = generate(tmp_path)
    assert manifest["physical_verified"] is manifest["human_reviewed"] is False
    assert report["schema_validation_errors"] == 0
    assert len((tmp_path / "raw_train.jsonl").read_text().splitlines()) == manifest["files"]["train"]["count"]
    assert json.loads((tmp_path / "raw_val.jsonl").read_text().splitlines()[0])["expected"]["reply"]
    with pytest.raises(FileExistsError, match="Frozen data"):
        generate(tmp_path)
