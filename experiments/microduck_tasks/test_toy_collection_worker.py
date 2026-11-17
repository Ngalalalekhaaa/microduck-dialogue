"""Independent label and goal-order gates for real collection workers."""

import copy

import pytest

from .toy_collection_worker import ExpectedTeacher, audit_goals


def test_expected_teacher_rejects_wrong_label_before_recording():
    class Teacher:
        def decide(self, request):
            return {"decision": "replace", "goals": [{"type": "unexpected"}], "reply": "错误标签"}

    spec = {"commands": ["你好"], "expected": [{"decision": "respond", "goals": []}]}
    teacher = ExpectedTeacher(Teacher(), spec)
    with pytest.raises(ValueError, match="Semantic label mismatch"):
        teacher.decide({"event": {"kind": "user", "text": "你好"}})
    assert teacher.index == 0


def test_host_controls_still_require_the_independent_expected_label():
    spec = {
        "commands": ["暂停", "继续", "停下"],
        "expected": [{"decision": value, "goals": []} for value in ("pause", "resume", "cancel")],
    }
    teacher = ExpectedTeacher(None, spec)
    teacher.verify_turn("暂停", {"decision": "pause", "goals": []})
    with pytest.raises(ValueError, match="Semantic label mismatch"):
        teacher.verify_turn("继续", {"decision": "replace", "goals": []})
    assert teacher.index == 1
    teacher.verify_turn("继续", {"decision": "resume", "goals": []})
    teacher.verify_turn("停下", {"decision": "cancel", "goals": []})
    assert teacher.index == 3


def _start(index=0, revision=1):
    return {
        "event": "goal_start",
        "task_id": "task_001",
        "goal_index": index,
        "revision": revision,
        "goal": {"type": "perform", "action": "nod", "repeat": 1},
    }


def _result(start, status="succeeded"):
    return dict(start, event="goal_result", status=status, reason="verified")


def test_audit_rejects_skipped_goal_and_failed_queue_advancement():
    first = _start()
    second = _start(index=1)
    accepted = {"task_001": [first["goal"], second["goal"]]}
    with pytest.raises(ValueError, match="skipped or reordered"):
        audit_goals([second, _result(second)], accepted)
    with pytest.raises(ValueError, match="failed goal incorrectly advanced"):
        audit_goals([first, _result(first, "failed"), second, _result(second)], accepted)


def test_resume_requires_explicit_user_turn_and_new_revision():
    first = _start()
    resumed = _start(revision=3)
    accepted = {"task_001": [first["goal"]]}
    events = [first, _result(first, "cancelled"), resumed, _result(resumed, "cancelled")]
    with pytest.raises(ValueError, match="without an explicit user resume"):
        audit_goals(events, accepted)
    allowed = copy.deepcopy(events)
    allowed.insert(2, {"event": "dialogue", "task_id": "task_001", "response": {"decision": "resume"}})
    assert audit_goals(allowed, accepted)["goal_attempts"] == 2


def test_accepted_task_cannot_disappear_or_silently_drop_its_tail():
    first = _start()
    accepted = {"task_001": [first["goal"], first["goal"]]}
    with pytest.raises(ValueError, match="never reached the executor"):
        audit_goals([], accepted)
    with pytest.raises(ValueError, match="silently dropped"):
        audit_goals([first, _result(first)], accepted)
