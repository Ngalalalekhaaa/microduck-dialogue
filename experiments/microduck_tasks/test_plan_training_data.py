"""V2 labels reflect executed, independently verified teacher trajectories."""

import copy

import pytest

from .plan_training_data import task_rows
from .test_tasks import ball, observed


def event(index, phase, skill=None, seconds=0.2):
    obs = observed([ball(x=0.4)], time_s=index * 0.3)
    return {
        "event_id": f"event_{index}",
        "task_id": "task_001",
        "instruction": "踢红球",
        "intent": {"kind": "kick", "color": "red"},
        "phase": phase,
        "decision": {"skill": skill, "seconds": seconds} if skill else None,
        "observation": obs,
        "done": skill is None,
        "status": "success" if skill is None else "running",
        "reply": "完成",
        "execution": {
            "started": bool(skill),
            "completed": bool(skill),
            "start_time_s": obs["time_s"] if skill else None,
            "end_time_s": obs["time_s"] + seconds if skill else None,
        },
    }


def test_continuation_uses_committed_prefix_without_future_sensor_inputs():
    events = [
        event(0, "approach", "move_forward"),
        event(1, "approach", "move_forward"),
        event(2, "settle", "stand"),
        event(3, "done"),
    ]
    rows = list(task_rows(events, {"evaluation": {"success": True}}))
    normal, future = rows[0][1], rows[1][1]
    assert normal["target"]["steps"][0] == {"skill": "approach_target", "seconds": 0.6}
    assert future["context"]["committed_steps"] == [{"skill": "move_forward", "seconds": 0.2}]
    assert normal["state"] == future["state"]
    assert "phase" not in normal["state"] and "evaluation" not in normal["state"]
    assert rows[2][1]["context"]["recent_outcomes"][0]["step"]["seconds"] == 0.2


@pytest.mark.parametrize("corruption", ["failed", "missing_verification", "partial", "future"])
def test_failure_partial_execution_and_future_observation_cannot_be_labels(corruption):
    events = [event(0, "approach", "move_forward"), event(1, "done")]
    result = {"evaluation": {"success": True}}
    if corruption == "failed":
        events[-1]["status"] = "verification_failed"
    elif corruption == "missing_verification":
        result = None
    elif corruption == "partial":
        events[0]["execution"]["completed"] = False
    else:
        events[0]["observation"]["time_s"] = 1.0
    with pytest.raises(ValueError):
        list(task_rows(events, result))


def test_direct_movement_has_no_teacher_recovery_label():
    events = [event(0, "direct", "move_forward"), event(1, "settle", "stand", 1.0), event(2, "done")]
    for e in events:
        e["intent"] = {"kind": "move"}
    rows = list(task_rows(events, {}))
    assert len(rows) == 2
    assert rows[0][1]["target"]["steps"] == [{"skill": "move_forward", "seconds": 0.2}]
    assert rows[-1][1]["target"]["status"] == "finish"


def test_sensor_whitelist_refuses_oracle_fields_in_v2_training():
    from .plan_protocol import validate_input

    rows = list(task_rows([event(0, "approach", "move_forward"), event(1, "done")], {"evaluation": {"success": True}}))
    row = copy.deepcopy(rows[0][1])
    row["state"]["ball_world_xyz"] = [1, 2, 3]
    with pytest.raises(ValueError):
        validate_input(row["state"], row["context"])
