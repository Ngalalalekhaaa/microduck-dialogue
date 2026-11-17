"""Bounded plans are task-general, strict and separate from v1 training artifacts."""

import copy
import json
import math

import pytest

from .plan_backend import LocalPlanBackend
from .plan_protocol import VERSION, parse_plan, resolve_target
from .test_tasks import ball, observed

TARGET = {"id": "ball:red", "category": "ball", "attributes": [{"name": "color", "value": "red"}]}


def plan(steps=None, target=None, status="continue"):
    return {
        "schema_version": VERSION,
        "status": status,
        "target": target,
        "steps": [{"skill": "move_forward", "seconds": 0.4}] if steps is None else steps,
        "reply": "",
    }


def test_target_attributes_are_not_ball_specific():
    value = plan(target={"id": "door:1", "category": "door", "attributes": [{"name": "material", "value": "wood"}]})
    assert parse_plan(json.dumps(value)).target["category"] == "door"


@pytest.mark.parametrize(
    "steps",
    [
        [{"skill": "stand", "seconds": 1.0}] * 4,
        [{"skill": "stand", "seconds": 0.1}] * 5,
        [{"skill": "move_forward", "seconds": math.nan}],
        [{"skill": "move_forward", "seconds": 0.03}],
        [{"skill": "unknown", "seconds": 0.2}],
        [{"skill": "kick_left", "seconds": 0.5}, {"skill": "stand", "seconds": 0.1}],
        [{"skill": "approach_target", "seconds": 2.02}],
    ],
)
def test_invalid_or_unbounded_plan_is_rejected(steps):
    with pytest.raises(ValueError):
        parse_plan(json.dumps(plan(steps=steps, target=TARGET)))


def test_approach_and_kick_require_target_and_kick_is_last():
    steps = [{"skill": "approach_target", "seconds": 1.0}, {"skill": "kick_right", "seconds": 0.5}]
    assert len(parse_plan(json.dumps(plan(steps, TARGET))).steps) == 2
    with pytest.raises(ValueError):
        parse_plan(json.dumps(plan(steps)))
    with pytest.raises(ValueError):
        parse_plan(json.dumps(plan([], status="continue")))


def test_target_identity_and_attributes_must_match_current_measurement():
    assert resolve_target(TARGET, observed([ball()])) is not None
    assert resolve_target(TARGET, observed([])) is None
    wrong = copy.deepcopy(TARGET)
    wrong["attributes"][0]["value"] = "blue"
    assert resolve_target(wrong, observed([ball()])) is None


def test_existing_adapters_cannot_silently_receive_v2_prompt():
    with pytest.raises(ValueError, match="v1"):
        LocalPlanBackend("http://127.0.0.1:8083/v1/chat/completions", "microduck-onnx")
