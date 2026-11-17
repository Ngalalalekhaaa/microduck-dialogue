"""Shared planning projection and checks for the staged offline experiments."""

import copy
import hashlib
import json

from .toy_protocol import DEFAULT_MAX_GOALS, VERSION, keys, validate_output
from .toy_schema import obj, response_schema
from .toy_selector_labels import MAX_GOALS, decode_goal, goal_label


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def checksum(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def plan_schema(*, max_goals=DEFAULT_MAX_GOALS):
    """Derive every goal field from the real protocol, including move.limit."""
    original = response_schema(max_goals=max_goals)
    properties = original["properties"]
    branches = []
    for decisions, minimum, maximum in (
        (["replace", "append"], 1, None),
        (["respond", "clarify", "decline", "cancel", "pause", "resume"], 0, 0),
    ):
        goals = copy.deepcopy(properties["goals"])
        goals["minItems"] = minimum
        if maximum is not None:
            goals["maxItems"] = maximum
        branches.append(obj({"decision": {"enum": decisions}, "goals": goals}))
    return {"$schema": original["$schema"], "oneOf": branches}


def wrap_plan(plan):
    """Offline validation envelope; this is never a submission acknowledgement."""
    keys(plan, "decision goals", "model plan")
    return dict(schema_version=VERSION, reply="离线规划评估，尚未提交执行。", **copy.deepcopy(plan))


def validate_context(response, request):
    validate_output(response, request)
    return response


def common_domain(goals):
    """The old head must represent the exact target, without numeric rounding."""
    if len(goals) > MAX_GOALS:
        return False
    try:
        decoded = []
        for goal in goals:
            token, fields = goal_label(goal)
            decoded.append(decode_goal(token, {name: max(0, value) for name, value in fields.items()}))
        return decoded == goals
    except (ValueError, KeyError, TypeError):
        return False
