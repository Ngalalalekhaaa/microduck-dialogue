"""Factorized labels for a small semantic selector; no complete-plan classes."""

import json

import numpy as np

from .toy_catalog import SELF_ACTIONS
from .toy_protocol import DECISIONS, DIRECTIONS, validate_output

TOKENS = (
    *(f"perform:{name}" for name in SELF_ACTIONS),
    *(f"move:{name}" for name in DIRECTIONS),
    "turn:left",
    "turn:right",
    "kick",
    "navigate",
    "posture:sitting",
    "posture:standing",
    "EOS",
)
EOS = TOKENS.index("EOS")
BOS = len(TOKENS)
MAX_GOALS = 6
FIELDS = {
    "color": ("red", "yellow", "blue"),
    "repeat": (1, 2, 3),
    "limit_kind": ("duration", "until_cancelled"),
    "seconds": (0.6, 8.0),
    "angle_deg": (30.0, 90.0, 180.0, 360.0),
}
MOTION = ("replace", "append")
LIMITATIONS = [
    "Frozen Qwen features with a learned decision and ordered-goal decoder; replies are not trained.",
    "Targets currently support ball/color red,yellow,blue, entity_id=null, and standoff_m=0.25 only.",
    "Durations are restricted to 0.6 and 8 seconds; turn angles to 30,90,180,360 degrees.",
    "No learned motor control and no claim of arbitrary numerical or open-world object generalization.",
    "Decoding enforces 1..6 goals for replace/append and ends after an until_cancelled goal.",
]


def goal_label(goal):
    """Reject unsupported values rather than silently rounding labels."""
    kind = goal["type"]
    fields = {name: -100 for name in FIELDS}
    if kind == "perform":
        token = f"perform:{goal['action']}"
        fields["repeat"] = FIELDS["repeat"].index(goal["repeat"])
    elif kind == "move":
        token = f"move:{goal['direction']}"
        fields["limit_kind"] = FIELDS["limit_kind"].index(goal["limit"]["kind"])
        if goal["limit"]["kind"] == "duration":
            fields["seconds"] = FIELDS["seconds"].index(goal["limit"]["seconds"])
    elif kind == "turn":
        token = f"turn:{goal['direction']}"
        fields["angle_deg"] = FIELDS["angle_deg"].index(goal["angle_deg"])
    elif kind == "posture":
        token = f"posture:{goal['pose']}"
    else:
        token = kind
        target = goal["target"]
        if target["category"] != "ball" or target["entity_id"] is not None:
            raise ValueError("Selector supports ball targets with entity_id=null only.")
        if len(target["attributes"]) != 1 or target["attributes"][0]["name"] != "color":
            raise ValueError("Selector currently requires exactly one color target attribute.")
        fields["color"] = FIELDS["color"].index(target["attributes"][0]["value"])
        if kind == "navigate" and goal["standoff_m"] != 0.25:
            raise ValueError("Selector training currently supports standoff_m=0.25 only.")
    return TOKENS.index(token), fields


def decode_goal(token, fields):
    name = TOKENS[int(token)]
    kind, _, argument = name.partition(":")
    values = {name: choices[int(fields[name])] for name, choices in FIELDS.items()}
    if kind == "perform":
        return dict(type=kind, action=argument, repeat=values["repeat"])
    if kind == "move":
        limit = dict(kind=values["limit_kind"])
        if limit["kind"] == "duration":
            limit["seconds"] = values["seconds"]
        return dict(type=kind, direction=argument, limit=limit)
    if kind == "turn":
        return dict(type=kind, direction=argument, angle_deg=values["angle_deg"])
    if kind == "posture":
        return dict(type=kind, pose=argument)
    if kind not in ("kick", "navigate"):
        raise ValueError("EOS is a sequence delimiter, not an executable goal.")
    target = dict(category="ball", attributes=[dict(name="color", value=values["color"])], entity_id=None)
    result = dict(type=kind, target=target)
    if kind == "navigate":
        result["standoff_m"] = 0.25
    return result


def encode_targets(rows):
    count = len(rows)
    result = {
        "decision": np.empty(count, dtype=np.int64),
        "token": np.full((count, MAX_GOALS + 1), -100, dtype=np.int64),
        "user": np.array([row["request"]["event"]["kind"] == "user" for row in rows]),
    }
    result.update({name: np.full((count, MAX_GOALS + 1), -100, dtype=np.int64) for name in FIELDS})
    for index, row in enumerate(rows):
        target = validate_output(row["target"])
        if len(target["goals"]) > MAX_GOALS:
            raise ValueError(
                f"Legacy small head supports at most {MAX_GOALS} goals; use a generative planner for longer plans."
            )
        result["decision"][index] = DECISIONS.index(target["decision"])
        for position, goal in enumerate(target["goals"]):
            token, fields = goal_label(goal)
            result["token"][index, position] = token
            for name, value in fields.items():
                result[name][index, position] = value
        result["token"][index, len(target["goals"])] = EOS
    return result


def canonical_plan(goals):
    """Normalize equivalent int/float numeric labels for exact-match metrics."""
    return json.dumps([dict(token=token, fields=fields) for token, fields in map(goal_label, goals)], sort_keys=True)


def check_support(encoded):
    report = {}
    for name in ("decision", "token", *FIELDS):
        values = {split: set(labels[name].ravel().tolist()) - {-100} for split, labels in encoded.items()}
        for split in ("val", "test"):
            if values[split] - values["train"]:
                raise ValueError(
                    f"{split} has unsupported training labels for {name}: {values[split] - values['train']}"
                )
        report[name] = {split: sorted(items) for split, items in values.items()}
    return report
