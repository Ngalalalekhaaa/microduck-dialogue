"""Strict draft dialogue goals, deliberately separate from v2 motor chunks."""

import copy
import math

from .toy_catalog import SELF_ACTIONS

VERSION = "microduck-dialogue-3-pilot1"
DEFAULT_MAX_GOALS = 15
DECISIONS = ("respond", "clarify", "decline", "replace", "append", "cancel", "pause", "resume")
GOALS = ("perform", "kick", "navigate", "move", "turn", "posture")
DIRECTIONS = ("forward", "backward", "left", "right")


def validate_capacity(value):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("Executor capacity must be null (no fixed limit) or a positive integer.")
    return value


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError(f"{label}: expected exactly {expected}.")


def text(value, label, limit=500):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label}: require nonempty text, at most {limit} characters.")


def number(value, lower, upper, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: require a finite number.")
    if not lower <= value <= upper:
        raise ValueError(f"{label}: require {lower} <= value <= {upper}.")


def sequence(value, maximum, label):
    if not isinstance(value, list):
        raise ValueError(f"{label}: require a list.")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{label}: require a list of at most {maximum} entries.")


def attributes(value):
    sequence(value, 8, "attributes")
    names = []
    for item in value:
        keys(item, "name value", "attribute")
        text(item["name"], "attribute name", 40)
        text(item["value"], "attribute value", 80)
        names.append(item["name"])
    if len(names) != len(set(names)):
        raise ValueError("Duplicate target attributes are ambiguous.")


def target(value):
    keys(value, "category attributes entity_id", "target")
    if value["category"] != "ball":
        raise ValueError("Only ball targets are currently supported.")
    attributes(value["attributes"])
    if value["entity_id"] is not None:
        text(value["entity_id"], "entity_id", 100)
    if not value["attributes"] and value["entity_id"] is None:
        raise ValueError("Specify a target attribute or an observed entity_id.")


def goal(value):
    if not isinstance(value, dict) or value.get("type") not in GOALS:
        raise ValueError("Unknown semantic goal type.")
    kind = value["type"]
    if kind == "perform":
        keys(value, "type action repeat", kind)
        if value["action"] not in SELF_ACTIONS:
            raise ValueError("Action is outside the toy self-action candidates.")
        if type(value["repeat"]) is not int or not 1 <= value["repeat"] <= 3:
            raise ValueError("Bound repeats to one through three executions.")
    elif kind in ("kick", "navigate"):
        keys(value, "type target" + (" standoff_m" if kind == "navigate" else ""), kind)
        target(value["target"])
        if kind == "navigate":
            number(value["standoff_m"], 0.15, 0.5, "standoff_m")
    elif kind == "move":
        keys(value, "type direction limit", kind)
        if value["direction"] not in DIRECTIONS:
            raise ValueError("Unknown movement direction.")
        limit = value["limit"]
        if isinstance(limit, dict) and limit.get("kind") == "until_cancelled":
            keys(limit, "kind", "limit")
        else:
            keys(limit, "kind seconds", "limit")
            if limit["kind"] != "duration":
                raise ValueError("Use a duration or an until_cancelled intent.")
            number(limit["seconds"], 0.12, 10, "seconds")
    elif kind == "turn":
        keys(value, "type direction angle_deg", kind)
        if value["direction"] not in ("left", "right"):
            raise ValueError("Turn left or right.")
        number(value["angle_deg"], 15, 360, "angle_deg")
    else:
        keys(value, "type pose", kind)
        if value["pose"] not in ("sitting", "standing"):
            raise ValueError("Use sitting or standing as the persistent posture.")


def validate_output(value, request=None, *, max_goals=DEFAULT_MAX_GOALS):
    keys(value, "schema_version reply decision goals", "assistant response")
    if value["schema_version"] != VERSION or value["decision"] not in DECISIONS:
        raise ValueError("Use the draft dialogue version and a known decision.")
    text(value["reply"], "reply")
    sequence(value["goals"], validate_capacity(max_goals) if request is None else None, "goals")
    planning = value["decision"] in ("replace", "append")
    if bool(value["goals"]) != planning:
        raise ValueError("Only replace/append carry a nonempty goal queue.")
    for index, item in enumerate(value["goals"]):
        goal(item)
        if item["type"] == "move" and item["limit"]["kind"] == "until_cancelled":
            if index != len(value["goals"]) - 1:
                raise ValueError("An until_cancelled goal must be last; later goals would be unreachable.")
    if request is not None:
        from .toy_inputs import validate_input

        validate_input(request)
        _check_context(value, request)
    return copy.deepcopy(value)


def _check_context(value, request):
    caps, execution = request["capabilities"], request["execution"]
    decision = value["decision"]
    capacity = caps["max_goals"]
    if capacity is not None and len(value["goals"]) > capacity:
        raise ValueError("The current executor cannot accept this many goals.")
    if decision in ("append", "pause", "resume") and not caps["queue_control"]:
        raise ValueError("Queue control is not available in this capability snapshot.")
    if decision == "resume" and execution["state"] != "paused":
        raise ValueError("Only explicitly paused tasks may resume; cancellation clears the queue.")
    if decision == "pause" and execution["state"] != "running":
        raise ValueError("Only a running task may pause.")
    if decision == "append":
        if execution["state"] not in ("running", "paused"):
            raise ValueError("Append requires an existing task.")
        existing = ([execution["active"]] if execution["active"] else []) + execution["pending"]
        if any(g["type"] == "move" and g["limit"]["kind"] == "until_cancelled" for g in existing):
            raise ValueError("Cannot append behind an unbounded task; replace or clarify.")
        if capacity is not None and len(existing) + len(value["goals"]) > capacity:
            raise ValueError("Combined queue exceeds the executor's goal capacity.")
    checked_goals = value["goals"]
    if decision == "append":
        checked_goals = existing + value["goals"]
    if decision == "resume":
        checked_goals = [execution["active"], *execution["pending"]]
    pose = request["observation"]["posture"]
    for item in checked_goals:
        if item["type"] not in caps["goal_types"]:
            raise ValueError("This goal executor is not available in the current capability snapshot.")
        if item["type"] == "perform" and item["action"] not in caps["self_actions"]:
            raise ValueError("This self action is currently unavailable.")
        if item["type"] == "posture" and item["pose"] == "sitting" and not caps["sit_hold"]:
            raise ValueError("sit_stand is not a sitting hold executor.")
        if item["type"] == "move" and item["limit"]["kind"] == "until_cancelled" and not caps["continuous_move"]:
            raise ValueError("Continuous motion is unavailable.")
        if item["type"] == "posture":
            if pose not in ("standing", "sitting"):
                raise ValueError("Cannot infer a safe posture transition from the observed state.")
            pose = item["pose"]
        elif pose != "standing":
            raise ValueError("Include a verified standing transition before this goal.")
        if item["type"] in ("kick", "navigate"):
            _check_target(item["target"], request["observation"]["entities"], caps["search_targets"])
    if planning_or_resume(value):
        observation = request["observation"]
        environment = observation["environment"]
        if environment["motion_allowed"] is not True:
            raise ValueError("The local motion gate is closed.")
        stationary_gestures = ("nod", "look_around", "open_beak", "close_beak")
        needs_space = any(g["type"] != "perform" or g["action"] not in stationary_gestures for g in checked_goals)
        if needs_space and environment["surface"] == "desktop" and environment["edge_guard"] != "available":
            raise ValueError("Desktop execution requires a verified local edge guard.")
        if observation["age_s"] > 0.5:
            raise ValueError("Refresh observations before accepting motion.")


def planning_or_resume(value):
    return value["decision"] in ("replace", "append", "resume")


def _check_target(query, entities, searchable):
    matching = []
    for entity in entities:
        attrs = {a["name"]: a["value"] for a in entity["attributes"]}
        if entity["category"] == query["category"] and all(
            attrs.get(a["name"]) == a["value"] for a in query["attributes"]
        ):
            if query["entity_id"] is None or query["entity_id"] == entity["id"]:
                matching.append(entity)
    if query["entity_id"] is not None and len(matching) != 1:
        raise ValueError("The bound entity is absent or contradicts target attributes.")
    if len(matching) > 1:
        raise ValueError("Multiple objects match; clarify instead of silently choosing.")
    if not matching and query not in searchable:
        raise ValueError("An unseen target needs a supported visual search capability.")
