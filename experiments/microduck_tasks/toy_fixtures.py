"""Explicitly fictional contract fixtures; never actual camera or execution data."""

import copy

from .toy_catalog import SELF_ACTIONS
from .toy_protocol import DEFAULT_MAX_GOALS, GOALS, VERSION, validate_output


def ball_target(color, entity_id=None):
    return {"category": "ball", "attributes": [{"name": "color", "value": color}], "entity_id": entity_id}


def kick(color):
    return {"type": "kick", "target": ball_target(color)}


def perform(action, repeat=1):
    return {"type": "perform", "action": action, "repeat": repeat}


def move(direction="forward", seconds=0.32, continuous=False):
    return {
        "type": "move",
        "direction": direction,
        "limit": {"kind": "until_cancelled"} if continuous else {"kind": "duration", "seconds": seconds},
    }


def turn(direction="left", angle=360):
    return {"type": "turn", "direction": direction, "angle_deg": angle}


def posture(pose):
    return {"type": "posture", "pose": pose}


def navigate(color):
    return {"type": "navigate", "target": ball_target(color), "standoff_m": 0.25}


def observed_ball(color, index=0):
    return {
        "id": f"observed:{color}:{index}",
        "category": "ball",
        "attributes": [{"name": "color", "value": color}],
        "position_body_m": [0.5, index * 0.12, 0.03],
        "confidence": 0.95,
    }


def request(utterance, *, planned=False):
    return {
        "schema_version": VERSION,
        "identity": {"name": "小鸭"},
        "event": {"kind": "user", "text": utterance},
        "history": [],
        "observation": {
            "age_s": 0.05,
            "entities": [],
            "posture": "standing",
            "stationary": True,
            "environment": {"surface": "simulation_floor", "motion_allowed": True, "edge_guard": "unavailable"},
        },
        "execution": {"state": "idle", "revision": 0, "active": None, "pending": [], "last_result": None},
        "capabilities": {
            "goal_types": list(GOALS) if planned else ["perform", "kick", "move", "posture"],
            "self_actions": list(SELF_ACTIONS),
            "continuous_move": planned,
            "sit_hold": planned,
            "search_targets": [ball_target(color) for color in ("red", "yellow", "blue")],
            "max_goals": DEFAULT_MAX_GOALS if planned else 1,
            "queue_control": planned,
        },
    }


def active(goal, *, paused=False, pending=()):
    return {
        "state": "paused" if paused else "running",
        "revision": 1,
        "active": copy.deepcopy(goal),
        "pending": list(copy.deepcopy(pending)),
        "last_result": None,
    }


def history(*pairs):
    return [
        item
        for user, assistant in pairs
        for item in (
            {"role": "user", "text": user},
            {"role": "assistant", "text": assistant},
        )
    ]


def add(rows, family, utterances, reply, decision="respond", goals=(), *, planned=False, modify=None):
    for i, utterance in enumerate(utterances):
        state = request(utterance, planned=planned)
        if modify is not None:
            modify(state)
        output = {"schema_version": VERSION, "reply": reply, "decision": decision, "goals": list(copy.deepcopy(goals))}
        validate_output(output, state)
        rows.append(
            {
                "case_id": f"{family}:{'planned' if planned else 'current'}:{i}",
                "family_id": family,
                "source": "hand_authored_design_fixture",
                "training_ready": False,
                "physical_verified": False,
                "capability_profile": "planned_semantics" if planned else "existing_primitives_without_v3_executor",
                "media": [],
                "input": state,
                "target": output,
            }
        )
