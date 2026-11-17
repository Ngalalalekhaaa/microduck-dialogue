"""Task-general v2 skill plans; independent of the existing trained v1 prompt."""

import json
from dataclasses import asdict, dataclass

from .skill_vlm import SKILLS, _keys, _loads, _number, _text, parse_proposal

VERSION = "microduck-plans-2"
CATALOG = {name: (0.5 if name.startswith("kick_") else 0.6 if name.startswith("move_") else 1.0) for name in SKILLS}
CATALOG["approach_target"] = 2.0
TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "category": {"type": "string"},
        "attributes": {},
    },
    "required": ["id", "category", "attributes"],
    "additionalProperties": False,
}
# Attributes are generic key/value entries on the wire, not task-specific fields.
TARGET_SCHEMA["properties"]["attributes"] = {
    "type": "array",
    "maxItems": 8,
    "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
        "required": ["name", "value"],
        "additionalProperties": False,
    },
}
STEP_SCHEMA = {
    "type": "object",
    "properties": {"skill": {"enum": list(CATALOG)}, "seconds": {"type": "number", "minimum": 0.02, "maximum": 2.0}},
    "required": ["skill", "seconds"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "schema_version": {"const": VERSION},
        "status": {"enum": ["continue", "answer", "clarify", "finish"]},
        "target": {"anyOf": [TARGET_SCHEMA, {"type": "null"}]},
        "steps": {"type": "array", "maxItems": 4, "items": STEP_SCHEMA},
        "reply": {"type": "string", "maxLength": 500},
    },
    "required": ["schema_version", "status", "target", "steps", "reply"],
    "additionalProperties": False,
}
SYSTEM = """You select bounded skills for a simulated mobile robot from RGB, proprioception,
the original user instruction and execution feedback. Return exactly the supplied JSON schema.
Use schema_version microduck-plans-2. Choose a target from observed entities by id/category/attributes,
or null for an untargeted motion/question. Object color is an attribute, not the task definition.
Do not invent visible targets. If a named target is not visible, turn to search with target=null.
Outputs are at most four steps and THREE seconds total, all durations multiples of 0.02 seconds.
Existing primitive skills: stand, move_forward/backward/left/right, turn_left/right, kick_left/right.
Primitive translations last at most 0.6s; stand/turns at most 1s; kicks exactly 0.5s.
approach_target is a closed-loop navigation skill, at most 2s, using the camera to face and approach
the selected visible entity. It stops when the entity is reachable, lost or obstructed. It cannot kick.
Prefer approach_target for navigation over repetitive tiny motion commands. You may return several
approach steps within the total budget. Kicks must be the last step, never blindly repeated;
the executor rechecks reach and stability and performs standing recovery after a kick.
The currently committed steps in context WILL execute before your continuation: plan what follows them,
not their repetition. Any interrupted prefix invalidates its continuation. Feedback overrides intent.
For a single move request execute it once and finish; for view questions answer without motion.
finish is a model claim, not physical verification. Missing objects may be outside the view.
No hidden reasoning text, joint commands or simulator coordinates. Reply in concise Chinese.

接口示例（只是格式示例，目标必须来自本次输入）：
用户“看看眼前有什么”：status=answer，target=null，steps=[]，用 reply 回答，禁止移动。
用户“往前走一点”，还未执行：
{"schema_version":"microduck-plans-2","status":"continue","target":null,
 "steps":[{"skill":"move_forward","seconds":0.32}],"reply":"向前走一小段。"}
当 committed_steps 或 recent_outcomes 已包含这次前进，返回：
{"schema_version":"microduck-plans-2","status":"finish","target":null,"steps":[],"reply":"已前进。"}
用户“左转一下”：先输出一个 turn_left 0.6 秒，之后 finish；没有指定物体时 target 必须 null。
用户“踢红球”且看不到红球：target=null，输出 turn_left 1.0 秒搜索，不要选择其他颜色。
不要为了凑满四步复制同一个动作。0.02 秒几乎没有实际位移，正常行走用 0.2～0.6 秒，
目标接近技能用 1～2 秒。continue 必须有动作；answer/clarify/finish 的 steps 必须是空数组。
"""


@dataclass(frozen=True)
class Plan:
    schema_version: str
    status: str
    target: dict | None
    steps: list
    reply: str

    def to_dict(self):
        return asdict(self)


def parse_plan(raw):
    if not isinstance(raw, str) or len(raw) > 16384:
        raise ValueError("Supply a v2 JSON plan of at most 16384 characters.")
    value = _loads(raw)
    fields = {"schema_version", "status", "target", "steps", "reply"}
    _keys(value, fields, "plan", fields)
    if value["schema_version"] != VERSION or value["status"] not in ("continue", "answer", "clarify", "finish"):
        raise ValueError("Use the v2 schema and a supported plan status.")
    _text(value["reply"], "reply", 500)
    target = value["target"]
    if target is not None:
        _keys(target, {"id", "category", "attributes"}, "target", {"id", "category", "attributes"})
        for key in ("id", "category"):
            _text(target[key], key, 120)
            if not target[key].strip():
                raise ValueError("Target id/category cannot be empty.")
        attrs = target["attributes"]
        if not isinstance(attrs, list) or len(attrs) > 8:
            raise ValueError("Target attributes must be at most eight name/value entries.")
        names = set()
        for attr in attrs:
            _keys(attr, {"name", "value"}, "attribute", {"name", "value"})
            for key in ("name", "value"):
                _text(attr[key], key, 120)
            if not attr["name"] or attr["name"] in names:
                raise ValueError("Attribute names must be unique and nonempty.")
            names.add(attr["name"])
    steps = value["steps"]
    if not isinstance(steps, list) or len(steps) > 4 or bool(steps) != (value["status"] == "continue"):
        raise ValueError("Continue needs 1..4 steps; other statuses need an empty step list.")
    total = 0
    for index, step in enumerate(steps):
        _keys(step, {"skill", "seconds"}, "step", {"skill", "seconds"})
        skill, seconds = step["skill"], _number(step["seconds"], "seconds")
        if not isinstance(skill, str) or skill not in CATALOG:
            raise ValueError("Choose a registered skill.")
        if skill == "approach_target":
            if target is None or not 0.02 <= seconds <= 2 or abs(seconds / 0.02 - round(seconds / 0.02)) > 1e-8:
                raise ValueError("Approach needs a target and 0.02..2 seconds on the 50 Hz grid.")
        else:
            parse_proposal(
                json.dumps({"type": "skill", "skill": skill, "seconds": seconds, "target_color": None, "reply": ""})
            )
        if skill.startswith("kick_") and (target is None or index != len(steps) - 1):
            raise ValueError("A kick needs a target and must be the final step.")
        total += seconds
    if total > 3.000001:
        raise ValueError("A plan may authorize at most three seconds; request another observation.")
    return Plan(**value)


def entities(observation):
    """Adapt this experiment's color detector; future detectors share the entity contract."""
    return [
        {
            "id": f"ball:{d['color']}",
            "category": "ball",
            "attributes": [{"name": "color", "value": d["color"]}],
            "position_body_m": d["center_body_m"],
            "confidence": d["confidence"],
            "source": "rgbd_color_sphere_detector",
            "camera": d.get("camera"),
        }
        for d in observation["detections"]
    ]


def resolve_target(target, observation):
    if target is None:
        return None
    for entity in entities(observation):
        if all(target[key] == entity[key] for key in ("id", "category", "attributes")):
            return entity
    return None


def validate_input(state, context):
    fields = {
        "yaw",
        "stationary",
        "tilt_deg",
        "entities",
        "observation_id",
        "task_elapsed_s",
        "available_skills",
        "schema_version",
    }
    _keys(state, fields, "v2 sensor state", fields)
    for key in ("yaw", "tilt_deg", "task_elapsed_s"):
        _number(state[key], key)
    if state["tilt_deg"] < 0 or state["task_elapsed_s"] < 0 or not isinstance(state["stationary"], bool):
        raise ValueError("Invalid v2 posture/time flags.")
    if state["schema_version"] != VERSION or state["available_skills"] != CATALOG:
        raise ValueError("Use the current version and registered capability catalog.")
    _text(state["observation_id"], "observation_id", 120)
    if not isinstance(state["entities"], list) or len(state["entities"]) > 20:
        raise ValueError("Supply at most twenty sensor entities.")
    ids = set()
    for entity in state["entities"]:
        keys = {"id", "category", "attributes", "position_body_m", "confidence", "source", "camera"}
        _keys(entity, keys, "sensor entity", keys)
        target = {k: entity[k] for k in ("id", "category", "attributes")}
        parse_plan(
            json.dumps(
                {"schema_version": VERSION, "status": "answer", "target": target, "steps": [], "reply": ""},
                allow_nan=False,
            )
        )
        if entity["id"] in ids:
            raise ValueError("Entity identities must be unique.")
        ids.add(entity["id"])
        point = entity["position_body_m"]
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError("Entity positions require three body-frame metre coordinates.")
        for coordinate in point:
            _number(coordinate, "body-frame coordinate")
        if not 0 <= _number(entity["confidence"], "confidence") <= 1:
            raise ValueError("Confidence must be in [0, 1].")
        if entity["source"] != "rgbd_color_sphere_detector" or entity["camera"] not in (None, "front", "near"):
            raise ValueError("Use a supported robot sensor source, never oracle/external views.")
    fields = {"task_id", "committed_steps", "recent_outcomes"}
    _keys(context, fields, "v2 context", fields)
    _text(context["task_id"], "task_id", 120)
    for key, limit in (("committed_steps", 4), ("recent_outcomes", 6)):
        if not isinstance(context[key], list) or len(context[key]) > limit:
            raise ValueError(f"Bound {key} to {limit} entries.")
    for step in context["committed_steps"]:
        _keys(step, {"skill", "seconds"}, "committed step", {"skill", "seconds"})
        if step["skill"] not in CATALOG or not 0.02 <= _number(step["seconds"], "seconds") <= CATALOG[step["skill"]]:
            raise ValueError("Invalid committed skill.")
    for outcome in context["recent_outcomes"]:
        _keys(outcome, {"step", "outcome", "target"}, "outcome", {"step", "outcome", "target"})
        _text(outcome["outcome"], "outcome", 120)
        parse_plan(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "status": "continue",
                    "target": outcome["target"],
                    "steps": [outcome["step"]],
                    "reply": "",
                },
                allow_nan=False,
            )
        )
