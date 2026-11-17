"""Shared target binding and skill durations for the teacher and learned selector."""

import json

from .language import COLORS, parse_instruction
from .plan_protocol import VERSION, parse_plan


def bind_target(instruction, state):
    color = parse_instruction(instruction).color
    matches = [
        e for e in state["entities"] if e["category"] == "ball" and {"name": "color", "value": color} in e["attributes"]
    ]
    if len(matches) != 1:
        return None
    return {key: matches[0][key] for key in ("id", "category", "attributes")}


def to_plan(selected, request):
    """Format the predicted symbol without replacing it with a rule-selected skill."""
    target, steps = None, []
    if selected in ("answer", "clarify", "finish"):
        status = selected
        reply = {"answer": "", "clarify": "请给出一个明确任务。", "finish": "模型判断本次任务已结束。"}[status]
        if selected == "answer":
            colors = [a["value"] for e in request.state["entities"] for a in e["attributes"] if a["name"] == "color"]
            reply = (
                "感知模块检测到：" + "、".join(COLORS.get(c, c) + "球" for c in colors)
                if colors
                else "感知模块未检测到球。"
            )
    else:
        status, reply = "continue", f"下一步：{selected}"
        if selected == "approach_target" or selected.startswith("kick_"):
            target = bind_target(request.instruction, request.state)
            if target is None:
                raise ValueError(f"Predicted {selected}, but the requested target cannot be bound to a visible entity.")
        if selected == "approach_target":
            seconds = 2.0
        elif selected.startswith("kick_"):
            seconds = 0.5
        elif selected.startswith("move_"):
            seconds = 0.32
        elif selected.startswith("turn_"):
            seconds = 0.6 if parse_instruction(request.instruction).kind == "turn" else 1.0
        else:
            seconds = 0.2
        steps = [dict(skill=selected, seconds=seconds)]
    return parse_plan(json.dumps(dict(schema_version=VERSION, status=status, target=target, steps=steps, reply=reply)))
