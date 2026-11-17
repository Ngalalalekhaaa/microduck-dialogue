"""Small Chinese command grammar; unknown or ambiguous commands never move a robot."""

import math
import re
from dataclasses import asdict, dataclass
from typing import Literal

COLORS = {"red": "红", "yellow": "黄", "blue": "蓝"}
KINDS = ("kick", "move", "turn", "describe", "stop", "clarify")
DIRECTIONS = ("forward", "backward", "left", "right")
NEGATIVE = r"(?:不要|不许|不能|禁止|别|不想|不打算|不需要|无需|不)(?:再|去|帮我|给我|替我)*"


@dataclass(frozen=True)
class Intent:
    kind: Literal["kick", "move", "turn", "describe", "stop", "clarify"]
    color: str | None = None
    direction: str | None = None
    seconds: float = 0.32
    reply: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError("Use a supported intent kind.")
        if self.color is not None and self.color not in COLORS:
            raise ValueError("Use red, yellow or blue for color.")
        if self.direction is not None and self.direction not in DIRECTIONS:
            raise ValueError("Use forward, backward, left or right for direction.")
        if self.kind == "kick" and (self.color is None or self.direction is not None):
            raise ValueError("A kick requires one color and no direction.")
        if self.kind in ("move", "turn") and (self.direction is None or self.color is not None):
            raise ValueError("A movement requires one direction and no color.")
        if self.kind == "turn" and self.direction not in ("left", "right"):
            raise ValueError("A turn must be left or right.")
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)):
            raise ValueError("seconds must be a finite number between 0.12 and 0.8.")
        if not math.isfinite(self.seconds) or not 0.12 <= self.seconds <= 0.8:
            raise ValueError("seconds must be a finite number between 0.12 and 0.8.")
        if not math.isclose(self.seconds / 0.02, round(self.seconds / 0.02), abs_tol=1e-9):
            raise ValueError("seconds must be a multiple of the 0.02 second control interval.")
        if not isinstance(self.reply, str):
            raise ValueError("reply must be text.")

    def to_dict(self) -> dict:
        return asdict(self)


INTENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "color", "direction", "seconds", "reply"],
    "properties": {
        "kind": {"type": "string", "enum": list(KINDS)},
        "color": {"enum": [*COLORS, None]},
        "direction": {"enum": [*DIRECTIONS, None]},
        "seconds": {"type": "number", "minimum": 0.12, "maximum": 0.8, "multipleOf": 0.02},
        "reply": {"type": "string"},
    },
}


def intent_from_dict(value: dict) -> Intent:
    """Validate decoded model output before giving it to the task controller."""
    if not isinstance(value, dict):
        raise ValueError("A decoded intent must be an object.")
    return Intent(**value)


def _clarify(reply="请说清楚一个任务，例如‘踢红球’、‘往前走’或‘你看到了什么’。"):
    return Intent("clarify", reply=reply)


def _colors(text):
    return [color for color, chinese in COLORS.items() if chinese in text or color in text]


def _without_negative_targets(text):
    target = r"(?:那个|这个)?(?:红|黄|蓝)(?:色)?(?:的)?(?:小)?(?:球|那个|这个)?"
    text = re.sub(NEGATIVE + r"踢(?:一下|一脚)?" + target + r"(?:踢走|踢开)?", "，", text)
    text = re.sub(NEGATIVE + r"把" + target + r"踢(?:走|开)?", "，", text)
    text = re.sub(target + NEGATIVE + r"踢(?:了)?", "，", text)
    return re.sub(NEGATIVE + target, "，", text)


def _referenced_color(context):
    for key in ("target_color", "selected_color"):
        color = context.get(key)
        if color in COLORS:
            return color
    colors = context.get("visible_colors", [])
    if isinstance(colors, list) and len(set(colors)) == 1 and colors[0] in COLORS:
        return colors[0]
    return None


def parse_instruction(text: str, context: dict | None = None) -> Intent:
    """Map supported paraphrases to one intent, preserving negation and ambiguity.

    ``context`` may contain ``target_color``, ``selected_color`` or an unambiguous
    ``visible_colors`` list to resolve pronouns. It must come from the conversation
    and current visual observations, never the simulator's hidden object table.
    """
    if not isinstance(text, str) or not text.strip():
        return _clarify()
    text = re.sub(r"\s+", "", text.lower())
    if re.search(r"(?:看|视野|眼前).*(?:什么|哪些|啥|有没有|有球|吗|么)|(?:说说|描述).*(?:看|视野|眼前)", text):
        return Intent("describe")
    if re.search(r"(?:眼前|视野|面前|前面|周围).*(?:有什么|有哪些|有啥)", text):
        return Intent("describe")
    if re.search(r"吗|么|能否|是否|会不会|能不能|可不可以|可以不|怎么|如何|为什么|想知道|\?|？", text):
        return _clarify("我支持踢指定颜色的球、短距离移动和转身；要执行时请直接下达指令。")
    if re.search(r"刚才|昨天|已经|上次|曾经|以前", text):
        return _clarify("请直接说明现在要执行的任务。")
    if re.search(r"如果|假如|要是|除非|否则|或者|还是", text):
        return _clarify("请先确定一个要执行的动作和目标。")
    if re.search(r"(?:不要|别|不许)停", text):
        return _clarify("请说明接下来要执行的具体动作。")
    if re.search(r"停下|停止|停住|停一停|先停|别动|不要动|别(?:再)?走|不要(?:再)?走|站住|暂停", text):
        return Intent("stop", reply="已停止任务。")

    positive = _without_negative_targets(text)
    if re.search(NEGATIVE + r"(?:踢|走|动|前进|后退|转)", positive):
        return Intent("stop", reply="好的，不执行这个动作。")
    if re.search(r"(?:\d|[两二三四五六七八九十百])+(?:次|下|步|秒|米|度)", positive):
        return _clarify("当前移动和转身按一个短动作执行，请先省略次数、距离或角度。")
    if re.search(r"转.*(?:圈|一周)|一圈", positive):
        return _clarify("请说‘向左转’或‘向右转’；指定球不在视野时，找球任务会自动搜索。")

    turn = []
    remaining = positive
    for direction, pattern in (
        ("left", r"(?:往|向|朝)?左(?:边|侧)?(?:转|拐)|转向左(?:边|侧)?"),
        ("right", r"(?:往|向|朝)?右(?:边|侧)?(?:转|拐)|转向右(?:边|侧)?"),
    ):
        if re.search(pattern, remaining):
            turn.append(direction)
            remaining = re.sub(pattern, "", remaining)
    moves = []
    for direction, pattern in (
        ("forward", r"前进|(?:往|向|朝)前|前(?:面|方)?(?:走|挪|移|迈)"),
        ("backward", r"后退|退后|倒退|(?:往|向|朝)后|后(?:面|方)?(?:走|挪|移)"),
        ("left", r"左移|(?:往|向|朝)左|左(?:边|侧)?(?:走|挪|移)"),
        ("right", r"右移|(?:往|向|朝)右|右(?:边|侧)?(?:走|挪|移)"),
    ):
        if re.search(pattern, remaining):
            moves.append(direction)
    kick = "踢" in positive or "kick" in positive
    if int(kick) + len(turn) + len(moves) > 1:
        return _clarify("请一次给出一个动作；踢球任务会自动完成寻找、接近和对齐。")
    if kick:
        colors = _colors(positive)
        if len(colors) > 1:
            return _clarify("这次要踢红球、黄球还是蓝球？请指定一种颜色。")
        color = colors[0] if colors else None
        if color is None and re.search(r"它|那个|这个", positive):
            color = _referenced_color(context or {})
        if color is None:
            return _clarify("要踢哪种颜色的球？请说红球、黄球或蓝球。")
        return Intent("kick", color=color)
    if turn:
        return Intent("turn", direction=turn[0], seconds=0.6)
    if moves:
        return Intent("move", direction=moves[0])
    if positive != text and re.search(NEGATIVE, text):
        return Intent("stop", reply="好的，不踢这个球。")
    if positive.strip("，,。.!！;；") == "":
        return Intent("stop", reply="好的，不踢这个球。")
    return _clarify()


def describe_view(detections: list[dict]) -> str:
    """Describe only validated, currently visible detections in the supplied frame."""
    visible = {}
    for detection in detections:
        if not isinstance(detection, dict) or detection.get("visible", True) is not True:
            continue
        color = detection.get("color")
        if color not in COLORS:
            continue
        center = detection.get("center_body_m")
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            continue
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in center):
            continue
        x, y, _ = center
        if x >= 0:
            bearing = "左前方" if y > 0.03 else "右前方" if y < -0.03 else "前方"
        else:
            bearing = "左后方" if y > 0.03 else "右后方" if y < -0.03 else "后方"
        visible.setdefault(color, f"{bearing}有{COLORS[color]}球")
    if not visible:
        return "当前画面中没有检测到红球、黄球或蓝球。"
    return "我现在看到" + "，".join(visible[color] for color in COLORS if color in visible) + "。"
