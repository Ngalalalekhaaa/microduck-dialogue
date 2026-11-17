"""Construct plans first, then explicit language; never obtain labels from a teacher."""

import copy

from .ablation_contract import common_domain, validate_context, wrap_plan
from .toy_fixtures import active, history, kick, move, navigate, perform, posture, request, turn
from .toy_protocol import DEFAULT_MAX_GOALS

VERSION = "microduck-dev240-v3-cap15"
GROUPS = ("single", "two_three", "four_six", "queue", "boundary", "state")
ATOMS = [
    ("点头一次", perform("nod")),
    ("左右看一次", perform("look_around")),
    ("张嘴一次", perform("open_beak")),
    ("闭嘴一次", perform("close_beak")),
    ("轻晃身体一次", perform("body_sway")),
    ("俯身再起一次", perform("ground_pick")),
    ("向前滚翻一次", perform("roll")),
    ("抬左脚一次", perform("flamingo_left")),
    ("抬右脚一次", perform("flamingo_right")),
    ("表演后退滑步一次", perform("moonwalk")),
    ("表演节奏舞一次", perform("groove")),
    ("表演害羞舞一次", perform("caught_dancing")),
    ("坐下再站起一次", perform("sit_stand")),
    ("向左转半圈", turn("left", 180)),
    ("向右转一整圈", turn("right", 360)),
    ("前进0.6秒", move(seconds=0.6)),
    ("后退8秒", move("backward", seconds=8)),
    ("踢红球", kick("red")),
    ("踢黄球", kick("yellow")),
    ("走到蓝球旁边", navigate("blue")),
]


def _context(state, index):
    mode = index % 4
    if mode == 1:
        state["history"] = history(("你好。", "你好，我是小鸭。"))
    elif mode == 2:
        state["history"] = history(("我们聊聊名字吧。", "我叫小鸭。"))
    elif mode == 3:
        state["execution"] = active(perform("look_around"))
        state["observation"]["stationary"] = False
    return ("cold_start", "greeting", "unrelated_chat", "running")[mode]


def build_cases(observation, *, max_goals=DEFAULT_MAX_GOALS):
    """Build current cases, or an explicit limited-capacity historical comparison."""
    rows = []

    def add(group, text, goals=(), decision="replace", *, modify=None, family=None, alternatives=()):
        index = sum(row["group"] == group for row in rows)
        state = request(text, planned=True)
        state["capabilities"]["max_goals"] = max_goals
        state["observation"] = copy.deepcopy(observation)
        context = _context(state, index) if group in GROUPS[:3] else "explicit_state"
        if modify:
            modify(state)
        expected = dict(decision=decision, goals=copy.deepcopy(list(goals)))
        validate_context(wrap_plan(expected), state)
        changed = [key for key in observation if state["observation"][key] != observation[key]]
        rows.append(
            dict(
                id=f"{group}_{index:02}",
                group=group,
                family_id=family or f"{group}:{index}",
                request=state,
                expected=expected,
                acceptable_decisions=[decision, *alternatives],
                source="synthetic_semantic",
                physical_verified=False,
                human_reviewed=False,
                context=context,
                requires_pixels=False,
                observation_changes=changed,
                domain="common" if common_domain(expected["goals"]) else "full_only",
                screen=index < 16,
                goal_count=len(goals),
            )
        )

    for i in range(40):
        phrase, goal = ATOMS[i % len(ATOMS)]
        if i in (12, 15, 32, 35):
            phrase, goal = {
                12: ("向左转45度", turn("left", 45)),
                15: ("前进2.3秒", move(seconds=2.3)),
                32: ("向右转135度", turn("right", 135)),
                35: ("走近蓝球，在距离0.3米处停下", dict(navigate("blue"), standoff_m=0.3)),
            }[i]
        text = f"小鸭，请{phrase}。" if i < 20 else f"我现在要你完成这个动作：{phrase}。"
        add("single", text, [goal], family=f"single:{i % 20}")
    for i in range(40):
        length = 2 + i % 2
        atoms = [ATOMS[(i * 3 + j * 7) % 20] for j in range(length)]
        text = "先" + "，然后".join(item[0] for item in atoms) + "。"
        if i % 4 == 1:
            text = "请按这个顺序做：" + "；".join(item[0] for item in atoms) + "。"
        if i >= 20:
            text = "顺序不要变，" + text
        add("two_three", text, [item[1] for item in atoms], family=f"two_three:{i % 20}")
    for i in range(40):
        length = 4 + i % 3
        atoms = [ATOMS[(i * 7 + j * 3) % 20] for j in range(length)]
        text = "这次有" + str(length) + "项，请依次完成：" + "；".join(item[0] for item in atoms) + "。"
        add("four_six", text, [item[1] for item in atoms])
    for i in range(40):
        goal_text, goal = ATOMS[(i // 8) * 3]
        mode = i % 8

        def modify(state, mode=mode):
            state["execution"] = active(move(seconds=8, continuous=mode >= 4), paused=mode == 5)
            state["observation"]["stationary"] = mode == 5

        text, decision, goals = (
            (f"在当前任务后再加上{goal_text}。", "append", [goal]),
            (f"把当前任务替换成{goal_text}。", "replace", [goal]),
            ("你叫什么？只回答，不改变当前任务。", "respond", []),
            ("取消当前任务。", "cancel", []),
            ("请暂停一下。", "pause", []),
            ("继续刚才暂停的行走任务。", "resume", []),
            (f"在持续行走后加上{goal_text}。", "clarify", []),
            (f"不再执行原任务，改成{goal_text}。", "replace", [goal]),
        )[mode]
        prefix = ("", "小鸭，", "现在，", "麻烦你，", "这轮请注意：")[i // 8]
        add(
            "queue",
            prefix + text,
            goals,
            decision,
            modify=modify,
            family=f"queue_mode:{mode}",
            alternatives=["decline"] if mode == 6 else [],
        )
    _boundary(add, max_goals=max_goals)
    _state(add, max_goals=max_goals)
    if len(rows) != 240 or any(sum(r["group"] == g for r in rows) != 40 for g in GROUPS):
        raise ValueError("The development suite must contain six groups of forty cases.")
    return rows


def _boundary(add, *, max_goals):
    for i in range(40):
        phrase, _ = ATOMS[i % 20]
        mode = i % 8
        count = (7, 15, 16, 15, 16)[i // 8] if max_goals == DEFAULT_MAX_GOALS else 7
        unsupported = ("后空翻", "叼起球", "搬运玩具", "倒立", "投篮")[i // 8]
        text, decision = (
            (f"我只是询问能力：你能{phrase}吗？", "respond"),
            (f"这轮不要{phrase}，只跟我聊天。", "respond"),
            (f"请{unsupported}。", "decline"),
            (f"先{phrase}，然后{unsupported}。", "decline"),
            ("踢红球还是蓝球，我还没选好，你先问清楚。", "clarify"),
            ("依次点头、张嘴、闭嘴、左右看、俯身再起、抬左脚、抬右脚，共七项。", "clarify"),
            (f"一直前进不要停，然后{phrase}。", "clarify"),
            (f"我在举例而非下指令：假如有人让你{phrase}，你会怎么回答？", "respond"),
        )[mode]
        goals = []
        if mode == 5:
            goals = [
                perform(name)
                for name in (
                    "nod",
                    "open_beak",
                    "close_beak",
                    "look_around",
                    "ground_pick",
                    "flamingo_left",
                    "flamingo_right",
                )
            ]
            if count != 7:
                atoms = [ATOMS[index % len(ATOMS)] for index in range(count)]
                text = "请依次完成：" + "；".join(phrase for phrase, _ in atoms) + f"，共{count}项。"
                goals = [goal for _, goal in atoms]
            if max_goals is None or count <= max_goals:
                decision = "replace"
            else:
                goals = []
        prefix = ("", "小鸭，", "现在，", "请理解我的意思：", "这轮请注意：")[i // 8]
        add(
            "boundary",
            prefix + text,
            goals,
            decision,
            family=f"boundary_mode:{mode}",
            alternatives=["decline"] if decision == "clarify" and mode in (5, 6) else [],
        )


def _state(add, *, max_goals):
    for i in range(40):
        mode = i % 10
        color = ("red", "yellow", "blue", "red")[i // 10]
        color_name = {"red": "红", "yellow": "黄", "blue": "蓝"}[color]
        existing_count = (6, 14, 15, 14)[i // 10] if max_goals == DEFAULT_MAX_GOALS else 6
        permits_append = max_goals is None or existing_count < max_goals

        def modify(state, mode=mode, color_name=color_name, existing_count=existing_count):
            if mode == 0:
                state["observation"]["posture"] = "sitting"
            elif mode == 1:
                state["history"] = history((f"我们接下来讨论{color_name}球。", "好的，记住了。"))
            elif mode == 2:
                state["history"] = history(("我们讨论红球和蓝球。", "明白，这两个球。"))
            elif mode == 3:
                state["execution"]["last_result"] = dict(
                    goal=move(continuous=True), status="cancelled", reason="user_cancelled", verified=False
                )
            elif mode == 4:
                state["observation"]["age_s"] = 1.0
            elif mode == 5:
                state["observation"]["environment"]["motion_allowed"] = False
            elif mode == 6:
                state["capabilities"]["self_actions"].remove("nod")
            elif mode == 7:
                state["observation"]["entities"] = []
            elif mode == 8:
                state["history"] = history(("请张嘴一次。", "已执行张嘴一次。"))
                state["execution"]["last_result"] = dict(
                    goal=perform("open_beak"), status="succeeded", reason="verified_completed", verified=True
                )
            else:
                state["execution"] = active(perform("nod"), pending=[perform("nod")] * (existing_count - 1))

        text, decision, goals = (
            ("向右转90度。", "replace", [posture("standing"), turn("right", 90)]),
            ("请踢刚才提到的那种颜色的球。", "replace", [kick(color)]),
            ("刚才提到的两个球，踢其中那个吧。", "clarify", []),
            ("继续之前取消的任务。", "clarify", []),
            ("现在前进0.6秒。", "decline", []),
            ("现在后退8秒。", "decline", []),
            ("点头一次。", "decline", []),
            (f"去找{color_name}球并踢它。", "replace", [kick(color)]),
            ("刚才成功的动作再做一次。", "replace", [perform("open_beak")]),
            ("在已有六项目标后再加一次闭嘴。", "clarify", []),
        )[mode]
        if mode == 9:
            if existing_count != 6:
                text = f"在已有{existing_count}项目标后再加一次闭嘴。"
            if permits_append:
                decision, goals = "append", [perform("close_beak")]
        prefix = ("", "小鸭，", "现在，", "请结合当前情况：")[i // 10]
        add(
            "state",
            prefix + text,
            goals,
            decision,
            modify=modify,
            family=f"state_mode:{mode}",
            alternatives=["decline"]
            if mode == 3 or (mode == 9 and not permits_append)
            else ["clarify"]
            if mode in (4, 5)
            else [],
        )
