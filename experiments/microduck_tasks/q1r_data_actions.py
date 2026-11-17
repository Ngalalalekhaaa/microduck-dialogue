"""Rich supported-action expressions and ordered queues, without motor claims."""

import copy

from .q1r_data_core import COLOR_NAMES, base_context
from .q1r_language import (
    ACTION_WORDINGS,
    KICK_WORDINGS,
    MOVE_WORDINGS,
    NAVIGATE_WORDINGS,
    POSTURE_WORDINGS,
    SEQUENCE_WORDINGS,
    TURN_WORDINGS,
)
from .toy_fixtures import kick, move, navigate, perform, posture, turn
from .toy_teacher import _goal_label

DURATIONS = (
    ("3秒", 3),
    ("3s", 3),
    ("三秒", 3),
    ("0.5秒", 0.5),
    ("500毫秒", 0.5),
    ("两秒", 2),
    ("1s", 1),
    ("一秒半", 1.5),
    ("半秒", 0.5),
    ("10秒", 10),
    ("八秒", 8),
    ("0.2 s", 0.2),
)


def generate_actions(builder):
    for action, wordings in ACTION_WORDINGS.items():
        builder.commands("self_action", action, wordings, lambda w, i, a=action: (w, [perform(a)]))
    for direction, wordings in MOVE_WORDINGS.items():

        def build(w, i, d=direction):
            word, seconds = DURATIONS[i % len(DURATIONS)]
            return w.format(t=word), [move(d, seconds)]

        builder.commands("duration_move", direction, wordings, build, variants=18, val_variants=2)
    for kind, wordings, create in (("kick", KICK_WORDINGS, kick), ("navigate", NAVIGATE_WORDINGS, navigate)):

        def build(w, i, create=create):
            color = list(COLOR_NAMES)[i % 3]
            return w.format(c=COLOR_NAMES[color]), [create(color)]

        builder.commands("target", kind, wordings, build, variants=18, val_variants=2)
    for pose, wordings in POSTURE_WORDINGS.items():
        for p, wording in enumerate(wordings):
            split = "val" if p >= len(wordings) - 3 else "train"
            for i in range(1 if split == "val" else 12):
                initial = "sitting" if i % 2 == 0 else "standing"
                req = base_context(wording, i, posture_name=initial)
                builder.add(
                    f"posture_{pose}_{p}",
                    split,
                    "posture",
                    req,
                    "replace",
                    [posture(pose)],
                    rationale="保持坐姿和坐站表演不同；明确站起或坐下转换只输出请求的姿态。",
                )

    def build_turn(w, i):
        direction = ("left", "right")[i % 2]
        angle, word = ((30, "30度"), (90, "九十度"), (180, "半圈"), (360, "一圈"), (45, "45°"), (15, "十五度"))[
            (i // 2) % 6
        ]
        return w.format(d="左" if direction == "left" else "右", a=word), [turn(direction, angle)]

    builder.commands("turn", "angle", TURN_WORDINGS, build_turn, variants=24, val_variants=2)
    continuous = (
        "一直往{d}走，等我喊停",
        "向{d}持续走",
        "往{d}走，不用自己停",
        "一直朝{d}移动吧",
        "小鸭，持续向{d}走到我说停为止",
        "朝{d}方向一直走",
        "向{d}走，我会叫停",
        "往{d}移动，直到取消",
        "保持向{d}行走，收到停止再停",
        "请持续朝{d}方行进，等待我的停止指令",
    )

    def build_continuous(w, i):
        direction = ("forward", "backward", "left", "right")[i % 4]
        name = {"forward": "前", "backward": "后", "left": "左", "right": "右"}[direction]
        return w.format(d=name), [move(direction, continuous=True)]

    builder.commands("continuous", "move", continuous, build_continuous, variants=12, val_variants=2, heldout=2)
    repeat_wordings = (
        "{a}，重复{n}次",
        "连续做{n}遍{a}",
        "给我来{n}次{a}",
        "{a}这个动作做{n}遍",
        "请重复{n}次{a}动作",
        "{a}做完再做，合计{n}次",
        "来{n}轮{a}表演",
        "把{a}反复做{n}次",
        "请将{a}连续展示{n}次",
        "一共演示{n}遍{a}",
    )

    def build_repeat(w, i):
        action = list(ACTION_WORDINGS)[i % len(ACTION_WORDINGS)]
        count = (1, 2, 3, 4, 6, 7, 9)[i % 7]
        goals = [perform(action, min(3, count - j)) for j in range(0, count, 3)]
        return w.format(a=_goal_label(perform(action)), n=count), goals

    builder.commands("repeat", "bounded", repeat_wordings, build_repeat, variants=26, val_variants=2, heldout=2)
    _sequences(builder)


def _sequences(builder):
    pool = [perform(a) for a in ACTION_WORDINGS] + [kick(c) for c in COLOR_NAMES]
    pool += [navigate(c) for c in COLOR_NAMES]
    pool += [move("backward", 3), move("left", 0.5), move("right", 2), turn("left", 90), turn("right", 180)]
    for p, wording in enumerate(SEQUENCE_WORDINGS):
        split = "val" if p >= len(SEQUENCE_WORDINGS) - 4 else "train"
        for i in (0, 6, 13) if split == "val" else range(28):
            count = 2 + i % 14
            goals = [copy.deepcopy(pool[(p * 7 + i * 3 + j * 5) % len(pool)]) for j in range(count)]
            # Include explicit posture transitions without invalidating following movement.
            if i % 5 == 0 and count >= 3:
                goals[:2] = [posture("sitting"), posture("standing")]
            labels = [_goal_label(g) for g in goals]
            joiner = ("，然后", "，接着", "；", "，再")[p % 4]
            text = wording.format(items=joiner.join(labels))
            req = base_context(text, i, state=("idle", "running", "paused")[i % 3])
            builder.add(
                f"sequence_{p}",
                split,
                "ordered_queue",
                req,
                "replace",
                goals,
                rationale="按本轮完整清单逐项排列，保留 2 至 15 项；旧对话里的紫球、黄球不得擅自加入。",
            )
    overflow_wordings = (
        "按顺序做这16项：{items}",
        "这一组共十六个任务：{items}",
        "请把以下16个目标全部排进去：{items}",
        "我现在要求十六项连着做：{items}",
        "接下来这十六项请依次执行：{items}",
    )
    for p, wording in enumerate(overflow_wordings):
        split = "val" if p == 4 else "train"
        for i in range(1 if split == "val" else 14):
            goals = [pool[(i + j) % len(pool)] for j in range(16)]
            req = base_context(wording.format(items="，然后".join(_goal_label(g) for g in goals)), i)
            builder.add(
                f"queue_overflow_{p}",
                split,
                "capacity",
                req,
                "clarify",
                reply="这次有16项任务，超过15项上限。请先选最多15项，剩下的稍后再安排。",
                rationale="不得截断队列或暗中少执行一项；整组确认后再开始。",
            )
