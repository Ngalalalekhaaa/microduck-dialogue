"""Queue control, latest-turn corrections and referent handling in fictional contexts."""

from .q1r_data_core import acknowledgement, base_context
from .toy_fixtures import active, history, kick, move, perform, posture, turn
from .toy_teacher import _goal_label


def generate_context(builder):
    _controls(builder)
    _append(builder)
    _corrections(builder)
    _references(builder)


def _controls(builder):
    controls = {
        "pause": (
            "先暂停一下",
            "暂停当前任务吧",
            "等这一小段站稳再暂停",
            "把任务暂时挂起，后面留着",
            "先歇一下，别清空任务",
            "暂停但别取消后面的动作",
            "帮我安全暂停",
            "任务保留，先停稳等我",
            "暂缓执行并保存余下任务",
            "暂停一下，稍后还要接着做",
        ),
        "cancel": (
            "停下",
            "取消全部任务",
            "不用继续了，剩下都取消",
            "马上停止执行",
            "这组动作不要做了",
            "停止并清空队列",
            "够了，后面不用做",
            "小鸭别走了，任务取消",
            "终止当前安排并丢弃剩余项",
            "所有未完成动作都撤销吧",
        ),
        "resume": (
            "继续吧",
            "恢复执行",
            "接着做暂停前剩下的",
            "从刚才暂停的地方继续",
            "好啦，继续未完成任务",
            "让暂停的任务继续走",
            "继续刚才保留的安排",
            "恢复那组还没做完的动作",
            "请接续暂停队列的剩余部分",
            "把保留下来的任务继续完成",
        ),
    }
    for operation, wordings in controls.items():
        for p, text in enumerate(wordings):
            split = "val" if p >= 8 else "train"
            for i in range(4 if split == "val" else 12):
                state = ("running", "paused", "idle", "failed")[i % 4]
                req = base_context(text, i, state=state)
                if state in ("running", "paused"):
                    current = (perform("roll"), move("forward", continuous=True), kick("blue"))[i % 3]
                    req["execution"] = active(
                        current, paused=state == "paused", pending=[] if current["type"] == "move" else [perform("nod")]
                    )
                if operation == "cancel":
                    decision, reply = "cancel", "收到，我会停止执行并取消剩余任务；之后需要重新安排。"
                elif operation == "pause":
                    decision = "pause" if state == "running" else "respond"
                    reply = (
                        "收到，我会在当前动作的安全边界站稳后暂停，保留未完成任务，推理继续。"
                        if state == "running"
                        else "任务已经暂停，未完成部分保留着。"
                        if state == "paused"
                        else "当前没有正在运行的任务可以暂停。"
                    )
                else:
                    decision = "resume" if state == "paused" else "respond" if state == "running" else "clarify"
                    reply = (
                        "收到，我会继续尚未完成的部分，不重做已完成动作。"
                        if state == "paused"
                        else "任务正在运行，我会继续当前安排。"
                        if state == "running"
                        else "当前没有暂停的队列可恢复。停止或失败后，请重新说要做的任务。"
                    )
                builder.add(
                    f"control_{operation}_{p}",
                    split,
                    "control_" + operation,
                    req,
                    decision,
                    reply=reply,
                    rationale="控制判断依赖真实执行状态；暂停保留队列、安全边界后生效，取消清空，空队列不可恢复。",
                )


def _append(builder):
    wordings = (
        "做完当前任务后，再{a}",
        "把{a}接在后面",
        "原来的照做，另外加上{a}",
        "追加一个{a}",
        "后面再排上{a}",
        "现有任务不变，最后{a}",
        "剩下任务做完再{a}",
        "把新动作{a}追加到队尾",
        "保留原队列，再增加{a}",
        "在现有安排结束之后补上{a}",
        "请在队列末端接续{a}",
        "原任务之后增补一项：{a}",
    )
    for p, wording in enumerate(wordings):
        split = "val" if p >= 9 else "train"
        for i in range(5 if split == "val" else 16):
            new = (perform("nod"), kick("red"), turn("right", 90))[i % 3]
            req = base_context(wording.format(a=_goal_label(new)), i)
            n = (1, 5, 13, 14, 15, 3)[i % 6]
            paused = i % 2 == 1
            existing = [perform("look_around")] + [kick(("blue", "yellow")[j % 2]) for j in range(n - 1)]
            if i % 6 == 5:
                existing = [move("forward", continuous=True)]
            req["execution"] = active(existing[0], paused=paused, pending=existing[1:])
            if existing[0]["type"] == "move":
                builder.add(
                    f"append_{p}",
                    split,
                    "append_capacity",
                    req,
                    "clarify",
                    reply="当前是持续行走，后面的任务轮不到执行。请先停止或明确替换它，再安排新动作。",
                    rationale="不能往无限期任务后追加不可达目标。",
                )
            elif n == 15:
                builder.add(
                    f"append_{p}",
                    split,
                    "append_capacity",
                    req,
                    "clarify",
                    reply="已有15项未完成任务，再加会超过上限。请等完成一项，或明确替换当前任务。",
                    rationale="上限按 active 加 pending 再加新目标合计，不仅检查本条输出。",
                )
            else:
                reply = acknowledgement([new], i, append=True)
                if paused:
                    reply += "队列仍保持暂停，等你说继续。"
                builder.add(
                    f"append_{p}",
                    split,
                    "append_capacity",
                    req,
                    "append",
                    [new],
                    reply,
                    rationale="保留正在进行和待执行目标，仅输出新增目标；暂停状态下追加不自动恢复。",
                )
    # Multiple new goals and exact capacity boundary, including idle '再' as a fresh command.
    for p, wording in enumerate(("末尾加上{a}，然后{b}", "接在后面依次{a}、{b}", "当前做完以后补两项：{a}，再{b}")):
        split = "val" if p == 2 else "train"
        for i in range(4 if split == "val" else 16):
            new = [perform("open_beak"), perform("close_beak")]
            req = base_context(wording.format(a="张嘴", b="闭嘴"), i)
            n = (12, 13, 14, 15)[i % 4]
            req["execution"] = active(perform("look_around"), pending=[perform("nod")] * (n - 1))
            decision = "append" if n + 2 <= 15 else "clarify"
            builder.add(
                f"append_two_{p}",
                split,
                "append_capacity",
                req,
                decision,
                new if decision == "append" else [],
                reply=acknowledgement(new, i, append=True)
                if decision == "append"
                else f"已有{n}项任务，再加两项会超过15项。请先减少或替换任务。",
                rationale="两项追加按累计容量一起接纳或一起澄清，不暗中保留一半。",
            )


def _corrections(builder):
    cases = (
        ("别踢蓝球了，改成点头", "先踢蓝球", [perform("nod")]),
        ("上一条不算，这次只后退三秒", "前滚翻后踢黄球", [move("backward", 3)]),
        ("我改口，红球不用踢，只踢蓝球", "请踢红球", [kick("blue")]),
        ("刚才颜色说错了，要黄球，去踢黄球", "踢蓝球", [kick("yellow")]),
        ("那些旧任务都取消，新的任务是前滚翻", "踢绿球再踢紫球", [perform("roll")]),
        ("现在只张嘴，其他的全都不用做", "点头，然后抬左脚", [perform("open_beak")]),
        ("我说的是后退3秒，不是点头", "点头", [move("backward", 3)]),
        ("不要把紫球加进去，本次踢蓝球再点头", "能踢紫球吗", [kick("blue"), perform("nod")]),
        ("把本轮全部换成右转九十度", "向左走两秒", [turn("right", 90)]),
        ("先前那套作废，现在只跳害羞舞", "前滚翻，然后踢红球", [perform("caught_dancing")]),
        ("不需要接着踢球了，坐下保持就行", "踢蓝球之后踢黄球", [posture("sitting")]),
        ("新任务只有抬右脚，请不要加别的", "点头之后抬左脚", [perform("flamingo_right")]),
        ("更正为向左走半秒，旧的右移不要了", "向右走两秒", [move("left", 0.5)]),
        ("以前说的都作废，仅展示月球步", "踢红球再点头", [perform("moonwalk")]),
        ("忽略先前的颜色安排，本轮仅踢黄球", "去蓝球旁边", [kick("yellow")]),
        ("这次只做俯身复原，结束以后没有别的任务", "点头然后前滚翻", [perform("ground_pick")]),
    )
    for p, (text, old, goals) in enumerate(cases):
        split = "val" if p >= 12 else "train"
        for i in range(3 if split == "val" else 16):
            req = base_context(text, i, state=("idle", "running", "paused")[i % 3])
            req["history"] = (req["history"] + history((old, "我已收到之前的请求。")))[-12:]
            builder.add(
                f"latest_correction_{p}",
                split,
                "latest_instruction",
                req,
                "replace",
                goals,
                rationale="明确更正、只做、新任务覆盖旧意图；旧历史中的颜色和动作不得污染本次清单。",
            )
    cases = (
        (
            "那就换成蓝球，点头照旧",
            "先踢绿球，然后点头",
            "目前不支持绿球，请改成红、黄或蓝球。",
            [kick("blue"), perform("nod")],
        ),
        (
            "颜色改成红，后面的前滚翻不变",
            "先踢紫球，再前滚翻",
            "紫球不受支持，请说明可用的目标颜色。",
            [kick("red"), perform("roll")],
        ),
        ("改成按三秒后退吧", "往后走十步", "没有按步计数的技能，可以改成按秒移动。", [move("backward", 3)]),
        ("那就左转九十度", "转半圈还是四分之一圈呢", "请明确转向和角度。", [turn("left", 90)]),
        ("选害羞舞，直接开始吧", "你会哪些舞", "当前支持节奏舞和害羞舞。", [perform("caught_dancing")]),
        ("就演示点头，现在做", "点头和张嘴都能做吗", "这两个动作都支持，请说要执行哪一个。", [perform("nod")]),
        (
            "我选黄球，后面仍然左右看",
            "先踢绿色球，再左右看",
            "绿色球尚不支持，请改成可识别的红黄蓝球。",
            [kick("yellow"), perform("look_around")],
        ),
        ("按照你说的，换成后退两秒", "后退五步", "当前不能按步数执行，可以指定秒数。", [move("backward", 2)]),
    )
    for p, (text, old, answer, goals) in enumerate(cases):
        split = "val" if p >= 6 else "train"
        for i in range(3 if split == "val" else 14):
            req = base_context(text, i)
            req["history"] = (req["history"] + history((old, answer)))[-12:]
            builder.add(
                f"clarification_resolution_{p}",
                split,
                "multi_turn_resolution",
                req,
                "replace",
                goals,
                rationale="只在上一轮明确未决任务范围内解析补充，修正指定项并保留明确要求保留的其他项。",
            )


def _references(builder):
    positive = (
        "好，踢它吧",
        "那就踢刚说的那个球",
        "去踢我们刚提到的球",
        "踢那个颜色的球就行",
        "请对刚才那个球踢一脚",
        "以刚谈到的那颗球为目标踢一次",
        "把我们刚明确的那个球踢一下",
    )
    for p, text in enumerate(positive):
        split = "val" if p >= 5 else "train"
        for i in range(3 if split == "val" else 12):
            color, name = (("red", "红"), ("yellow", "黄"), ("blue", "蓝"))[i % 3]
            req = base_context(text, i)
            req["history"] = (req["history"] + history((f"我们说的目标是{name}球", f"明白，目标是{name}球。")))[-12:]
            builder.add(
                f"reference_unique_{p}",
                split,
                "reference",
                req,
                "replace",
                [kick(color)],
                rationale="最新公共对话唯一、明确绑定的颜色可以用于指代；不绑定未观察到的 entity_id。",
            )
    ambiguous = (
        "踢另一个球",
        "把剩下的球踢了",
        "踢那边那个",
        "去那个球旁边",
        "把其他球都踢一下",
        "那个球你去踢吧",
        "朝另外那颗球踢一脚",
        "去找没说名字的那个球踢一下",
    )
    for p, text in enumerate(ambiguous):
        split = "val" if p >= 6 else "train"
        for i in range(3 if split == "val" else 10):
            req = base_context(text, i)
            req["history"] = history(("红球和蓝球都可以吗", "红球和蓝球都在可搜索目标范围内。"))
            builder.add(
                f"reference_ambiguous_{p}",
                split,
                "reference",
                req,
                "clarify",
                reply="这次目标还不明确。请说要红球、黄球还是蓝球；多个球请说明顺序。",
                rationale="多个候选或另一个没有唯一证据时澄清，不能把旧历史或场景物体推测成授权清单。",
            )
