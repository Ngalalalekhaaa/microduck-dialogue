"""Finite, independently labelled language curriculum for real simulator collection.

These are episode requests, never recorded observations or training examples.
The collector must run each request, capture current cameras and verify results.
All paraphrases of one semantic family share their dataset split group.
"""

import copy

SELF_PHRASES = {
    "nod": ("点头", "点个头", "点点头", "点一下头", "点头示意", "点头打个招呼"),
    "look_around": ("左右看", "环顾四周", "看看左右", "向两边看看", "转头环视", "左右转头"),
    "sit_stand": ("做个坐站动作", "坐站循环", "坐站练习", "坐站演示", "做一下坐站动作", "坐站"),
    "ground_pick": ("俯身复原", "俯身动作", "低头触地动作", "俯身触地复原", "俯身演示", "做一下俯身动作"),
    "roll": ("前滚翻", "翻滚", "向前翻滚", "前翻滚", "向前滚翻", "往前滚翻"),
    "flamingo_left": ("抬左脚", "抬起左脚", "左脚抬起来", "把左脚抬起", "抬高左脚", "用右脚单脚站立"),
    "flamingo_right": ("抬右脚", "抬起右脚", "右脚抬起来", "把右脚抬起", "抬高右脚", "用左脚单脚站立"),
    "moonwalk": ("后退滑步", "月球步", "月球漫步", "后退滑步动作", "滑个太空步", "太空步"),
    "body_sway": ("轻晃身体", "左右摇摆", "晃晃身体", "摇摇身体", "轻轻晃动身体", "身体左右晃动"),
    "open_beak": ("张嘴", "张开嘴", "张开鸭嘴", "把嘴张开", "嘴巴张开", "张大嘴巴"),
    "close_beak": ("闭嘴", "合上嘴", "合上鸭嘴", "把嘴合上", "嘴巴合上", "闭上嘴巴"),
    "groove": ("节奏舞步", "跳舞", "跳个舞", "跳一段舞", "来段节奏舞", "跳节奏舞"),
    "caught_dancing": ("害羞舞步", "跳个害羞舞", "跳害羞舞", "害羞舞", "跳一段害羞舞", "表演害羞舞步"),
}
COLOR_NAMES = {"red": "红", "yellow": "黄", "blue": "蓝"}
GREETINGS = ("你好", "hello", "嗨", "Hi", "哈喽", "小鸭你好")
VISUAL = ("你现在能看到什么", "描述一下画面", "现在看见什么了", "说说你的视野", "眼前有什么", "你面前有什么")
THANKS = ("谢谢", "辛苦了", "多谢", "感谢你", "做得好", "Thanks")
CANCEL = ("停下", "立即停止", "取消当前任务", "请停住", "别再走了", "停止所有任务")
PAUSE = ("暂停", "请暂停一下", "先暂停", "暂停任务", "请暂停任务", "先暂停一下")
RESUME = ("继续", "恢复", "请继续执行", "恢复任务", "继续吧", "请恢复任务")


def _perform(action, repeat=1):
    return {"type": "perform", "action": action, "repeat": repeat}


def _target(color):
    return {"category": "ball", "attributes": [{"name": "color", "value": color}], "entity_id": None}


def _object_task(kind, color):
    result = {"type": kind, "target": _target(color)}
    if kind == "navigate":
        result["standoff_m"] = 0.25
    return result


def _move(direction, seconds=0.6, *, continuous=False):
    return {
        "type": "move",
        "direction": direction,
        "limit": {"kind": "until_cancelled"} if continuous else {"kind": "duration", "seconds": seconds},
    }


def _turn(direction="left", angle=90):
    return {"type": "turn", "direction": direction, "angle_deg": angle}


def _posture(pose):
    return {"type": "posture", "pose": pose}


def _expect(decision="respond", goals=()):
    return {"decision": decision, "goals": copy.deepcopy(list(goals))}


def _phrase(goal, variant):
    index = variant % 6
    kind = goal["type"]
    if kind == "perform":
        phrase = SELF_PHRASES[goal["action"]][index]
        return phrase + (f"{goal['repeat']}次" if goal["repeat"] > 1 else "")
    if kind == "posture":
        phrases = (
            ("坐下", "坐好", "保持坐姿", "蹲下", "蹲好", "保持蹲姿")
            if goal["pose"] == "sitting"
            else ("站起来", "起立", "站好", "站立", "站起", "请站起来")
        )
        return phrases[index]
    if kind in ("kick", "navigate"):
        color = COLOR_NAMES[goal["target"]["attributes"][0]["value"]]
        phrases = (
            (
                f"踢{color}球",
                f"把{color}球踢走",
                f"踢一下{color}色球",
                f"把{color}色的球踢开",
                f"踢{color}球一脚",
                f"踢一脚{color}色小球",
            )
            if kind == "kick"
            else (
                f"走到{color}球旁边",
                f"靠近{color}球",
                f"去到{color}色球附近",
                f"移动到{color}球跟前",
                f"接近{color}色的球",
                f"走到{color}球边上",
            )
        )
        return phrases[index]
    if kind == "turn":
        side = "左" if goal["direction"] == "left" else "右"
        angle = goal["angle_deg"]
        amount = "一圈" if angle == 360 else "半圈" if angle == 180 else f"{angle}度"
        return (
            f"向{side}转{angle}度",
            f"往{side}转{amount}",
            f"朝{side}转动{angle}度",
            f"转向{side}{amount}",
            f"原地向{side}边转{angle}度",
            f"原地往{side}转{amount}",
        )[index]
    stems = {
        "forward": ("前进", "向前走", "往前走", "朝前走", "向前", "往前"),
        "backward": ("后退", "退后", "向后走", "往后走", "朝后走", "向后"),
        "left": ("左移", "向左横移", "往左移动", "朝左边走", "向左侧挪", "往左走"),
        "right": ("右移", "向右横移", "往右移动", "朝右边走", "向右侧挪", "往右走"),
    }
    stem = stems[goal["direction"]][index]
    limit = goal["limit"]
    if limit["kind"] == "until_cancelled":
        return (f"一直{stem}", f"{stem}不要停", f"持续{stem}", f"不停{stem}", f"一直{stem}别停", f"{stem}不停")[index]
    return stem + f"{limit['seconds']:g}秒"


def _plan_text(goals, variant):
    clauses = [_phrase(goal, variant + index) for index, goal in enumerate(goals)]
    if len(clauses) == 1:
        return clauses[0]
    style = variant % 6
    if style == 0:
        return "先" + "，再".join(clauses)
    if style == 1:
        return "，然后".join(clauses)
    if style == 2:
        return "请先" + "，随后".join(clauses[:-1]) + "，最后" + clauses[-1]
    if style == 3:
        return "麻烦" + "；接着".join(clauses)
    if style == 4:
        return "能不能帮我先" + "，再".join(clauses) + "？"
    return "帮我依次" + "，之后".join(clauses[:-1]) + "，最后" + clauses[-1]


def _family(name, *, goals=None, utterances=None, decision="respond", kind="ordinary", metadata=None):
    return {
        "name": name,
        "goals": goals,
        "utterances": utterances,
        "decision": decision,
        "kind": kind,
        "metadata": metadata or {},
    }


def families():
    """Return the finite semantic families; no teacher is used to create labels."""
    result = [_family(f"self_{action}", goals=[_perform(action)]) for action in SELF_PHRASES]
    result += [
        _family(f"repeat_{action}_{count}", goals=[_perform(action, count)])
        for action, count in (("nod", 2), ("look_around", 2), ("body_sway", 3))
    ]
    result += [
        _family(f"move_{direction}", goals=[_move(direction)]) for direction in ("forward", "backward", "left", "right")
    ]
    result += [
        _family(f"turn_{direction}_{angle}", goals=[_turn(direction, angle)])
        for direction in ("left", "right")
        for angle in (30, 90, 180)
    ]
    result += [_family(f"posture_{pose}", goals=[_posture(pose)]) for pose in ("sitting", "standing")]
    result += [
        _family(f"{kind}_{color}", goals=[_object_task(kind, color)])
        for kind in ("kick", "navigate")
        for color in COLOR_NAMES
    ]
    actions = list(SELF_PHRASES)
    for index, action in enumerate(actions):
        result.append(
            _family(
                f"pair_{action}_{actions[(index + 1) % len(actions)]}",
                goals=[_perform(action), _perform(actions[(index + 1) % len(actions)])],
            )
        )
    triples = (
        ("nod", "open_beak", "close_beak"),
        ("flamingo_left", "flamingo_right", "look_around"),
        ("groove", "caught_dancing", "nod"),
        ("ground_pick", "look_around", "body_sway"),
        ("moonwalk", "nod", "body_sway"),
        ("roll", "nod", "look_around"),
    )
    for index, sequence in enumerate(triples):
        result.append(_family(f"triple_{index}", goals=[_perform(action) for action in sequence]))
    combinations = (
        [_perform("nod"), _posture("sitting"), _posture("standing"), _perform("look_around")],
        [_move("forward"), _turn("right", 90), _move("backward"), _perform("nod")],
        [_perform("open_beak"), _perform("close_beak"), _turn("left", 30), _perform("body_sway")],
        [_perform("flamingo_left"), _perform("flamingo_right"), _perform("groove"), _perform("caught_dancing")],
        [
            _perform("nod"),
            _perform("look_around"),
            _perform("open_beak"),
            _perform("close_beak"),
            _perform("body_sway"),
        ],
        [_move("left"), _turn("right", 30), _move("right"), _posture("sitting"), _posture("standing")],
        [
            _perform("nod"),
            _perform("open_beak"),
            _perform("close_beak"),
            _posture("sitting"),
            _posture("standing"),
            _perform("look_around"),
        ],
        [
            _perform("flamingo_left"),
            _perform("flamingo_right"),
            _perform("body_sway"),
            _perform("ground_pick"),
            _perform("groove"),
            _perform("nod"),
        ],
    )
    for index, goals in enumerate(combinations):
        result.append(_family(f"mixed_{len(goals)}_{index}", goals=goals))
    object_combinations = (
        [_object_task("kick", "blue"), _object_task("kick", "red"), _turn("left", 360)],
        [_object_task("navigate", "yellow"), _perform("nod"), _object_task("kick", "yellow")],
        [_object_task("kick", "red"), _perform("look_around"), _object_task("kick", "blue")],
        [_object_task("navigate", "blue"), _turn("right", 90), _object_task("navigate", "red")],
        [_perform("nod"), _object_task("navigate", "red"), _object_task("kick", "red"), _perform("groove")],
        [_object_task("kick", "yellow"), _posture("sitting"), _posture("standing"), _perform("nod")],
    )
    for index, goals in enumerate(object_combinations):
        result.append(_family(f"object_sequence_{index}", goals=goals))
    chats = {
        "greeting": GREETINGS,
        "identity": ("你叫什么", "你的名字是什么", "你是谁", "你好，你叫什么", "说说你的名字", "告诉我你是谁"),
        "capabilities": ("你都会做什么", "有哪些动作", "有什么本领", "你能做什么", "你有什么技能", "支持哪些动作"),
        "visual_scene": VISUAL,
        "visual_count": (
            "你看到几个球",
            "有几个球",
            "画面里有多少个球",
            "眼前有多少球",
            "现在看见几个球",
            "视野里有几个球",
        ),
        "visual_nearest": (
            "哪个球离你最近",
            "你看到哪个球最近",
            "画面里哪个球离你近",
            "你眼前哪个球最近",
            "视野里哪一个球最近",
            "你看见的哪个球最近",
        ),
        "gratitude": THANKS,
        "farewell": ("再见", "拜拜", "下次见", "回头见", "bye", "Goodbye"),
        "discuss_actions": (
            "如果我说踢红球呢",
            "昨天你点头了",
            "刚才你坐下了",
            "‘停下’是什么意思",
            "假如让你转圈呢",
            "上次你踢了蓝球",
        ),
    }
    result += [_family(f"chat_{name}", utterances=phrases) for name, phrases in chats.items()]
    seven_actions = ("nod", "look_around", "open_beak", "close_beak", "body_sway", "groove", "caught_dancing")
    result.append(
        _family(
            "sequence_seven_supported",
            goals=[_perform(action) for action in seven_actions],
            utterances=tuple("先" + "，再".join(SELF_PHRASES[action][i] for action in seven_actions) for i in range(6)),
        )
    )
    negatives = {
        "unsupported": ("后空翻", "请倒立", "做一下跳跃", "帮我跑步", "飞起来", "叼起红球"),
        "unsupported_tail": (
            "先点头再后空翻",
            "先坐下再倒立",
            "先踢蓝球再叼起红球",
            "先张嘴再抓取",
            "先转一圈再投篮",
            "先点点头再搬运",
        ),
        "unbounded_tail": (
            "先一直前进再点头",
            "先持续后退再张嘴",
            "先左移不停再踢蓝球",
            "先向右走不要停再坐下",
            "先一直往前走再站起来",
            "先持续向后走再转半圈",
        ),
        "range": ("向左转720度", "向右转10度", "前进20秒", "后退0.01秒", "向左走15秒", "向右走0.01秒"),
    }
    result += [
        _family(f"decline_{name}", utterances=phrases, decision="decline") for name, phrases in negatives.items()
    ]
    unclear = {
        "target": ("踢球", "踢一下球", "走到球旁边", "把球踢开", "靠近球", "移动到球附近"),
        "negation": ("不要点头", "别踢红球", "不许转圈", "禁止前进", "不要张嘴", "别坐下"),
        "choice": ("踢红球还是蓝球", "点头或者坐下", "左转还是右转", "前进或者后退", "张嘴还是闭嘴", "转一圈或者踢球"),
        "question": ("踢红球吗", "前滚翻吗", "转一圈么", "点头吗", "坐下吗", "向前走吗"),
        "unknown_units": ("向前走一米", "向左转八圈", "后退三步", "向右走半米", "转十五圈", "踢绿色球"),
    }
    result += [_family(f"clarify_{name}", utterances=phrases, decision="clarify") for name, phrases in unclear.items()]
    for color in COLOR_NAMES:
        result.append(
            _family(
                f"reference_{color}", kind="reference", goals=[_object_task("kick", color)], metadata={"color": color}
            )
        )
    for direction in ("forward", "backward", "left", "right"):
        result.append(
            _family(f"continuous_{direction}", kind="continuous_control", goals=[_move(direction, continuous=True)])
        )
    for direction in ("forward", "backward"):
        result.append(
            _family(
                f"append_cancel_{direction}",
                kind="append_cancel",
                goals=[_move(direction, 8)],
                metadata={"append_goal": _perform("nod")},
            )
        )
    for kind, color in (("navigate", "yellow"), ("kick", "red")):
        result.append(
            _family(
                f"missing_{kind}_{color}",
                kind="missing_target",
                goals=[_object_task(kind, color), _perform("nod")],
                metadata={"missing_color": color},
            )
        )
    return result


def build_curriculum(repeats=6, base_seed=210000):
    """Build 4–6 reviewed paraphrases per family, with independent expected goals."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 4 <= repeats <= 6:
        raise ValueError("Choose 4, 5 or 6 reviewed paraphrases per family; unreviewed permutations are not generated.")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer.")
    episodes = []
    for family in families():
        for variant in range(repeats):
            goals, kind = family["goals"], family["kind"]
            if kind == "continuous_control":
                commands = [
                    _plan_text(goals, variant),
                    VISUAL[variant],
                    PAUSE[variant],
                    GREETINGS[variant],
                    RESUME[variant],
                    CANCEL[variant],
                ]
                expected = [
                    _expect("replace", goals),
                    _expect(),
                    _expect("pause"),
                    _expect(),
                    _expect("resume"),
                    _expect("cancel"),
                ]
            elif kind == "append_cancel":
                addition = family["metadata"]["append_goal"]
                commands = [
                    _plan_text(goals, variant),
                    "然后" + _phrase(addition, variant),
                    VISUAL[variant],
                    CANCEL[variant],
                ]
                expected = [_expect("replace", goals), _expect("append", [addition]), _expect(), _expect("cancel")]
            elif kind == "reference":
                name = COLOR_NAMES[family["metadata"]["color"]]
                refer = ("踢它", "把它踢走", "踢那个球", "把那个球踢开", "踢这个球", "把这个球踢一脚")
                questions = (
                    f"看得到{name}球吗",
                    f"你看到{name}球了吗",
                    f"{name}球在你哪边",
                    f"画面里有{name}球吗",
                    f"能看见{name}色球吗",
                    f"现在看到了{name}球吗",
                )
                commands = [questions[variant], refer[variant], THANKS[variant]]
                expected = [_expect(), _expect("replace", goals), _expect()]
                kind = "ordinary"
            elif goals is not None:
                phrase = (
                    family["utterances"][variant] if family["utterances"] is not None else _plan_text(goals, variant)
                )
                commands = [GREETINGS[variant], phrase, VISUAL[variant]]
                expected = [_expect(), _expect("replace", goals), _expect()]
            else:
                commands = [family["utterances"][variant], THANKS[variant]]
                expected = [_expect(family["decision"]), _expect()]
            episodes.append(
                {
                    "episode_id": f"{family['name']}_p{variant:02d}",
                    "seed": base_seed + len(episodes),
                    "split_group": "toy_curriculum_v1:" + family["name"],
                    "kind": kind,
                    "commands": commands,
                    "expected": expected,
                    "metadata": {
                        "family": family["name"],
                        "paraphrase_index": variant,
                        "source": "reviewed_episode_request",
                        "training_ready": False,
                        **copy.deepcopy(family["metadata"]),
                    },
                }
            )
    return episodes
