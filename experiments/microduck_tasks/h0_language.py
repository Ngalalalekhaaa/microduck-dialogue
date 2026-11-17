"""Independent wording books for synthetic training and sealed evaluation cases."""

import copy
import random

from .toy_fixtures import kick, move, navigate, perform, posture, turn

ACTION_WORDS = {
    "nod": ("点头", "做点头动作"),
    "look_around": ("左右看看", "向两侧张望"),
    "open_beak": ("张开嘴巴", "把嘴张开"),
    "close_beak": ("合上嘴巴", "把嘴合拢"),
    "body_sway": ("晃动身体", "做摇摆身体的动作"),
    "ground_pick": ("俯身触地再起身", "弯下身子后再抬起来"),
    "roll": ("向前滚翻", "做前滚翻动作"),
    "flamingo_left": ("抬左脚单脚站立", "抬起左脚保持单脚站姿"),
    "flamingo_right": ("抬右脚单脚站立", "抬起右脚保持单脚站姿"),
    "moonwalk": ("表演后退滑步", "做后滑舞步"),
    "groove": ("跳节奏舞", "表演节奏舞动作"),
    "caught_dancing": ("跳害羞舞", "表演害羞舞动作"),
    "sit_stand": ("坐下后立刻站起来", "完成坐下再站起的动作"),
}
COLORS = {"red": "红", "yellow": "黄", "blue": "蓝"}
DIRECTIONS = {"forward": "前", "backward": "后", "left": "左", "right": "右"}
TEMPLATES = {
    "train": (
        "现在开始做：{steps}。",
        "我要看这些动作，顺序如下：{steps}。",
        "收到这句话就执行以下安排：{steps}。",
        "请完成我列出的任务：{steps}。",
        "这是一条动作指令：{steps}。",
    ),
    "blind": (
        "接下来按这份清单行动——{steps}。",
        "照我说的次序来一遍：{steps}。",
        "我下达的执行安排为：{steps}。",
        "下面这些事请实际做出来：{steps}。",
    ),
}


def inventory():
    goals = [perform(name, repeat) for name in ACTION_WORDS for repeat in (1, 2, 3)]
    goals += [move(direction, seconds) for direction in DIRECTIONS for seconds in (0.6, 8)]
    goals += [turn(direction, angle) for direction in ("left", "right") for angle in (30, 90, 180, 360)]
    goals += [fn(color) for fn in (kick, navigate) for color in COLORS]
    goals += [posture("sitting"), posture("standing")]
    goals += [move(direction, continuous=True) for direction in DIRECTIONS]
    return goals


def phrase(goal, split):
    word = 0 if split == "train" else 1
    kind = goal["type"]
    if kind == "perform":
        return f"{ACTION_WORDS[goal['action']][word]}{goal['repeat']}次"
    if kind == "posture":
        return {"sitting": ("坐下并保持坐姿", "坐下后不要站起"), "standing": ("站起来", "调整为站姿")}[goal["pose"]][
            word
        ]
    if kind == "turn":
        direction = DIRECTIONS[goal["direction"]]
        angle = goal["angle_deg"]
        return f"向{direction}转{angle:g}度" if word == 0 else f"往{direction}旋转{angle:g}度"
    if kind == "move":
        direction = DIRECTIONS[goal["direction"]]
        if goal["limit"]["kind"] == "until_cancelled":
            return f"一直向{direction}走，直到我叫停" if word == 0 else f"持续往{direction}移动，等取消指令再停"
        seconds = goal["limit"]["seconds"]
        return f"向{direction}移动{seconds:g}秒" if word == 0 else f"往{direction}走{seconds:g}秒后停住"
    color = COLORS[goal["target"]["attributes"][0]["value"]]
    if kind == "kick":
        return f"踢{color}色的球" if word == 0 else f"把{color}球踢一下"
    return f"靠近{color}球但不踢" if word == 0 else f"走到{color}色球旁停住，不踢球"


def action_request(goals, split, index):
    words = [phrase(g, split) for g in goals]
    if len(goals) == 1 and index % 7 == 6:
        distractor = next(g for g in inventory()[index % 30 :] if g != goals[0])
        previous = phrase(distractor, split)
        text = (
            f"原先说要{previous}，现在改为{words[0]}，只执行修改后的动作。"
            if split == "train"
            else f"更正刚才的要求：不要{previous}了，这次只需{words[0]}。"
        )
        return text, f"{split}:correction"
    if len(goals) == 2 and index % 5 in (0, 1):
        if split == "train":
            text = f"在{words[1]}之前，先{words[0]}。" if index % 5 == 0 else f"{words[0]}之后，再{words[1]}。"
        else:
            text = f"先完成{words[0]}，做完才{words[1]}。" if index % 5 == 0 else f"{words[1]}要放在{words[0]}后面。"
        return text, f"{split}:order:{index % 5}"
    number = index % len(TEMPLATES[split])
    steps = "；然后".join(words) if split == "train" else "，随后".join(words)
    return TEMPLATES[split][number].format(steps=steps), f"{split}:list:{number}"


def random_plans(count, length, seed):
    rng = random.Random(seed)
    atoms = inventory()
    ordinary = [g for g in atoms if g["type"] != "posture" and g.get("limit", {}).get("kind") != "until_cancelled"]
    seen, result = set(), []
    while len(result) < count:
        goals = copy.deepcopy([rng.choice(ordinary) for _ in range(length)])
        index = len(result)
        if index % 11 == 0:
            goals[-1] = posture("sitting")
        elif index % 11 == 1:
            goals[-1] = move(rng.choice(list(DIRECTIONS)), continuous=True)
        elif length >= 3 and index % 11 == 2:
            goals[:2] = [posture("sitting"), posture("standing")]
        if any(
            a["type"] == "perform" and b.get("action") == a["action"] for a, b in zip(goals, goals[1:], strict=False)
        ):
            continue
        key = repr(goals)
        if key in seen:
            continue
        seen.add(key)
        result.append(goals)
    return result
