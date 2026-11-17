"""State-dependent synthetic labels, constructed before their language variants."""

from .h0_language import COLORS, phrase
from .toy_fixtures import active, history, kick, move, observed_ball, perform, posture, turn


def context(state, index):
    mode = index % 4
    if mode == 1:
        state["history"] = history(("早上好，小鸭。", "早上好。"))
    elif mode == 2:
        state["history"] = history(("今天我们讨论名字。", "我的名字是小鸭。"))
    elif mode == 3:
        state["execution"] = active(move(seconds=8))
        state["observation"]["stationary"] = False
    return ("cold_start", "greeting", "unrelated_chat", "running")[mode]


def queue_case(state, goal, split, index):
    mode = index % 12
    word = phrase(goal, split)
    direction = ("forward", "backward", "left", "right")[(index // 12) % 4]
    state["execution"] = active(move(direction, seconds=8, continuous=mode in (3, 4, 7)), paused=mode == 4)
    state["observation"]["stationary"] = mode == 4
    train = (
        f"请保留现在的任务，在它后面追加{word}。",
        f"舍弃还没做完的安排，换成{word}。",
        "请立即取消这一轮任务，不再继续。",
        "暂时挂起正在进行的持续行走，保留它。",
        "把已经暂停的持续行走恢复起来。",
        "刚才那项任务已取消，现在我说继续。",
        "我想临时暂停当前这个有时限的动作。",
        f"一直走的任务照常继续，走完它再加{word}。",
        f"六项安排都别删，末尾另外增加{word}。",
        f"现有安排后再排一项：{word}。",
        "此刻只和我说说你的名字，原任务保持不变。",
        f"不用做之前的了，只做{word}。",
    )
    blind = (
        f"别动已有清单，做完以后接着{word}。",
        f"余下的都作废，新的任务是{word}。",
        "本次任务到此为止，请清空并停下。",
        "正在无限行走对吧？暂停它，但别取消。",
        "恢复被挂起的无限行走，让它接着走。",
        "已经作废的那次行走，再接着来吧。",
        "把这次定时行走先挂起等我恢复。",
        f"不终止无限行走，在其完成之后{word}。",
        f"保持原有六个目标，再排上{word}。",
        f"原清单做完后接上{word}，已有目标照旧。",
        "照原样执行任务就行，顺便告诉我你叫什么。",
        f"接下来只需{word}，此前任务不作数。",
    )
    decisions = (
        "append",
        "replace",
        "cancel",
        "pause",
        "resume",
        "clarify",
        "clarify",
        "clarify",
        "clarify",
        "append",
        "respond",
        "replace",
    )
    if mode == 5:
        state["execution"] = dict(
            state["execution"],
            state="idle",
            active=None,
            last_result=dict(
                goal=move(direction, continuous=True), status="cancelled", reason="user_cancelled", verified=False
            ),
        )
        state["observation"]["stationary"] = True
    elif mode in (8, 9):
        state["execution"]["pending"] = [perform("nod")] * (5 if mode == 8 else 4)
    decision = decisions[mode]
    return (
        (train if split == "train" else blind)[mode],
        decision,
        [goal] if decision in ("append", "replace") else [],
        mode,
    )


def boundary_case(state, goal, split, index):
    mode = index % 10
    word = phrase(goal, split)
    unsupported = ("后空翻", "搬运箱子", "把球叼起来", "倒立", "把球投进篮筐")[(index // 10) % 5]
    train = (
        f"不要{word}，这不是动作请求。",
        f"我要你{unsupported}，请实际执行。",
        f"先{word}，接着{unsupported}，两件都要完成。",
        "我没确定球的颜色，先别踢，问问我选哪种。",
        "要求七个独立目标：点头、张嘴、闭嘴、左右看、抬左脚、抬右脚、前滚翻。",
        f"先无期限向前移动，在这项任务结束后{word}。",
        f"如果在故事里让小鸭{word}会怎样？我现在只是讨论故事。",
        "我要求转身，但必须保持坐姿，不能站起来。",
        "往前走二十秒再停。",
        "去踢那个球，没有其他说明。",
    )
    blind = (
        f"{word}这件事本轮禁止做，听清就好。",
        f"现在就表演{unsupported}给我看。",
        f"安排为{word}之后再{unsupported}，不许只做一半。",
        "红球或黄球我还拿不定主意，踢球前向我确认。",
        "我要分七步做完：张嘴、点头、闭嘴、抬右脚、抬左脚、左右看、前滚翻。",
        f"保持向前走且永不停下，走完才{word}。",
        f"这是举例说明：机器人可能被要求{word}。我没有要求你照做。",
        "不许离开坐姿，却要向左旋转一圈。",
        "持续往后走满十五秒才停住。",
        "把那只球踢走，目标就用‘那只’代指。",
    )
    if mode == 7:
        state["observation"]["posture"] = "sitting"
    if mode == 9:
        state["observation"]["entities"] = [observed_ball("red"), observed_ball("blue", 1)]
    decision = (
        "respond",
        "decline",
        "decline",
        "clarify",
        "clarify",
        "clarify",
        "respond",
        "clarify",
        "clarify",
        "clarify",
    )[mode]
    return (train if split == "train" else blind)[mode], decision, [], mode


def state_case(state, split, index):
    mode = index % 12
    variant = index // 12
    color = tuple(COLORS)[variant % 3]
    name = COLORS[color]
    direction = ("left", "right")[variant % 2]
    angle = (30, 90, 180, 360)[(variant // 2) % 4]
    requested_turn = turn(direction, angle)
    turn_words = phrase(requested_turn, split)
    previous = perform(("nod", "open_beak", "close_beak", "look_around")[variant % 4], 1 + variant // 4)
    if mode in (3, 4, 11) and variant % 3:
        state["history"] = history(
            ("早上好。", "早上好。") if variant % 3 == 1 else ("我想聊聊名字。", "我的名字是小鸭。")
        )
    if mode == 0:
        state["observation"]["posture"] = "sitting"
    elif mode == 1:
        state["history"] = history((f"我只选{name}色的球。", "记住了你的选择。"))
    elif mode == 2:
        other = tuple(COLORS.values())[(variant + 1) % 3]
        state["history"] = history((f"{name}色与{other}色两只球都不错。", "我们还没有选定一只。"))
    elif mode == 3:
        state["execution"] = active(
            move(tuple(("forward", "backward", "left", "right"))[variant % 4], continuous=True), paused=True
        )
    elif mode == 4:
        state["execution"]["last_result"] = dict(
            goal=previous, status="succeeded", reason="verified_completed", verified=True
        )
    elif mode == 5:
        state["observation"]["age_s"] = 1.0
    elif mode == 6:
        state["observation"]["environment"]["motion_allowed"] = False
    elif mode == 7:
        state["observation"]["entities"] = []
    elif mode == 8:
        state["observation"]["entities"] = [observed_ball(color, 0), observed_ball(color, 1)]
    elif mode == 9:
        state["history"] = history((phrase(previous, split), "这项动作已确认完成。"))
        state["execution"]["last_result"] = dict(
            goal=previous, status="succeeded", reason="verified_completed", verified=True
        )
    train = (
        f"现在执行一次{turn_words}。",
        "踢我们刚选定颜色的球。",
        "踢前面讨论的那只球。",
        "接着做吧。",
        "接着做吧。",
        f"立刻{turn_words}。",
        f"立刻{turn_words}。",
        f"搜索并踢{name}色球。",
        f"请踢{name}色球。",
        "把刚刚成功做过的动作再做一次。",
        f"立刻{turn_words}。",
        "接着做吧。",
    )
    blind = (
        f"请{turn_words}，先完成必要的姿态准备。",
        "把我之前选的那种颜色的球踢一下。",
        "把上面说的那个球踢一下。",
        "继续未结束的任务。",
        "继续未结束的任务。",
        f"执行{turn_words}这项动作。",
        f"执行{turn_words}这项动作。",
        f"找一下{name}球，找到后踢它。",
        f"把{name}球踢开。",
        "重做上一项已确认完成的动作。",
        f"执行{turn_words}这项动作。",
        "继续未结束的任务。",
    )
    decisions = (
        "replace",
        "replace",
        "clarify",
        "resume",
        "clarify",
        "clarify",
        "clarify",
        "replace",
        "clarify",
        "replace",
        "replace",
        "clarify",
    )
    goals = (
        [posture("standing"), requested_turn],
        [kick(color)],
        [],
        [],
        [],
        [],
        [],
        [kick(color)],
        [],
        [previous],
        [requested_turn],
        [],
    )
    suffixes = (
        ("", "这就是我这一轮的要求。", "请按这层意思理解。")
        if split == "train"
        else ("", "这条要求现在生效。", "以上是本次的意图。")
    )
    text = (train if split == "train" else blind)[mode] + suffixes[(variant // 3) % 3]
    return text, decisions[mode], goals[mode], mode
