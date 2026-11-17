"""Independent agent-authored synthetic development probes; never include these rows in training."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from .toy_fixtures import active, history, kick, move, navigate, observed_ball, perform, posture, request, turn
from .toy_inputs import validate_input
from .toy_protocol import VERSION, validate_output

VERSION_SUITE = "q1r-development-20261116-v1"


def cases():
    rows = []

    def add(group, family, text, decision="replace", goals=(), reply=None, rationale=None, **changes):
        state = request(text, planned=True)
        for name, value in changes.items():
            if name in ("history", "execution"):
                state[name] = copy.deepcopy(value)
            elif name in state["observation"]:
                state["observation"][name] = copy.deepcopy(value)
            elif name in state["capabilities"]:
                state["capabilities"][name] = copy.deepcopy(value)
            else:
                raise KeyError(name)
        reply = (
            reply
            or {
                "replace": "收到，我会按你说的动作和顺序执行。",
                "append": "好，当前任务保留，把这些动作排在后面。",
                "respond": "我在，你可以继续和我说话。",
                "clarify": "请确认具体目标或改成当前支持的动作，我先保留原任务。",
                "decline": "现在的观察或安全条件不允许开始这个动作，请先处理当前状态。",
                "cancel": "收到，停止并清空剩余任务。",
                "pause": "收到，到安全动作边界站稳后暂停，保留剩余任务。",
                "resume": "好，从尚未完成的任务继续。",
            }[decision]
        )
        expected = dict(decision=decision, goals=list(copy.deepcopy(goals)), reply=reply)
        validate_input(state)
        validate_output(dict(schema_version=VERSION, **expected), state)
        rows.append(
            dict(
                id=f"dev_{len(rows):04d}",
                group=group,
                family_id=f"dev/{family}",
                request=state,
                expected=expected,
                rationale=rationale or f"{family}：只依据当前明确请求和可信能力生成计划。",
                reply_expectations=dict(
                    nonempty_chinese=True,
                    no_unverified_completion=True,
                    semantic_reference=reply,
                    exact_match_required=False,
                ),
                source="agent_authored_synthetic_semantic",
                split="development_only",
                training_allowed=False,
                human_reviewed=False,
                physical_executed=False,
            )
        )

    actions = {
        "nod": ("给我点个头嘛", "小鸭，连续点头两次", "帮忙点三下头", "点头致意一下"),
        "look_around": ("左右看看", "脑袋往两边看一看", "来两遍左右看", "小家伙，环顾一下左右"),
        "sit_stand": ("坐下接着自己起来", "演示一次坐下再站起", "坐下再起来两遍", "做个坐站动作吧"),
        "ground_pick": ("俯身再起来", "做一次低头触地再起身的动作", "俯身再起做两回", "弯下身子再复原"),
        "roll": ("给我翻个前滚翻", "麻烦前滚翻两遍", "往前翻滚一次", "做一个前滚翻吧小鸭"),
        "flamingo_left": ("左脚抬一下", "用右脚站着，抬起左脚", "抬左脚两回", "表演左脚抬起的动作"),
        "flamingo_right": ("抬一下右脚", "用左脚站着，把右脚抬起来", "抬右脚两次", "做个抬右脚动作"),
        "moonwalk": ("来一段月球步", "给我表演后退滑步", "后退滑步两遍", "走个太空步瞧瞧"),
        "body_sway": ("轻轻晃一下身体", "身体左右晃一晃", "轻晃身体两遍", "摇摆一下身子"),
        "open_beak": ("嘴巴张开", "张一下你的小嘴", "做两次张嘴动作", "把嘴张开吧"),
        "close_beak": ("嘴巴闭上", "把张开的嘴合起来", "闭嘴动作做两次", "合拢嘴巴"),
        "groove": ("表演节奏舞步", "来一段律动舞", "节奏舞跳两遍", "做那个有节奏的舞步"),
        "caught_dancing": ("来段害羞舞步", "表演被发现偷跳舞的动作", "害羞舞步跳两遍", "做一次害羞跳舞"),
    }
    for action, texts in actions.items():
        for i, text in enumerate(texts):
            count = 2 if any(word in text for word in ("两次", "两遍", "两回")) else 3 if "三下" in text else 1
            add("self_actions", f"self_{action}_{i}", text, goals=[perform(action, count)])

    moving = {
        "forward": ("向前走", "前进", "往前挪", "向前移动"),
        "backward": ("后退", "向后走", "往后退", "退后"),
        "left": ("向左横移", "左移", "往左挪", "向左走"),
        "right": ("向右横移", "右移", "往右挪", "向右走"),
    }
    for direction, aliases in moving.items():
        for i, unit in enumerate(("3秒", "3s", " 3 s", "三秒", "半秒", "0.5秒", "两秒", "0.12秒")):
            seconds = 0.5 if i in (4, 5) else 2 if i == 6 else 0.12 if i == 7 else 3
            add("move_units", f"move_{direction}_{i}", aliases[i % 4] + unit, goals=[move(direction, seconds)])
        add(
            "continuous",
            f"continuous_{direction}",
            f"一直{aliases[0]}，直到我喊停",
            goals=[move(direction, continuous=True)],
        )
    for text, direction, angle in (
        ("朝右转90°", "right", 90),
        ("左转九十度", "left", 90),
        ("向右转半圈", "right", 180),
        ("左边转一整圈", "left", 360),
        ("逆时针转180度", "left", 180),
        ("顺时针转45度", "right", 45),
        ("向右转动15度", "right", 15),
        ("朝左转三十度", "left", 30),
        ("原地转身一圈", "left", 360),
        ("向左转", "left", 30),
        ("转右边30度", "right", 30),
        ("左转270度", "left", 270),
    ):
        add("turn", f"turn_{direction}_{angle}_{len(rows)}", text, goals=[turn(direction, angle)])
    for text in ("坐下别起来", "坐着保持这个姿势", "请坐好", "蹲下保持"):
        add("posture", "sit_hold_" + text, text, goals=[posture("sitting")])
    for text in ("站起来", "起立吧", "别坐着了，站好", "从坐姿站起来"):
        add("posture", "stand_" + text, text, goals=[posture("standing")], posture="sitting")
    for text, goals in (
        ("站起来再踢红球", [posture("standing"), kick("red")]),
        ("先起立，然后往后走2秒", [posture("standing"), move("backward", 2)]),
        ("坐下，站起来，再点头", [posture("sitting"), posture("standing"), perform("nod")]),
        ("站起来然后左转90度", [posture("standing"), turn("left", 90)]),
    ):
        add("posture", "transition_" + text, text, goals=goals, posture="sitting")

    for color, cn in (("red", "红"), ("yellow", "黄"), ("blue", "蓝")):
        for i, text in enumerate((f"把{cn}色小球踢走", f"对{cn}球来一脚", f"踢一下{cn}色的球吧", f"请你去踢{cn}球")):
            add("target", f"kick_{color}_{i}", text, goals=[kick(color)])
        for i, text in enumerate(
            (f"靠近{cn}球，到了别踢", f"走到{cn}色球旁边停住", f"去{cn}球边上", f"到{cn}球附近就行")
        ):
            add("target", f"navigate_{color}_{i}", text, goals=[navigate(color)])

    unsupported = (
        "把紫色方块踢走",
        "踢绿球",
        "帮我踢那个红方块",
        "叼起红球",
        "把黄球搬到我这里",
        "推一下蓝色方块",
        "做个后空翻",
        "跳上桌子",
        "跑步冲过去",
        "倒立给我看",
        "飞起来",
        "转个芭蕾舞圈",
        "先踢蓝球再踢紫色方块",
        "点头，然后抓住那只杯子",
        "先前滚翻再踢绿球",
        "踢红球，然后把它叼回来",
        "后退10步",
        "前进一米",
        "左边移动30厘米",
        "右转10度",
        "前进11秒",
        "朝右转720度",
        "同时点头和前滚翻",
        "每隔一小时自动跳舞",
        "跟着我走",
        "用左脚把球踢进球门",
    )
    for i, text in enumerate(unsupported):
        add(
            "unsupported",
            f"unsupported_{i}",
            text,
            "clarify",
            reply="这个动作或参数目前还不支持，请换成已有技能，例如点头、前滚翻或踢红黄蓝球。",
            rationale="本轮用户明确要求库外技能澄清，不执行部分任务，也不擅自换成近似动作。",
        )

    multi = (
        ("先张嘴再闭嘴", [perform("open_beak"), perform("close_beak")]),
        ("点头，左转90度，再后退2秒", [perform("nod"), turn("left", 90), move("backward", 2)]),
        ("先踢蓝球，然后踢黄球，最后前滚翻", [kick("blue"), kick("yellow"), perform("roll")]),
        ("先做前滚翻，再踢蓝球，再踢红球", [perform("roll"), kick("blue"), kick("red")]),
        ("往右走2秒，然后向右转90度", [move("right", 2), turn("right", 90)]),
        ("先转左边30度，再左移半秒", [turn("left", 30), move("left", 0.5)]),
        ("坐下待一会儿", [posture("sitting")]),
        ("靠近红球然后点头两次", [navigate("red"), perform("nod", 2)]),
        ("点头三次再抬右脚", [perform("nod", 3), perform("flamingo_right")]),
        ("左右看看，然后一直往前走", [perform("look_around"), move("forward", continuous=True)]),
        ("跳害羞舞，再跳节奏舞", [perform("caught_dancing"), perform("groove")]),
        ("先抬右脚再抬左脚", [perform("flamingo_right"), perform("flamingo_left")]),
        ("俯身再起，然后月球步", [perform("ground_pick"), perform("moonwalk")]),
        ("先踢红球再靠近黄球", [kick("red"), navigate("yellow")]),
        ("连续点头3次，然后坐下保持", [perform("nod", 3), posture("sitting")]),
        ("前进0.5秒、后退0.5秒、左移0.5秒、右移0.5秒", [move(d, 0.5) for d in moving]),
        ("点个头吧，我今天心情不太好", [perform("nod")]),
        ("你先踢蓝球，顺便说声收到", [kick("blue")]),
        ("你好小鸭，帮我往后退三秒呗", [move("backward", 3)]),
        ("刚才说错了，这回只踢红球", [kick("red")]),
    )
    for i, (text, goals) in enumerate(multi):
        add("composition", f"composition_{i}", text, goals=goals)

    dirty = history(
        ("你好", "你好。"),
        ("后退10步", "请改成秒数。"),
        ("后退3s", "请再说具体些。"),
        ("踢绿球和紫方块", "当前只能踢红黄蓝球。"),
        ("那踢蓝球和黄球", "好的。"),
        ("顺便踢紫球", "紫球不支持，请换个目标。"),
    )
    for i, (text, goals) in enumerate(
        (
            ("后退3秒", [move("backward", 3)]),
            ("后退3s", [move("backward", 3)]),
            ("只踢蓝色球，然后踢黄球，做一个前滚翻", [kick("blue"), kick("yellow"), perform("roll")]),
            ("现在只点头", [perform("nod")]),
            ("算了，改成踢红球", [kick("red")]),
            ("前面那些都不要，只向右走1秒", [move("right", 1)]),
            ("我说的是前滚翻，不是后空翻，现在做一次", [perform("roll")]),
            ("别踢绿球了，踢蓝球", [kick("blue")]),
        )
    ):
        add("history_correction", f"dirty_repair_{i}", text, goals=goals, history=dirty)
    for color, cn in (("red", "红"), ("yellow", "黄"), ("blue", "蓝")):
        prior = history((f"先靠近{cn}球", f"好的，我会走到{cn}球旁边。"))
        add("reference", f"it_{color}", "现在踢它", goals=[kick(color)], history=prior)
        add(
            "reference",
            f"repeat_{color}",
            "刚才那一脚再来一次",
            goals=[kick(color)],
            history=history((f"踢{cn}球", f"踢{cn}球已通过执行验收。")),
        )
        add("reference", f"only_visible_{color}", "踢这个球", goals=[kick(color)], entities=[observed_ball(color)])
    for i, text in enumerate(("踢另一个球", "把它踢走", "再来一遍", "踢那个", "另外的也踢了")):
        add(
            "reference",
            f"ambiguous_{i}",
            text,
            "clarify",
            entities=[observed_ball("red"), observed_ball("blue")],
            reply="你指的是哪一个球或动作？请说清颜色或动作名。",
        )

    for i, (text, decision, goals) in enumerate(
        (
            ("做完这个，再点头", "append", [perform("nod")]),
            ("后面再加一个前滚翻", "append", [perform("roll")]),
            ("原来的保留，再踢红球", "append", [kick("red")]),
            ("把后续全部换成坐下", "replace", [posture("sitting")]),
            ("不用踢了，改为左右看", "replace", [perform("look_around")]),
            ("从现在开始只后退2秒", "replace", [move("backward", 2)]),
            ("现在忙什么呢", "respond", []),
            ("你边做边回我一声好吗", "respond", []),
            ("继续干活，跟我说说你是谁", "respond", []),
            ("我没让你停，接着忙", "respond", []),
        )
    ):
        add(
            "active_dialogue",
            f"active_{i}",
            text,
            decision,
            goals,
            execution=active(kick("blue"), pending=[perform("roll")]),
            stationary=False,
        )
    for count in (6, 7, 14, 15, 16):
        text = "按顺序做这些动作：" + "，".join("点头" if i % 2 == 0 else "左右看" for i in range(count))
        goals = [perform("nod" if i % 2 == 0 else "look_around") for i in range(count)]
        add(
            "capacity",
            f"new_queue_{count}",
            text,
            "replace" if count <= 15 else "clarify",
            goals if count <= 15 else [],
        )
    for existing, extra in ((13, 2), (14, 1), (14, 2), (15, 1)):
        execution = active(perform("nod"), pending=[perform("look_around")] * (existing - 1))
        goals = [perform("open_beak"), perform("close_beak")][:extra]
        text = "原任务后追加张嘴" + ("再闭嘴" if extra == 2 else "")
        decision = "append" if existing + extra <= 15 else "clarify"
        add(
            "capacity",
            f"append_{existing}_{extra}",
            text,
            decision,
            goals if decision == "append" else [],
            execution=execution,
        )
    add(
        "capacity",
        "replace_full",
        "原来的都换掉，现在只踢红球",
        goals=[kick("red")],
        execution=active(perform("nod"), pending=[perform("look_around")] * 14),
    )
    add("capacity", "behind_continuous", "在这之后再加一次点头", "clarify", execution=active(move(continuous=True)))
    add("capacity", "continuous_then", "一直前进，然后点头", "clarify")

    for i, text in enumerate(("停", "停止", "别动了", "取消当前任务", "够了，停下来", "不用继续走了")):
        add("control", f"cancel_{i}", text, "cancel", execution=active(move(continuous=True)))
    for i, text in enumerate(("暂停", "先暂停一下", "先站稳暂停，留着后面的任务", "等会儿再做，先安全暂停")):
        add("control", f"pause_{i}", text, "pause", execution=active(kick("yellow"), pending=[perform("roll")]))
    for i, text in enumerate(("继续", "恢复执行", "接着做还没做完的", "继续刚才暂停的任务")):
        add("control", f"resume_{i}", text, "resume", execution=active(kick("yellow"), paused=True))
    add("control", "idle_resume", "继续", "clarify")
    add("control", "idle_pause", "暂停", "respond", reply="现在没有正在执行的任务。")
    for i, text in enumerate(
        (
            "不要停止，继续走",
            "我说的是‘停’这个字，不是在让你停",
            "假如我说暂停你会怎样",
            "昨天我让你踢球了",
            "别做前滚翻，我只是问问",
        )
    ):
        add("negation_quote", f"non_authorize_{i}", text, "respond", execution=active(move(continuous=True)))

    for i, text in enumerate(("踢红球", "向前走2秒", "前滚翻", "坐下")):
        add(
            "safety",
            f"motion_gate_{i}",
            text,
            "decline",
            environment=dict(surface="simulation_floor", motion_allowed=False, edge_guard="unavailable"),
        )
    for pose in ("fallen", "transitioning", "unknown"):
        add("safety", f"unsafe_pose_{pose}", "踢蓝球", "decline", posture=pose)
    add("safety", "stale_observation", "往前走1秒", "decline", age_s=1.5)
    add(
        "safety",
        "desktop_roll",
        "做个前滚翻",
        "decline",
        environment=dict(surface="desktop", motion_allowed=True, edge_guard="unavailable"),
    )
    add(
        "safety",
        "desktop_nod",
        "点个头",
        goals=[perform("nod")],
        environment=dict(surface="desktop", motion_allowed=True, edge_guard="unavailable"),
    )
    add(
        "safety",
        "sit_direct_move",
        "往前走2秒",
        goals=[posture("standing"), move("forward", 2)],
        posture="sitting",
        reply="我先站起来，再往前走2秒。",
    )
    add(
        "safety",
        "sit_direct_kick",
        "踢蓝球",
        goals=[posture("standing"), kick("blue")],
        posture="sitting",
        reply="我先站起来，再去踢蓝球。",
    )
    add("safety", "duplicate_red", "踢红球", "clarify", entities=[observed_ball("red", 0), observed_ball("red", 1)])

    chat = (
        ("你好呀鸭鸭", "你好呀，我是小鸭，有什么想聊的？"),
        ("早呀，睡醒了吗", "早上好，我已经准备好陪你聊天和做动作了。"),
        ("你叫什么名字", "我叫小鸭，是和你互动的小鸭机器人。"),
        ("谢谢你陪着我", "不客气，我很乐意陪你。"),
        ("今天有点烦，陪我聊聊", "听起来今天有些不顺，愿意说说发生了什么吗？"),
        ("我考试没考好", "考砸了会难过很正常，要不要聊聊是哪一部分没发挥好？"),
        ("讲个短笑话吧", "小鸭去买鞋，老板问多大码，小鸭说：能装下我的快乐就行。"),
        ("给我加个油", "加油呀，慢慢来，把眼前这一步做好就很棒。"),
        ("你真可爱", "谢谢，我会继续陪你玩。"),
        ("你喜欢红球还是蓝球", "我没有人的偏好，不过红球和蓝球我都可以帮你踢。"),
        ("你是不是已经踢到球了", "要以执行器的确认结果为准，目前没有完成记录。"),
        ("你能做什么", "我会点头、左右看、前滚翻、舞步、移动转身，也能靠近和踢红黄蓝球。"),
        ("你会前滚翻吗", "会，我支持前滚翻；要现在做的话直接告诉我。"),
        ("会踢方块吗", "目前不会踢方块，交互支持踢红、黄、蓝球。"),
        ("你能叼起球吗", "现在不具备叼球和搬运能力，俯身动作不代表能捡东西。"),
        ("能不能帮我点头一下", "可以，我会点一下头。"),
        ("我想让你帮忙踢一下红球", "好，我会去踢红球。"),
        ("bye，下次玩", "下次见，我还在这里。"),
        ("1加1等于几", "1加1等于2。"),
        ("你会不会摔倒", "有可能，仿真测试不能保证每次成功，我会根据执行反馈告诉你结果。"),
        ("你能保证100%踢中吗", "不能保证每次踢中，目标运动和观察误差都会影响结果。"),
        ("你今天的天气怎么样", "我没有实时天气数据，可以告诉我你所在的城市或天气。"),
        ("你会读心吗", "不会，我只能根据你告诉我的话和当前观察理解需求。"),
        ("我说中文你听得懂吗", "可以，你可以用中文告诉我要做什么。"),
    )
    for i, (text, reply) in enumerate(chat):
        goals = [perform("nod")] if i == 15 else [kick("red")] if i == 16 else []
        add("chat", f"chat_{i}", text, "replace" if goals else "respond", goals, reply)
    for i, (text, reply) in enumerate(chat[:8]):
        add(
            "chat_during_motion",
            f"busy_chat_{i}",
            text,
            "respond",
            reply=reply,
            execution=active(kick("yellow"), pending=[perform("roll")]),
            stationary=False,
        )
    for i, text in enumerate(("你看到了什么", "红球在哪里", "现在有几个球", "哪个球比较近", "绿球在不在")):
        add(
            "visual_question",
            f"visual_{i}",
            text,
            "respond",
            entities=[observed_ball("red"), observed_ball("blue", 1)],
            reply="当前可靠检测到红球和蓝球，红球在正前方，蓝球在左前方；没有可靠的绿球检测。",
        )
    for status, verified, reason in (
        ("succeeded", True, "target_contact_and_displacement_verified"),
        ("failed", False, "target_not_found"),
        ("cancelled", True, "user_or_local_stop"),
    ):
        execution = dict(
            state="idle",
            revision=3,
            active=None,
            pending=[],
            last_result=dict(goal=kick("blue"), status=status, reason=reason, verified=verified),
        )
        add(
            "result_question",
            f"result_{status}",
            "刚才蓝球踢中了吗",
            "respond",
            execution=execution,
            reply="蓝球已通过接触和位移验收。"
            if status == "succeeded"
            else f"没有完成，执行器记录为{status}，原因是{reason}。",
        )
    return rows


def extra_cases():
    """A separate small noisy-input probe, not a source of training paraphrases."""
    inputs = (
        ("dian tou yi xia", [perform("nod")]),
        ("houtui 3 miao", [move("backward", 3)]),
        ("xiang you zhuan 90 du", [turn("right", 90)]),
        ("ti hong qiu", [kick("red")]),
        ("zou dao lan qiu pang bian", [navigate("blue")]),
        ("zuo xia bing bao chi", [posture("sitting")]),
        ("先 dian tou，再 ti lan qiu", [perform("nod"), kick("blue")]),
        ("请右转９０度", [turn("right", 90)]),
        ("往左横移 ０．５ 秒", [move("left", 0.5)]),
        ("小鸭鸭～向后走 2.0 秒啦！", [move("backward", 2)]),
        ("蓝球来一jio", [kick("blue")]),
        ("点头\n然后张嘴\n最后闭嘴", [perform("nod"), perform("open_beak"), perform("close_beak")]),
        ("依次：①左右看；②右转45°；③前进1s", [perform("look_around"), turn("right", 45), move("forward", 1)]),
        ("啊那个……不是左边，是右边，往右挪2秒", [move("right", 2)]),
        ("不不不，红的不要，踢黄色的那个球", [kick("yellow")]),
        ("dui lv qiu lai yi jiao", []),
        ("ti fang kuai", []),
        ("hou kong fan", []),
        ("wo jin tian hen kai xin", None),
        ("ni hui zuo shen me", None),
    )
    rows = []
    for index, (text, goals) in enumerate(inputs):
        decision = "respond" if goals is None else "replace" if goals else "clarify"
        expected = dict(
            decision=decision,
            goals=goals or [],
            reply="好，我会执行你说的动作。"
            if goals
            else "请改成当前支持的动作。"
            if goals == []
            else "我在，愿意陪你聊聊。",
        )
        state = request(text, planned=True)
        validate_output(dict(schema_version=VERSION, **expected), state)
        rows.append(
            dict(
                id=f"noise_{index:04d}",
                group="pinyin_and_noisy_text",
                family_id=f"dev/noisy_input_{index}",
                request=state,
                expected=expected,
                rationale="拼音、混合字符、全角数字与口语纠正保留明确语义；库外技能先澄清。",
                reply_expectations=dict(
                    nonempty_chinese=True, no_unverified_completion=True, exact_match_required=False
                ),
                source="agent_authored_synthetic_semantic",
                split="development_only",
                training_allowed=False,
                human_reviewed=False,
                physical_executed=False,
            )
        )
    return rows


def write_suite(output, *, extra=False, combined=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = cases() + extra_cases() if combined else extra_cases() if extra else cases()
    serialized = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    (output / "suite.jsonl").write_text(serialized)
    metadata = dict(
        version=VERSION_SUITE,
        cases=len(rows),
        groups=dict(Counter(r["group"] for r in rows)),
        sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        old_blind_access=False,
        physical_executed=False,
        training_allowed=False,
        human_reviewed=False,
        provenance=(
            "Expected semantics authored independently by a coding agent; "
            "no tested-model gold labels; not human reviewed."
        ),
    )
    (output / "suite_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/microduck_tasks/outputs/q1r_20261116/baseline")
    print(json.dumps(write_suite(parser.parse_args().output), ensure_ascii=False, indent=2))
