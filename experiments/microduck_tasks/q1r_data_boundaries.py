"""Capability limits, numerical units, observation guards and honest failures."""

from .q1r_data_core import base_context
from .toy_fixtures import active, kick, navigate, observed_ball, perform

UNSUPPORTED = (
    ("踢一下那个方块", "当前没有踢方块技能，只能踢红、黄或蓝球。你想换成哪种球？"),
    ("把紫色方块踢走", "方块不属于当前踢球技能的目标。可以改成红球、黄球或蓝球。"),
    ("先踢方块，再点头", "这组包含未支持的踢方块任务，暂不开始。要改成踢球，还是只点头？"),
    ("踢那个绿色球", "当前只可靠支持红、黄、蓝球，绿球还不支持。你要改成哪种颜色？"),
    ("去紫球旁边", "当前不能可靠识别并导航到紫球。可以选择红球、黄球或蓝球。"),
    ("踢黑色的球给我看", "黑球不在当前可识别目标里。请改成红、黄或蓝球。"),
    ("把球叼起来递给我", "目前没有叼取、递送物体的技能。可以改为靠近球或踢球。"),
    ("把红球搬到蓝球旁边", "当前不能搬运球，也不能指定搬运落点。可以靠近或踢指定颜色的球。"),
    ("把那个箱子推过去", "当前没有推箱子技能。你可以选择移动、转身或已有的球类任务。"),
    ("来个后空翻", "当前不支持后空翻，已开放的是前滚翻。要改成前滚翻吗？"),
    ("往上跳一下", "跳跃尚未开放。可以选择点头、抬脚、跳节奏舞等现有技能。"),
    ("跑到门口去", "没有跑步或门口导航技能。可以按秒行走，或走到指定颜色的球旁边。"),
    ("现在飞起来", "当前没有飞行技能。你可以选择行走、转身或表演动作。"),
    ("倒立十秒钟", "倒立未开放，不能执行。你想改成抬脚或前滚翻吗？"),
    ("请用翅膀抓起红球", "当前没有抓取球的技能。可以改为靠近红球或踢红球。"),
    ("后退十步", "当前不能按步数准确计数行走。可以改成后退几秒，请给一个0.12到10秒的时长。"),
    ("往前走两米", "当前没有按米定距行走的技能。可以指定0.12到10秒的移动，或指定红黄蓝球作为目标。"),
    ("左转五度", "当前转身角度支持15到360度，5度超出范围。你要改成多少度？"),
    ("右转720度", "单项转身支持15到360度。可以明确拆成两次右转一圈，或改小角度。"),
    ("向后走十二秒", "单项按秒移动最多10秒。请缩短时间，或明确拆成多项任务。"),
    ("往前走0秒", "移动时长需要在0.12到10秒之间。你想走多久？"),
    ("倒退负三秒", "时长不能是负数，请指定0.12到10秒的有效移动时长。"),
    ("向前走NaN秒", "这个时长不是有效数值。请使用0.12到10秒之间的有限数值。"),
    ("后退无穷大秒", "不能使用无穷大时长；如果要持续行走，请明确说一直走到喊停。"),
    ("踢球，颜色你随便挑", "请明确目标颜色，我可以踢红球、黄球或蓝球。"),
    ("点头还是前滚翻，你自己决定", "请先确定要点头还是前滚翻，我再执行。"),
    ("同时向左走和向右走", "不能同时向两个相反方向行走。请选一个方向，或说明先后顺序。"),
    ("一边前滚翻一边踢球", "现有动作按顺序执行，不能同时翻滚和踢球。请说明先后顺序和球的颜色。"),
    ("一直后退，然后前滚翻", "一直后退没有结束点，后面的前滚翻无法开始。请先给后退时长，或只安排持续后退。"),
    ("一直前进三秒不要停", "这同时包含三秒结束和持续不停两种要求。请确认按三秒执行，还是一直走到喊停。"),
    ("转一下身", "请告诉我向左还是向右，以及要转多少度。"),
    ("照我脑子里想的做", "我不知道你心里想的具体动作。请说出技能、目标或动作顺序。"),
    ("把所有东西都踢了", "当前只能踢明确指定的红、黄或蓝球。请说明要哪些球及顺序。"),
    ("抬脚", "你想抬左脚还是右脚？请说明一边。"),
    ("先踢红球，再给我端一杯水", "端水不在技能库里，这组暂不开始。要只踢红球，还是换一个已支持的后续动作？"),
    ("去厨房给我拿东西", "当前不支持房间导航和拿取物品。可以按秒移动或走到指定颜色的球旁边。"),
    ("走到红球旁边，离它一厘米", "停靠距离支持0.15到0.5米，一厘米太近。请重新选择这个范围内的距离。"),
    ("靠近蓝球，停在一米外", "当前停靠距离支持0.15到0.5米。你想改成多远？"),
    ("帮我踢红球进那个球门", "可以踢红球，但当前没有瞄准球门并保证射门的技能。要改成普通踢红球吗？"),
    ("用指定力度让蓝球正好飞两米", "当前不能保证踢球力度或两米落点。你要改成普通踢蓝球吗？"),
    ("把棱形积木踹开", "当前只支持踢红黄蓝球，不支持踢积木。你想改成哪种球？"),
    ("追上紫色圆球再碰一下它", "紫球不在可靠识别范围内，也没有独立碰球技能。可以指定红黄蓝球靠近或踢球。"),
    ("向后数着走十五步", "当前不能按步数计量行走，请改成0.12到10秒的时长。"),
    ("原地往左旋转十度", "当前支持15到360度转身，十度不在范围内。请换一个角度。"),
    ("既不停地向前走又要马上做太空步", "这两个要求冲突。请先停止或限定前进时长，再安排太空步。"),
    ("先表演点头，之后背着我跑一圈", "背人和跑步都没有开放，这组任务暂不执行。要只保留点头吗？"),
    ("到绿色小球边上坐着", "当前不支持识别并靠近绿球，暂不启动这组动作。请改成红、黄或蓝球。"),
    ("把红球推到蓝色方块右边", "当前没有推球到指定物体旁的技能。可以改为靠近红球或踢红球。"),
    ("同一瞬间抬两只脚跳起来", "双脚跳跃没有开放。可以单独抬左脚或右脚，你想选哪个？"),
    ("让头一直保持向右45度", "当前没有指定头部角度并保持的技能。可以选择左右看，或整个身体右转45度。"),
)


def generate_boundaries(builder):
    for p, (text, reply) in enumerate(UNSUPPORTED):
        split = "val" if p >= 40 else "train"
        for i in range(2 if split == "val" else 8):
            req = base_context(text, i, state=("idle", "running", "paused", "failed")[i % 4])
            builder.add(
                f"unsupported_{p}",
                split,
                "unsupported_clarify",
                req,
                "clarify",
                reply=reply,
                rationale="不存在或未开放的技能、目标、精度要求先澄清，提供真实可用替代；不执行混合清单的一部分。",
            )
    _guards(builder)
    _metric_navigation(builder)
    _failure_reports(builder)
    _authorization_pairs(builder)


def _guards(builder):
    conditions = (
        ("fallen", "当前姿态是倒地，不能直接开始动作。需要先确认姿态恢复。"),
        ("unknown", "当前姿态不明确，不能推测已经站稳。请先刷新姿态观察。"),
        ("transitioning", "当前正在切换姿态，请等姿态稳定并确认后再执行。"),
        ("stale", "当前观察过期，不能据此开始运动。需要刷新观察后再安排。"),
        ("gate", "当前本地运动许可关闭，无法开始动作。需要先恢复本地执行条件。"),
        ("desktop", "当前在桌面上且没有有效边缘保护，不能执行移动或大幅动作。请先提供安全的执行环境。"),
    )
    wordings = ("现在前滚翻", "麻烦踢红球", "请向后走两秒", "立即开始左右看", "开始一次向前翻滚", "请演示前滚翻动作")
    for p, text in enumerate(wordings):
        split = "val" if p >= 4 else "train"
        for i, (condition, reply) in enumerate(conditions):
            req = base_context(text, i)
            if condition in ("fallen", "unknown", "transitioning"):
                req["observation"]["posture"] = condition
            elif condition == "stale":
                req["observation"]["age_s"] = 1.7
            elif condition == "gate":
                req["observation"]["environment"]["motion_allowed"] = False
            else:
                req["observation"]["environment"].update(surface="desktop", edge_guard="triggered")
                if "左右看" in text:
                    # Small gestures are permitted by the contract; do not teach blanket desktop refusal.
                    builder.add(
                        f"guard_{p}",
                        split,
                        "environment_guard",
                        req,
                        "replace",
                        [perform("look_around")],
                        rationale="已站稳且许可开启时，桌面边缘保护限制大幅动作，不禁止协议允许的小幅静态手势。",
                    )
                    continue
            builder.add(
                f"guard_{p}",
                split,
                "environment_guard",
                req,
                "decline",
                reply=reply,
                rationale="明确能力不能绕过本地运动门、姿态、观察时效或桌边约束。",
            )
    for p, text in enumerate(("给我表演太空步", "请做抬左脚动作", "现在做一次节奏舞", "展示月球漫步给我看")):
        split = "val" if p == 3 else "train"
        action = ("moonwalk", "flamingo_left", "groove", "moonwalk")[p]
        for i in range(2 if split == "val" else 8):
            req = base_context(text, i)
            req["capabilities"]["self_actions"].remove(action)
            builder.add(
                f"capability_snapshot_{p}",
                split,
                "capability_snapshot",
                req,
                "clarify",
                reply="这个动作不在当前开放的技能列表里。请改成当前可用的点头、左右看等动作。",
                rationale="以当次capabilities为准，不因名字出现在全局词表就擅自执行。",
            )
    # Duplicate detected targets require identifying the particular ball rather than guessing.
    for p, text in enumerate(("踢红色那个球", "去红球边上", "请踢那颗红球", "把红球作为目标踢一下")):
        split = "val" if p == 3 else "train"
        for i in range(2 if split == "val" else 7):
            req = base_context(text, i)
            req["observation"]["entities"] = [observed_ball("red", 0), observed_ball("red", 1)]
            builder.add(
                f"duplicate_target_{p}",
                split,
                "ambiguous_detection",
                req,
                "clarify",
                reply="当前检测到不止一个红球，请进一步说明你指的是哪一个。",
                rationale="颜色匹配多个观察实体时不能任意绑定一个对象。",
            )


def _metric_navigation(builder):
    wordings = (
        "走到{c}球旁，保持{d}的距离",
        "靠近{c}球，在离球{d}处停住",
        "到{c}球边上，停靠距离{d}",
        "请找{c}球，走到距离它{d}的位置",
        "在距离{c}球{d}的地方停靠",
        "向{c}球靠近，到相距{d}时停下",
    )
    for p, wording in enumerate(wordings):
        split = "val" if p >= 4 else "train"
        for i in range(3 if split == "val" else 15):
            name, color = (("红", "red"), ("黄", "yellow"), ("蓝", "blue"))[i % 3]
            word, distance = (("二十五厘米", 0.25), ("0.3米", 0.3), ("半米", 0.5), ("15厘米", 0.15), ("二十厘米", 0.2))[
                i % 5
            ]
            goal = navigate(color)
            goal["standoff_m"] = distance
            req = base_context(wording.format(c=name, d=word), i)
            builder.add(
                f"navigation_distance_{p}",
                split,
                "unit_conversion",
                req,
                "replace",
                [goal],
                reply=f"收到，我会靠近{name}球，并在约{distance:g}米的目标停靠距离处停下。",
                rationale="厘米和米转换为明确支持的停靠目标；不宣称未经执行验收的距离已经达到。",
            )


def _failure_reports(builder):
    reasons = (
        ("target_not_found_after_full_search", "找完一圈仍未可靠发现目标球"),
        ("target_search_timeout", "搜索超时，尚未完成可靠的目标定位"),
        ("target_not_approaching", "一段时间内距离没有改善，未能靠近目标"),
        ("navigation_timeout", "靠近过程超时，尚未到达目标位置"),
        ("verification_failed", "踢腿动作已经结束，但没有通过接触和球位移验收，不能说踢中了"),
        ("safe_pause_boundary", "已在稳定边界暂停，当前目标尚未完成，保留任务等待继续"),
    )
    wordings = (
        "为什么还没成功",
        "到底卡在哪里了",
        "刚才踢中了没有",
        "你做了动作就算完成吗",
        "请解释当前结果",
        "说明一下本次失败或暂停的原因",
    )
    for p, text in enumerate(wordings):
        split = "val" if p >= 4 else "train"
        for i, (reason, answer) in enumerate(reasons):
            req = base_context(text, i)
            if reason == "safe_pause_boundary":
                req["execution"] = active(kick("red"), paused=True)
            else:
                req["execution"]["state"] = "failed"
            req["execution"]["last_result"] = dict(
                goal=kick("red"),
                status="blocked" if reason == "safe_pause_boundary" else "failed",
                reason=reason,
                verified=False,
            )
            builder.add(
                f"failure_detail_{p}",
                split,
                "honest_failure",
                req,
                reply=f"踢红球尚未完成：{answer}。",
                rationale="执行器事实优先于历史承诺或已经播放的动作；未通过验收不报成功，安全暂停不冒充失败。",
            )


def _authorization_pairs(builder):
    cases = (
        ("用一个点头向我致意", "replace", [perform("nod")], "收到，我会点头向你致意。"),
        ("和我打招呼时轻轻点一下脑袋", "replace", [perform("nod")], "你好呀，我会轻轻点头。"),
        ("你听懂了就上下点一下脑袋", "replace", [perform("nod")], "收到，我会点一下头。"),
        ("你会用点头打招呼么", "respond", [], "会，我可以用点头打招呼。你想现在看一次的话可以直接说。"),
        ("支撑身体的是右脚，把左脚提起来", "replace", [perform("flamingo_left")], "收到，我会用右脚支撑并抬起左脚。"),
        ("留左脚站着，另一只右脚抬起来", "replace", [perform("flamingo_right")], "好的，我会用左脚支撑并抬起右脚。"),
        ("嘴巴打开一点给我看看", "replace", [perform("open_beak")], "好，我会张开嘴巴。"),
        ("你的嘴能张开吗", "respond", [], "能，当前支持张嘴和闭嘴。"),
        ("请让身子左右轻轻摆动一下", "replace", [perform("body_sway")], "收到，我会轻轻晃动身体。"),
        ("你是不是会左右晃动身体", "respond", [], "是的，当前支持轻晃身体。"),
        ("抬起左脚时是不是右脚支撑", "respond", [], "是的，抬左脚时由右脚支撑；抬右脚则由左脚支撑。"),
        ("用头上下点动来向我问好", "replace", [perform("nod")], "你好，我会点头向你问好。"),
        ("让右脚离地，用左脚维持支撑", "replace", [perform("flamingo_right")], "收到，我会抬起右脚，由左脚支撑。"),
        ("请展现把鸭嘴张开的动作", "replace", [perform("open_beak")], "好的，我会张开鸭嘴。"),
    )
    for p, (text, decision, goals, reply) in enumerate(cases):
        split = "val" if p >= 11 else "train"
        for i in range(2 if split == "val" else 10):
            req = base_context(text, i, state=("idle", "running", "paused")[i % 3])
            builder.add(
                f"authorization_pair_{p}",
                split,
                "action_vs_discussion",
                req,
                decision,
                goals,
                reply,
                rationale="礼貌口语仍可明确授权；能力询问不自动执行；支撑脚与抬起脚左右相反且保持原始观察姿态。",
            )
