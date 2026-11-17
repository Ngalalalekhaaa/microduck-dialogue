"""Language coverage with explicit limits, negation, and per-goal object binding."""

from .toy_fixtures import add, ball_target, kick, move, navigate, observed_ball, perform, posture, turn


def examples():
    rows = []
    add(
        rows,
        "greeting",
        ["你好，你叫什么", "嗨，小家伙，你的名字是什么？"],
        "你好，我叫小鸭！可以陪你聊天，也能表演动作和踢球。",
    )
    add(
        rows,
        "capabilities",
        ["你都会做什么", "说说你有什么本领"],
        "我能点头、左右看、坐下再站起、抬脚、跳舞和前滚翻，也能走动、转向和踢指定的球；滚翻需要宽敞空间。",
    )
    add(rows, "companionship", ["今天有点无聊", "陪我聊一会儿吧"], "我在呢。想说说今天的事，还是听个小故事？")
    add(rows, "thanks", ["谢谢你", "你刚才帮上忙了"], "不客气，很高兴陪你玩。")
    add(rows, "knowledge", ["一加一等于几", "1加1是多少？"], "一加一等于二。")
    add(
        rows,
        "story",
        ["讲个小故事", "给我讲一段鸭子的故事"],
        "小鸭追着一片树叶跑，跑到池塘边才发现，树叶已经坐上了小鱼的顺风车。",
    )
    add(rows, "embodiment", ["你能像真鸭子一样飞吗", "你会游泳吗？"], "我是机器人小鸭，现在不会飞也不会游泳。")
    add(
        rows,
        "offline_fact",
        ["现在外面下雨了吗", "今天室外天气怎么样"],
        "我目前没有室外天气信息，不能确定外面是否下雨。",
    )
    add(
        rows,
        "name_change",
        ["以后叫你豆豆吧", "我给你起名叫豆豆"],
        "豆豆这个名字很好听！当前版本还没有保存新名字的功能，我暂时仍叫小鸭。",
    )
    add(
        rows,
        "ability_question",
        ["你会前滚翻吗？", "你是不是能翻跟头？"],
        "我有前滚翻动作，需要宽敞空间。要表演时可以说‘做一下前滚翻’。",
    )
    for action, phrases, reply in (
        ("nod", ["点个头", "向我点头示意"], "好，我点个头。"),
        ("look_around", ["左右看看", "看看左边和右边"], "好，我左右看看。"),
        ("sit_stand", ["坐下然后站起来", "做一次坐站动作"], "好，我坐下后再站起来。"),
        ("ground_pick", ["低头俯身再起来", "做一个俯身动作"], "好，我俯下身再起身。"),
        ("roll", ["做一下前滚翻", "给我表演一个前滚翻"], "好，我做一次前滚翻，然后恢复站立。"),
        ("flamingo_left", ["抬一下左脚", "把左脚抬起来保持一下"], "好，我抬起左脚，再放下站稳。"),
        ("flamingo_right", ["抬一下右脚", "换右脚抬起来"], "好，我抬起右脚，再放下站稳。"),
        ("moonwalk", ["做个后退滑步", "表演后退滑步"], "好，我做一段后退滑步。"),
        ("body_sway", ["轻轻晃晃身体", "摆动一下身体"], "好，我轻轻晃动身体。"),
        ("open_beak", ["张开嘴巴", "把嘴张开"], "好，我张开嘴巴。"),
        ("close_beak", ["闭上嘴巴", "把嘴合上"], "好，我合上嘴巴。"),
        ("groove", ["跳个节奏舞", "表演一段节奏舞步"], "好，我跳一段节奏舞。"),
        ("caught_dancing", ["来段害羞舞", "表演害羞舞步"], "好，我表演一段害羞舞步。"),
    ):
        add(rows, f"self_{action}", phrases, reply, "replace", [perform(action)])
    add(
        rows,
        "polite_action",
        ["能不能帮我点个头？", "可以请你点一下头吗？"],
        "可以，我来点头。",
        "replace",
        [perform("nod")],
    )
    add(
        rows,
        "polite_roll",
        ["能不能给我做个前滚翻？", "可以请你表演一次前滚翻吗？"],
        "好，我做一次前滚翻，然后恢复站立。",
        "replace",
        [perform("roll")],
    )
    add(
        rows,
        "repeat_action",
        ["连续点两次头", "点头两次"],
        "好，我完整做两次点头动作。",
        "replace",
        [perform("nod", 2)],
    )
    add(
        rows,
        "excessive_repeat",
        ["前滚翻一百次", "一直翻跟头不要停"],
        "滚翻只能按有限次数执行，这次要做一到三次中的几次？",
        "clarify",
    )
    for direction, phrases, reply in (
        ("forward", ["往前走一点", "向前挪一点点"], "好，我短暂向前走一下。"),
        ("backward", ["往后退一点", "往后挪一下"], "好，我短暂后退一下。"),
        ("left", ["向左横移一点", "往左边平移，不要转身"], "好，我向左横移一点。"),
        ("right", ["向右横移一点", "往右边平移，不要转身"], "好，我向右横移一点。"),
    ):
        add(rows, f"move_{direction}", phrases, reply, "replace", [move(direction)])
    add(
        rows,
        "timed_move",
        ["向前走两秒后停下", "往前走2秒就停"],
        "好，我向前走两秒，然后停下。",
        "replace",
        [move(seconds=2)],
    )
    add(
        rows,
        "distance_move",
        ["向前走十厘米", "往前挪0.1米"],
        "当前还没有按距离验收的直走任务，可以改成短暂前进。",
        "decline",
    )
    add(
        rows,
        "turn_angle",
        ["向左转九十度", "朝左转90度"],
        "好，我向左转九十度。",
        "replace",
        [turn(angle=90)],
        planned=True,
    )
    add(
        rows,
        "turn_angle",
        ["向左转九十度", "朝左转90度"],
        "当前只能按短时指令转向，九十度转角还没有验收，不能保证准确执行。",
        "decline",
    )
    add(
        rows,
        "full_turn",
        ["原地转一圈", "向左转完整一周"],
        "好，我向左转一整圈；完成后停下。",
        "replace",
        [turn()],
        planned=True,
    )
    add(
        rows,
        "full_turn",
        ["原地转一圈", "向左转完整一周"],
        "完整转圈的累计角度验收还没接好，我先不把一次短转当成转满一圈。",
        "decline",
    )
    add(
        rows,
        "sit_hold",
        ["坐下", "坐下来，先别起来"],
        "好，我坐下后保持坐姿。",
        "replace",
        [posture("sitting")],
        planned=True,
    )
    add(rows, "sit_hold", ["坐下", "坐下来，先别起来"], "目前能做坐下再站起，保持坐姿还没接好；我先不动。", "decline")
    add(rows, "stand", ["站起来", "起立"], "好，我保持站立。", "replace", [posture("standing")])
    for color, chinese in (("red", "红"), ("yellow", "黄"), ("blue", "蓝")):
        add(
            rows,
            f"kick_{color}",
            [f"踢{chinese}球", f"把{chinese}色的球踢一下"],
            f"好，我先找{chinese}球，接近后踢它。",
            "replace",
            [kick(color)],
        )
        add(
            rows,
            f"navigate_{color}",
            [f"走到{chinese}球旁边", f"靠近{chinese}球后停下，不要踢"],
            f"好，我靠近{chinese}球，在约二十五厘米处停下，不踢它。",
            "replace",
            [navigate(color)],
            planned=True,
        )
    add(
        rows,
        "navigate_yellow",
        ["走到黄球旁边", "靠近黄球后停下，不要踢"],
        "目前接近控制器是为踢球准备的，走到旁边停下还没独立验收；我先不执行。",
        "decline",
    )
    add(
        rows,
        "negated_target",
        ["别踢蓝球，踢红球", "红的踢一下，蓝的不要踢"],
        "好，我只去踢红球。",
        "replace",
        [kick("red")],
    )
    add(
        rows,
        "negative_only",
        ["别动", "不要踢球", "停下", "算了，全部取消"],
        "好，我取消动作任务并请求停下。",
        "cancel",
    )
    add(
        rows,
        "conditional",
        ["如果红球靠过来就踢它", "等它滚过来再踢"],
        "当前还不支持条件触发任务，你希望我现在踢哪颗球？",
        "clarify",
    )
    add(
        rows,
        "ambiguous_choice",
        ["踢红球还是蓝球呢", "红的蓝的，你说踢哪个？"],
        "这次由你选，要踢红球还是蓝球？",
        "clarify",
    )
    add(
        rows,
        "hypothetical",
        ["如果我说前滚翻，你会怎么做？", "解释一下‘前滚翻’这个命令"],
        "这条命令表示执行一次前滚翻再站稳；现在你是在提问，我先不表演。",
    )
    add(rows, "quoted_command", ["读一下这句话：停下", "念出引号里的词：‘停下’"], "停下。")
    add(rows, "quoted_kick", ["把‘踢红球’这几个字念出来"], "踢红球。")
    add(
        rows,
        "unsupported_perception",
        ["踢紫色球", "把紫球踢走"],
        "当前找球模块只支持红、黄、蓝三种球，还不能可靠地找紫球。",
        "decline",
    )
    add(
        rows,
        "unsupported_grasp",
        ["把球叼起来", "叼起那个绿色小球"],
        "当前玩具模式只保留踢球交互，不执行叼取。",
        "decline",
    )
    add(rows, "unsupported_push", ["把方块推过去", "帮我推一下球"], "当前玩具模式不执行推物任务。", "decline")
    add(rows, "unsupported_throw", ["把球扔过来", "叼起球然后甩出去"], "当前玩具模式不执行叼取或投掷。", "decline")
    add(
        rows,
        "unstable_action",
        ["做个后空翻", "倒立给我看", "往前跑", "跳一下"],
        "这个动作尚未进入当前稳定候选库，我先不执行。",
        "decline",
    )
    add(
        rows,
        "unsupported_sequence",
        ["先踢红球再叼回来", "先推球再点头"],
        "这组任务包含当前不支持的交互，我先不执行整组；可以去掉叼取或推物后再安排。",
        "decline",
    )
    add(
        rows,
        "sequence_required",
        ["先踢蓝球，再踢红球，再原地转一圈", "按顺序踢蓝球、踢红球，最后向左转一圈"],
        "好，我先踢蓝球，再踢红球，最后向左转一圈；每一步确认完成后再继续。",
        "replace",
        [kick("blue"), kick("red"), turn()],
        planned=True,
    )
    add(
        rows,
        "sequence_required",
        ["先踢蓝球，再踢红球，再原地转一圈"],
        "踢球动作已有，但整圈转向和组合任务队列还没接入当前播放器；我先不启动这一组。",
        "decline",
    )
    add(
        rows,
        "self_sequence",
        ["先点头，再左右看看，最后跳节奏舞", "点个头，然后左右看一下，接着来段节奏舞"],
        "好，我按点头、左右看、节奏舞的顺序表演。",
        "replace",
        [perform("nod"), perform("look_around"), perform("groove")],
        planned=True,
    )
    add(
        rows,
        "navigation_sequence",
        ["先走到黄球旁边，再点头", "到黄球旁边停好，然后点个头"],
        "好，我在黄球附近停下，再点头。",
        "replace",
        [navigate("yellow"), perform("nod")],
        planned=True,
    )
    add(
        rows,
        "continuous",
        ["一直往前走不要停", "往前走，等我叫停"],
        "好，我持续向前走；你叫停或本地安全检查要求停止时，我就停下。",
        "replace",
        [move(continuous=True)],
        planned=True,
    )
    add(
        rows,
        "continuous",
        ["一直往前走不要停", "往前走，等我叫停"],
        "持续行走任务还没接入当前播放器，我不能把一次短走当作一直走。",
        "decline",
    )
    add(
        rows,
        "unreachable_sequence",
        ["一直前进，然后再踢红球", "不停地走，最后点头"],
        "前面的持续动作没有结束条件，后面的任务无法开始；要先走多久？",
        "clarify",
    )
    add(
        rows,
        "seven_goals",
        ["点头、向左转30度、向右转30度、坐下、站起、踢红球、踢蓝球"],
        "好，我会依次点头、向左转30度、向右转30度、坐下、站起，再踢红球和蓝球。",
        "replace",
        [
            perform("nod"),
            turn("left", 30),
            turn("right", 30),
            posture("sitting"),
            posture("standing"),
            kick("red"),
            kick("blue"),
        ],
        planned=True,
    )

    def duplicate(state):
        state["observation"]["entities"] = [observed_ball("red", 0), observed_ball("red", 1)]

    add(
        rows,
        "duplicate_target",
        ["踢红球", "踢那个红色的球"],
        "我看到两颗红球，你指哪一颗？",
        "clarify",
        modify=duplicate,
    )

    def visible(state):
        state["observation"]["entities"] = [observed_ball("red"), observed_ball("blue")]

    add(
        rows,
        "visual_count",
        ["你看到了什么", "现在有几个球在眼前", "你现在在能看到什么", "你现在能看到什么", "描述一下眼前的画面"],
        "当前观察中有两颗球，一颗红色、一颗蓝色。",
        modify=visible,
    )
    add(rows, "visual_empty", ["你看到了什么", "现在能看到红球吗"], "当前观察里没有检测到球，这不代表周围一定没有球。")

    def spatial(state):
        red, blue = observed_ball("red"), observed_ball("blue")
        red["position_body_m"], blue["position_body_m"] = [0.5, 0.15, 0.03], [0.5, -0.15, 0.03]
        state["observation"]["entities"] = [red, blue]

    add(
        rows,
        "visual_spatial",
        ["红球和蓝球分别在哪边", "按你自己的方向说说球的位置"],
        "以我自己的朝向为准，红球在左前方，蓝球在右前方。",
        modify=spatial,
    )

    def other_object(state):
        block = observed_ball("purple")
        block["id"], block["category"] = "observed:block:0", "block"
        state["observation"]["entities"] = [block]

    add(
        rows,
        "visual_other_object",
        ["你现在能看到什么", "眼前除了球还有什么"],
        "当前观察中有一个紫色方块，没有检测到球。",
        planned=True,
        modify=other_object,
    )

    def missing_now(state):
        state["history"] = [
            {"role": "user", "text": "你看到了什么"},
            {"role": "assistant", "text": "当前观察中有一颗红球和一颗蓝球。"},
        ]
        state["observation"]["entities"] = [observed_ball("blue")]

    add(
        rows,
        "visual_missing_now",
        ["现在还能看到红球吗", "刚才那个红球去哪了"],
        "当前观察里只检测到蓝球，没有检测到红球；仅凭现在的观察不能判断红球去了哪里。",
        modify=missing_now,
    )
    add(rows, "vague_target", ["踢那颗球", "把它踢走"], "你指哪颗球？可以告诉我颜色。", "clarify", modify=visible)

    def bound(state):
        state["observation"]["entities"] = [observed_ball("blue")]

    add(
        rows,
        "unique_reference",
        ["踢眼前唯一那颗球", "把眼前那颗蓝球踢走"],
        "好，我去踢眼前这颗蓝球。",
        "replace",
        [{"type": "kick", "target": ball_target("blue", "observed:blue:0")}],
        modify=bound,
    )
    return rows
