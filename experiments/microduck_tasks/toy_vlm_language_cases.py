"""Predeclared language/context diagnostics, never training labels or physical trials."""

import copy

from .toy_fixtures import active, history, kick, move, navigate, perform, posture, request, turn
from .toy_inputs import validate_input
from .toy_protocol import VERSION, validate_output

VERSION_CASES = "toy-vlm-language-context-1"


def cases(observation=None):
    """Use actual recorded visual observations with explicitly hypothetical dialogue state."""
    rows = []

    def add(name, text, decision="respond", goals=(), *, tags=(), changes=None, acceptable=()):
        state = request(text, planned=True)
        if observation is not None:
            state["observation"] = copy.deepcopy(observation)
        alterations = ["event.text", "history", "execution", "capabilities"]
        if changes:
            changes(state)
        if name == "seated_transition":
            alterations.append("observation.posture")
        if name == "desktop_guard":
            alterations.append("observation.environment.surface")
        expected = dict(decision=decision, goals=list(copy.deepcopy(goals)))
        validate_input(state)
        validate_output(dict(schema_version=VERSION, reply="诊断用期望，绝不输入模型。", **expected), state)
        rows.append(
            dict(
                id=name,
                tags=list(tags),
                request=state,
                expected=expected,
                acceptable_decisions=list(acceptable) or [decision],
                counterfactual_fields=alterations,
            )
        )

    add("identity", "认识一下吧，你叫什么名字？", tags=["chat", "identity"])
    add("smalltalk", "今天心情怎么样？陪我聊两句。", tags=["chat"])
    add("abilities", "你都会哪些本领？先讲讲，别表演。", tags=["chat", "capability_question"])
    add("ability_question", "你会用右脚单脚站立吗？", tags=["capability_question", "non_authorization"])
    add(
        "polite_request",
        "能不能给我表演一下用右脚单脚站立？",
        "replace",
        [perform("flamingo_left")],
        tags=["polite_imperative", "support_foot"],
    )
    add(
        "repeat_gestures",
        "把头点两回，接着朝左右看看，最后身体轻晃一下。",
        "replace",
        [perform("nod", 2), perform("look_around"), perform("body_sway")],
        tags=["combination", "repeat"],
    )
    add(
        "foot_moonwalk",
        "抬高右脚给我瞧瞧，然后做个太空步。",
        "replace",
        [perform("flamingo_right"), perform("moonwalk")],
        tags=["combination", "support_foot"],
    )
    add(
        "dance_pair",
        "来一段害羞舞步，结束后换成节奏舞。",
        "replace",
        [perform("caught_dancing"), perform("groove")],
        tags=["combination"],
    )
    add(
        "cycle_bow",
        "完成一遍坐站循环，随后低头触地再起身。",
        "replace",
        [perform("sit_stand"), perform("ground_pick")],
        tags=["combination", "skill_semantics"],
    )
    add(
        "roll_mouth",
        "先翻个前滚翻，恢复后张开鸭嘴，随后把嘴合拢。",
        "replace",
        [perform("roll"), perform("open_beak"), perform("close_beak")],
        tags=["combination"],
    )
    add(
        "color_order",
        "先把红色球踢开，接下来去踢黄色的，蓝色的放最后。",
        "replace",
        [kick("red"), kick("yellow"), kick("blue")],
        tags=["combination", "colors", "ellipsis"],
    )
    add(
        "unsupported_partial",
        "点下头，再把黄球叼起来带给我。",
        "decline",
        tags=["unsupported", "no_partial_execution"],
    )

    def referent(state):
        state["history"] = history(("我说的是蓝色球。", "明白，你指的是蓝球。"))

    add(
        "history_referent",
        "就把它踢走吧。",
        "replace",
        [kick("blue")],
        tags=["history", "pronoun"],
        changes=referent,
    )

    def finite(state):
        state["execution"] = active(move(seconds=3))

    add(
        "append_finite",
        "做完手头的事，再补一个点头动作。",
        "append",
        [perform("nod")],
        tags=["queue", "append"],
        changes=finite,
    )

    def continuous(state):
        state["execution"] = active(move(continuous=True))

    add(
        "append_unreachable",
        "走完以后追加一个点头动作。",
        "clarify",
        tags=["queue", "continuous", "no_implicit_cancel"],
        changes=continuous,
        acceptable=["clarify", "decline"],
    )

    def seated(state):
        state["observation"]["posture"] = "sitting"

    add(
        "seated_transition",
        "现在向右转45度。",
        "replace",
        [posture("standing"), turn("right", 45)],
        tags=["posture", "prerequisite"],
        changes=seated,
    )

    def desktop(state):
        state["observation"]["environment"].update(surface="desktop", edge_guard="unavailable")

    add(
        "desktop_guard",
        "向前挪一点给我看。",
        "decline",
        tags=["environment", "motion_guard"],
        changes=desktop,
    )
    add(
        "numeric_motion",
        "先往左横移1.25秒，再向右转45度。",
        "replace",
        [move("left", seconds=1.25), turn("right", 45)],
        tags=["combination", "numeric", "lateral_vs_turn"],
    )
    close = navigate("red")
    close["standoff_m"] = 0.3
    add(
        "numeric_navigation",
        "靠近红色的球，在距它三十厘米的位置停住。",
        "replace",
        [close],
        tags=["numeric", "navigate_without_kick"],
    )
    add(
        "visual_without_motion",
        "请描述一下这两路相机里的东西，机器人不要移动。",
        tags=["visual_chat", "non_authorization"],
    )
    return rows
