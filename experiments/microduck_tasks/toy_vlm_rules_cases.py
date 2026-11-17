"""Predeclared prompt comparison cases; labels never enter model inputs or training."""

import copy

from .toy_fixtures import active, history, kick, move, navigate, perform, posture, request, turn
from .toy_inputs import validate_input
from .toy_protocol import VERSION, validate_output

VERSION_CASES = "toy-vlm-rules-ablation-2"


def cases(observation=None):
    """Attach one recorded observation to hypothetical utterances and execution states."""
    rows = []

    def add(name, text, decision="respond", goals=(), *, tags=(), changes=None, acceptable=(), alternatives=()):
        state = request(text, planned=True)
        if observation is not None:
            state["observation"] = copy.deepcopy(observation)
        alterations = ["event.text", "history", "execution", "capabilities"]
        if changes:
            alterations.extend(changes(state))
        expected = dict(decision=decision, goals=list(copy.deepcopy(goals)))
        acceptable_decisions = list(acceptable) or [decision]
        acceptable_goals = list(copy.deepcopy(alternatives))
        validate_input(state)
        for variant_decision in acceptable_decisions:
            for variant_goals in [expected["goals"], *acceptable_goals]:
                validate_output(
                    dict(
                        schema_version=VERSION,
                        reply="评估期望，绝不输入模型。",
                        decision=variant_decision,
                        goals=variant_goals,
                    ),
                    state,
                )
        rows.append(
            dict(
                id=name,
                tags=list(tags),
                request=state,
                expected=expected,
                acceptable_decisions=acceptable_decisions,
                acceptable_goals=acceptable_goals,
                counterfactual_fields=alterations,
                visual_source="recorded_observation" if observation is not None else "contract_fixture_only",
                reply_review="manual; plan scores do not certify response truth or completeness",
                training_ready=False,
                physical_validation=False,
            )
        )

    add("first_nod", "请现在点一下头。", "replace", [perform("nod")], tags=["first_turn", "single"])
    add("first_scan", "左右看看吧。", "replace", [perform("look_around")], tags=["first_turn", "single"])
    add(
        "cycle_then_bow",
        "先做一遍坐站循环，然后俯身再起。",
        "replace",
        [perform("sit_stand"), perform("ground_pick")],
        tags=["first_turn", "two_goals", "skill_semantics"],
    )
    add(
        "roll_open_close",
        "做个前滚翻，接着张嘴，最后闭嘴。",
        "replace",
        [perform("roll"), perform("open_beak"), perform("close_beak")],
        tags=["first_turn", "three_goals"],
    )
    add(
        "left_then_right_foot",
        "先抬左脚，接下来抬右脚。",
        "replace",
        [perform("flamingo_left"), perform("flamingo_right")],
        tags=["first_turn", "two_goals", "foot_identity"],
    )
    add(
        "groove_then_shy",
        "先跳节奏舞，再来一段害羞舞步。",
        "replace",
        [perform("groove"), perform("caught_dancing")],
        tags=["first_turn", "two_goals"],
    )
    add(
        "moonwalk_then_sway",
        "做完后退滑步，再轻晃一下身体。",
        "replace",
        [perform("moonwalk"), perform("body_sway")],
        tags=["first_turn", "two_goals"],
    )
    add(
        "repeat_three_then_two",
        "点头三次，再左右看两次。",
        "replace",
        [perform("nod", 3), perform("look_around", 2)],
        tags=["repeat", "two_goals"],
    )
    add(
        "six_in_order",
        "按顺序来：左右看、点头、张嘴、闭嘴、轻晃身体、俯身再起。",
        "replace",
        [perform(name) for name in ("look_around", "nod", "open_beak", "close_beak", "body_sway", "ground_pick")],
        tags=["first_turn", "six_goals", "order"],
    )
    add(
        "yellow_blue_red",
        "踢球的顺序是黄球、蓝球、红球，请依次完成。",
        "replace",
        [kick("yellow"), kick("blue"), kick("red")],
        tags=["three_goals", "colors", "order"],
    )
    add(
        "unspecified_circle",
        "在原地转完整一圈。",
        "replace",
        [turn("left", 360)],
        tags=["angle", "direction_unspecified"],
        alternatives=[[turn("right", 360)]],
    )
    add(
        "numeric_walk_turn",
        "前进1.4秒，再向左转60度，最后后退0.8秒。",
        "replace",
        [move(seconds=1.4), turn("left", 60), move("backward", seconds=0.8)],
        tags=["three_goals", "numeric"],
    )
    approach = navigate("blue")
    approach["standoff_m"] = 0.3
    add(
        "approach_blue_only",
        "走近蓝色球，离它三十厘米时停下，不用踢。",
        "replace",
        [approach],
        tags=["navigate_without_kick", "numeric"],
    )
    add("introduce_identity", "介绍一下你自己和你的名字吧。", tags=["chat", "identity"])
    add("ask_roll_ability", "我想先了解一下：你会前滚翻吗？", tags=["capability_question", "non_authorization"])
    add("negated_nod", "这次不要点头，和我聊聊天就好。", tags=["negation", "non_authorization"])
    add(
        "unsupported_whole_plan",
        "先向左转90度，再做后空翻，最后点头。",
        "decline",
        tags=["unsupported", "no_partial_execution"],
    )

    def seated(state):
        state["observation"]["posture"] = "sitting"
        return ["observation.posture"]

    add(
        "stand_before_turn",
        "现在向左转60度。",
        "replace",
        [posture("standing"), turn("left", 60)],
        tags=["posture", "prerequisite"],
        changes=seated,
    )

    def running(state, *, continuous=False, full=False, paused=False):
        pending = [perform(name) for name in ("nod", "look_around", "open_beak", "close_beak", "body_sway")]
        state["execution"] = active(
            move(seconds=2, continuous=continuous), paused=paused, pending=pending if full else ()
        )
        state["observation"]["stationary"] = paused
        return ["observation.stationary"]

    add(
        "append_two",
        "把张嘴和闭嘴接在现有任务后面。",
        "append",
        [perform("open_beak"), perform("close_beak")],
        tags=["queue", "append"],
        changes=running,
    )
    add(
        "append_after_six",
        "别改已有顺序，在最后加一次点头。",
        "append",
        [perform("nod")],
        tags=["queue", "append", "capacity"],
        changes=lambda state: running(state, full=True),
    )
    add(
        "walk_until_called",
        "朝前一直走，直到我叫停。",
        "replace",
        [move(continuous=True)],
        tags=["continuous", "first_turn"],
    )
    add(
        "append_after_unbounded",
        "在当前一直走的任务后面加一次点头。",
        "clarify",
        tags=["queue", "continuous", "no_implicit_cancel"],
        changes=lambda state: running(state, continuous=True),
        acceptable=["clarify", "decline"],
    )
    add(
        "chat_during_walk",
        "你叫什么呀？回答就行，保持现在的任务。",
        tags=["chat", "continuous", "preserve_task"],
        changes=lambda state: running(state, continuous=True),
    )
    add(
        "pause_continuous",
        "请暂停一下。",
        "pause",
        tags=["continuous", "pause", "host_control"],
        changes=lambda state: running(state, continuous=True),
    )
    add(
        "resume_paused",
        "接着刚才暂停的任务走吧。",
        "resume",
        tags=["continuous", "resume"],
        changes=lambda state: running(state, continuous=True, paused=True),
    )

    def cancelled(state):
        state["history"] = history(("取消前进任务。", "前进任务已经取消，队列已清空。"))
        state["execution"]["last_result"] = dict(
            goal=move(continuous=True),
            status="cancelled",
            reason="user_cancelled",
            verified=False,
        )
        return []

    add(
        "resume_after_cancel",
        "把刚才的任务继续做完。",
        "clarify",
        tags=["cancelled", "no_resurrection"],
        changes=cancelled,
        acceptable=["clarify", "decline"],
    )
    add(
        "stop_continuous",
        "立即停住。",
        "cancel",
        tags=["continuous", "cancel", "host_control"],
        changes=lambda state: running(state, continuous=True),
    )

    def referent(state, *, ambiguous=False):
        pair = ("我提到红球和蓝球。", "明白，是红球和蓝球。") if ambiguous else ("目标是黄球。", "明白，目标是黄球。")
        state["history"] = history(pair)
        return []

    add(
        "resolved_yellow_pronoun",
        "就踢刚才提到的那个吧。",
        "replace",
        [kick("yellow")],
        tags=["history", "pronoun"],
        changes=referent,
    )
    add(
        "ambiguous_two_referents",
        "把它踢开吧。",
        "clarify",
        tags=["history", "pronoun", "ambiguity"],
        changes=lambda state: referent(state, ambiguous=True),
    )
    add(
        "describe_both_views",
        "分别说说前视和近处画面里能看见什么，先不做动作。",
        tags=["visual_chat", "non_authorization", "manual_visual_review"],
    )
    add("unknown_step_unit", "往前走三步。", "clarify", tags=["unsupported_unit", "no_invented_conversion"])
    add(
        "unreachable_new_plan",
        "先一直往前走不要停，再张嘴。",
        "clarify",
        tags=["continuous", "unreachable_suffix"],
        acceptable=["clarify", "decline"],
    )
    return rows
