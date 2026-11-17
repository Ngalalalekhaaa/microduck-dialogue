"""Dialogue teacher behavior before it supplies any real rollout labels."""

import copy

import pytest

from .toy_catalog import SELF_ACTIONS
from .toy_fixtures import active, kick, move, navigate, observed_ball, perform, posture, request, turn
from .toy_protocol import validate_output
from .toy_teacher import ToyTeacher, is_cancel, is_pause


def decide(text, *, mutate=None):
    state = request(text, planned=True)
    if mutate:
        mutate(state)
    before = copy.deepcopy(state)
    result = ToyTeacher().decide(state)
    assert state == before
    assert result["reply"].strip()
    assert validate_output(result, state) == result
    return result


@pytest.mark.parametrize(
    ("text", "goals"),
    [
        ("先踢蓝球，再踢红球，再原地转一圈", [kick("blue"), kick("red"), turn()]),
        ("先走到黄球旁边，再踢蓝球，再原地转一圈", [navigate("yellow"), kick("blue"), turn()]),
        ("先点头，再坐下，再站起来", [perform("nod"), posture("sitting"), posture("standing")]),
        ("走到黄球旁边", [navigate("yellow")]),
        ("坐下", [posture("sitting")]),
        ("能不能给我做一下前滚翻？", [perform("roll")]),
        ("点头三次", [perform("nod", 3)]),
        ("先俯身再起，然后左右看", [perform("ground_pick"), perform("look_around")]),
        ("往前走一点", [move()]),
        ("一直往前走不要停", [move(continuous=True)]),
        ("向后走两秒", [move("backward", 2)]),
        ("向左横移", [move("left")]),
        ("向右走", [move("right")]),
        ("向左转", [turn("left", 30)]),
        ("向右转九十度", [turn("right", 90)]),
        ("原地转半圈", [turn(angle=180)]),
        ("把红球踢开", [kick("red")]),
        ("先张嘴，再闭嘴", [perform("open_beak"), perform("close_beak")]),
        ("先环顾四周，再左右摇摆", [perform("look_around"), perform("body_sway")]),
        ("做一下月球漫步", [perform("moonwalk")]),
        ("跳个舞", [perform("groove")]),
        ("蹲下再站起来", [posture("sitting"), posture("standing")]),
        ("先俯身再起，再害羞舞步", [perform("ground_pick"), perform("caught_dancing")]),
        ("先抬左脚，再抬右脚", [perform("flamingo_left"), perform("flamingo_right")]),
    ],
)
def test_exact_goal_semantics_and_queue_order(text, goals):
    output = decide(text)
    assert output["decision"] == "replace"
    assert output["goals"] == goals
    assert "已经" not in output["reply"] and "完成" not in output["reply"]


@pytest.mark.parametrize("text", ["先点头，再后空翻", "先踢红球，再叼起蓝球", "先跑步，再坐下"])
def test_unsupported_tail_rejects_whole_plan(text):
    output = decide(text)
    assert output["decision"] == "decline"
    assert not output["goals"]


@pytest.mark.parametrize("text", ["先点头再做个未知动作", "先一直前进再点头", "踢球", "向左转八圈", "向前走一米"])
def test_ambiguous_or_unreachable_queue_never_executes_prefix(text):
    output = decide(text)
    assert output["decision"] in ("clarify", "decline")
    assert output["goals"] == []


@pytest.mark.parametrize(
    "text", ["你好，你叫什么", "你都会做什么", "你会前滚翻吗", "如果我让你停下呢", "昨天你踢了红球"]
)
def test_chat_and_nonimperative_statements_never_create_goals(text):
    output = decide(text)
    assert output["decision"] == "respond"
    assert output["goals"] == []


@pytest.mark.parametrize(
    "text",
    [
        "停",
        "停下",
        "停！",
        "请停止",
        "立即停止当前任务",
        "别再走了",
        "先停下，再描述画面",
        "停一下",
        "先别继续了",
        "不用做了",
        "小鸭，快停下来",
        "请你现在停止执行",
        "不要再执行了",
        "别继续下去了",
        "算了",
        "够了",
    ],
)
def test_stop_fast_path_and_teacher_agree(text):
    assert is_cancel(text)
    assert not is_pause(text)
    output = decide(text)
    assert output["decision"] == "cancel"
    assert "已经停" not in output["reply"]


@pytest.mark.parametrize(
    "text",
    [
        "一直往前走不要停",
        "不要暂停",
        "如果我说停下",
        "我刚才说停下",
        "‘停下’是什么意思",
        "不停",
        "不要停",
        "别停下",
        "如果我说不用做了呢",
        "刚才我说先别继续了",
        "不要取消任务",
    ],
)
def test_stop_detector_never_matches_negation_or_quotation(text):
    assert not is_cancel(text)
    assert not is_pause(text)


def test_pause_resume_and_cancellation_are_distinct():
    assert is_pause("请暂停一下")

    def running(state):
        return state.update(execution=active(move(continuous=True)))

    def paused(state):
        return state.update(execution=active(move(continuous=True), paused=True))

    assert decide("暂停", mutate=running)["decision"] == "pause"
    assert decide("继续", mutate=paused)["decision"] == "resume"
    assert decide("继续")["decision"] == "clarify"


@pytest.mark.parametrize("goal", [perform("roll"), kick("red"), posture("sitting"), move(seconds=1)])
def test_bounded_goals_support_pause_at_safe_boundary_and_resume_remaining_goal(goal):
    def running(state):
        state.update(execution=active(goal))

    def paused(state):
        state.update(execution=active(goal, paused=True))

    response = decide("暂停", mutate=running)
    assert response["decision"] == "pause" and "恢复稳定后" in response["reply"]
    assert decide("继续", mutate=paused)["decision"] == "resume"


def test_append_keeps_existing_queue_but_refuses_after_continuous_move():
    def running(state):
        return state.update(execution=active(perform("nod")))

    output = decide("然后抬左脚", mutate=running)
    assert output["decision"] == "append"
    assert output["goals"] == [perform("flamingo_left")]

    def endless(state):
        return state.update(execution=active(move(continuous=True)))

    assert decide("然后点头", mutate=endless)["decision"] == "decline"


@pytest.mark.parametrize("count", [7, 12])
def test_explicit_long_sequences_are_preserved_without_a_fixed_cap(count):
    output = decide("、".join(["点头"] * count))
    assert output["decision"] == "replace"
    assert output["goals"] == [perform("nod")] * count


@pytest.mark.parametrize("max_goals", [None, 6, 12])
def test_append_after_six_respects_only_explicit_capacity(max_goals):
    def running(state):
        state["capabilities"]["max_goals"] = max_goals
        state["execution"] = active(perform("nod"), pending=[perform("nod")] * 5)

    output = decide("追加闭嘴", mutate=running)
    if max_goals == 6:
        assert output["decision"] == "decline" and output["goals"] == []
    else:
        assert output["decision"] == "append"
        assert output["goals"] == [perform("close_beak")]


def test_visual_chat_describes_only_current_supported_observations_and_keeps_task():
    def setup(state):
        state["execution"] = active(move(continuous=True))
        left = observed_ball("red")
        left["position_body_m"] = [0.5, 0.2, 0.03]
        right = observed_ball("blue")
        right["position_body_m"] = [0.4, -0.2, 0.03]
        state["observation"]["entities"] = [left, right]

    output = decide("你现在在能看到什么", mutate=setup)
    assert output["decision"] == "respond" and not output["goals"]
    assert "红球在左前方" in output["reply"]
    assert "蓝球在右前方" in output["reply"]
    assert "黄球在" not in output["reply"]
    assert "只可靠识别" in output["reply"]


def test_visual_empty_stale_and_specific_target_queries_do_not_hallucinate():
    output = decide("你能看到什么")
    assert "不表示它们不存在" in output["reply"]

    def stale(state):
        return state["observation"].update(age_s=2)

    assert "已经过期" in decide("描述一下画面", mutate=stale)["reply"]

    def blue(state):
        return state["observation"].update(entities=[observed_ball("blue")])

    red_output = decide("看得到红球吗", mutate=blue)
    assert "没有可靠检测到红球" in red_output["reply"]
    assert "蓝球在" not in red_output["reply"]


def test_current_capabilities_gate_whole_queue():
    def setup(state):
        state["capabilities"]["goal_types"].remove("turn")

    output = decide("先点头再原地转一圈", mutate=setup)
    assert output["decision"] == "decline" and not output["goals"]


def test_sitting_needs_an_explicit_standing_transition_before_motion():
    def sitting(state):
        return state["observation"].update(posture="sitting")

    assert decide("点头", mutate=sitting)["decision"] == "decline"
    assert decide("先站起来，再点头", mutate=sitting)["goals"] == [posture("standing"), perform("nod")]


def test_referential_target_uses_public_history_but_not_hidden_world_state():
    def setup(state):
        state["history"] = [{"role": "user", "text": "我喜欢蓝球"}]

    assert decide("踢它", mutate=setup)["goals"] == [kick("blue")]
    assert decide("踢它")["decision"] == "clarify"


@pytest.mark.parametrize("success", [True, False])
def test_feedback_reports_only_verified_goal_result_not_entire_pending_plan(success):
    def setup(state):
        state["event"]["kind"] = "feedback"
        state["execution"] = active(kick("red"), pending=[turn()])
        state["execution"]["last_result"] = {
            "goal": kick("blue"),
            "status": "succeeded" if success else "failed",
            "reason": "verified displacement" if success else "target_lost",
            "verified": success,
        }

    output = decide("goal_result", mutate=setup)
    assert output["decision"] == "respond" and not output["goals"]
    assert "都完成" not in output["reply"] and "整组完成" not in output["reply"]
    assert ("通过执行验收" in output["reply"]) is success


def test_real_teacher_has_no_fixture_lookup_or_vlm_claim():
    import inspect

    from . import toy_teacher

    assert "toy_fixtures" not in inspect.getsource(toy_teacher)
    assert ToyTeacher.metadata["model_inference"] is False


@pytest.mark.parametrize("text", ["踢红球吗", "前滚翻吗", "转一圈么"])
def test_ambiguous_question_does_not_become_imperative_when_suffix_removed(text):
    output = decide(text)
    assert output["decision"] == "clarify" and not output["goals"]


def test_feedback_supports_valid_entity_only_targets():
    def setup(state):
        state["event"]["kind"] = "feedback"
        state["execution"]["last_result"] = {
            "goal": {"type": "kick", "target": {"category": "ball", "attributes": [], "entity_id": "observed:1"}},
            "status": "succeeded",
            "reason": "verified displacement",
            "verified": True,
        }

    assert "指定的球" in decide("goal_result", mutate=setup)["reply"]


SELF_ACTION_UTTERANCES = {
    "nod": "点个头",
    "look_around": "环顾四周",
    "sit_stand": "做个坐站动作",
    "ground_pick": "俯身再起",
    "roll": "做一下前滚翻",
    "flamingo_left": "抬起左脚",
    "flamingo_right": "抬起右脚",
    "moonwalk": "月球漫步",
    "body_sway": "左右摇摆",
    "open_beak": "张开嘴",
    "close_beak": "合上嘴",
    "groove": "跳个舞",
    "caught_dancing": "害羞舞步",
}


@pytest.mark.parametrize(("action", "text"), SELF_ACTION_UTTERANCES.items())
def test_all_thirteen_self_actions_have_natural_command_coverage(action, text):
    assert set(SELF_ACTION_UTTERANCES) == set(SELF_ACTIONS)
    output = decide(text)
    assert output["decision"] == "replace" and output["goals"] == [perform(action)]


def test_single_leg_stance_distinguishes_support_foot_from_lifted_foot():
    assert decide("用左脚单脚站立")["goals"] == [perform("flamingo_right")]
    assert decide("用右脚单脚站立")["goals"] == [perform("flamingo_left")]
    ambiguous = decide("先向左单脚站立，再向右单脚站立")
    assert ambiguous["decision"] == "clarify" and not ambiguous["goals"]


def test_crouch_is_a_persistent_low_posture_not_an_automatic_sit_stand_cycle():
    assert decide("蹲下")["goals"] == [posture("sitting")]


@pytest.mark.parametrize("text", ["hello", "HELLO!", "Hi", "hey", "Thanks", "thank you", "多谢", "bye"])
def test_common_short_chat_has_a_reply_and_no_action(text):
    output = decide(text)
    assert output["decision"] == "respond" and output["goals"] == []


def test_ordering_connectors_keep_all_six_typed_goals_in_order():
    output = decide("请先点点头，随后张开鸭嘴，接下来合上嘴，之后坐下，完成后站起来，最后左右转头")
    assert output["decision"] == "replace"
    assert output["goals"] == [
        perform("nod"),
        perform("open_beak"),
        perform("close_beak"),
        posture("sitting"),
        posture("standing"),
        perform("look_around"),
    ]


@pytest.mark.parametrize("text", ["你有什么技能", "支持哪些动作", "你会抬高左脚吗", "你会跳害羞舞吗"])
def test_capability_questions_with_new_aliases_never_execute(text):
    output = decide(text)
    assert output["decision"] == "respond" and output["goals"] == []


def test_unknown_or_unsupported_tail_after_new_connectors_never_executes_prefix():
    assert decide("先点头，随后叼起红球")["decision"] == "decline"
    assert decide("先点头，接下来做个未知动作")["decision"] == "clarify"
