"""Product semantics and data isolation, without pretending fixtures are robot trials."""

import copy
import json
from types import SimpleNamespace

import pytest

from .build_toy_casebook import build, check_case, require_training_ready
from .toy_catalog import CANDIDATES, INTERACTIONS, SELF_ACTIONS, SHOWCASE
from .toy_context_examples import examples as context_examples
from .toy_examples import examples
from .toy_fixtures import active, kick, move, observed_ball, perform, posture, request, turn
from .toy_protocol import VERSION, validate_output


def response(*goals, decision="replace", reply="好，我按顺序执行。"):
    return {"schema_version": VERSION, "reply": reply, "decision": decision, "goals": list(goals)}


def test_candidate_scope_excludes_other_interactions_and_unstable_actions():
    assert INTERACTIONS == ("kick",)
    assert {"pick_up", "carry", "place", "push_block", "throw", "run", "hop", "happy_hop", "pirouette"}.isdisjoint(
        CANDIDATES
    )
    assert set(SHOWCASE) <= set(CANDIDATES)
    assert "roll" in SELF_ACTIONS and "sit_stand" in SELF_ACTIONS and "sit" not in SELF_ACTIONS


def test_toy_profile_refuses_research_before_any_physics(tmp_path):
    from .action_session import ActionSession

    session = ActionSession(SimpleNamespace(output=tmp_path))
    for command in ("叼起来", "推方块", "轻跳", "左脚踢球"):
        assert session.run(command)["outcome"] == "outside_toy_candidates"


def test_all_design_cases_are_valid_and_can_never_be_promoted():
    rows = examples() + context_examples()
    assert len(rows) > 150
    assert {row["target"]["decision"] for row in rows} == {
        "respond",
        "clarify",
        "decline",
        "replace",
        "append",
        "cancel",
        "pause",
        "resume",
    }
    for row in rows:
        check_case(row)
        if row["family_id"].startswith("visual_"):
            assert row["target"]["decision"] == "respond" and not row["target"]["goals"]
        if row["family_id"] == "visual_running":
            assert row["input"]["execution"]["state"] == "running"
            assert row["input"]["execution"]["active"] == move(continuous=True)
        with pytest.raises(ValueError, match="Draft"):
            require_training_ready(row)
    falsified = copy.deepcopy(rows[0])
    falsified["training_ready"] = True
    with pytest.raises(ValueError, match="relabeled"):
        check_case(falsified)


def test_users_requested_sequence_keeps_order_and_distinct_targets():
    rows = [r for r in examples() if r["family_id"] == "sequence_required"]
    current = next(r for r in rows if r["capability_profile"].startswith("existing"))
    assert current["target"]["decision"] == "decline"
    planned = next(r for r in rows if r["capability_profile"] == "planned_semantics")
    assert planned["target"]["goals"] == [kick("blue"), kick("red"), turn()]
    assert all("target" not in r["target"] for r in rows)  # No single global target.


def test_navigation_never_adds_a_kick_and_sitting_never_means_sit_stand():
    for row in examples():
        goals = row["target"]["goals"]
        if row["family_id"].startswith("navigate_") and goals:
            assert [g["type"] for g in goals] == ["navigate"]
        if row["family_id"] == "sit_hold" and goals:
            assert goals == [posture("sitting")]


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_every_user_turn_requires_a_spoken_response(bad):
    with pytest.raises(ValueError, match="nonempty"):
        validate_output(response(decision="respond", reply=bad))


@pytest.mark.parametrize("decision", ["respond", "clarify", "decline", "cancel", "pause", "resume"])
def test_nonplanning_decisions_cannot_hide_motion(decision):
    with pytest.raises(ValueError, match="Only replace"):
        validate_output(response(kick("red"), decision=decision))


@pytest.mark.parametrize("seconds", [True, float("nan"), float("inf"), -1, 100])
def test_rejects_unsafe_or_unbounded_duration(seconds):
    with pytest.raises(ValueError):
        validate_output(response(move(seconds=seconds)))


def test_unreachable_tail_and_append_behind_continuous_motion_are_rejected():
    with pytest.raises(ValueError, match="last"):
        validate_output(response(move(continuous=True), kick("red")))
    state = request("然后点头", planned=True)
    state["execution"] = active(move(continuous=True))
    with pytest.raises(ValueError, match="unbounded"):
        validate_output(response(perform("nod"), decision="append"), state)


def test_incapable_tail_rejects_the_whole_plan_without_mutating_it():
    state = request("先踢红球再转一圈")
    state["capabilities"]["max_goals"] = 6
    value = response(kick("red"), turn())
    before = copy.deepcopy((state, value))
    with pytest.raises(ValueError, match="not available"):
        validate_output(value, state)
    assert (state, value) == before


def test_sit_hold_requires_capability_and_chat_keeps_posture():
    with pytest.raises(ValueError, match="sitting hold"):
        validate_output(response(posture("sitting")), request("坐下"))
    state = request("往前走", planned=True)
    state["observation"]["posture"] = "sitting"
    with pytest.raises(ValueError, match="standing transition"):
        validate_output(response(move()), state)
    validate_output(response(posture("standing"), move()), state)
    value = response(decision="respond", reply="你好，我叫小鸭。")
    validate_output(value, state)
    assert state["observation"]["posture"] == "sitting" and not value["goals"]


def test_cancelled_queue_cannot_be_resumed_and_resume_rechecks_capabilities():
    state = request("继续", planned=True)
    with pytest.raises(ValueError, match="paused"):
        validate_output(response(decision="resume"), state)
    state["execution"] = active(move(continuous=True), paused=True)
    state["capabilities"]["continuous_move"] = False
    with pytest.raises(ValueError, match="Continuous"):
        validate_output(response(decision="resume"), state)


def test_duplicate_and_contradictory_entity_binding_are_not_silently_chosen():
    state = request("踢红球")
    state["observation"]["entities"] = [observed_ball("red", 0), observed_ball("red", 1)]
    with pytest.raises(ValueError, match="Multiple"):
        validate_output(response(kick("red")), state)
    value = response(kick("blue"))
    value["goals"][0]["target"]["entity_id"] = "observed:red:0"
    with pytest.raises(ValueError, match="contradicts"):
        validate_output(value, state)
    state["observation"]["entities"] = []
    state["capabilities"]["search_targets"] = []
    with pytest.raises(ValueError, match="search capability"):
        validate_output(response(kick("red")), state)


@pytest.mark.parametrize("guard", ["stale", "motion_closed", "desktop", "fallen"])
def test_local_state_gates_motion_but_never_cancellation_or_chat(guard):
    state = request("往前走", planned=True)
    if guard == "stale":
        state["observation"]["age_s"] = 2
    elif guard == "motion_closed":
        state["observation"]["environment"]["motion_allowed"] = False
    elif guard == "desktop":
        state["observation"]["environment"]["surface"] = "desktop"
    else:
        state["observation"]["posture"] = "fallen"
    with pytest.raises(ValueError):
        validate_output(response(move()), state)
    for decision in ("cancel", "respond"):
        validate_output(response(decision=decision), state)


def test_design_export_contains_no_training_splits_or_fake_media(tmp_path):
    output = tmp_path / "review"
    report = build(output)
    assert report["training_ready"] is False and not report["physical_verified"]
    assert not any((output / f"{split}.jsonl").exists() for split in ("train", "val", "test"))
    rows = [json.loads(line) for line in (output / "design_cases.jsonl").read_text().splitlines()]
    assert len(rows) == report["cases"] and all(not row["media"] for row in rows)
    with pytest.raises(FileExistsError):
        build(output)


def test_desktop_head_gesture_does_not_require_locomotion_permission():
    state = request("点头")
    state["observation"]["environment"]["surface"] = "desktop"
    validate_output(response(perform("nod")), state)
    with pytest.raises(ValueError, match="edge guard"):
        validate_output(response(perform("roll")), state)


def test_current_profile_does_not_claim_to_execute_a_multi_goal_queue():
    with pytest.raises(ValueError, match="many goals"):
        validate_output(response(perform("nod"), perform("look_around")), request("点头然后左右看"))


def test_append_respects_the_posture_after_existing_queued_tasks():
    state = request("然后向前走", planned=True)
    state["execution"] = active(posture("sitting"))
    with pytest.raises(ValueError, match="standing transition"):
        validate_output(response(move(), decision="append"), state)
    validate_output(response(posture("standing"), move(), decision="append"), state)
