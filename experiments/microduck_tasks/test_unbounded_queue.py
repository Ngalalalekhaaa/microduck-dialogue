"""Explicit historical capacity snapshots retain compatibility with current limits."""

import copy
from types import SimpleNamespace

import pytest

from .toy_fixtures import active, move, perform, request, turn
from .toy_inputs import model_payload, validate_input
from .toy_live_inputs import capabilities
from .toy_protocol import VERSION, sequence, validate_output


def _goals(count):
    return [turn("left" if index % 2 else "right", angle=15 + index) for index in range(count)]


def _response(goals=(), *, decision="replace"):
    return dict(schema_version=VERSION, reply="好，我按顺序执行。", decision=decision, goals=list(goals))


@pytest.mark.parametrize("count", [7, 20, 100])
def test_long_response_retains_every_goal_with_or_without_request(count):
    state = request("按顺序完成所有任务", planned=True)
    state["capabilities"]["max_goals"] = None
    value = _response(_goals(count))
    before = copy.deepcopy((state, value))
    assert validate_output(value, max_goals=None) == value
    checked = validate_output(value, state)
    assert checked == value and len(checked["goals"]) == count
    assert (state, value) == before
    checked["goals"][-1]["angle_deg"] = 360
    assert value == before[1]


@pytest.mark.parametrize("count", [6, 19, 99])
def test_input_and_model_payload_preserve_pending_queue_longer_than_five(count):
    state = request("队列现在如何", planned=True)
    state["capabilities"]["max_goals"] = None
    state["execution"] = active(perform("nod"), pending=_goals(count))
    before = copy.deepcopy(state)
    assert validate_input(state) == before
    payload = model_payload(state)
    assert payload["execution"]["pending"] == before["execution"]["pending"]
    assert len(payload["execution"]["pending"]) == count
    assert state == before


@pytest.mark.parametrize("capacity", [None, 1, 6, 7, 20, 100, 1_000_000])
def test_capacity_accepts_null_or_any_positive_integer(capacity):
    state = request("点头", planned=True)
    state["capabilities"]["max_goals"] = capacity
    assert validate_input(state)["capabilities"]["max_goals"] == capacity
    assert validate_output(_response([perform("nod")]), state)["decision"] == "replace"


@pytest.mark.parametrize("capacity", [True, False, 0, -1, -100, 1.0, "6", [], {}, float("nan"), float("inf")])
def test_capacity_rejects_boolean_zero_negative_and_noninteger_values(capacity):
    state = request("点头", planned=True)
    state["capabilities"]["max_goals"] = capacity
    with pytest.raises(ValueError, match="capacity"):
        validate_input(state)


def test_current_runtime_and_planned_fixture_limit_fifteen_but_legacy_fixture_keeps_one():
    assert capabilities(SimpleNamespace())["max_goals"] == 15
    assert request("任务", planned=True)["capabilities"]["max_goals"] == 15
    assert request("任务")["capabilities"]["max_goals"] == 1


@pytest.mark.parametrize("capacity", [6, 20, 99])
def test_explicit_capacity_accepts_exact_boundary_and_rejects_next_goal(capacity):
    state = request("执行队列", planned=True)
    state["capabilities"]["max_goals"] = capacity
    exact = _response(_goals(capacity))
    assert validate_output(exact, state) == exact
    with pytest.raises(ValueError, match="many goals"):
        validate_output(_response(_goals(capacity + 1)), state)


@pytest.mark.parametrize("capacity", [6, 20, 100])
def test_append_checks_combined_capacity_including_active_and_pending(capacity):
    state = request("然后点头", planned=True)
    state["capabilities"]["max_goals"] = capacity
    state["execution"] = active(perform("nod"), pending=_goals(capacity - 2))
    value = _response([perform("look_around")], decision="append")
    assert validate_output(value, state) == value
    state["execution"]["pending"].append(perform("open_beak"))
    before = copy.deepcopy((state, value))
    with pytest.raises(ValueError, match="Combined queue"):
        validate_output(value, state)
    assert (state, value) == before


def test_null_capacity_allows_append_after_six_existing_goals_without_mutation():
    state = request("追加全部任务", planned=True)
    state["capabilities"]["max_goals"] = None
    state["execution"] = active(perform("nod"), pending=_goals(5))
    value = _response(_goals(100), decision="append")
    before = copy.deepcopy((state, value))
    assert validate_output(value, state) == value
    assert (state, value) == before


def test_unlimited_sequence_still_requires_list_and_bounded_sequences_keep_their_limit():
    sequence(list(range(100)), None, "queue")
    sequence([], None, "queue")
    for invalid in (None, tuple(range(7)), {"goals": []}, "1234567"):
        with pytest.raises(ValueError, match="list"):
            sequence(invalid, None, "queue")
    with pytest.raises(ValueError, match="at most 6"):
        sequence(list(range(7)), 6, "queue")


@pytest.mark.parametrize("count", [7, 20, 100])
def test_illegal_tail_of_long_queue_rejects_whole_response_without_truncating(count):
    state = request("执行全部任务", planned=True)
    value = _response(_goals(count - 1) + [perform("unsupported_action")])
    before = copy.deepcopy((state, value))
    with pytest.raises(ValueError, match="outside"):
        validate_output(value, state)
    assert (state, value) == before


def test_continuous_motion_may_end_long_queue_but_cannot_hide_unreachable_tail():
    state = request("做完所有动作然后一直走", planned=True)
    state["capabilities"]["max_goals"] = None
    value = _response(_goals(99) + [move(continuous=True)])
    assert validate_output(value, state) == value
    value["goals"].append(perform("nod"))
    with pytest.raises(ValueError, match="must be last"):
        validate_output(value, state)


@pytest.mark.parametrize("paused", [False, True])
def test_append_behind_continuous_tail_of_long_existing_queue_is_rejected(paused):
    state = request("然后点头", planned=True)
    state["capabilities"]["max_goals"] = None
    state["execution"] = active(perform("nod"), paused=paused, pending=_goals(18) + [move(continuous=True)])
    with pytest.raises(ValueError, match="unbounded task"):
        validate_output(_response([perform("nod")], decision="append"), state)


def test_resume_rechecks_goal_after_old_six_goal_boundary():
    state = request("继续", planned=True)
    state["capabilities"]["max_goals"] = None
    state["execution"] = active(perform("nod"), paused=True, pending=_goals(19))
    value = _response(decision="resume")
    assert validate_output(value, state) == value
    state["execution"]["pending"][-1] = perform("flamingo_left")
    state["capabilities"]["self_actions"].remove("flamingo_left")
    with pytest.raises(ValueError, match="currently unavailable"):
        validate_output(value, state)
