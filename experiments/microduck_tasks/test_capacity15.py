"""Current plans and unfinished queues stop at fifteen without truncation."""

import copy
import json
from types import SimpleNamespace

import pytest

from .ablation_cases import build_cases
from .ablation_contract import plan_schema, validate_context, wrap_plan
from .ablation_grammar import grammar_schema
from .ablation_prompt import PARENTS, messages_for
from .toy_fixtures import active, perform, request
from .toy_inputs import validate_input
from .toy_live_inputs import capabilities
from .toy_protocol import DEFAULT_MAX_GOALS, validate_output
from .toy_schema import response_schema
from .toy_teacher import ToyTeacher


def plan(count, decision="replace"):
    return wrap_plan(dict(decision=decision, goals=[perform("nod") for _ in range(count)]))


def test_current_defaults_are_fifteen_and_legacy_fixture_stays_single_goal():
    assert DEFAULT_MAX_GOALS == 15
    assert capabilities(SimpleNamespace())["max_goals"] == 15
    assert request("点头", planned=True)["capabilities"]["max_goals"] == 15
    assert request("点头")["capabilities"]["max_goals"] == 1
    assert validate_output(plan(15))["goals"] == plan(15)["goals"]
    with pytest.raises(ValueError, match="at most 15"):
        validate_output(plan(16))
    assert len(validate_output(plan(20), max_goals=None)["goals"]) == 20


@pytest.mark.parametrize("count", [7, 14, 15, 16, 20])
def test_current_response_and_rule_teacher_accept_fifteen_and_decline_sixteen(count):
    state = request("、".join(["点头"] * count), planned=True)
    output, original = plan(count), copy.deepcopy(state)
    if count <= 15:
        assert validate_output(output, state) == output
    else:
        with pytest.raises(ValueError, match="cannot accept"):
            validate_output(output, state)
    predicted = ToyTeacher().decide(state)
    assert predicted["decision"] == ("replace" if count <= 15 else "decline")
    assert predicted["goals"] == (output["goals"] if count <= 15 else [])
    assert len(output["goals"]) == count and state == original


@pytest.mark.parametrize("existing,new", [(14, 1), (13, 2), (15, 1), (14, 2)])
@pytest.mark.parametrize("paused", [False, True])
def test_append_counts_active_pending_and_new_without_truncation(existing, new, paused):
    state = request("然后" + "、".join(["点头"] * new), planned=True)
    state["execution"] = active(perform("nod"), pending=plan(existing - 1)["goals"], paused=paused)
    output = plan(new, "append")
    original = copy.deepcopy((state, output))
    if existing + new <= 15:
        assert validate_output(output, state) == output
    else:
        with pytest.raises(ValueError, match="Combined queue"):
            validate_output(output, state)
    predicted = ToyTeacher().decide(state)
    assert predicted["decision"] == ("append" if existing + new <= 15 else "decline")
    assert predicted["goals"] == (output["goals"] if existing + new <= 15 else [])
    assert (state, output) == original


@pytest.mark.parametrize("paused", [False, True])
def test_existing_queue_snapshot_cannot_exceed_current_capacity(paused):
    state = request("当前任务如何", planned=True)
    state["execution"] = active(perform("nod"), pending=plan(14)["goals"], paused=paused)
    assert validate_input(state) == state
    state["execution"]["pending"].append(perform("nod"))
    with pytest.raises(ValueError, match="Existing active and pending queue"):
        validate_input(state)


@pytest.mark.parametrize("factory", [response_schema, plan_schema, grammar_schema])
@pytest.mark.parametrize("decision", ["replace", "append"])
def test_current_json_schema_accepts_fifteen_rejects_sixteen(factory, decision):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(factory())
    for count in (15, 16):
        value = plan(count, decision)
        if factory is not response_schema:
            value = {key: value[key] for key in ("decision", "goals")}
        assert validator.is_valid(value) == (count == 15)


@pytest.mark.parametrize("factory", [response_schema, plan_schema, grammar_schema])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.0, "15"])
def test_schema_capacity_rejects_invalid_overrides(factory, invalid):
    with pytest.raises(ValueError, match="capacity"):
        factory(max_goals=invalid)


@pytest.mark.parametrize("variant", PARENTS)
def test_current_prompt_and_schema_both_state_fifteen(variant):
    state = request("点头", planned=True)
    system = messages_for(state, variant)[0]["content"]
    assert "当前默认最多15个未完成目标" in system and "不能静默截断" in system
    schema = json.loads(system.split("\nOUTPUT_JSON_SCHEMA=", 1)[1])
    planning = (schema.get("oneOf") or [schema])[0]
    assert planning["properties"]["goals"]["maxItems"] == 15


def test_development_cases_include_plan_and_combined_queue_boundaries():
    cases = {row["id"]: row for row in build_cases(request("样本", planned=True)["observation"])}
    for case_id, count in (("boundary_13", 15), ("boundary_21", 16), ("boundary_29", 15), ("boundary_37", 16)):
        row = cases[case_id]
        assert row["expected"]["decision"] == ("replace" if count == 15 else "clarify")
        assert len(row["expected"]["goals"]) == (15 if count == 15 else 0)
        assert f"共{count}项" in row["request"]["event"]["text"]
    for case_id, total in (("state_19", 15), ("state_29", 16), ("state_39", 15)):
        row = cases[case_id]
        existing = 1 + len(row["request"]["execution"]["pending"])
        assert existing + 1 == total
        assert row["expected"]["decision"] == ("append" if total == 15 else "clarify")
        assert len(row["expected"]["goals"]) == (1 if total == 15 else 0)


@pytest.mark.parametrize("decision", ["pause", "resume"])
def test_context_allows_safe_pause_and_resume_of_fifteen_finite_goals(decision):
    state = request("暂停" if decision == "pause" else "继续", planned=True)
    state["execution"] = active(perform("nod"), pending=plan(14)["goals"], paused=decision == "resume")
    response = plan(0, decision)
    assert validate_context(response, state) == response
