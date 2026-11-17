"""Historical unbounded snapshots preserve contextual capacity and old-head boundaries."""

import copy
import json

import pytest

from . import ablation_prompt, toy_vlm_prompt, toy_vlm_prompt_compact, toy_vlm_prompt_rules
from .ablation_contract import common_domain, plan_schema, validate_context, wrap_plan
from .toy_fixtures import active, perform, request
from .toy_protocol import DECISIONS
from .toy_schema import SYSTEM, response_schema

PROMPTS = (toy_vlm_prompt, toy_vlm_prompt_compact, toy_vlm_prompt_rules)


@pytest.mark.parametrize("schema_factory", [response_schema, plan_schema])
@pytest.mark.parametrize("decision", ["replace", "append"])
@pytest.mark.parametrize("count", [1, 7, 20, 100])
def test_actual_json_schema_accepts_complete_long_plans(schema_factory, decision, count):
    jsonschema = pytest.importorskip("jsonschema")
    schema = schema_factory(max_goals=None)
    jsonschema.Draft202012Validator.check_schema(schema)
    plan = dict(decision=decision, goals=[perform("nod") for _ in range(count)])
    output = wrap_plan(plan) if schema_factory is response_schema else plan
    jsonschema.Draft202012Validator(schema).validate(output)
    assert len(output["goals"]) == count


@pytest.mark.parametrize("schema_factory", [response_schema, plan_schema])
@pytest.mark.parametrize("decision", DECISIONS)
def test_actual_json_schema_retains_empty_and_nonempty_decision_contract(schema_factory, decision):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(schema_factory())
    planning = decision in ("replace", "append")
    for populated in (False, True):
        plan = dict(decision=decision, goals=[perform("nod")] if populated else [])
        output = wrap_plan(plan) if schema_factory is response_schema else plan
        assert validator.is_valid(output) == (planning == populated)


def test_explicit_null_planning_and_grammar_schema_have_no_numeric_goal_ceiling():
    from .ablation_grammar import grammar_schema

    assert "maxItems" not in response_schema(max_goals=None)["properties"]["goals"]
    for schema in (plan_schema(max_goals=None), grammar_schema(max_goals=None)):
        planning, nonplanning = schema["oneOf"]
        assert "maxItems" not in planning["properties"]["goals"]
        assert planning["properties"]["goals"]["minItems"] == 1
        assert nonplanning["properties"]["goals"]["maxItems"] == 0


@pytest.mark.parametrize("schema_factory", [response_schema, plan_schema])
def test_long_schema_still_validates_late_goal_parameters(schema_factory):
    jsonschema = pytest.importorskip("jsonschema")
    plan = dict(decision="replace", goals=[perform("nod") for _ in range(20)])
    plan["goals"][-1]["repeat"] = 4
    output = wrap_plan(plan) if schema_factory is response_schema else plan
    assert not jsonschema.Draft202012Validator(schema_factory(max_goals=None)).is_valid(output)


@pytest.mark.parametrize("capacity,count", [(None, 100), (6, 6), (7, 7), (20, 20)])
def test_plan_context_honors_null_or_explicit_integer_capacity(capacity, count):
    state = request("按顺序执行全部动作", planned=True)
    state["capabilities"]["max_goals"] = capacity
    plan = dict(decision="replace", goals=[perform("nod") for _ in range(count)])
    assert validate_context(wrap_plan(plan), state)["goals"] == plan["goals"]
    if capacity is not None:
        plan["goals"].append(perform("nod"))
        with pytest.raises(ValueError, match="cannot accept"):
            validate_context(wrap_plan(plan), state)


@pytest.mark.parametrize("capacity,accepted", [(None, True), (9, True), (8, False)])
def test_plan_context_counts_active_pending_and_new_only_for_explicit_capacity(capacity, accepted):
    state = request("在后面追加四次点头", planned=True)
    state["capabilities"]["max_goals"] = capacity
    state["execution"] = active(perform("nod"))
    state["execution"]["pending"] = [perform("nod") for _ in range(4)]
    output = wrap_plan(dict(decision="append", goals=[perform("nod") for _ in range(4)]))
    if accepted:
        assert validate_context(output, state) == output
    else:
        with pytest.raises(ValueError, match="Combined queue"):
            validate_context(output, state)


def test_old_selector_domain_does_not_claim_new_long_plan_support():
    assert common_domain([perform("nod") for _ in range(6)])
    assert not common_domain([perform("nod") for _ in range(7)])
    assert not common_domain([perform("nod") for _ in range(100)])


def test_old_selector_training_reports_unsupported_length_without_truncating():
    from .toy_selector_labels import encode_targets

    goals = [perform("nod") for _ in range(7)]
    row = dict(request=request("完成七项目标", planned=True), target=wrap_plan(dict(decision="replace", goals=goals)))
    original = copy.deepcopy(row)
    with pytest.raises(ValueError, match="use a generative planner"):
        encode_targets([row])
    assert row == original


def test_all_prompt_variants_explain_null_explicit_capacity_and_no_truncation():
    texts = [SYSTEM, *(module.system_prompt() for module in PROMPTS)]
    texts.extend(ablation_prompt.system_prompt(variant) for variant in ablation_prompt.PARENTS)
    for text in texts:
        assert "capabilities.max_goals=null" in text
        assert "当前默认最多15个未完成目标" in text
        assert "不设固定目标数量上限" in text
        assert "正整数" in text and "active+pending+新增" in text
        assert "不能静默截断" in text
        assert "持续" in text and "追加" in text
        for legacy in ("最多六个", "超过六个", "min(6,", "1至6个"):
            assert legacy not in text


@pytest.mark.parametrize("capacity", [None, 6, 20])
def test_renderers_preserve_current_and_historical_capacity_snapshot(capacity):
    state = request("依次完成所有目标", planned=True)
    state["capabilities"]["max_goals"] = capacity
    original = copy.deepcopy(state)
    messages = [module.messages_for(state, vision=False) for module in PROMPTS]
    messages.extend(ablation_prompt.messages_for(state, variant) for variant in ablation_prompt.PARENTS)
    expected = '"max_goals":' + json.dumps(capacity)
    for rendered in messages:
        text = "\n".join(item["text"] for item in rendered[1]["content"] if item["type"] == "text")
        assert expected in text
        schema = json.loads(rendered[0]["content"].split("\nOUTPUT_JSON_SCHEMA=", 1)[1])
        goals = (schema.get("oneOf") or [schema])[0]["properties"]["goals"]
        assert goals.get("maxItems") == capacity
    assert state == original
