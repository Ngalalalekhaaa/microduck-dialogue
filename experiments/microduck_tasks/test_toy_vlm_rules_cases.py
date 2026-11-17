"""The comparison cases are fixed labels, with explicit coverage and provenance boundaries."""

import copy
import json

import pytest

from .toy_catalog import SELF_ACTIONS
from .toy_fixtures import perform, request, turn
from .toy_inputs import validate_input
from .toy_protocol import VERSION, validate_output
from .toy_teacher import is_cancel, is_pause
from .toy_vlm_rules_cases import VERSION_CASES, cases


def test_predeclared_coverage_and_contract():
    rows = cases()
    assert VERSION_CASES == "toy-vlm-rules-ablation-2"
    assert len(rows) == 32 and len({row["id"] for row in rows}) == len(rows)
    actions = {goal["action"] for row in rows for goal in row["expected"]["goals"] if goal["type"] == "perform"}
    assert actions == set(SELF_ACTIONS)
    assert {2, 3, 6} <= {len(row["expected"]["goals"]) for row in rows}
    tags = {tag for row in rows for tag in row["tags"]}
    assert {
        "first_turn",
        "repeat",
        "colors",
        "angle",
        "capacity",
        "pause",
        "resume",
        "ambiguity",
        "visual_chat",
    } <= tags
    for row in rows:
        validate_input(row["request"])
        validate_output(dict(schema_version=VERSION, reply="Expected", **row["expected"]), row["request"])
        assert not row["training_ready"] and not row["physical_validation"]
        assert "expected" not in row["request"] and "acceptable_goals" not in row["request"]
        if "first_turn" in row["tags"]:
            assert row["request"]["history"] == []


def test_recorded_observation_is_preserved_except_declared_counterfactual_fields():
    observation = request("test", planned=True)["observation"]
    original = copy.deepcopy(observation)
    rows = cases(observation)
    assert observation == original
    for row in rows:
        assert row["visual_source"] == "recorded_observation"
        changed = row["request"]["observation"]
        for key in observation:
            if changed[key] != observation[key]:
                assert f"observation.{key}" in row["counterfactual_fields"]
        assert changed["entities"] == original["entities"]
    seated = next(row for row in rows if row["id"] == "stand_before_turn")
    assert seated["request"]["observation"]["posture"] == "sitting"
    assert "observation.posture" in seated["counterfactual_fields"]
    assert seated["request"]["observation"] is not observation


def test_case_generation_is_deterministic_and_independent():
    original = cases()
    encoded = json.dumps(original, ensure_ascii=False, sort_keys=True)
    assert json.dumps(cases(), ensure_ascii=False, sort_keys=True) == encoded
    original[0]["expected"]["goals"][0]["action"] = "roll"
    original[0]["request"]["capabilities"]["self_actions"].clear()
    assert json.dumps(cases(), ensure_ascii=False, sort_keys=True) == encoded


def test_circle_equivalence_is_explicit_and_only_for_unspecified_direction():
    rows = cases()
    circle = next(row for row in rows if row["id"] == "unspecified_circle")
    assert circle["expected"]["goals"] == [turn("left", 360)]
    assert circle["acceptable_goals"] == [[turn("right", 360)]]
    assert all(not row["acceptable_goals"] for row in rows if row["id"] != "unspecified_circle")
    for alternative in circle["acceptable_goals"]:
        validate_output(
            dict(schema_version=VERSION, reply="Expected", decision="replace", goals=alternative), circle["request"]
        )
    assert [turn("right", 30)] not in circle["acceptable_goals"]


def test_two_control_cases_are_explicitly_host_paths():
    rows = cases()
    controls = [row for row in rows if "host_control" in row["tags"]]
    assert len(controls) == 2
    for row in rows:
        text = row["request"]["event"]["text"]
        assert bool(is_cancel(text) or is_pause(text)) == (row in controls)


@pytest.mark.parametrize("case_id", ["append_after_unbounded", "unreachable_new_plan"])
def test_unreachable_requests_do_not_allow_partial_motion(case_id):
    row = next(row for row in cases() if row["id"] == case_id)
    assert set(row["acceptable_decisions"]) == {"clarify", "decline"}
    assert row["expected"]["goals"] == []
    assert row["acceptable_goals"] == []


def test_six_existing_goals_do_not_block_an_append():
    row = next(row for row in cases() if row["id"] == "append_after_six")
    state = row["request"]
    assert state["capabilities"]["max_goals"] == 15
    assert state["execution"]["active"] is not None
    assert len(state["execution"]["pending"]) == 5
    assert row["expected"] == dict(decision="append", goals=[perform("nod")])
    output = dict(schema_version=VERSION, reply="Expected", **row["expected"])
    validate_output(output, state)
    state["capabilities"]["max_goals"] = 6
    with pytest.raises(ValueError):
        validate_output(output, state)


def test_label_sequences_preserve_requested_number_color_and_order():
    rows = {row["id"]: row for row in cases()}
    repeats = rows["repeat_three_then_two"]["expected"]["goals"]
    assert [(goal["action"], goal["repeat"]) for goal in repeats] == [("nod", 3), ("look_around", 2)]
    colors = rows["yellow_blue_red"]["expected"]["goals"]
    assert [goal["target"]["attributes"][0]["value"] for goal in colors] == ["yellow", "blue", "red"]
    assert rows["unsupported_whole_plan"]["expected"] == dict(decision="decline", goals=[])
    assert rows["ambiguous_two_referents"]["expected"] == dict(decision="clarify", goals=[])
