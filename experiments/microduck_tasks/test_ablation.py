"""Regression checks for false-start metrics and the predeclared P comparisons."""

import copy
import json
from collections import Counter

import pytest

from .ablation_audit import coverage
from .ablation_cases import VERSION, build_cases
from .ablation_contract import checksum, common_domain, validate_context, wrap_plan
from .ablation_prompt import messages_for
from .ablation_report import details, paired, report
from .ablation_score import score, summarize
from .toy_fixtures import kick, perform, request, turn
from .toy_inputs import model_payload
from .toy_vlm_prompt_compact import messages_for as compact_messages


def test_suite_sizes_labels_domains_and_membership():
    cases = build_cases(request("样本", planned=True)["observation"])
    assert VERSION == "microduck-dev240-v3-cap15"
    assert len(cases) == 240
    assert set(Counter(c["group"] for c in cases).values()) == {40}
    assert set(Counter(c["group"] for c in cases if c["screen"]).values()) == {16}
    assert len({checksum(c["request"]) for c in cases}) == 240
    assert sum(c["domain"] == "full_only" for c in cases) == 7
    for case in cases:
        assert case["request"]["capabilities"]["max_goals"] == 15
        validate_context(wrap_plan(case["expected"]), case["request"])
        if case["domain"] == "common":
            assert common_domain(case["expected"]["goals"])


@pytest.mark.parametrize("max_goals", [None, 6, 7, 12])
def test_capacity_cases_follow_the_explicit_capability_snapshot(max_goals):
    cases = build_cases(request("样本", planned=True)["observation"], max_goals=max_goals)
    permitted = max_goals is None or max_goals >= 7
    assert sum(case["domain"] == "common" for case in cases) == (231 if permitted else 236)
    for case in cases:
        if case["family_id"] == "boundary_mode:5":
            expected = [
                perform(name)
                for name in (
                    "nod",
                    "open_beak",
                    "close_beak",
                    "look_around",
                    "ground_pick",
                    "flamingo_left",
                    "flamingo_right",
                )
            ]
            assert case["expected"] == dict(
                decision="replace" if permitted else "clarify", goals=expected if permitted else []
            )
        elif case["family_id"] == "state_mode:9":
            assert case["expected"] == dict(
                decision="append" if permitted else "clarify", goals=[perform("close_beak")] if permitted else []
            )
        else:
            continue
        assert case["acceptable_decisions"] == ([case["expected"]["decision"]] if permitted else ["clarify", "decline"])
        assert score(json.dumps(case["expected"]), case, plan_only=True)["accepted_exact"]


def test_p0_exact_and_single_factor_edges():
    state = request("先踢红球，再踢蓝球", planned=True)
    assert messages_for(state, "P0") == compact_messages(state)
    assert messages_for(state, "P1")[1] == messages_for(state, "P0")[1]
    assert messages_for(state, "P3")[0] == messages_for(state, "P2")[0]
    compact = messages_for(state, "P3")[1]["content"][-1]["text"]
    assert json.loads(compact) == model_payload(state)
    assert list(json.loads(compact))[-1] == "event"
    assert messages_for(state, "P5") == messages_for(state, "P3")
    assert messages_for(state, "P4")[1] == messages_for(state, "P3")[1]
    assert messages_for(state, "P6")[0] == messages_for(state, "P3")[0]
    assert messages_for(state, "P6")[1]["content"][-1] == messages_for(state, "P3")[1]["content"][-1]


def _case(decision="replace", goals=None):
    return dict(
        request=request("测试", planned=True),
        expected=dict(decision=decision, goals=goals or []),
        acceptable_decisions=[decision],
        group="boundary",
        domain="common",
        context="cold_start",
    )


def test_raw_refusal_never_gets_action_credit_and_order_matters():
    case = _case(goals=[kick("red"), kick("blue")])
    for goals, decision in (
        ([], "decline"),
        ([kick("blue"), kick("red")], "replace"),
        ([kick("red")], "replace"),
        ([kick("red"), kick("blue"), perform("nod")], "replace"),
    ):
        result = score(json.dumps(dict(decision=decision, goals=goals)), case, plan_only=True)
        assert not result["accepted_exact"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"decision":"respond","decision":"replace","goals":[]}',
        '{"decision":"replace","goals":[],"value":NaN}',
        "```json\n{}\n```",
    ],
)
def test_strict_parser_does_not_repair(raw):
    assert not score(raw, _case("respond"), plan_only=True)["json_parsed"]


def test_invalid_attempt_is_counted_even_if_guard_refuses():
    result = score('{"decision":"replace","goals":[]}', _case("respond"), plan_only=True)
    assert result["attempted_false_start"]
    assert not result["accepted_false_start"]
    assert not result["context_valid"]


def test_context_and_head_domain_are_not_interchangeable():
    case = _case(goals=[turn("left", 45)])
    raw = json.dumps(case["expected"])
    assert score(raw, case, plan_only=True)["accepted_exact"]
    assert not common_domain(case["expected"]["goals"])
    case["request"]["observation"]["posture"] = "sitting"
    result = score(raw, case, plan_only=True)
    assert result["schema_valid"] and not result["context_valid"]


@pytest.mark.parametrize("count", [6, 7, 12])
def test_long_goals_are_not_truncated_by_scoring(count):
    goals = [perform("nod"), kick("yellow"), perform("open_beak"), perform("close_beak"), turn(), kick("blue")]
    goals += [perform("look_around")] * (count - 6)
    case = _case(goals=goals)
    assert score(json.dumps(case["expected"]), case, plan_only=True)["accepted_exact"]
    bad = copy.deepcopy(case["expected"])
    bad["goals"].pop()
    assert not score(json.dumps(bad), case, plan_only=True)["accepted_exact"]


def test_screen_and_partial_runs_cannot_pass_prototype_gate():
    case = _case("respond")
    row = dict(**case, timing={}, score=score(json.dumps(case["expected"]), case, plan_only=True))
    assert not summarize([row] * 96, expected_count=96)["prototype_gate_eligible"]
    assert not summarize([row] * 96, expected_count=240)["prototype_gate_pass"]


def test_report_does_not_select_incomplete_candidates(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(case_ids=["unrun"])))
    (tmp_path / "P0").mkdir()
    result = report(tmp_path)
    assert not result["complete_screen"] and result["finalists"] == []


def test_comparison_tracks_fixed_and_regressed_cases_and_host_route():
    case = _case(goals=[perform("nod")])
    correct = dict(
        **case,
        id="case",
        family_id="family",
        host_control="host_pause",
        timing={},
        score=score(json.dumps(case["expected"]), case, plan_only=True),
    )
    wrong = dict(correct, score=score('{"decision":"respond","goals":[]}', case, plan_only=True))
    assert paired([wrong], [correct])["improved_ids"] == ["case"]
    assert paired([correct], [wrong])["regressed_ids"] == ["case"]
    result = details([correct])
    assert result["host_route_count"] == 1
    assert result["model_route_exact"]["total"] == 0
    assert result["host_route_model_exact"]["total"] == 1


def test_semantic_equivalence_never_changes_strict_plan_score():
    from .toy_fixtures import observed_ball

    case = _case(goals=[kick("red")])
    entity = observed_ball("red")
    case["request"]["observation"]["entities"] = [entity]
    prediction = copy.deepcopy(case["expected"])
    prediction["goals"][0]["target"]["entity_id"] = entity["id"]
    result = score(json.dumps(prediction), case, plan_only=True)
    assert result["semantic"] and result["context_valid"] and not result["accepted_exact"]


def test_audit_treats_unparseable_nonaction_as_unknown_not_safe():
    case = _case("respond")
    raw = '{"decision":"clarify","goals":[' + "\t" * 110
    row = dict(
        **case,
        id="cut_off",
        raw_text=raw,
        timing=dict(finish_reason="limit", output_tokens=768),
        score=score(raw, case, plan_only=True),
    )
    result = coverage([row])
    assert result["nonaction_known_attempts"] == 0
    assert result["nonaction_unassessable"] == 1
    assert result["nonaction_attempt_sensitivity_bounds"] == [0.0, 1.0]
    assert result["invalid_trailing_tab_loop"] == 1
