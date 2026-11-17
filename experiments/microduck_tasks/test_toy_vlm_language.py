"""Diagnostics must preserve input boundaries and distinguish host fallback from model quality."""

import copy
import json

import pytest

from .evaluate_toy_vlm_language import evaluate, score_prediction, source_snapshot
from .test_toy_dataset import trace
from .toy_catalog import SELF_ACTIONS
from .toy_fixtures import observed_ball, request, turn
from .toy_protocol import VERSION, validate_output
from .toy_vlm_language_cases import cases


def test_predeclared_cases_cover_all_actions_and_validate_without_labels_in_request():
    rows = cases()
    assert len(rows) == 20
    actions = {goal["action"] for row in rows for goal in row["expected"]["goals"] if goal["type"] == "perform"}
    assert actions == set(SELF_ACTIONS)
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert "target" not in row["request"] and "expected" not in row["request"]
        validate_output(dict(schema_version=VERSION, reply="Expected", **row["expected"]), row["request"])


def test_behavioral_comparison_preserves_color_order_and_checks_entity_binding():
    row = next(row for row in cases() if row["id"] == "color_order")
    row["request"]["observation"]["entities"] = [observed_ball("red")]
    prediction = copy.deepcopy(row["expected"])
    prediction["goals"][0]["target"]["entity_id"] = "observed:red:0"
    assert score_prediction(prediction, row) == dict(exact=False, behavioral=True)
    prediction["goals"][0]["target"]["entity_id"] = "unknown"
    assert score_prediction(prediction, row)["behavioral"] is False
    prediction = copy.deepcopy(row["expected"])
    prediction["goals"].reverse()
    assert score_prediction(prediction, row)["behavioral"] is False


def test_fallback_decline_does_not_count_as_correct_model_plan(tmp_path):
    source = tmp_path / "source"
    trace(source)

    class Backend:
        name = "test_fake_backend"
        metadata = {"real_model": False}

        def decide_with_images(self, request, images):
            assert set(images) == {"front", "near"}
            assert "expected" not in request and "target" not in request
            self.last_trace = dict(
                model_called=True,
                raw_response=dict(schema_version=VERSION, reply="", decision="respond", goals=[]),
                raw_prediction=dict(decision="respond", goals=[]),
                rejected=dict(reason="empty generated reply"),
            )
            return dict(schema_version=VERSION, reply="Rejected by validation", decision="decline", goals=[])

    result = evaluate(Backend(), source, tmp_path / "report", selected=["unsupported_partial"])
    assert result["summary"]["accepted_response_behaviorally_correct"] == 1
    assert result["summary"]["model_accepted_behaviorally_correct"] == 0
    assert result["summary"]["model_contract_valid"] == 0
    assert result["physical_validation"] is False and result["training_ready"] is False
    assert result["source"]["original_response_used"] is False
    assert (tmp_path / "report" / "cases.json").is_file()


def test_source_refuses_image_paths_outside_recorded_episode(tmp_path):
    source = tmp_path / "source"
    trace(source)
    events_path = source / "events.jsonl"
    event = json.loads(events_path.read_text())
    event["media"]["front"] = "../not_an_episode_image.png"
    events_path.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="escapes"):
        source_snapshot(source)


@pytest.mark.parametrize("variant", [None, "full", "compact", "rules"])
def test_cli_selects_explicit_context_variant_without_loading_model(tmp_path, monkeypatch, variant):
    from . import toy_vlm_backend
    from .evaluate_toy_vlm_language import main

    source = tmp_path / "source"
    trace(source)

    class Backend:
        name = "test_fake_backend"
        metadata = {"real_model": False}

        def __init__(self, model, *, prompt_variant):
            assert prompt_variant == (variant or "compact")

        def decide_with_images(self, request, images):
            response = dict(schema_version=VERSION, reply="你好，我叫小鸭。", decision="respond", goals=[])
            self.last_trace = dict(model_called=True, raw_response=response, raw_prediction=response)
            return response

    monkeypatch.setattr(toy_vlm_backend, "ToyVLMBackend", Backend)
    args = ["--source-episode", str(source), "--output", str(tmp_path / "report"), "--case", "identity"]
    if variant:
        args.extend(["--vlm-context", variant])
    assert main(args) == 0


def test_behavioral_alternatives_require_an_explicit_complete_plan():
    case = dict(
        request=request("原地转一圈", planned=True),
        expected=dict(decision="replace", goals=[turn("left", 360)]),
        acceptable_decisions=["replace"],
        acceptable_goals=[[turn("right", 360)]],
    )
    prediction = dict(decision="replace", goals=[turn("right", 360)])
    assert score_prediction(prediction, case) == dict(exact=False, behavioral=True)
    prediction["goals"][0]["angle_deg"] = 90
    assert score_prediction(prediction, case)["behavioral"] is False
    prediction["goals"] = [turn("right", 360), turn("right", 360)]
    assert score_prediction(prediction, case)["behavioral"] is False
    prediction["goals"] = [turn("right", 360)]
    del case["acceptable_goals"]
    case["request"]["event"]["text"] = "向左转一整圈"
    assert score_prediction(prediction, case) == dict(exact=False, behavioral=False)


def test_rules_suite_keeps_expectations_outside_backend_inputs_and_training(tmp_path):
    from .toy_vlm_rules_cases import VERSION_CASES
    from .toy_vlm_rules_cases import cases as rules_cases

    source = tmp_path / "source"
    trace(source)
    observation, image_paths, _ = source_snapshot(source)
    selected = next(case for case in rules_cases(observation) if case["expected"]["decision"] == "respond")
    seen = []

    class Backend:
        name = "test_fake_backend"
        metadata = {"real_model": False}

        def decide_with_images(self, current, images):
            seen.append(copy.deepcopy(current))
            assert current == selected["request"]
            assert images == image_paths
            assert (
                not {"expected", "acceptable_goals", "acceptable_decisions", "counterfactual_fields"} & current.keys()
            )
            response = dict(schema_version=VERSION, reply="测试回复。", decision="respond", goals=[])
            self.last_trace = dict(model_called=True, raw_response=response, raw_prediction=response)
            return response

    result = evaluate(Backend(), source, tmp_path / "report", selected=[selected["id"]], case_suite="rules")
    assert len(seen) == result["summary"]["cases"] == 1
    assert result["cases_version"] == VERSION_CASES
    assert result["expectations_passed_to_model"] is False
    assert result["physical_validation"] is False and result["training_ready"] is False
    assert result["summary"]["physical_executions"] == 0
    assert result["source"]["original_response_used"] is False


def test_cli_selects_rules_case_suite_separately_from_prompt_variant(tmp_path, monkeypatch):
    from . import evaluate_toy_vlm_language as diagnostic
    from . import toy_vlm_backend
    from .toy_vlm_rules_cases import cases as rules_cases

    source = tmp_path / "source"
    trace(source)
    selected = rules_cases()[0]["id"]
    constructed, evaluated = [], []

    class Backend:
        def __init__(self, model, *, prompt_variant):
            constructed.append(prompt_variant)

    def evaluate(backend, source_episode, output, *, event_id, selected, case_suite):
        evaluated.append(dict(selected=selected, case_suite=case_suite))
        return dict(summary=dict(errors=0))

    monkeypatch.setattr(toy_vlm_backend, "ToyVLMBackend", Backend)
    monkeypatch.setattr(diagnostic, "evaluate", evaluate)
    assert (
        diagnostic.main(
            [
                "--source-episode",
                str(source),
                "--output",
                str(tmp_path / "report"),
                "--vlm-context",
                "rules",
                "--suite",
                "rules",
                "--case",
                selected,
            ]
        )
        == 0
    )
    assert constructed == ["rules"]
    assert evaluated == [dict(selected=[selected], case_suite="rules")]
