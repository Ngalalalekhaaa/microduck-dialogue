"""Generated plans use the existing evaluator without being mistaken for teacher labels."""

import copy
import sys
from types import SimpleNamespace

import pytest

from . import evaluate_toy_selector as evaluation
from .toy_demo import _backend_label
from .toy_fixtures import perform
from .toy_selector_rollout import raw_plan_exact


def _report(source, *, error=None):
    expected = dict(decision="replace", goals=[perform("nod")])
    trace = dict(decision_source=source, raw_prediction=expected, seconds=2.5)
    return dict(
        turns=[
            dict(
                raw_selector_exact=raw_plan_exact(trace, expected, selector_only=True),
                raw_model_exact=raw_plan_exact(trace, expected),
                semantic_exact=True,
                model_trace=trace,
            )
        ],
        goal_status={},
        expected_finite_plan_verified=False,
        semantic_all_correct=True,
        control_checks={},
        error=error,
        runtime_failure=None,
        timed_out=bool(error and error["type"] == "TimeoutError"),
    )


def test_generated_plan_scoring_excludes_host_and_preserves_selector_metric():
    report = _report("vlm_generation")
    summary = evaluation.summarize([report, _report("host_stop")])
    assert summary["model_turns"] == 1 and summary["model_raw_plan_exact"] == 1
    assert summary["learned_selector_turns"] == 0 and summary["learned_raw_plan_exact"] is None
    assert summary["host_control_turns"] == 1 and summary["model_response_seconds_mean"] == 2.5
    summary = evaluation.summarize([_report("learned_selector")])
    assert summary["model_raw_plan_exact"] == summary["learned_raw_plan_exact"] == 1
    # An older selector report has no generic metric yet and must remain readable.
    legacy = _report("learned_selector")
    del legacy["turns"][0]["raw_model_exact"]
    assert evaluation.summarize([legacy])["model_raw_plan_exact"] == 1


def test_malformed_model_output_is_a_model_failure_not_an_excluded_host_turn():
    expected = dict(decision="replace", goals=[perform("nod")])
    assert raw_plan_exact(dict(decision_source="vlm_generation", raw_prediction=None), expected) is False
    assert raw_plan_exact(dict(decision_source="host_stop", raw_prediction=expected), expected) is None
    assert raw_plan_exact(None, expected) is None


def test_generation_report_uses_actual_sources_and_does_not_grade_reply_quality(tmp_path, monkeypatch):
    backend = SimpleNamespace(
        name="qwen35_toy_zero_shot", metadata=dict(goal_source="vlm_generation", reply_source="vlm_generation")
    )
    expected_cases = copy.deepcopy(evaluation.cases())
    visited = []

    def run_case(case, actual_backend, output, **kwargs):
        assert actual_backend is backend and "expected" not in kwargs
        visited.append(copy.deepcopy(case))
        output.mkdir()
        return _report("vlm_generation")

    monkeypatch.setattr(evaluation, "run_case", run_case)
    result = evaluation.evaluate(backend, tmp_path / "evaluation")
    assert visited == expected_cases and len(visited) == 14
    assert result["model_evaluation"] and not result["teacher_diagnostic_only"]
    assert result["reply_source"] == result["goal_source"] == "vlm_generation"
    assert any("do not grade reply truthfulness" in text for text in result["limitations"])
    assert result["summary"]["model_turns"] == 14
    assert result["summary"]["learned_selector_turns"] == 0


def test_ordinary_timeout_is_not_a_program_error():
    result = evaluation.summarize([_report("vlm_generation", error=dict(type="TimeoutError", message="budget"))])
    assert result["timed_out"] == result["errors"] == 1
    assert result["program_errors"] == 0


@pytest.mark.parametrize("wall_argument, expected", [([], 300), (["--wall-limit", "25"], 25)])
@pytest.mark.parametrize(
    "context_argument, expected_context",
    [([], "compact"), (["--vlm-context", "full"], "full"), (["--vlm-context", "rules"], "rules")],
)
def test_vlm_evaluation_needs_no_head_checkpoint_and_uses_explicit_budget(
    tmp_path, monkeypatch, wall_argument, expected, context_argument, expected_context
):
    from . import bootstrap

    constructed = []
    evaluated = []

    def create(model, *, prompt_variant):
        constructed.append((model, prompt_variant))
        return SimpleNamespace(name="qwen35_toy_zero_shot")

    def evaluate(backend, output, **kwargs):
        evaluated.append((backend.name, output, kwargs))
        return dict(summary=dict(program_errors=0, runtime_failures=0))

    monkeypatch.setattr(bootstrap, "configure", lambda: None)
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    monkeypatch.setitem(
        sys.modules,
        f"{evaluation.__package__}.toy_vlm_backend",
        SimpleNamespace(ToyVLMBackend=create),
    )
    assert (
        evaluation.main(["--backend", "vlm", "--output", str(tmp_path / "eval"), *wall_argument, *context_argument])
        == 0
    )
    assert len(constructed) == len(evaluated) == 1
    assert constructed[0][1] == expected_context
    assert evaluated[0][0] == "qwen35_toy_zero_shot" and evaluated[0][2]["wall_limit"] == expected


def test_vlm_ui_label_and_checkpoint_argument_do_not_claim_trained_selector():
    from .toy_demo import main

    assert "原始 Qwen" in _backend_label(SimpleNamespace(name="qwen35_toy_zero_shot"))
    with pytest.raises(SystemExit) as error:
        main(["--backend", "vlm", "--head-checkpoint", "unused.pt"])
    assert error.value.code == 2
