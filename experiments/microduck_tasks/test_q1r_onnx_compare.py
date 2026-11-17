"""Analysis regressions use invented durations; no model inference or speed claims."""

import copy
import json

import pytest

from .q1r_onnx_compare import compare
from .q1r_onnx_inputs import checksum
from .toy_fixtures import kick, navigate, request


def example(identifier="red_yellow"):
    state = request("先踢红球再踢黄球", planned=True)
    expected = dict(decision="replace", goals=[kick("red"), kick("yellow")], reply="我会按顺序踢球。")
    historical = dict(decision="replace", goals=[kick("red"), navigate("yellow")], reply="我会执行。")
    return dict(
        id=identifier,
        request=state,
        request_sha256=checksum(state),
        expected=expected,
        frozen_reference=dict(raw_text=json.dumps(historical), timing=dict(seconds=9999)),
    )


def record(row, backend="fixture_baseline", seconds=4, **changes):
    value = dict(
        id=row["id"],
        raw_text=json.dumps(row["expected"]),
        timing=dict(seconds=seconds, input_tokens=10, output_tokens=20),
        backend=backend,
        request_sha256=row["request_sha256"],
        device="fictional_cpu",
        precision="fp32",
        prompt_sha256="fixture_prompt",
        checkpoint_sha256="fixture_weights",
        max_new_tokens=100,
        generation_config_sha256="fixture_configuration",
    )
    value.update(changes)
    return value


def test_historical_mistake_is_not_gold_or_timing_baseline():
    row = example()
    baseline = record(row, raw_text=row["frozen_reference"]["raw_text"])
    candidate = record(row, "fixture_onnx", seconds=2)
    report = compare([row], [baseline], [candidate])
    assert report["paired_total_latency_ratio"] == 2
    assert report["baseline"]["assessments"]["semantic_plan_exact"] == 0
    assert report["onnx"]["assessments"]["semantic_plan_exact"] == 1
    pair = report["pairs"][0]
    assert pair["baseline_matches_historical_raw"]
    assert not pair["onnx_matches_historical_raw"]
    assert not pair["historical_reference_is_gold"]
    assert not report["historical_latency_used_for_speedup"]


def test_pairing_uses_ids_and_trials_excludes_warmup_and_reports_unmatched():
    first, second = example("first"), example("second")
    left = [record(first, warmup=True, seconds=100), record(first, repetition=0), record(second)]
    right = [record(second, "fixture_onnx", seconds=2), record(first, "fixture_onnx", repetition=1)]
    report = compare([first, second], left, right)
    assert report["baseline"]["measured_trials"] == 2
    assert report["paired_trials"] == 1
    assert report["pairs"][0]["id"] == "second"
    assert report["unmatched"]["baseline_only"] == [dict(id="first", repetition=0)]
    assert report["unmatched"]["onnx_only"] == [dict(id="first", repetition=1)]


def test_runner_repeat_alias_pairs_by_trial_even_when_order_differs():
    row = example()
    left = [record(row, repeat=1, seconds=4), record(row, repeat=0, seconds=8)]
    right = [record(row, "fixture_onnx", repeat=0, seconds=4), record(row, "fixture_onnx", repeat=1, seconds=2)]
    report = compare([row], left, right)
    assert [pair["repetition"] for pair in report["pairs"]] == [0, 1]
    assert all(pair["observed_latency_ratio_baseline_over_onnx"] == 2 for pair in report["pairs"])
    with pytest.raises(ValueError, match="disagree"):
        compare([row], [record(row, repeat=1, repetition=0)], [])


def test_length_or_metadata_differences_prevent_backend_only_claim():
    row = example()
    left, right = record(row), record(row, "fixture_onnx", seconds=2)
    right["timing"]["output_tokens"] = 10
    report = compare([row], [left], [right])
    assert not report["pairs"][0]["conditions"]["backend_only_speed_claim_supported"]
    assert not report["pairs"][0]["conditions"]["output_token_count_equal"]
    right["timing"]["output_tokens"] = 20
    right.pop("checkpoint_sha256")
    report = compare([row], [left], [right])
    assert "checkpoint_sha256" in report["pairs"][0]["conditions"]["metadata_missing"]
    assert not report["pairs"][0]["conditions"]["backend_only_speed_claim_supported"]


@pytest.mark.parametrize(
    "change",
    [
        {"request_sha256": "a_different_input"},
        {"timing_role": "historical_reference_only"},
        {"fair_benchmark": False},
        {"timing": dict(seconds=float("nan"), input_tokens=10, output_tokens=20)},
    ],
)
def test_untrustworthy_measurement_is_rejected(change):
    row = example()
    with pytest.raises(ValueError):
        compare([row], [record(row, **change)], [])


def test_invalid_action_output_and_empty_candidate_are_visible():
    row = example()
    invalid = copy.deepcopy(row["expected"])
    invalid["goals"][0]["target"]["attributes"][0]["value"] = "green"
    report = compare([row], [record(row, raw_text=json.dumps(invalid))], [])
    assert report["paired_trials"] == 0
    assert report["paired_total_latency_ratio"] is None
    assert report["baseline"]["assessments"]["json_valid"] == 1
    assert report["baseline"]["assessments"]["context_valid"] == 0
    assert report["missing_input_ids"]["onnx"] == [row["id"]]
