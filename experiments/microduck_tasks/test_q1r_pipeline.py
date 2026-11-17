"""Continuation preserves answer supervision, honest scores, and epoch boundaries."""

import copy
import json

import pytest

from .q1r_evaluate import assess, canonical, screen, selection_key
from .q1r_prompt import messages_for, validate_prediction
from .q1r_training_state import batches, learning_rate
from .toy_fixtures import active, kick, perform, request


def row(text="点一下头", decision="replace", goals=None):
    return dict(
        id="test",
        group="test",
        family_id="test",
        request=request(text, planned=True),
        expected=dict(decision=decision, goals=[perform("nod")] if goals is None else goals, reply="好，我会点头。"),
    )


def test_prompt_contains_latest_event_history_and_trusted_capabilities_without_labels():
    sample = row("先别动，我只是打招呼", "respond", [])
    sample["request"]["execution"] = active(kick("red"))
    messages = messages_for(sample["request"])
    payload = json.loads(messages[1]["content"][0]["text"])
    assert payload["event"]["text"] == "先别动，我只是打招呼"
    assert payload["capabilities"]["max_goals"] == 15
    assert payload["execution"]["active"] == kick("red")
    assert "expected" not in payload and "rationale" not in payload


def test_plan_accuracy_does_not_require_reference_reply_equality():
    sample = row()
    predicted = dict(sample["expected"], reply="收到，我来点一下头。")
    score = assess(json.dumps(predicted), sample)
    assert score["plan_exact"] and score["semantic_plan_exact"] and score["reply_present"]
    assert not score["reply_reference_exact"]


def test_semantically_wrong_valid_output_is_not_repaired():
    sample = row("踢蓝球", goals=[kick("blue")])
    predicted = dict(sample["expected"], goals=[kick("red")], reply="好，我会踢红球。")
    score = assess(json.dumps(predicted), sample)
    assert score["context_valid"] and not score["semantic_plan_exact"]


def test_accepted_and_rejected_false_starts_are_distinct():
    sample = row("你好呀", "respond", [])
    predicted = dict(decision="replace", goals=[perform("nod")], reply="我来点头。")
    score = assess(json.dumps(predicted), sample)
    assert score["attempted_false_start"] and score["accepted_false_start"]
    sample["request"]["observation"]["environment"]["motion_allowed"] = False
    score = assess(json.dumps(predicted), sample)
    assert score["attempted_false_start"] and not score["accepted_false_start"]


def test_missing_reply_and_fabricated_skill_fail_contract():
    sample = row()
    for prediction in (
        dict(decision="replace", goals=[perform("nod")]),
        dict(decision="replace", goals=[perform("fly")], reply="我来飞。"),
    ):
        with pytest.raises(ValueError):
            validate_prediction(prediction, sample["request"])


def test_clear_premature_completion_is_reported_separately():
    sample = row()
    score = assess(json.dumps(dict(sample["expected"], reply="已经完成点头。")), sample)
    assert score["semantic_plan_exact"] and score["premature_completion"]


def test_canonicalization_does_not_mutate_predictions_or_erase_colors():
    original = [kick("red")]
    before = copy.deepcopy(original)
    assert canonical(original) == original == before
    assert canonical(original) != canonical([kick("blue")])


def test_every_training_row_used_once_each_epoch_including_partial_accumulation():
    schedule = list(batches(19, 2, 16, 1611))
    assert [len(indices) for _, indices in schedule] == [16, 3, 16, 3]
    for epoch in (1, 2):
        indices = [i for e, group in schedule if e == epoch for i in group]
        assert sorted(indices) == list(range(19))
    assert schedule == list(batches(19, 2, 16, 1611))
    assert schedule[:2] != schedule[2:]


def test_lr_warmup_and_cosine_floor_are_finite_and_bounded():
    rates = [learning_rate(step, 100, 5e-5, 20) for step in range(1, 101)]
    assert rates[0] == pytest.approx(5e-5 / 20)
    assert rates[19] == 5e-5
    assert rates[-1] == pytest.approx(5e-5 * 0.05)
    assert all(0 < rate <= 5e-5 for rate in rates)


def test_screen_stratifies_and_cannot_select_a_checkpoint():
    rows = [dict(id=f"r{i}", row=dict(group="frequent" if i < 80 else "rare")) for i in range(100)]
    subset = screen(rows, 12)
    assert sum(r["row"]["group"] == "rare" for r in subset) == 6
    with pytest.raises(ValueError, match="complete held-out"):
        selection_key(dict(full=False))
