"""Semantic labels, loss masking, held-out evaluation and online checkpoint tests."""

# ruff: noqa: E402

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from .toy_protocol import DECISIONS, VERSION, validate_output
from .toy_selector_labels import BOS, EOS, FIELDS, TOKENS, canonical_plan, check_support, encode_targets, goal_label
from .toy_selector_metrics import evaluate, majority_user_baseline
from .toy_selector_model import MODEL_VERSION, Selector, SelectorModel, decode_outputs, load_selector, training_loss
from .toy_selector_training import fit, loss_weights


def row(goals=None, decision="replace", event="user"):
    return dict(
        split_group="fixture",
        request=dict(event=dict(kind=event)),
        target=dict(schema_version=VERSION, reply="收到。", decision=decision, goals=goals or []),
    )


def nod(repeat=1):
    return dict(type="perform", action="nod", repeat=repeat)


@pytest.mark.parametrize("repeat", [1, 2, 3])
def test_goal_fields_and_eos_have_explicit_masks(repeat):
    encoded = encode_targets([row([nod(repeat)]), row(decision="respond", event="feedback")])
    assert encoded["token"][0].tolist() == [TOKENS.index("perform:nod"), EOS, -100, -100, -100, -100, -100]
    assert encoded["repeat"][0].tolist() == [repeat - 1, -100, -100, -100, -100, -100, -100]
    assert np.all(encoded["color"] == -100)
    assert encoded["token"][1, 0] == EOS


def test_unknown_duration_and_target_attributes_are_rejected():
    with pytest.raises(ValueError):
        goal_label(dict(type="move", direction="forward", limit=dict(kind="duration", seconds=1.5)))
    with pytest.raises(ValueError, match="exactly one color"):
        goal_label(dict(type="kick", target=dict(category="ball", attributes=[], entity_id=None)))


def test_numeric_canonicalization_does_not_mark_integer_angles_as_new_plans():
    assert canonical_plan([dict(type="turn", direction="left", angle_deg=180)]) == canonical_plan(
        [dict(type="turn", direction="left", angle_deg=180.0)]
    )


def test_holdout_label_values_need_training_support():
    encoded = {
        "train": encode_targets([row([nod(1)])]),
        "val": encode_targets([row([nod(2)])]),
        "test": encode_targets([row([nod(1)])]),
    }
    with pytest.raises(ValueError, match="unsupported training labels for repeat"):
        check_support(encoded)


def test_padded_tokens_and_inapplicable_fields_have_no_loss_gradient():
    labels = {
        name: torch.from_numpy(value) for name, value in encode_targets([row([nod()]), row(decision="respond")]).items()
    }
    output = {
        "decision": torch.randn(2, len(DECISIONS), requires_grad=True),
        "token": torch.randn(2, 7, len(TOKENS), requires_grad=True),
        **{name: torch.randn(2, 7, len(values), requires_grad=True) for name, values in FIELDS.items()},
    }
    training_loss(output, labels, torch.ones(len(DECISIONS)), torch.ones(len(TOKENS))).backward()
    assert output["token"].grad[0, :2].abs().sum() > 0
    assert output["token"].grad[0, 2:].abs().sum() == 0
    assert output["token"].grad[1].abs().sum() == 0
    assert output["repeat"].grad[0, 0].abs().sum() > 0
    assert output["repeat"].grad[0, 1:].abs().sum() == 0
    assert output["color"].grad is None


def test_feedback_eos_never_changes_action_token_class_weights():
    actions = [row([nod()]), row([nod(), nod()])]
    augmented = actions + [row(decision="respond", event="feedback")] * 100
    encoded = [
        {name: torch.from_numpy(value) for name, value in encode_targets(rows).items()} for rows in (actions, augmented)
    ]
    assert torch.equal(loss_weights(encoded[0])[1], loss_weights(encoded[1])[1])


def test_decoder_stops_after_continuous_motion_and_hides_goals_on_respond():
    output = {
        "decision": torch.full((2, len(DECISIONS)), -100.0),
        "token": torch.full((2, 7, len(TOKENS)), -100.0),
        **{name: torch.zeros(2, 7, len(values)) for name, values in FIELDS.items()},
    }
    output["decision"][0, DECISIONS.index("replace")] = 0
    output["decision"][1, DECISIONS.index("respond")] = 0
    output["token"][:, :, TOKENS.index("perform:nod")] = 0
    output["token"][:, 0, TOKENS.index("move:forward")] = 1
    output["limit_kind"][:, :, 1] = 1
    predictions = decode_outputs(output)
    assert predictions[0]["goals"] == [dict(type="move", direction="forward", limit=dict(kind="until_cancelled"))]
    assert predictions[1] == dict(decision="respond", goals=[])
    for prediction in predictions:
        validate_output(dict(schema_version=VERSION, reply="收到。", **prediction))


def test_inference_does_not_require_or_accept_targets():
    torch.set_num_threads(1)
    selector = Selector(SelectorModel(dim=8, hidden=32), {})
    vectors = np.zeros((2, 8), dtype=np.float32)
    one = selector.predict(vectors)
    assert one == selector.predict(vectors)
    with pytest.raises(TypeError):
        selector.predict(vectors, target=row([nod()]))
    with pytest.raises(ValueError, match="finite"):
        selector.predict(np.full((2, 8), np.nan))
    assert sum(p.numel() for p in SelectorModel().parameters()) < 1_000_000
    assert SelectorModel().token_embedding.num_embeddings == BOS + 1


def test_checkpoint_roundtrip_and_feature_contract_checks(tmp_path):
    model = SelectorModel(dim=8, hidden=32).eval()
    checkpoint = dict(
        version=MODEL_VERSION,
        state_dict=model.state_dict(),
        dim=8,
        hidden=32,
        decisions=list(DECISIONS),
        tokens=list(TOKENS),
        fields={name: list(values) for name, values in FIELDS.items()},
        feature_config={"example": 1},
        contract_sha256="contract",
    )
    torch.save(checkpoint, tmp_path / "selector.pt")
    loaded = load_selector(tmp_path, feature_config={"example": 1}, contract_sha256="contract")
    x = np.zeros((2, 8), dtype=np.float32)
    assert loaded.predict(x) == Selector(model, {}).predict(x)
    with pytest.raises(ValueError, match="Feature configuration"):
        load_selector(tmp_path, feature_config={"example": 2})
    with pytest.raises(ValueError, match="Dialogue contract"):
        load_selector(tmp_path, contract_sha256="another")


def test_training_can_learn_opposite_orders_with_shared_skill_heads():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    look = dict(type="perform", action="look_around", repeat=1)
    rows = [row([nod(), look]), row([look, nod()])]
    labels = {name: torch.from_numpy(value) for name, value in encode_targets(rows).items()}
    features = torch.eye(2, 4)
    model = SelectorModel(dim=4, hidden=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(200):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(
            model(features, teacher=labels), labels, torch.ones(len(DECISIONS)), torch.ones(len(TOKENS))
        )
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        predicted = decode_outputs(model(features))
    assert [item["goals"] for item in predicted] == [item["target"]["goals"] for item in rows]


def test_easy_feedback_does_not_hide_wrong_action_plans():
    rows = [row([nod()]), row(decision="respond", event="feedback"), row(decision="cancel")]
    predictions = [dict(decision="respond", goals=[])] * 3
    result = evaluate(rows, predictions, {canonical_plan([nod()])})
    assert result["decision_all"]["accuracy"] == pytest.approx(1 / 3)
    assert result["action_plan_exact"]["accuracy"] == 0
    assert result["missed_cancel"]["rate"] == 1
    assert result["decision_user"]["per_class"]["pause"]["support"] == 0
    assert result["decision_user"]["per_class"]["pause"]["f1"] is None
    assert result["action_by_length"]["6"] == dict(correct=0, support=0, accuracy=None)


def test_false_start_user_excludes_easy_feedback_and_baseline_uses_users_only():
    rows = [row(decision="respond")] + [row(decision="respond", event="feedback")] * 9
    predictions = [dict(decision="replace", goals=[nod()])] + [dict(decision="respond", goals=[])] * 9
    result = evaluate(rows, predictions, set())
    assert result["false_start"]["rate"] == pytest.approx(0.1)
    assert result["false_start_user"] == dict(count=1, support=1, rate=1.0)
    baseline = majority_user_baseline([row([nod()]), *rows[1:]])
    assert baseline["prediction"] == dict(decision="replace", goals=[nod()])
    assert baseline["training_user_rows"] == 1


def test_fit_normalizes_only_training_and_refuses_test_access():
    torch.set_num_threads(1)
    rows = {split: [row([nod()]), row(decision="respond")] for split in ("train", "val")}
    arrays = {"train": torch.tensor([[1.0, 3.0], [3.0, 5.0]]), "val": torch.full((2, 2), 100.0)}
    labels = {
        split: {name: torch.from_numpy(value) for name, value in encode_targets(items).items()}
        for split, items in rows.items()
    }
    config = dict(hidden=32, lr=0.001, weight_decay=0.01, seed=17)
    model, result = fit(arrays, labels, rows, config, lambda record: None, epochs=1)
    assert model.mean.tolist() == [2.0, 4.0]
    assert model.scale.tolist() == [1.0, 1.0]
    assert result["best_epoch"] == 1
    leaked = copy.copy(arrays)
    leaked["test"] = arrays["val"]
    with pytest.raises(ValueError, match="train and validation splits only"):
        fit(leaked, labels, rows, config, lambda record: None, epochs=1)
