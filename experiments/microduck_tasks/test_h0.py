"""Prevent data leakage, label corruption, and accidental blind-set evaluation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from .ablation_cases import build_cases
from .ablation_contract import checksum, dumps
from .ablation_prompt import messages_for
from .h0_data import audit, generate, validation_rows
from .h0_features import config, read_data, record
from .h0_training import train
from .run_ablation import save
from .toy_dataset import contract_digest, digest
from .toy_fixtures import request
from .toy_selector_labels import EOS, encode_targets


def test_generated_training_quotas_and_six_step_eos():
    rows = generate("train")
    assert all(row["request"]["capabilities"]["max_goals"] == 6 for row in rows)
    report = audit({"train": rows})["train"]
    assert report["rows"] == 1200
    assert report["groups"] == dict(
        single=300, multi2=100, multi3=100, multi4=100, multi5=100, multi6=100, interaction=180, chat=120, boundary=100
    )
    assert report["unique_action_plans"] > 500
    labels = encode_targets(rows)
    six = [i for i, row in enumerate(rows) if row["goal_count"] == 6]
    assert len(six) == 100
    assert (labels["token"][six, 6] == EOS).all()
    assert (labels["token"][six, :6] != EOS).all()
    for name in ("nod", "roll", "moonwalk", "kick", "navigate"):
        assert {"cold_start", "greeting", "unrelated_chat", "running"} <= report["action_contexts"][name].keys()


@pytest.mark.parametrize("max_goals", [None, 6, 12])
def test_historical_training_requires_matching_validation_capabilities(tmp_path, max_goals):
    rows = build_cases(request("样本", planned=True)["observation"], max_goals=max_goals)
    save(tmp_path / "dev240.json", rows)
    if max_goals == 6:
        assert len(validation_rows(tmp_path)) == 240
    else:
        with pytest.raises(ValueError, match="explicit max_goals=6"):
            validation_rows(tmp_path)


def test_blind_quota_and_no_cross_split_input_or_family():
    report = audit({"train": generate("train"), "blind": generate("blind")})
    assert report["blind"]["rows"] == 480
    assert set(report["blind"]["groups"].values()) == {80}
    assert report["cross_split_exact_input_or_utterance_overlap"] == 0


def test_cross_split_counterfactual_family_is_rejected():
    row = generate("train")[0]
    other = copy.deepcopy(row)
    other.update(id="blind_probe", split="blind")
    other["request"]["event"]["text"] = "不同措辞不能洗掉家庭来源"
    with pytest.raises(ValueError, match="family_id"):
        audit({"train": [row], "blind": [other]})


def test_answers_and_metadata_cannot_change_input_messages():
    row = generate("train")[0]
    before = messages_for(row["request"], "P6")
    row["target"] = {"secret_answer": "never_leak"}
    row["expected"] = row["target"]
    row["family_id"] = "never_leak"
    assert messages_for(row["request"], "P6") == before
    assert "never_leak" not in dumps(before)
    assert all(part["type"] == "text" for part in before[1]["content"])


def test_permission_and_freshness_contrasts_include_a_valid_action():
    rows = generate("train")
    families = {r["family_id"] for r in rows if r["family_id"].startswith("train:state_pair:5:")}
    assert len(families) == 5
    for family in families:
        group = [r for r in rows if r["family_id"] == family]
        assert len(group) == 3
        assert len({r["request"]["event"]["text"] for r in group}) == 1
        assert sorted(r["expected"]["decision"] for r in group) == ["clarify", "clarify", "replace"]
        allowed = next(r for r in group if r["expected"]["decision"] == "replace")
        assert allowed["request"]["observation"]["age_s"] <= 0.5
        assert allowed["request"]["observation"]["environment"]["motion_allowed"]


def test_current_utterance_correction_keeps_only_the_final_goal():
    rows = [r for r in generate("train") if r["template_family"] == "train:correction"]
    assert len(rows) >= 40
    from .h0_language import phrase

    for row in rows:
        assert row["expected"]["decision"] == "replace"
        assert len(row["expected"]["goals"]) == 1
        final = phrase(row["expected"]["goals"][0], "train")
        assert f"现在改为{final}，只执行修改后的动作" in row["request"]["event"]["text"]


@pytest.mark.parametrize("hidden", (128, 256))
def test_three_seed_training_never_opens_blind_and_preserves_full_only(tmp_path, monkeypatch, hidden):
    dataset, features, output = (tmp_path / name for name in ("dataset", "features", "output"))
    dataset.mkdir()
    features.mkdir()
    # Random inputs test plumbing only; no model quality is inferred from this test.
    train_rows = generate("train")[:32]
    val = copy.deepcopy(train_rows[:8])
    for row in val:
        row.update(id="val_" + row["id"], split="val")
    val[-1]["expected"] = dict(decision="replace", goals=[dict(type="turn", direction="left", angle_deg=45)])
    val[-1]["domain"] = "full_only"
    from .ablation_contract import wrap_plan

    val[-1]["target"] = wrap_plan(val[-1]["expected"])
    rows = dict(train=train_rows, val=val)
    for split, items in rows.items():
        (dataset / f"{split}.jsonl").write_text("".join(dumps(r) + "\n" for r in items))
    save(
        dataset / "manifest.json",
        dict(
            training_splits=["train", "val"],
            input_view="P6",
            files_sha256={s: digest(dataset / f"{s}.jsonl") for s in rows},
        ),
    )
    (dataset / "sealed").mkdir()
    (dataset / "sealed/blind480.jsonl").write_text("DO NOT READ THIS FILE")
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if "sealed" in path.parts:
            raise AssertionError("The blind split was opened before route selection.")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    read_data(dataset)
    cfg = dict(config(), diagnostic_random_features=True)
    manifest = dict(
        complete=True,
        config=cfg,
        config_sha256=checksum(cfg),
        splits={},
        dataset_manifest_sha256=digest(dataset / "manifest.json"),
        contract_sha256=contract_digest(),
    )
    rng = np.random.default_rng(17)
    for split, items in rows.items():
        np.savez_compressed(
            features / f"{split}.npz",
            x=rng.normal(size=(len(items), 2048)).astype(np.float16),
            ids=np.array([r["id"] for r in items]),
        )
        save(features / f"{split}_records.json", [record(r, split) for r in items])
        manifest["splits"][split] = dict(
            cache_sha256=digest(features / f"{split}.npz"), records_sha256=digest(features / f"{split}_records.json")
        )
    save(features / "manifest.json", manifest)
    result = train(dataset, features, output, epochs=1, hidden=hidden)
    assert [r["seed"] for r in result["seeds"]] == [17, 43, 71]
    assert result["blind_access"] is False
    settings = json.loads((output / "training_config.json").read_text())
    assert settings["selection_samples"] == 7
    assert settings["total_validation_samples"] == 8
    assert settings["hidden"] == hidden
    assert all(r["trial"]["config"]["hidden"] == hidden for r in result["seeds"])
    report = json.loads((output / "seed_17/all240.json").read_text())
    assert report["predictions"][-1]["expected"]["goals"][0]["angle_deg"] == 45
