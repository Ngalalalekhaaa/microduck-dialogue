"""Check full-epoch coverage, leakage guards, selection and the actual optimizer loop."""

import copy
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from .ablation_contract import dumps
from .ablation_score import score
from .q0_model import weights_digest
from .q1_evaluate import common_gate, report, selection_key
from .q1_schedule import batches, learning_rate, require_q0
from .run_ablation import save


def q0_result():
    return dict(
        complete=True,
        overfit_check_pass=True,
        reload_pass=True,
        optimizer_steps=100,
        total_microbatches=800,
        weights_before="initial",
        weights_after="trained",
        after=dict(
            completed=32,
            expected=32,
            errors=0,
            metrics={
                **{k: dict(rate=1) for k in ("accepted_exact", "action_plan_exact", "schema_valid", "context_valid")},
                "nonaction_attempted_start": dict(correct=0),
            },
        ),
    )


def test_q1_uses_each_training_row_twice_and_only_150_updates():
    groups = list(batches())
    assert len(groups) == 150 and all(len(indices) == 16 for _, indices in groups)
    for epoch in (1, 2):
        assert sorted(i for e, indices in groups if e == epoch for i in indices) == list(range(1200))
    assert groups == list(batches()) and groups[:75] != groups[75:]
    assert learning_rate(1) > 0 and learning_rate(8) == 1e-4 and learning_rate(150) == 0
    with pytest.raises(ValueError):
        list(batches(count=1201))


def test_q0_guard_checks_outcomes_not_just_normal_exit():
    good = q0_result()
    require_q0(good)
    for key, value in (("complete", False), ("reload_pass", False), ("weights_after", "initial")):
        bad = copy.deepcopy(good)
        bad[key] = value
        with pytest.raises(ValueError):
            require_q0(bad)
    for key, value in (("completed", 31), ("errors", 1)):
        bad = copy.deepcopy(good)
        bad["after"][key] = value
        with pytest.raises(ValueError):
            require_q0(bad)
    bad = copy.deepcopy(good)
    bad["after"]["metrics"]["action_plan_exact"]["rate"] = 0.89
    with pytest.raises(ValueError):
        require_q0(bad)


def test_selection_penalizes_invalid_boundary_and_ignores_full_only():
    from .ablation_cases import build_cases
    from .toy_fixtures import request

    rows = build_cases(request("test", planned=True)["observation"], max_goals=6)
    common = [r for r in rows if r["domain"] == "common"]
    predictions = [dict(r, score=score(dumps(r["expected"]), r, plan_only=True)) for r in common]
    good = dict(step=50, common236=report(predictions, 236))
    assert common_gate(good["common236"])
    later = dict(good, step=100, full_only4={"arbitrary": "does not enter selection"})
    assert selection_key(good) > selection_key(later)
    bad = copy.deepcopy(predictions)
    index = next(i for i, r in enumerate(common) if r["expected"]["decision"] == "decline")
    bad[index]["score"] = score("unparseable", common[index], plan_only=True)
    broken = dict(step=50, common236=report(bad, 236))
    assert broken["common236"]["metrics"]["nonaction_attempted_start"]["correct"] == 0
    assert broken["common236"]["nonaction_answer_cases"]["unparseable"] == 1
    assert not common_gate(broken["common236"])
    assert selection_key(good) > selection_key(broken)


def test_real_frozen_data_has_all_rows_and_never_reads_blind(monkeypatch):
    from .q1_data import load
    from .q1_verify import reload_rows

    root = Path("experiments/microduck_tasks/outputs/staged_plan_20261116")
    if not (root / "q1/data/manifest.json").exists():
        pytest.skip("Local frozen experiment artifacts are absent.")
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "blind" not in str(path) and "sealed" not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    _, encoded = load(root / "q1/data", root / "h0/data")
    assert {s: len(rr) for s, rr in encoded.items()} == {"train": 1200, "val": 240}
    assert sum(r["row"]["domain"] == "common" for r in encoded["val"]) == 236
    selected = reload_rows(encoded["val"])
    assert len({r["id"] for r in selected}) == 3
    assert len(selected[1]["row"]["expected"]["goals"]) == 6


def test_actual_training_loop_updates_only_training_rows_and_validates_at_fixed_steps(tmp_path, monkeypatch):
    from . import q1_training as training

    work = tmp_path / "q1"
    (work / "data").mkdir(parents=True)
    save(work / "data/manifest.json", {})
    (tmp_path / "q0/run100").mkdir(parents=True)
    save(tmp_path / "CURRENT_STATUS.json", {})
    rows = {"train": [dict(id=f"train_{i}") for i in range(1200)], "val": [dict(id="val_only")]}
    calls, validations = [], []

    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def save_pretrained(self, path, **options):
            path.mkdir()
            save(path / "tiny.json", dict(weight=self.weight.item()))

    model = TinyAdapter()
    initial = weights_digest(dict(model.named_parameters()))
    q0 = q0_result()
    q0["weights_before"] = initial
    save(tmp_path / "q0/run100/results.json", q0)
    processor = type("Processor", (), {"tokenizer": type("Tokenizer", (), {"pad_token_id": 0})()})()
    monkeypatch.setattr(training, "load", lambda *_: ({"splits": {s: dict(sha256=s) for s in rows}}, rows))
    monkeypatch.setattr(training, "load_model", lambda *_: (model, processor, {}))
    monkeypatch.setattr(training, "memory", lambda: {})
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kw: nullcontext())

    def tiny_loss(current, row, pad):
        calls.append(row["id"])
        assert row["id"].startswith("train_")
        return current.weight.square().mean()

    def evaluation(current, proc, actual, output, step, notify):
        assert actual == rows["val"]
        validations.append(step)
        return dict(step=step, all240=dict(prototype_gate_pass=False))

    monkeypatch.setattr(training, "loss", tiny_loss)
    monkeypatch.setattr(training, "evaluate", evaluation)
    monkeypatch.setattr(training, "selection_key", lambda r: (-r["step"],))
    save(work / "planned_config.json", training.configuration(tmp_path))
    training.run(tmp_path, tmp_path)
    assert Counter(calls) == Counter({r["id"]: 2 for r in rows["train"]})
    assert validations == [50, 100, 150] and model.weight.item() < 1
    result = json.loads((work / "train1200/results.json").read_text())
    assert result["training_complete"] and result["reload_pending"] and not result["complete"]
    assert result["selected_step"] == 50 and result["weights_after"] != initial
