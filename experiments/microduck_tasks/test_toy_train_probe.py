"""Validation reads held-out inputs without allowing them into optimizer rows."""

import json
import sys
from types import SimpleNamespace

import pytest

from . import toy_train_probe
from .test_toy_dataset import trace
from .toy_dataset import export


@pytest.fixture
def dataset(tmp_path):
    for index in range(3):
        trace(tmp_path / "input" / str(index), episode=f"episode{index}", seed=index, family=f"family{index}")
    output = tmp_path / "dataset"
    export(tmp_path / "input", output, three_way=True)
    return output


def test_processor_selection_includes_test_and_every_validation_row(dataset):
    rows, splits = toy_train_probe._probe_rows(dataset, processor_only=True, limit=1)
    assert len(rows) == 3
    assert set(splits.values()) == {"train", "val", "test"}


def test_optimizer_selection_cannot_read_or_train_on_heldout_labels(dataset, monkeypatch):
    original = toy_train_probe.read_records
    requested = []

    def train_only(path):
        requested.append(path.stem)
        assert path.stem == "train"
        return original(path)

    monkeypatch.setattr(toy_train_probe, "read_records", train_only)
    (dataset / "val.jsonl").write_text("held-out label must never be read")
    (dataset / "test.jsonl").write_text("held-out label must never be read")
    rows, splits = toy_train_probe._probe_rows(dataset, processor_only=False, limit=8)
    assert len(rows) == 1 and requested == ["train"]
    assert set(splits.values()) == {"train"}


def test_processor_validation_rejects_changed_test_split(dataset):
    (dataset / "test.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed since export"):
        toy_train_probe._probe_rows(dataset, processor_only=True, limit=8)


def test_processor_probe_reports_all_splits_without_loading_model_or_optimizing(dataset, tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    threads = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    fake = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()))
    monkeypatch.setitem(sys.modules, "transformers", fake)

    def encoding(row, root, processor):
        assert root == dataset
        return {"labels": torch.tensor([[-100, -100, 4, 5]]), "image_grid_thw": torch.ones((2, 3))}, 2

    monkeypatch.setattr(toy_train_probe, "encode", encoding)
    output = tmp_path / "validation"
    report = toy_train_probe.probe(dataset, "no-model-is-loaded", output, processor_only=True, limit=1)
    assert threads == [2]
    assert report["optimizer_steps"] == 0
    assert report["all_splits_tokenized_read_only"]
    assert not report["training_rows_only"]
    assert report["tokenized_split_counts"] == {"train": 1, "test": 1, "val": 1}
    assert report["max_input_tokens"] == 2 and report["max_total_tokens"] == 4
    assert not report["generative_backward_tested"]
    assert not (output / "probe_router.pt").exists()
    saved = json.loads((output / "probe_report.json").read_text())
    assert len(saved["rows"]) == 3
    progress = json.loads(capsys.readouterr().out.strip())
    assert progress["tokenized"] == progress["total"] == 3
