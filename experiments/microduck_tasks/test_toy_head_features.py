"""Input boundaries, batch padding, and resumable provenance without loading a VLM."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import extract_toy_head_features as extractor
from .test_toy_dataset import trace
from .toy_dataset import digest, export
from .toy_fixtures import request
from .toy_head_features import ToyFrozenFeatures, _pool, _right_padded_lengths, input_digest, messages_for
from .toy_train_probe import messages_for as training_messages


def test_only_current_request_is_serialized_with_same_training_prompt():
    state = request("先点头，再坐下")
    assert messages_for(state) == training_messages(state)
    changed = copy.deepcopy(state)
    changed["execution"]["revision"] = 123
    assert messages_for(changed) == messages_for(state)
    assert input_digest(changed) == input_digest(state)
    changed["target"] = {"reply": "supervised answer must never enter input"}
    with pytest.raises(ValueError):
        messages_for(changed)
    assert sum(item["type"] == "image" for item in messages_for(state)[1]["content"]) == 2
    assert all(item["type"] == "text" for item in messages_for(state, vision=False)[1]["content"])


def test_variable_length_batch_is_right_padded_and_pools_final_real_token():
    torch = pytest.importorskip("torch")

    class Processor:
        tokenizer = SimpleNamespace(padding_side="left", pad_token_id=99)

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            assert [item["role"] for item in messages] == ["system", "user"]
            return json.dumps(messages)

        def __call__(self, *, text, images, return_tensors, padding):
            assert len(text) == 2 and images is None
            assert return_tensors == "pt" and padding is True and self.tokenizer.padding_side == "right"
            return {
                "input_ids": torch.tensor([[1, 2, 3, 99, 99], [4, 5, 6, 7, 8]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
                "mm_token_type_ids": torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
            }

    model = object.__new__(ToyFrozenFeatures)
    model.processor = Processor()
    batch, lengths = model.encode_batch([request("你好"), request("点头")], [{}, {}], vision=False)
    assert lengths == [3, 5]
    assert all(value.shape == (2, 1024) for value in batch.values())
    assert batch["input_ids"][0, -1] == 99
    assert batch["attention_mask"].sum().item() == 8
    hidden = torch.arange(2 * 1024 * 2).reshape(2, 1024, 2)
    assert np.array_equal(_pool(hidden, batch["attention_mask"]), hidden[[0, 1], [2, 4]].numpy())
    with pytest.raises(ValueError, match="right-padded"):
        _right_padded_lengths(torch.tensor([[0, 1, 1]]))
    with pytest.raises(ValueError, match="right-padded"):
        _right_padded_lengths(torch.tensor([[1, 0, 1]]))
    with pytest.raises(ValueError, match="right-padded"):
        _right_padded_lengths(torch.zeros((1, 2)))


@pytest.fixture
def dataset(tmp_path):
    for index in range(3):
        trace(tmp_path / "source" / str(index), episode=f"episode{index}", seed=index, family=f"family{index}")
    output = tmp_path / "dataset"
    export(tmp_path / "source", output, three_way=True)
    return output


def test_partial_cache_resumes_without_reextracting_or_resplitting(dataset, tmp_path, monkeypatch):
    calls = []

    class Model:
        load_seconds = 0.1
        torch = SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0))

        def __init__(self, path):
            assert path == "unused"

        def extract_batch(self, requests, images, *, vision):
            assert all(set(state) == set(request("你好")) for state in requests)
            calls.extend(state["event"]["text"] for state in requests)
            return np.ones((len(requests), 2048), np.float32), [{"tokens": 23} for _ in requests]

    monkeypatch.setattr(extractor, "ToyFrozenFeatures", Model)
    output = tmp_path / "features"
    partial = extractor.extract(dataset, "unused", output, batch_size=2, limit=1)
    assert partial["complete"] is False and len(calls) == 1
    complete = extractor.extract(dataset, "unused", output, batch_size=2, resume=True)
    assert complete["complete"] is True and len(calls) == 3
    for split, rows in extractor.read_splits(dataset).items():
        with np.load(output / f"{split}.npz", allow_pickle=False) as saved:
            assert saved["x"].shape == (1, 2048) and saved["x"].dtype == np.float16
            assert saved["ids"].tolist() == [row["id"] for row in rows]
        assert complete["splits"][split]["cache_sha256"] == digest(output / f"{split}.npz")
        record = json.loads((output / f"{split}_records.json").read_text())[0]
        assert record["input_sha256"] == input_digest(rows[0]["request"])
        assert record["image_sha256"] == rows[0]["image_sha256"]
        assert "target" not in record
    extractor.extract(dataset, "unused", output, resume=True)
    assert len(calls) == 3
    with pytest.raises(ValueError, match="configuration or dataset changed"):
        extractor.extract(dataset, "unused", output, resume=True, vision=False)


def test_cache_rejects_wrong_input_identity_and_nonfinite_vector(tmp_path):
    path = tmp_path / "row.npz"
    expected = {"id": "a", "input_sha256": "input"}
    extractor._save_cache(path, np.zeros(2048), {**expected, "config_sha256": "config"})
    with pytest.raises(ValueError, match="provenance differs"):
        extractor._read_cache(path, {**expected, "input_sha256": "changed"}, "config")
    extractor._save_cache(path, np.full(2048, np.nan), {**expected, "config_sha256": "config"})
    with pytest.raises(ValueError, match="Invalid feature vector"):
        extractor._read_cache(path, expected, "config")


def test_changed_dataset_and_cross_split_family_are_refused(dataset):
    test = dataset / "test.jsonl"
    val = json.loads((dataset / "val.jsonl").read_text())
    row = json.loads(test.read_text())
    row["split_group"] = val["split_group"]
    test.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="changed since export"):
        extractor.read_splits(dataset)
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files_sha256"]["test"] = digest(test)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="split_group crosses"):
        extractor.read_splits(dataset)
