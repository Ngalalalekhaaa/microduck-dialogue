"""Verify H2 memory boundaries, padding invariance, and unchanged decision input."""

import json

import numpy as np
import pytest
import torch

from .h0_data import generate
from .h2_memory import identity, layout, pool, read_cache, write_cache
from .h2_model import AttentionSelector
from .toy_selector_labels import encode_targets
from .toy_selector_model import SelectorModel, training_loss
from .toy_selector_training import loss_weights


def test_layout_preserves_every_dynamic_token_and_matches_input():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(".cache/microduck_head/model", local_files_only=True)
    row = generate("train")[400]
    text, ids, spans = layout(tokenizer, row["request"])
    assert ids == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert spans[0]["start"] == 0 and spans[-1]["end"] == len(ids)
    assert all(a["end"] == b["start"] for a, b in zip(spans[:-1], spans[1:], strict=True))
    assert all(s["end"] == s["start"] + 1 for s in spans[32:])
    assert {"observation", "execution", "history", "event"} <= {s["role"] for s in spans}
    changed = json.loads(json.dumps(row))
    changed["expected"] = {"answer_secret": "never_include_in_input"}
    assert layout(tokenizer, changed["request"]) == (text, ids, spans)


def test_memory_pooling_and_cache_identity(tmp_path):
    hidden = torch.arange(6, dtype=torch.float32)[:, None].repeat(1, 2048)
    spans = [dict(start=0, end=4, role="prefix_mean"), dict(start=4, end=5, role="event")]
    memory = pool(hidden, spans)
    assert (memory[0] == 1.5).all() and (memory[1] == 4).all() and not memory[2:].any()
    rec = dict(identity({"id": "probe"}, "train", "configuration"), spans=spans)
    path = tmp_path / "row.npz"
    write_cache(path, memory, rec)
    assert np.array_equal(read_cache(path, rec)[0], memory)
    with pytest.raises(ValueError, match="identity"):
        read_cache(path, dict(rec, config_sha256="wrong"))


def test_attention_masks_padding_and_does_not_change_decision_branch():
    torch.manual_seed(17)
    original = SelectorModel(dim=16, hidden=8).eval()
    torch.manual_seed(17)
    model = AttentionSelector(dim=16, hidden=8).eval()
    x, memory = torch.randn(3, 16), torch.randn(3, 5, 16)
    mask = torch.tensor([[True, True, False, False, False]] * 3)
    first = model(x, memory, mask)
    other = memory.clone()
    other[:, 2:] = 10000
    second = model(x, other, mask)
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    torch.testing.assert_close(first["decision"], original(x)["decision"])
    changed = model(x, memory * 3, mask)
    torch.testing.assert_close(first["decision"], changed["decision"])
    assert not torch.allclose(first["token"], changed["token"])
    assert first["token"].shape[1] == 7
    with pytest.raises(ValueError, match="unmasked"):
        model(x, memory, torch.zeros_like(mask))


def test_teacher_forcing_trains_attention_and_uses_original_masked_loss():
    torch.manual_seed(17)
    model = AttentionSelector(dim=16, hidden=8)
    rows = generate("train")[:4]
    labels = {k: torch.from_numpy(v) for k, v in encode_targets(rows).items()}
    output = model(torch.randn(4, 16), torch.randn(4, 5, 16), torch.ones(4, 5, dtype=torch.bool), teacher=labels)
    loss = training_loss(output, labels, *loss_weights(labels))
    loss.backward()
    for name in ("memory_key", "memory_value", "query", "energy"):
        gradient = getattr(model, name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_h2_training_keeps_blind_closed_and_normalises_only_valid_training_memory(tmp_path, monkeypatch):
    from pathlib import Path

    from .ablation_contract import checksum, dumps, wrap_plan
    from .h0_features import config, record
    from .h2_memory import MAX_MEMORY
    from .h2_training import train
    from .run_ablation import save
    from .toy_dataset import contract_digest, digest

    dataset, features, memory, output = [tmp_path / n for n in ("data", "global", "memory", "output")]
    for folder in (dataset, features, memory):
        folder.mkdir()
    rows = dict(train=generate("train")[:8])
    rows["val"] = json.loads(json.dumps(rows["train"][:4]))
    for row in rows["val"]:
        row.update(id="val_" + row["id"], split="val")
    rows["val"][-1].update(
        domain="full_only", expected=dict(decision="replace", goals=[dict(type="turn", direction="left", angle_deg=45)])
    )
    rows["val"][-1]["target"] = wrap_plan(rows["val"][-1]["expected"])
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
    (dataset / "sealed/blind480.jsonl").write_text("never read")
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if "sealed" in path.parts:
            raise AssertionError("H2 opened blind data.")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    cfg = config()
    global_manifest = dict(
        complete=True,
        config=cfg,
        config_sha256=checksum(cfg),
        splits={},
        dataset_manifest_sha256=digest(dataset / "manifest.json"),
        contract_sha256=contract_digest(),
    )
    memory_cfg = dict(diagnostic_random_features=True)
    mem_manifest = dict(
        complete=True,
        config=memory_cfg,
        config_sha256=checksum(memory_cfg),
        splits={},
        dataset_manifest_sha256=digest(dataset / "manifest.json"),
        contract_sha256=contract_digest(),
    )
    rng = np.random.default_rng(17)
    for split, items in rows.items():
        np.savez_compressed(
            features / f"{split}.npz",
            ids=np.array([r["id"] for r in items]),
            x=rng.normal(size=(len(items), 2048)).astype(np.float16),
        )
        save(features / f"{split}_records.json", [record(r, split) for r in items])
        global_manifest["splits"][split] = dict(
            cache_sha256=digest(features / f"{split}.npz"), records_sha256=digest(features / f"{split}_records.json")
        )
        values = np.zeros((len(items), MAX_MEMORY, 2048), dtype=np.float16)
        recs = []
        for i, row in enumerate(items):
            values[i, :3] = i + 1 if split == "train" else 1000
            recs.append(
                dict(
                    identity(row, split, checksum(memory_cfg)),
                    tokens=3,
                    global_vector_exact_match=True,
                    spans=[dict(start=j, end=j + 1, role="event") for j in range(3)],
                )
            )
        np.save(memory / f"{split}.npy", values)
        save(memory / f"{split}_records.json", recs)
        mem_manifest["splits"][split] = dict(
            memory_sha256=digest(memory / f"{split}.npy"), records_sha256=digest(memory / f"{split}_records.json")
        )
    save(features / "manifest.json", global_manifest)
    mem_manifest["global_manifest_sha256"] = digest(features / "manifest.json")
    save(memory / "manifest.json", mem_manifest)
    result = train(dataset, features, memory, output, epochs=1, device="cpu")
    assert [r["seed"] for r in result["seeds"]] == [17, 43, 71]
    checkpoint = torch.load(output / "seed_17/selector.pt", weights_only=True)
    torch.testing.assert_close(checkpoint["state_dict"]["memory_mean"], torch.full((2048,), 4.5))
    assert checkpoint["training_config"]["selection_samples"] == 3
    report = json.loads((output / "seed_17/all240.json").read_text())
    assert len(report["predictions"]) == 4
    assert report["predictions"][-1]["expected"]["goals"][0]["angle_deg"] == 45
