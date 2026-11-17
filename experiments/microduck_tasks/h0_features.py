"""Resumable P6 input features; this entry never opens the sealed blind dataset."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .ablation_contract import checksum, dumps
from .ablation_prompt import messages_for, metadata, system_prompt
from .extract_toy_head_features import _cache_path, _read_cache, _save_cache
from .run_ablation import save
from .toy_dataset import contract_digest, digest
from .toy_head_features import ToyFrozenFeatures, _right_padded_lengths
from .toy_head_features import feature_config as old_feature_config


def read_data(dataset):
    """Only train.jsonl and val.jsonl may be opened, regardless of other files."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["training_splits"] != ["train", "val"] or manifest["input_view"] != "P6":
        raise ValueError("Require the frozen H0 train/val contract and P6 input view.")
    rows = {}
    for split in ("train", "val"):
        path = dataset / f"{split}.jsonl"
        if digest(path) != manifest["files_sha256"][split]:
            raise ValueError(f"Frozen {split} data changed.")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        if any(r["split"] != split or r["requires_pixels"] for r in rows[split]):
            raise ValueError("Wrong split or a pixel-dependent sample in the fixed text view.")
        ids = [r["id"] for r in rows[split]]
        if len(set(ids)) != len(ids) or any(not name.replace("_", "").isalnum() for name in ids):
            raise ValueError("Require unique path-safe row identifiers.")
    return manifest, rows


def config():
    result = old_feature_config(False)
    result.update(
        version="h0-p6-input-feature-1",
        implementation_sha256=digest(__file__),
        renderer=metadata("P6"),
        prompt_sha256=checksum(system_prompt("P6")),
        input_view="P6",
        answer_tokens_in_input=False,
        blind_access=False,
    )
    return result


class Features(ToyFrozenFeatures):
    def encode_batch(self, requests, images, *, vision=False):
        import torch

        if vision or not 1 <= len(requests) <= 4:
            raise ValueError("H0 requires 1..4 text-only inputs.")
        self.processor.tokenizer.padding_side = "right"
        texts = [
            self.processor.apply_chat_template(
                messages_for(r, "P6"), tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for r in requests
        ]
        batch = dict(self.processor(text=texts, images=None, return_tensors="pt", padding=True))
        lengths = _right_padded_lengths(batch["attention_mask"])
        if int(lengths.max()) > 4096:
            raise ValueError("Input exceeds 4096 tokens; truncation is forbidden.")
        extra = (-batch["input_ids"].shape[1]) % 1024
        for key, value in (
            ("input_ids", self.processor.tokenizer.pad_token_id),
            ("attention_mask", 0),
            ("mm_token_type_ids", 0),
            ("token_type_ids", 0),
        ):
            if key in batch:
                batch[key] = torch.nn.functional.pad(batch[key], (0, extra), value=value)
        self.last_rendered_hashes = [checksum(t) for t in texts]
        return batch, lengths.tolist()


def record(row, split):
    return dict(
        id=row["id"],
        split=split,
        source_row_sha256=checksum(row),
        messages_sha256=checksum(messages_for(row["request"], "P6")),
    )


def extract(dataset, model_dir, output, *, batch_size=1):
    if not 1 <= batch_size <= 4:
        raise ValueError("Choose batch size 1..4.")
    data_manifest, rows = read_data(dataset)
    cfg = config()
    identity = dict(
        config=cfg,
        config_sha256=checksum(cfg),
        dataset_manifest_sha256=digest(dataset / "manifest.json"),
        source_hashes=data_manifest["files_sha256"],
        contract_sha256=contract_digest(),
        expected_rows={s: len(r) for s, r in rows.items()},
        blind_access=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else dict(identity, complete=False, splits={})
    if any(manifest.get(k) != v for k, v in identity.items()):
        raise ValueError("Feature identity changed; start a separate output directory.")
    save(path, manifest)
    pending, cached = [], 0
    for split, items in rows.items():
        for row in items:
            target = _cache_path(output, split, row["id"])
            expected = record(row, split)
            if target.exists():
                _read_cache(target, expected, identity["config_sha256"])
                cached += 1
            else:
                pending.append((split, row, expected))
    started = time.monotonic()
    if pending:
        model = Features(model_dir)
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            vectors, timings = model.extract_batch([r["request"] for _, r, _ in chunk], [{}] * len(chunk), vision=False)
            for (split, row, rec), vector, timing, rendered in zip(
                chunk, vectors, timings, model.last_rendered_hashes, strict=True
            ):
                _save_cache(
                    _cache_path(output, split, row["id"]),
                    vector,
                    dict(rec, **timing, config_sha256=identity["config_sha256"], rendered_sha256=rendered),
                )
            progress = dict(
                stage="features",
                completed=cached + start + len(chunk),
                total=sum(map(len, rows.values())),
                new_rows=start + len(chunk),
                elapsed_s=time.monotonic() - started,
                last_batch=timings[-1],
                blind_access=False,
            )
            save(output / "progress.json", progress)
            print(dumps(progress), flush=True)
    for split, items in rows.items():
        pairs = [
            _read_cache(_cache_path(output, split, r["id"]), record(r, split), identity["config_sha256"]) for r in items
        ]
        vectors, records = zip(*pairs, strict=True)
        with (output / f"{split}.npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, x=np.stack(vectors), ids=np.array([r["id"] for r in items]))
        (output / f"{split}.npz.tmp").replace(output / f"{split}.npz")
        save(output / f"{split}_records.json", list(records))
        manifest["splits"][split] = dict(
            samples=len(items),
            cache_sha256=digest(output / f"{split}.npz"),
            records_sha256=digest(output / f"{split}_records.json"),
        )
    manifest.update(complete=True, elapsed_s=time.monotonic() - started)
    save(path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    extract(args.dataset, args.model, args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
