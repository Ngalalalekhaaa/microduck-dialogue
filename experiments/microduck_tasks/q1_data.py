"""Freeze all H0 training/development rows for full-answer Q1 supervision."""

import argparse
import json
from pathlib import Path

from .ablation_contract import checksum, dumps
from .h0_features import read_data
from .q0_data import encode
from .run_ablation import save
from .toy_dataset import digest


def prepare(dataset, model, output):
    from transformers import AutoProcessor

    source, original = read_data(dataset)
    if {s: len(rr) for s, rr in original.items()} != {"train": 1200, "val": 240}:
        raise ValueError("Q1 requires the frozen 1200 training and 240 development rows.")
    if sum(r["domain"] == "common" for r in original["val"]) != 236:
        raise ValueError("Retain the 236 common-domain and four full-protocol development rows.")
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        version="q1-p6-full-answer-1",
        source_manifest_sha256=digest(dataset / "manifest.json"),
        source_sha256=source["files_sha256"],
        input_view="P6",
        eos_token_id=processor.tokenizer.eos_token_id,
        maximum_sequence_length=4096,
        padding_multiple=1024,
        truncated=0,
        discarded=0,
        blind_access=False,
        human_reviewed=False,
        splits={},
    )
    for split, rows in original.items():
        encoded = [encode(processor, row) for row in rows]
        path = output / f"{split}.jsonl"
        path.write_text("".join(dumps(r) + "\n" for r in encoded))
        manifest["splits"][split] = dict(
            count=len(rows),
            ids=[r["id"] for r in encoded],
            sha256=digest(path),
            minimum_tokens=min(r["total_tokens"] for r in encoded),
            maximum_tokens=max(r["total_tokens"] for r in encoded),
            maximum_supervised_tokens=max(r["supervised_tokens"] for r in encoded),
        )
    save(output / "manifest.json", manifest)
    return manifest


def load(data, source):
    manifest = json.loads((data / "manifest.json").read_text())
    original_manifest, original = read_data(source)
    if (
        manifest["source_manifest_sha256"] != digest(source / "manifest.json")
        or manifest["source_sha256"] != original_manifest["files_sha256"]
    ):
        raise ValueError("Q1 must use the same frozen source data as H0/H1/H2.")
    encoded = {}
    for split in ("train", "val"):
        path = data / f"{split}.jsonl"
        info = manifest["splits"][split]
        if digest(path) != info["sha256"]:
            raise ValueError(f"Q1 encoded {split} data changed.")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != info["count"] or [r["id"] for r in rows] != info["ids"]:
            raise ValueError("Q1 encoded IDs or counts changed.")
        for row, expected in zip(rows, original[split], strict=True):
            n = row["input_tokens"]
            if (
                row["row"] != expected
                or row["row"]["split"] != split
                or row["source_row_sha256"] != checksum(expected)
                or row["input_ids"][:n] != row["prefix_ids"]
                or row["labels"] != [-100] * n + row["answer_ids"]
                or row["input_ids"][n:] != row["answer_ids"]
                or not row["answer_ids"]
                or row["answer_ids"][-1] != manifest["eos_token_id"]
                or len(row["input_ids"]) != row["total_tokens"]
                or row["padded_tokens"] != ((row["total_tokens"] + 1023) // 1024) * 1024
                or row["padded_tokens"] > 4096
            ):
                raise ValueError("Q1 original row, complete answer/EOS or input boundaries changed.")
        encoded[split] = rows
    if set(manifest["splits"]["train"]["ids"]) & set(manifest["splits"]["val"]["ids"]):
        raise ValueError("Training and development IDs overlap.")
    return manifest, encoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.dataset, args.model, args.output)
    print(dumps({s: {k: v for k, v in info.items() if k != "ids"} for s, info in result["splits"].items()}))
