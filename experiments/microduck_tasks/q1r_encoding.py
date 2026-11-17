"""Lossless continuation tokens, separated from synthetic semantic source rows."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .ablation_contract import checksum, dumps
from .q1r_prompt import messages_for, metadata, validate_prediction
from .run_ablation import save
from .toy_dataset import digest


def encode(processor, row):
    validate_prediction(row["expected"], row["request"])
    tokenizer = processor.tokenizer
    messages = messages_for(row["request"])
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    answer = dumps(row["expected"])
    full = processor.apply_chat_template(
        messages + [dict(role="assistant", content=answer)],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full.startswith(prefix) or full[len(prefix) :] != answer + tokenizer.eos_token + "\n":
        raise ValueError("Continuation template must preserve the complete answer and EOS.")
    prefix_ids = processor(text=[prefix], images=None, return_tensors="pt", padding=False)["input_ids"][0].tolist()
    ids = processor(text=[full[:-1]], images=None, return_tensors="pt", padding=False)["input_ids"][0].tolist()
    target = ids[len(prefix_ids) :]
    if ids[: len(prefix_ids)] != prefix_ids or target[-1] != tokenizer.eos_token_id:
        raise ValueError("Continuation token prefix or answer EOS differs from generation.")
    if tokenizer.decode(target[:-1], skip_special_tokens=False) != answer:
        raise ValueError("Continuation answer tokens do not reproduce the original JSON.")
    padded = ((len(ids) + 1023) // 1024) * 1024
    if padded > 4096:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceed 4096; shorten the authored case, never truncate.")
    return dict(
        id=row["id"],
        row=row,
        input_ids=ids,
        prefix_ids=prefix_ids,
        answer_ids=target,
        labels=[-100] * len(prefix_ids) + target,
        input_tokens=len(prefix_ids),
        supervised_tokens=len(target),
        total_tokens=len(ids),
        padded_tokens=padded,
        prefix_sha256=checksum(prefix),
        source_row_sha256=checksum(row),
    )


def prepare(data, model_dir, output):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        version="q1r-lossless-tokens-1",
        prompt=metadata(),
        splits={},
        maximum_sequence_length=4096,
        padding_multiple=1024,
        truncated=0,
        discarded=0,
        blind_access=False,
        human_reviewed=False,
    )
    for split in ("train", "val"):
        source = data / f"raw_{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        path = output / f"{split}.jsonl"
        lengths, groups, histories, families = [], Counter(), Counter(), set()
        with path.open("x") as stream:
            for row in rows:
                if row["split"] != split:
                    raise ValueError("A raw row is in the wrong split.")
                encoded = encode(processor, row)
                stream.write(dumps(encoded) + "\n")
                lengths.append(encoded["total_tokens"])
                groups[row["group"]] += 1
                histories[len(row["request"]["history"])] += 1
                families.add(row["family_id"])
        report["splits"][split] = dict(
            samples=len(rows),
            source_sha256=digest(source),
            sha256=digest(path),
            min_tokens=min(lengths),
            max_tokens=max(lengths),
            mean_tokens=sum(lengths) / len(lengths),
            groups=dict(groups),
            history_lengths=dict(histories),
            families=sorted(families),
        )
    if set(report["splits"]["train"]["families"]) & set(report["splits"]["val"]["families"]):
        raise ValueError("Training and validation language families overlap.")
    save(output / "manifest.json", report)
    return report


def load(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["prompt"] != metadata():
        raise ValueError("The encoded continuation prompt changed.")
    rows = {}
    for split, info in manifest["splits"].items():
        path = output / f"{split}.jsonl"
        if digest(path) != info["sha256"]:
            raise ValueError("Frozen continuation tokens changed.")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows[split]) != info["samples"]:
            raise ValueError("Token count and split manifest disagree.")
        for row in rows[split]:
            n = row["input_tokens"]
            if (
                checksum(row["row"]) != row["source_row_sha256"]
                or row["input_ids"][:n] != row["prefix_ids"]
                or row["input_ids"][n:] != row["answer_ids"]
                or row["labels"] != [-100] * n + row["answer_ids"]
                or row["total_tokens"] != len(row["input_ids"])
            ):
                raise ValueError("Continuation labels or source identity changed.")
    return manifest, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    print(dumps(prepare(args.data, args.model, args.output)))
