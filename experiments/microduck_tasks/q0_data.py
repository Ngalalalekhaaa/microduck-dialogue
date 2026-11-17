"""Freeze a training-only Q0 subset and audit exact P6 answer supervision."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .ablation_contract import checksum, dumps
from .ablation_prompt import messages_for
from .h0_features import read_data
from .run_ablation import save
from .toy_dataset import digest


def encode(processor, row):
    tokenizer = processor.tokenizer
    messages = messages_for(row["request"], "P6")
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    answer = dumps(row["expected"])
    full = processor.apply_chat_template(
        messages + [dict(role="assistant", content=answer)],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full.startswith(prefix) or full[len(prefix) :] != answer + tokenizer.eos_token + "\n":
        raise ValueError("SFT answer prefix differs from actual P6 generation.")
    # Supervise through EOS; the template newline after EOS is outside the answer.
    full = full[:-1]
    p = processor(text=[prefix], images=None, return_tensors="pt", padding=False)
    f = processor(text=[full], images=None, return_tensors="pt", padding=False)
    prefix_ids, ids = p["input_ids"][0].tolist(), f["input_ids"][0].tolist()
    if ids[: len(prefix_ids)] != prefix_ids or ids[-1] != tokenizer.eos_token_id:
        raise ValueError("Token boundaries changed or answer EOS is missing.")
    target = ids[len(prefix_ids) :]
    if tokenizer.decode(target[:-1], skip_special_tokens=False) != answer or not target:
        raise ValueError("The entire answer JSON must be preserved exactly.")
    padded = ((len(ids) + 1023) // 1024) * 1024
    if padded > 4096:
        raise ValueError("Full input and answer exceed 4096 tokens; no truncation or deletion is allowed.")
    labels = [-100] * len(prefix_ids) + target
    return dict(
        id=row["id"],
        row=row,
        input_ids=ids,
        prefix_ids=prefix_ids,
        labels=labels,
        answer_ids=target,
        prefix_sha256=checksum(prefix),
        source_row_sha256=checksum(row),
        input_tokens=len(prefix_ids),
        supervised_tokens=len(target),
        total_tokens=len(ids),
        padded_tokens=padded,
    )


def choose(encoded):
    rng = np.random.default_rng(17)
    chosen = [max(encoded, key=lambda r: r["total_tokens"])]
    ids = {chosen[0]["id"]}

    def take(predicate, count):
        available = [r for r in encoded if predicate(r["row"]) and r["id"] not in ids]
        if len(available) < count:
            raise ValueError("Insufficient training rows for the predeclared Q0 coverage.")
        for index in rng.choice(len(available), count, replace=False):
            row = available[int(index)]
            chosen.append(row)
            ids.add(row["id"])

    for kind in ("move", "turn"):
        take(lambda r, kind=kind: r["group"] == "single" and r["expected"]["goals"][0]["type"] == kind, 2)
    take(lambda r: r["group"] == "single" and r["expected"]["goals"][0]["type"] == "perform", 4)
    take(lambda r: r["group"] == "multi3", 6)
    take(lambda r: r["group"] == "multi6", 6 - sum(x["row"]["group"] == "multi6" for x in chosen))
    for decision, count in (("decline", 4), ("append", 4), ("clarify", 2), ("cancel", 2)):
        take(lambda r, decision=decision: r["expected"]["decision"] == decision, count)
    if len(chosen) != 32:
        raise ValueError("Q0 must contain exactly 32 training-only rows, including the globally longest training row.")
    return chosen


def prepare(dataset, model, output):
    from transformers import AutoProcessor

    manifest, rows = read_data(dataset)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    encoded = {s: [encode(processor, r) for r in rr] for s, rr in rows.items()}
    selected = choose(encoded["train"])
    output.mkdir(parents=True, exist_ok=False)
    path = output / "train32.jsonl"
    path.write_text("".join(dumps(r) + "\n" for r in selected))
    report = dict(
        version="q0-p6-label-audit-1",
        input_view="P6",
        blind_access=False,
        source_dataset_manifest_sha256=digest(dataset / "manifest.json"),
        source_train_sha256=manifest["files_sha256"]["train"],
        data_sha256=digest(path),
        samples=len(selected),
        selection_seed=17,
        ids=[r["id"] for r in selected],
        decisions=dict(Counter(r["row"]["expected"]["decision"] for r in selected)),
        goal_lengths=dict(Counter(len(r["row"]["expected"]["goals"]) for r in selected)),
        longest_training_id=max(encoded["train"], key=lambda r: r["total_tokens"])["id"],
        maximum_sequence_length=4096,
        padding_multiple=1024,
        truncated=0,
        discarded=0,
        loss="only final answer JSON and EOS; logits at corresponding preceding input positions",
        implementation_sha256=digest(__file__),
        full_input_answer_lengths={
            s: dict(
                samples=len(rr), minimum=min(r["total_tokens"] for r in rr), maximum=max(r["total_tokens"] for r in rr)
            )
            for s, rr in encoded.items()
        },
        evaluation="same 32 training rows before/after fitting; no generalisation claim",
    )
    save(output / "manifest.json", report)
    return report


def load(data):
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["data_sha256"] != digest(data / "train32.jsonl"):
        raise ValueError("Q0 frozen subset changed.")
    rows = [json.loads(line) for line in (data / "train32.jsonl").read_text().splitlines()]
    if [r["id"] for r in rows] != manifest["ids"] or any(r["row"]["split"] != "train" for r in rows):
        raise ValueError("Q0 uses only its frozen training IDs.")
    for row in rows:
        n = row["input_tokens"]
        if (
            checksum(row["row"]) != row["source_row_sha256"]
            or row["labels"][:n] != [-100] * n
            or row["input_ids"][:n] != row["prefix_ids"]
            or row["labels"][n:] != row["answer_ids"]
            or row["input_ids"][n:] != row["answer_ids"]
        ):
            raise ValueError("Q0 labels, token boundaries or original row changed.")
    return manifest, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(dumps(prepare(args.dataset, args.model, args.output)))
