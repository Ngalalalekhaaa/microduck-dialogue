"""Free generation on the fixed training subset, without any answer tokens."""

import time

import torch

from .ablation_score import score, summarize
from .run_ablation import save


def generate(model, processor, row):
    ids = torch.tensor([row["prefix_ids"]], dtype=torch.long, device="cuda")
    tokenizer = processor.tokenizer
    options = dict(
        do_sample=False,
        use_cache=True,
        logits_to_keep=1,
        max_new_tokens=768,
        max_time=90.0,
        eos_token_id=list(dict.fromkeys([tokenizer.eos_token_id, tokenizer.pad_token_id])),
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.zeros_like(ids), **options
        )
    torch.cuda.synchronize()
    tokens = output[0, len(row["prefix_ids"]) :].tolist()
    raw = tokenizer.decode(tokens, skip_special_tokens=True)
    timing = dict(
        seconds=time.monotonic() - started,
        input_tokens=len(row["prefix_ids"]),
        output_tokens=len(tokens),
        finish_reason="eos" if tokens and tokens[-1] in options["eos_token_id"] else "limit",
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
    )
    return raw, timing


def evaluate(model, processor, rows, output, stage, notify):
    directory = output / stage
    directory.mkdir(exist_ok=False)
    records = []
    for row in rows:
        raw, timing = generate(model, processor, row)
        record = dict(row["row"])
        record.update(
            raw_text=raw,
            timing=timing,
            score=score(raw, row["row"], plan_only=True),
            generation_error=None,
            source="q0_same_training_subset",
            prefix_sha256=row["prefix_sha256"],
            host_control=None,
        )
        save(directory / f"{row['id']}.json", record)
        records.append(record)
        notify(stage, completed=len(records), total=len(rows), last_id=row["id"])
    report = dict(
        summary=summarize(records, expected_count=len(rows)),
        predictions=records,
        training_subset=True,
        generalisation_scored=False,
        blind_access=False,
    )
    save(directory / "report.json", report)
    return report
