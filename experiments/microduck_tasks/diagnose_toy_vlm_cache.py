"""Compare cached and uncached greedy generation on one identical recorded request.

This runs no robot actions and measures implementation agreement, not task accuracy.
Run after other GPU jobs have exited; uncached decoding repeats the full prefill.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .setup_skill_head import MODEL_REVISION, WEIGHTS, WEIGHTS_SHA256, digest


def source_audit(model_dir):
    root = Path(model_dir)
    rows = []
    for metadata in sorted((root / ".cache/huggingface/download").glob("*.metadata")):
        name = metadata.name.removesuffix(".metadata")
        revision, etag, *_ = metadata.read_text().splitlines()
        row = dict(file=name, revision_matches=revision == MODEL_REVISION)
        if name == WEIGHTS:
            row.update(etag_matches_pinned_sha=etag == WEIGHTS_SHA256, content_checked_by_pinned_model_loader=True)
        else:
            data = (root / name).read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            row.update(content_matches_etag=etag in (sha, blob), sha256=sha)
        rows.append(row)
    return rows


def recorded_request(episode):
    for line in (episode / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["event"] == "dialogue" and row["request"]["event"]["kind"] == "user":
            images = {name: episode / row["media"][name] for name in ("front", "near")}
            for name, image in images.items():
                if digest(image) != row["media_sha256"][name]:
                    raise ValueError(f"Recorded image changed: {image}")
            return row["request"], images
    raise ValueError("Episode contains no user request with recorded images.")


def generate_once(generator, batch, length, *, use_cache, tokens, max_time):
    torch = generator.torch
    options = dict(generator.options)
    options.update(
        use_cache=use_cache,
        max_new_tokens=tokens,
        max_time=max_time,
        return_dict_in_generate=True,
        output_scores=True,
    )
    # Cloning preserves exactly the same token and image tensors for all three calls.
    fresh = {name: value.clone() for name, value in batch.items()}
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        result = generator.model.generate(**fresh, **options)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    ids = result.sequences[0, length:].tolist()
    scores = [score[0].float().cpu().numpy().copy() for score in result.scores]
    summary = dict(
        use_cache=use_cache,
        seconds=elapsed,
        ids=ids,
        text=generator.processor.tokenizer.decode(ids, skip_special_tokens=False),
        output_tokens=len(ids),
        requested_tokens=tokens,
        finished_with_eos=bool(ids and ids[-1] in options["eos_token_id"]),
    )
    del result, fresh
    torch.cuda.empty_cache()
    return summary, scores


def comparison(left, right, left_scores, right_scores):
    a, b = left["ids"], right["ids"]
    first = next((index for index, pair in enumerate(zip(a, b, strict=False)) if pair[0] != pair[1]), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    stop = min(len(left_scores), len(right_scores), first + 1 if first is not None else len(a))
    steps = []
    for index in range(stop):
        x, y = left_scores[index], right_scores[index]
        finite = np.isfinite(x) & np.isfinite(y)
        delta = (x[finite] - y[finite]).astype(np.float64)
        top_x, top_y = np.argpartition(x, -2)[-2:], np.argpartition(y, -2)[-2:]
        steps.append(
            dict(
                index=index,
                left_token=a[index],
                right_token=b[index],
                logits_max_abs_difference=float(np.abs(delta).max()) if delta.size else None,
                logits_rms_difference=float(np.sqrt(np.mean(delta**2))) if delta.size else None,
                finite_masks_equal=bool(np.array_equal(np.isfinite(x), np.isfinite(y))),
                left_top2_margin=float(abs(x[top_x[1]] - x[top_x[0]])),
                right_top2_margin=float(abs(y[top_y[1]] - y[top_y[0]])),
            )
        )
    return dict(
        exact_token_match=a == b,
        first_divergent_token_index=first,
        shared_prefix_steps=steps,
        later_scores_omitted_because_prefixes_differ=first is not None and stop < min(len(a), len(b)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--prompt-variant", choices=("full", "compact", "rules"), default="full")
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--max-time", type=float, default=180)
    args = parser.parse_args(argv)
    if not 1 <= args.tokens <= 128 or not 0 < args.max_time <= 600:
        parser.error("Use 1..128 tokens and a time limit within (0, 600] seconds per call.")
    if args.output.exists():
        parser.error("Output exists; choose a new diagnostic output path.")
    request, images = recorded_request(args.episode.resolve())
    from .toy_vlm_generation import ToyVLMGenerator

    generator = ToyVLMGenerator(args.model, prompt_variant=args.prompt_variant)
    batch, length = generator.encode_request(request, images)
    result = dict(
        diagnostic="cache_equivalence_not_task_accuracy",
        episode=str(args.episode.resolve()),
        source_events_sha256=digest(args.episode / "events.jsonl"),
        source_audit=source_audit(args.model),
        generator=generator.metadata,
        input_tokens=length,
        input_tensor_sha256={
            name: hashlib.sha256(value.contiguous().view(generator.torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for name, value in batch.items()
        },
        implementation_sha256=digest(__file__),
        scores_comparable_only_until_first_different_token=True,
        limitations=[
            "NF4/bfloat16 chunked prefill and recurrent decode can differ numerically even with correct caches.",
            "A short common prefix checks gross divergence, not complete semantic-output equivalence.",
            "After the first token difference, later logits have different prefixes and cannot be compared.",
        ],
        runs=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_scores = []
    for mode in (True, False, True):
        summary, scores = generate_once(
            generator, batch, length, use_cache=mode, tokens=args.tokens, max_time=args.max_time
        )
        result["runs"].append(summary)
        all_scores.append(scores)
        args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    runs = result["runs"]
    result["cached_vs_uncached"] = comparison(runs[0], runs[1], all_scores[0], all_scores[1])
    result["cached_repeat"] = comparison(runs[0], runs[2], all_scores[0], all_scores[2])
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("cached_vs_uncached", "cached_repeat")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
