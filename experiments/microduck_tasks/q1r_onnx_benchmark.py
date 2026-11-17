"""Benchmark CUDA ORT GenAI on frozen Q1R token inputs without rewriting prompts."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

MAX_NEW_TOKENS = 1536
MAX_TIME_SECONDS = 180.0
BACKEND = "q1r_onnx_genai_cuda"


class SequenceMismatch(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__("Generator sequence mismatch: " + json.dumps(diagnostic, ensure_ascii=False))


def sequence_diagnostic(prefix, tokens, sequence):
    expected = prefix + tokens
    first = next((i for i, (want, got) in enumerate(zip(expected, sequence, strict=False)) if want != got), None)
    if first is None and len(expected) != len(sequence):
        first = min(len(expected), len(sequence))
    suffix = sequence[len(prefix) :]
    return dict(
        prefix_length=len(prefix),
        collected_output_length=len(tokens),
        expected_sequence_length=len(expected),
        actual_sequence_length=len(sequence),
        actual_output_length=len(suffix),
        prefix_matches=sequence[: len(prefix)] == prefix,
        first_difference_index=first,
        first_difference_output_index=first - len(prefix) if first is not None else None,
        expected_token=expected[first] if first is not None and first < len(expected) else None,
        actual_token=sequence[first] if first is not None and first < len(sequence) else None,
        prefix_tail=prefix[-8:],
        collected_output_tail=tokens[-16:],
        actual_sequence_tail=sequence[-16:],
        collected_output_head=tokens[:8],
        actual_output_head=suffix[:8],
    )


def sequence_contract(prefix, tokens, sequence, endings, done):
    if sequence == prefix + tokens:
        return "exact"
    # GenAI CUDA returns the sampled stop token but omits it from the completed sequence.
    if done and tokens and tokens[-1] in endings and sequence == prefix + tokens[:-1]:
        return "terminal_eos_omitted_by_genai"
    return None


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def load_inputs(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Provide a nonempty frozen input file with unique case IDs.")
    for row in rows:
        tokens = row.get("prefix_ids")
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError(f"{row['id']}: prefix_ids must contain nonnegative integer tokens.")
    return rows


def load_tokenizer(inputs, model):
    from tokenizers import Tokenizer

    candidates = [inputs.parent / "merged_effective_text", model]
    folder = next(
        (
            path
            for path in candidates
            if (path / "tokenizer.json").is_file() and (path / "tokenizer_config.json").is_file()
        ),
        None,
    )
    if folder is None:
        raise FileNotFoundError(
            "Keep tokenizer.json and tokenizer_config.json beside the merged model or ONNX package."
        )
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    config = json.loads((folder / "tokenizer_config.json").read_text())
    special_ids = {}
    for name in ("eos_token", "pad_token"):
        token = config.get(name)
        if isinstance(token, dict):
            token = token.get("content")
        token_id = tokenizer.token_to_id(token) if isinstance(token, str) else None
        if token_id is None:
            raise ValueError(f"The reference tokenizer does not define {name}; do not guess generation endings.")
        special_ids[name] = token_id
    return tokenizer, folder, list(dict.fromkeys(special_ids.values())), special_ids["pad_token"]


def generate(og, np, model, tokenizer, row, endings, *, context_length, stop_after_tokens=None):
    if stop_after_tokens is not None and (
        type(stop_after_tokens) is not int or not 1 <= stop_after_tokens <= MAX_NEW_TOKENS
    ):
        raise ValueError("stop_after_tokens must be an integer from 1 to 1536 or None.")
    prefix = row["prefix_ids"]
    maximum = len(prefix) + MAX_NEW_TOKENS
    if maximum > context_length:
        raise ValueError(
            f"{row['id']}: {len(prefix)} prompt + {MAX_NEW_TOKENS} output tokens exceed model context "
            f"{context_length}. Re-export with sufficient context; inputs and output budgets are never truncated."
        )
    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=maximum,
        min_length=0,
        num_beams=1,
        num_return_sequences=1,
        repetition_penalty=1.0,
    )
    input_ids = np.asarray([prefix], dtype=np.int32)
    tokens, first_token_seconds = [], None
    started = time.monotonic()
    generator = og.Generator(model, params)
    try:
        generator.append_tokens(input_ids)
        limit_reason = None
        while not generator.is_done() and len(tokens) < MAX_NEW_TOKENS:
            generator.generate_next_token()
            # The CPU array forces the generated token to be available to the caller.
            next_tokens = generator.get_next_tokens().reshape(-1).tolist()
            if len(next_tokens) != 1:
                raise RuntimeError("This benchmark requires batch-one, single-token greedy decoding.")
            if first_token_seconds is None:
                first_token_seconds = time.monotonic() - started
            tokens.extend(int(token) for token in next_tokens)
            if tokens[-1] in endings:
                break
            if stop_after_tokens is not None and len(tokens) >= stop_after_tokens:
                limit_reason = "warmup_token_limit"
                break
            if time.monotonic() - started >= MAX_TIME_SECONDS:
                limit_reason = "max_time"
                break
        done = generator.is_done()
        sequence = generator.get_sequence(0).tolist()
        validation = sequence_contract(prefix, tokens, sequence, endings, done)
        if validation is None:
            diagnostic = sequence_diagnostic(prefix, tokens, sequence)
            diagnostic.update(
                case_id=row["id"],
                generator_token_count=generator.token_count(),
                generator_done=done,
                elapsed_seconds=time.monotonic() - started,
                reference_eos_ids=endings,
                collected_raw_text=tokenizer.decode(tokens, skip_special_tokens=True),
            )
            raise SequenceMismatch(diagnostic)
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        elapsed = time.monotonic() - started
        finished_with_eos = bool(tokens and tokens[-1] in endings)
        if not finished_with_eos and limit_reason is None:
            limit_reason = "max_new_tokens" if len(tokens) == MAX_NEW_TOKENS else "generator_done_without_reference_eos"
        return dict(
            raw_text=raw,
            sequence_validation=validation,
            timing=dict(
                seconds=elapsed,
                input_tokens=len(prefix),
                output_tokens=len(tokens),
                ttft_seconds=first_token_seconds,
                finish_reason="eos" if finished_with_eos else "limit",
                limit_reason=limit_reason,
            ),
            generated_ids=tokens,
        )
    finally:
        del generator
        del params


def benchmark(args):
    root = Path(__file__).resolve().parents[2]
    overlay = root / ".cache/microduck_head/onnx_deps"
    if not overlay.is_dir():
        raise FileNotFoundError(f"Install the isolated ONNX dependencies first: {overlay}")
    sys.path.insert(0, str(overlay))

    import numpy as np
    import onnxruntime as ort
    import onnxruntime_genai as og

    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    inputs, model_path, output = args.inputs.resolve(), args.model.resolve(), args.output.resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Choose a new output filename; existing benchmark evidence is never overwritten.")
    rows = load_inputs(inputs)
    tokenizer, tokenizer_path, endings, pad = load_tokenizer(inputs, model_path)
    config_path = model_path / "genai_config.json"
    model_config = json.loads(config_path.read_text())
    context_length = int(model_config["model"]["context_length"])
    for row in rows:
        if len(row["prefix_ids"]) + MAX_NEW_TOKENS > context_length:
            raise ValueError(f"{row['id']}: full prompt plus 1536 output tokens exceed model context {context_length}.")
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available or not og.is_cuda_available():
        raise RuntimeError(f"CUDA is required; refusing CPU benchmark. ORT providers: {available}")
    configuration = og.Config(str(model_path))
    configuration.overlay(json.dumps(dict(model=dict(eos_token_id=endings, pad_token_id=pad))))
    configuration.clear_providers()
    configuration.append_provider("cuda")
    metadata = dict(
        status="loading",
        backend=BACKEND,
        model=str(model_path),
        genai_config_sha256=digest(config_path),
        inputs=str(inputs),
        inputs_sha256=digest(inputs),
        tokenizer=str(tokenizer_path),
        tokenizer_sha256=digest(tokenizer_path / "tokenizer.json"),
        script_sha256=digest(Path(__file__)),
        ort_version=ort.__version__,
        genai_version=og.__version__,
        ort_module=ort.__file__,
        genai_module=og.__file__,
        available_providers=available,
        requested_providers=["CUDAExecutionProvider"],
        eos_token_ids=endings,
        pad_token_id=pad,
        max_new_tokens=MAX_NEW_TOKENS,
        max_time_seconds=MAX_TIME_SECONDS,
        do_sample=False,
        context_length=context_length,
        repeats=args.repeats,
        case_count=len(rows),
        warmup_count=1,
        prefix_cache_shared=False,
        fresh_generator_per_request=True,
        sequence_validation="Exact prefix/body; only a final reference EOS may be absent when generator.is_done().",
        timing_scope="Generator allocation, append frozen tokens, greedy generation, CPU token reads, final decode",
        ttft_scope="Start of Generator allocation through first generated token returned to CPU",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save(metadata_path, metadata)
    try:
        started = time.monotonic()
        model = og.Model(configuration)
        metadata.update(load_seconds=time.monotonic() - started, device_type=model.device_type)
        if str(model.device_type).lower() != "cuda":
            raise RuntimeError(f"Loaded model reports {model.device_type}; refusing silent CPU fallback.")
        metadata["status"] = "warming_up"
        save(metadata_path, metadata)
        warmup = generate(og, np, model, tokenizer, rows[0], endings, context_length=context_length)
        metadata.update(status="running", warmup_case=rows[0]["id"], warmup_timing=warmup["timing"])
        save(metadata_path, metadata)
        completed = 0
        with output.open("x") as stream:
            for repeat in range(args.repeats):
                for row in rows:
                    prediction = generate(og, np, model, tokenizer, row, endings, context_length=context_length)
                    record = dict(id=row["id"], repeat=repeat, backend=BACKEND, **prediction)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    completed += 1
                    print(json.dumps(dict(id=row["id"], repeat=repeat, **prediction["timing"])), flush=True)
        metadata.update(status="complete", measured_records=completed, output_sha256=digest(output))
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        if isinstance(error, SequenceMismatch):
            metadata["sequence_diagnostic"] = error.diagnostic
        raise
    finally:
        save(metadata_path, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL filename; metadata uses .metadata.json.")
    parser.add_argument("--repeats", type=int, default=2)
    benchmark(parser.parse_args())
