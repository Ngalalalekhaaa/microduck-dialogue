"""Input-only memory layout and immutable per-row cache for H2."""

import json

import numpy as np

from .ablation_contract import checksum, dumps
from .ablation_prompt import messages_for
from .h0_features import config as global_config
from .toy_dataset import digest

PREFIX_BINS = 32
MAX_MEMORY = 416
ROLES = ("observation", "execution", "history", "event")


def config():
    return dict(
        version="h2-p6-memory-1",
        global_config=global_config(),
        implementation_sha256=digest(__file__),
        max_memory=MAX_MEMORY,
        prefix_bins=PREFIX_BINS,
        pooling="32 contiguous prefix means; every observation/execution/history/event/suffix token kept",
        layer="last_hidden_state",
        dtype="float16",
        causal=True,
        current_answer_in_input=False,
        blind_access=False,
    )


def layout(tokenizer, request):
    messages = messages_for(request, "P6")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    payload = messages[1]["content"][-1]["text"]
    offset = text.rfind(payload)
    if offset < 0:
        raise ValueError("The exact P6 payload must occur in the rendered input.")
    values = json.loads(payload)
    cursor, boundaries = offset + 1, {}
    for name, value in values.items():
        boundaries[name] = cursor
        cursor += len(dumps(name) + ":" + dumps(value)) + 1
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets, ids = encoding["offset_mapping"], encoding["input_ids"]
    first = next(i for i, (_, end) in enumerate(offsets) if end > boundaries["observation"])
    if first < PREFIX_BINS or len(ids) > 4096:
        raise ValueError("Unexpected P6 token layout or input beyond 4096; do not truncate.")
    edges = np.linspace(0, first, PREFIX_BINS + 1, dtype=int).tolist()
    spans = [dict(start=a, end=b, role="prefix_mean") for a, b in zip(edges[:-1], edges[1:], strict=True)]
    for index in range(first, len(ids)):
        # Tokens crossing a JSON key boundary belong to the following field.
        role = next((name for name in reversed(ROLES) if offsets[index][1] > boundaries[name]), "observation")
        if offsets[index][0] >= offset + len(payload):
            role = "generation_prefix"
        spans.append(dict(start=index, end=index + 1, role=role))
    if len(spans) > MAX_MEMORY:
        raise ValueError(f"Memory requires {len(spans)} slots, beyond {MAX_MEMORY}; truncation is forbidden.")
    return text, ids, spans


def pool(hidden, spans):
    """Float32 means over disjoint spans, then FP16 storage; no answer or padded token."""
    import torch

    values = torch.stack([hidden[s["start"] : s["end"]].float().mean(0) for s in spans])
    array = values.cpu().numpy().astype(np.float16)
    if array.shape != (len(spans), 2048) or not np.isfinite(array).all():
        raise ValueError("Require finite 2048-dimensional memory slots.")
    result = np.zeros((MAX_MEMORY, 2048), dtype=np.float16)
    result[: len(spans)] = array
    return result


def write_cache(path, memory, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, memory=memory, record=np.array(dumps(record)))
    tmp.replace(path)


def read_cache(path, expected):
    with np.load(path, allow_pickle=False) as data:
        memory, record = data["memory"], json.loads(data["record"].item())
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("H2 memory cache identity mismatch.")
    length = len(record["spans"])
    if memory.shape != (MAX_MEMORY, 2048) or not 1 <= length <= MAX_MEMORY or not np.isfinite(memory).all():
        raise ValueError("Invalid H2 memory shape, mask or values.")
    if np.any(memory[length:]):
        raise ValueError("Padded memory must be zero and masked.")
    return memory, record


def identity(row, split, config_sha256):
    return dict(id=row["id"], split=split, source_row_sha256=checksum(row), config_sha256=config_sha256)
