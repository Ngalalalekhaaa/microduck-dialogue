"""Audited Q1-weight initialization and resumable continuation checkpoints."""

import json
import math
import os
import random
import time
from datetime import UTC, datetime

import numpy as np
import torch

from .q0_model import load_model, weights_digest
from .run_ablation import save
from .toy_dataset import digest


def load_continuation(model_dir, checkpoint, output):
    identity = json.loads((checkpoint / "identity.json").read_text())
    if identity.get("step") != 100:
        raise ValueError("Continue from the previously selected Q1 checkpoint_100.")
    model, processor, metadata = load_model(model_dir, output)
    result = model.load_adapter(str(checkpoint), adapter_name="default", is_trainable=True, local_files_only=True)
    model.set_adapter("default")
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Q1 adapter load was incomplete: {result}")
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if list(params) != metadata["trainable_names"] or weights_digest(params) != identity["weights_sha256"]:
        raise ValueError("Continuation must begin with the exact selected Q1 weights, not a fresh adapter.")
    model.config.use_cache = False
    save(
        output / "initialization.json",
        dict(
            parent_checkpoint=str(checkpoint.resolve()),
            parent_adapter_sha256=digest(checkpoint / "adapter_model.safetensors"),
            parent_weights_sha256=identity["weights_sha256"],
            actual_weights_sha256=weights_digest(params),
            exact_parent_weights=True,
            new_adapter=False,
            optimizer="fresh AdamW state for the new dialogue objective",
            trained_output="decision+goals+reply",
        ),
    )
    return model, processor, params


def batches(count, epochs, accumulation, seed):
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(count).tolist()
        for start in range(0, count, accumulation):
            yield epoch, order[start : start + accumulation]


def learning_rate(step, total, peak, warmup):
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)


def checkpoint(model, optimizer, params, output, step, config, counters):
    path = output / f"checkpoint_{step:06d}"
    if path.exists():
        identity = json.loads((path / "identity.json").read_text())
        if (
            identity["step"] != step
            or identity["weights_sha256"] != weights_digest(params)
            or identity["data_sha256"] != config["data_manifest_sha256"]
            or identity["prompt"] != config["prompt"]
            or not (path / "trainer_state.pt").is_file()
        ):
            raise ValueError("An existing continuation checkpoint has a different or incomplete identity.")
        return path
    temporary = output / f".checkpoint_{step:06d}.tmp"
    if temporary.exists():
        temporary.rename(output / f".checkpoint_{step:06d}.interrupted_{time.time_ns()}")
    temporary.mkdir(exist_ok=False)
    model.save_pretrained(temporary, safe_serialization=True, save_embedding_layers=False)
    save(
        temporary / "identity.json",
        dict(
            step=step,
            weights_sha256=weights_digest(params),
            parent=config["parent_checkpoint"],
            data_sha256=config["data_manifest_sha256"],
            prompt=config["prompt"],
        ),
    )
    torch.save(
        dict(
            optimizer=optimizer.state_dict(),
            step=step,
            counters=counters,
            config=config,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        ),
        temporary / "trainer_state.pt",
    )
    temporary.rename(path)
    return path


class Status:
    def __init__(self, root):
        self.root = root
        self.value = dict(pid=os.getpid(), started_at=datetime.now(UTC).isoformat(), complete=False, blind_access=False)

    def __call__(self, state, **fields):
        self.value.update(state=state, updated_at=datetime.now(UTC).isoformat(), **fields)
        save(self.root / "status.json", self.value)
        print(json.dumps(self.value, ensure_ascii=False, allow_nan=False), flush=True)


def restore(path, model, optimizer, params, config):
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    if state["config"] != config:
        raise ValueError("Resume configuration differs from the original continuation.")
    result = model.load_adapter(str(path), adapter_name="default", is_trainable=True, local_files_only=True)
    identity = json.loads((path / "identity.json").read_text())
    if result.missing_keys or result.unexpected_keys or weights_digest(params) != identity["weights_sha256"]:
        raise ValueError("Resume adapter identity mismatch.")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return state["step"], state["counters"]
