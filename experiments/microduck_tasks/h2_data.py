"""Validated CPU memory maps; only each minibatch moves to the training device."""

import json

import numpy as np
import torch

from .ablation_contract import checksum
from .h0_training import load_data
from .h2_memory import MAX_MEMORY, identity
from .toy_dataset import digest


class MemoryData:
    def __init__(self, dataset, features, memory):
        self.global_manifest, self.rows, self.arrays = load_data(dataset, features)
        self.manifest = json.loads((memory / "manifest.json").read_text())
        manifest = self.manifest
        if not manifest["complete"] or set(manifest["splits"]) != {"train", "val"}:
            raise ValueError("Require completed train/val memory only.")
        if (
            manifest["global_manifest_sha256"] != digest(features / "manifest.json")
            or manifest["dataset_manifest_sha256"] != digest(dataset / "manifest.json")
            or checksum(manifest["config"]) != manifest["config_sha256"]
        ):
            raise ValueError("H2 memory provenance differs from frozen H0 inputs.")
        self.memories, self.lengths = {}, {}
        for split, items in self.rows.items():
            path, recpath = memory / f"{split}.npy", memory / f"{split}_records.json"
            expected = manifest["splits"][split]
            if digest(path) != expected["memory_sha256"] or digest(recpath) != expected["records_sha256"]:
                raise ValueError("H2 memory bytes or provenance changed.")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            records = json.loads(recpath.read_text())
            if values.shape != (len(items), MAX_MEMORY, 2048) or values.dtype != np.float16:
                raise ValueError("H2 memory shape or dtype differs.")
            lengths = []
            for index, (row, rec) in enumerate(zip(items, records, strict=True)):
                if any(rec.get(k) != v for k, v in identity(row, split, manifest["config_sha256"]).items()):
                    raise ValueError("H2 memory and labels are misaligned.")
                spans = rec["spans"]
                if (
                    not rec["global_vector_exact_match"]
                    or not 1 <= len(spans) <= MAX_MEMORY
                    or spans[0]["start"] != 0
                    or spans[-1]["end"] != rec["tokens"]
                    or any(a["end"] != b["start"] for a, b in zip(spans[:-1], spans[1:], strict=True))
                ):
                    raise ValueError("Incomplete input coverage or different last-token feature.")
                if not np.isfinite(values[index]).all() or np.any(values[index, len(spans) :]):
                    raise ValueError("Nonfinite memory or nonzero padding.")
                lengths.append(len(spans))
            self.memories[split], self.lengths[split] = values, np.array(lengths)

    def batch(self, split, indices, device):
        indices = np.asarray(indices)
        lengths = self.lengths[split][indices]
        count = int(lengths.max())
        memory = torch.from_numpy(self.memories[split][indices, :count].astype(np.float32)).to(device)
        mask = torch.from_numpy(np.arange(count)[None, :] < lengths[:, None]).to(device)
        features = self.arrays[split][torch.as_tensor(indices)].to(device)
        return features, memory, mask

    def normalise(self, model):
        # Streaming training-only moments omit masked slots and all validation rows.
        total, square, count = np.zeros(2048), np.zeros(2048), 0
        for row, length in zip(self.memories["train"], self.lengths["train"], strict=True):
            values = row[:length].astype(np.float64)
            total += values.sum(0)
            square += np.square(values).sum(0)
            count += int(length)
        mean = total / count
        scale = np.sqrt(np.maximum(square / count - mean**2, 0)).clip(min=0.1)
        with torch.no_grad():
            model.mean.copy_(self.arrays["train"].mean(0))
            model.scale.copy_(self.arrays["train"].std(0, unbiased=False).clamp_min(0.1))
            model.memory_mean.copy_(torch.from_numpy(mean).float())
            model.memory_scale.copy_(torch.from_numpy(scale).float())


def infer(model, data, split, indices, *, oracle=False, batch_size=128):
    from .toy_protocol import DECISIONS
    from .toy_selector_model import decode_outputs

    device = next(model.parameters()).device
    predictions, raw = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            output = model(*data.batch(split, selected, device))
            if oracle:
                logits = torch.full_like(output["decision"], -torch.inf)
                for index, i in enumerate(selected):
                    logits[index, DECISIONS.index(data.rows[split][i]["expected"]["decision"])] = 0
                output["decision"] = logits
            predictions.extend(decode_outputs(output))
            raw.extend(output["token"].argmax(-1).cpu().tolist())
    return predictions, raw
