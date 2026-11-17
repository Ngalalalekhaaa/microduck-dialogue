"""Extract resumable input-only features without modifying the recorded data splits."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .toy_dataset import contract_digest, digest, dumps
from .toy_head_features import ToyFrozenFeatures, feature_config, input_digest
from .toy_train_probe import read_records


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_splits(dataset):
    """Validate exported rows and preserve their declared train/val/test assignment."""
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    names = tuple(name for name in ("train", "val", "test") if name in manifest["files_sha256"])
    if set(names) != set(manifest["files_sha256"]) or "train" not in names:
        raise ValueError("Use the declared train, val, and test dataset splits.")
    splits, seen, owners = {}, set(), {key: {} for key in ("episode_id", "seed", "split_group")}
    for split in names:
        path = dataset / f"{split}.jsonl"
        if digest(path) != manifest["files_sha256"][split]:
            raise ValueError("Dataset split changed since export; regenerate its provenance manifest.")
        splits[split] = read_records(path) if path.stat().st_size else []
        for row in splits[split]:
            if (
                row["id"] in seen
                or not row["id"]
                or any(c not in "-_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in row["id"])
            ):
                raise ValueError("Require unique, path-safe sample identifiers.")
            seen.add(row["id"])
            for key, mapping in owners.items():
                previous = mapping.setdefault(row[key], split)
                if previous != split:
                    raise ValueError(f"A {key} crosses dataset splits; preserve the grouped export.")
    if not seen:
        raise ValueError("The exported dataset has no samples.")
    return splits


def _record(row, split):
    return {
        **{key: row[key] for key in ("id", "episode_id", "seed", "split_group")},
        "split": split,
        "input_sha256": input_digest(row["request"]),
        "image_sha256": row["image_sha256"],
        "source_row_sha256": hashlib.sha256(dumps(row).encode()).hexdigest(),
    }


def _cache_path(output, split, identifier):
    return output / "rows" / split / f"{identifier}.npz"


def _read_cache(path, expected, config_sha):
    with np.load(path, allow_pickle=False) as cached:
        vector, metadata = cached["x"], json.loads(str(cached["metadata"]))
    if (
        any(metadata.get(key) != value for key, value in expected.items())
        or metadata.get("config_sha256") != config_sha
    ):
        raise ValueError(f"Cached input provenance differs: {path}. Use a new feature output directory.")
    if vector.shape != (2048,) or vector.dtype != np.float16 or not np.isfinite(vector).all():
        raise ValueError(f"Invalid feature vector in {path}; restore or remove this damaged cache row.")
    return vector, metadata


def _save_cache(path, vector, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, x=vector.astype(np.float16), metadata=np.array(dumps(metadata)))
    temporary.replace(path)


def extract(dataset, model_dir, output, *, batch_size=1, limit=None, resume=False, vision=True):
    if type(batch_size) is not int or not 1 <= batch_size <= 4:
        raise ValueError("Use --batch-size from one through four.")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--limit must be a positive number of new rows to extract.")
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    datasets = read_splits(dataset)
    config = feature_config(vision)
    config_sha = hashlib.sha256(dumps(config).encode()).hexdigest()
    identity = {
        "config": config,
        "config_sha256": config_sha,
        "contract_sha256": contract_digest(),
        "dataset": str(dataset),
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "source_hashes": {split: digest(dataset / f"{split}.jsonl") for split in datasets},
        "expected_rows": {split: len(rows) for split, rows in datasets.items()},
    }
    if output.exists():
        if not resume:
            raise FileExistsError("Use a new output directory or --resume to verify and continue its cached inputs.")
        manifest = json.loads((output / "manifest.json").read_text())
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("Feature configuration or dataset changed; use a new output directory.")
    else:
        if resume:
            raise FileNotFoundError("--resume needs an existing extraction manifest.")
        output.mkdir(parents=True)
        manifest = {**identity, "complete": False, "splits": {}, "runs": []}
        _write_json(output / "manifest.json", manifest)
    pending, cached_counts = [], Counter()
    for split, rows in datasets.items():
        for row in rows:
            record = _record(row, split)
            path = _cache_path(output, split, row["id"])
            if path.exists():
                _read_cache(path, record, config_sha)
                cached_counts[split] += 1
            else:
                pending.append((split, row, record))
    selected = pending if limit is None else pending[:limit]
    started, processed = time.monotonic(), 0
    run = {"new_rows": 0, "requested_new_rows": len(selected), "batch_size": batch_size, "vision": vision}
    if selected:
        model = ToyFrozenFeatures(model_dir)
        run["model_load_seconds"] = model.load_seconds
        extraction_started = time.monotonic()
        for start in range(0, len(selected), batch_size):
            chunk = selected[start : start + batch_size]
            requests = [row["request"] for _, row, _ in chunk]
            images = [{name: dataset / relative for name, relative in row["images"].items()} for _, row, _ in chunk]
            vectors, timings = model.extract_batch(requests, images, vision=vision)
            for (split, row, record), vector, timing in zip(chunk, vectors, timings, strict=True):
                metadata = {**record, "config_sha256": config_sha, **timing}
                _save_cache(_cache_path(output, split, row["id"]), vector, metadata)
                cached_counts[split] += 1
            processed += len(chunk)
            elapsed = time.monotonic() - extraction_started
            progress = {
                "new_rows": processed,
                "cached_rows": sum(cached_counts.values()),
                "total_rows": sum(map(len, datasets.values())),
                "run_target": len(selected),
                "elapsed_s": round(elapsed, 3),
                "seconds_per_row": round(elapsed / processed, 4),
                "eta_remaining_s": round((len(pending) - processed) * elapsed / processed, 1),
                "last_batch": timings[-1],
            }
            _write_json(output / "progress.json", progress)
            print(dumps(progress), flush=True)
        run["peak_gpu_allocated_mib"] = model.torch.cuda.max_memory_allocated() / 1024**2
        run["extraction_seconds"] = time.monotonic() - extraction_started
    run.update(new_rows=processed, wall_seconds=time.monotonic() - started)
    manifest["runs"].append(run)
    manifest["cached_rows"] = dict(cached_counts)
    for split, rows in datasets.items():
        if cached_counts[split] != len(rows):
            continue
        vectors, records = [], []
        for row in rows:
            vector, record = _read_cache(_cache_path(output, split, row["id"]), _record(row, split), config_sha)
            vectors.append(vector)
            records.append(record)
        temporary = output / f"{split}.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                x=np.stack(vectors) if vectors else np.empty((0, 2048), dtype=np.float16),
                ids=np.array([r["id"] for r in rows]),
            )
        temporary.replace(output / f"{split}.npz")
        _write_json(output / f"{split}_records.json", records)
        manifest["splits"][split] = {
            "samples": len(rows),
            "cache_sha256": digest(output / f"{split}.npz"),
            "records_sha256": digest(output / f"{split}_records.json"),
        }
    manifest["complete"] = all(cached_counts[s] == len(rows) for s, rows in datasets.items())
    _write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--without-images", action="store_true")
    args = parser.parse_args()
    result = extract(
        args.dataset,
        args.model,
        args.output,
        batch_size=args.batch_size,
        limit=args.limit,
        resume=args.resume,
        vision=not args.without_images,
    )
    print(
        dumps({"complete": result["complete"], "cached_rows": result["cached_rows"], "last_run": result["runs"][-1]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
