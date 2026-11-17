"""Cache frozen features for current-decision samples, with input/image provenance."""

import argparse
import cProfile
import io
import json
import pstats
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .head_features import FrozenFeatures, feature_config, label, payload
from .plan_inputs import BOUNDARY_CONTRACT, LEGACY_CONTRACT, contract_digest
from .setup_skill_head import digest
from .train_vlm import read_records, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--without-images", action="store_true")
    parser.add_argument("--profile-first", action="store_true")
    parser.add_argument(
        "--allow-legacy-inputs", action="store_true", help="Reproduce the first, mismatched pilot only."
    )
    args = parser.parse_args()
    source_manifest = json.loads((args.dataset / "manifest.json").read_text())
    contract = source_manifest.get("input_contract", LEGACY_CONTRACT)
    if contract != BOUNDARY_CONTRACT and not (args.allow_legacy_inputs and contract == LEGACY_CONTRACT):
        raise ValueError(
            "Collect semantic boundary demonstrations first; --allow-legacy-inputs is for reproduction only."
        )
    if contract == BOUNDARY_CONTRACT and source_manifest.get("input_contract_sha256") != contract_digest():
        raise ValueError("Dataset executor contract changed; use its pinned source or collect new demonstrations.")
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {
        split: [
            r
            for r in read_records(args.dataset / f"{split}.jsonl")
            if r.get("variant") != "continuation" and not r["context"]["committed_steps"]
        ]
        for split in ("train", "val", "test")
    }
    seeds = {s: {r["scene_seed"] for r in rows} for s, rows in datasets.items()}
    if contract == BOUNDARY_CONTRACT and any(
        r.get("input_contract") != contract for rows in datasets.values() for r in rows
    ):
        raise ValueError("Every semantic sample must carry the same input contract as its manifest.")
    if any(seeds[a] & seeds[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Scene seeds must be disjoint across splits.")
    manifest = dict(
        config=feature_config(not args.without_images),
        input_contract=contract,
        input_contract_sha256=source_manifest.get("input_contract_sha256"),
        dataset=str(args.dataset.resolve()),
        source_hashes={s: digest(args.dataset / f"{s}.jsonl") for s in datasets},
        seeds={s: sorted(v) for s, v in seeds.items()},
        class_counts={s: dict(Counter(map(label, rows))) for s, rows in datasets.items()},
        splits={},
        complete=False,
    )
    write_json(args.output / "manifest.json", manifest)
    model = FrozenFeatures(args.model)
    manifest["model_load_seconds"] = model.load_seconds
    started = time.monotonic()
    for split, rows in datasets.items():
        vectors, records = [], []
        for index, row in enumerate(rows):
            paths = {name: args.dataset / path for name, path in row["images"].items()}
            profiler = cProfile.Profile() if args.profile_first and split == "train" and index == 0 else None
            if profiler:
                profiler.enable()
            vector, timing = model.extract(
                row["instruction"], row["state"], row["context"], paths, vision=not args.without_images
            )
            if profiler:
                profiler.disable()
                result = io.StringIO()
                pstats.Stats(profiler, stream=result).sort_stats("cumulative").print_stats(35)
                (args.output / "first_sample_profile.txt").write_text(result.getvalue())
            vectors.append(vector.astype(np.float16))
            records.append(
                dict(
                    id=row["id"],
                    label=label(row),
                    scene_seed=row["scene_seed"],
                    episode_id=row["episode_id"],
                    image_hashes={k: digest(v) for k, v in paths.items()},
                    input=payload(row["instruction"], row["state"], row["context"]),
                    **timing,
                )
            )
            if index == 0 or (index + 1) % 10 == 0:
                print(json.dumps(dict(split=split, done=index + 1, total=len(rows), **timing)), flush=True)
        np.savez_compressed(
            args.output / f"{split}.npz",
            x=np.stack(vectors),
            labels=np.array([r["label"] for r in records]),
            ids=np.array([r["id"] for r in records]),
        )
        write_json(args.output / f"{split}_records.json", records)
        manifest["splits"][split] = dict(samples=len(rows), cache_sha256=digest(args.output / f"{split}.npz"))
        write_json(args.output / "manifest.json", manifest)
    manifest.update(
        complete=True,
        extraction_seconds=time.monotonic() - started,
        peak_gpu_allocated_mib=model.torch.cuda.max_memory_allocated() / 1024**2,
    )
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
