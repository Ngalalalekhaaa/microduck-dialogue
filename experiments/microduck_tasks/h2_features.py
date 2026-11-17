"""Re-extract final-layer multi-position memory without opening sealed data."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .ablation_contract import checksum, dumps
from .h0_features import Features, read_data
from .h2_memory import MAX_MEMORY, config, identity, layout, pool, read_cache, write_cache
from .run_ablation import save
from .toy_dataset import contract_digest, digest


def extract(dataset, globals_dir, model_dir, output):
    import torch

    data_manifest, rows = read_data(dataset)
    old = json.loads((globals_dir / "manifest.json").read_text())
    cfg = config()
    if not old["complete"] or old["config"] != cfg["global_config"]:
        raise ValueError("H2 must use the completed, identical H0 input and Qwen configuration.")
    expected = dict(
        config=cfg,
        config_sha256=checksum(cfg),
        dataset_manifest_sha256=digest(dataset / "manifest.json"),
        global_manifest_sha256=digest(globals_dir / "manifest.json"),
        source_hashes=data_manifest["files_sha256"],
        contract_sha256=contract_digest(),
        blind_access=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else dict(expected, complete=False, splits={})
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Memory identity changed; choose a separate experiment directory.")
    save(path, manifest)
    vectors, records = {}, {}
    for split in rows:
        if digest(globals_dir / f"{split}.npz") != old["splits"][split]["cache_sha256"]:
            raise ValueError("H0 feature bytes changed.")
        with np.load(globals_dir / f"{split}.npz", allow_pickle=False) as data:
            if data["ids"].tolist() != [r["id"] for r in rows[split]]:
                raise ValueError("Global feature order differs.")
            vectors[split] = data["x"]
        records[split] = json.loads((globals_dir / f"{split}_records.json").read_text())
    pending, completed = [], 0
    for split, items in rows.items():
        for index, row in enumerate(items):
            target = output / "rows" / split / f"{row['id']}.npz"
            rec = identity(row, split, expected["config_sha256"])
            if target.exists():
                read_cache(target, rec)
                completed += 1
            else:
                pending.append((split, index, row, target, rec))
    started = time.monotonic()
    if pending:
        model = Features(model_dir)
        for split, index, row, target, rec in pending:
            tick = time.monotonic()
            text, ids, spans = layout(model.processor.tokenizer, row["request"])
            batch, lengths = model.encode_batch([row["request"]], [{}], vision=False)
            if batch["input_ids"][0, : lengths[0]].tolist() != ids:
                raise ValueError("Tokenizer positions differ from the actual model input.")
            if checksum(text) != records[split][index]["rendered_sha256"]:
                raise ValueError("H2 rendered input differs from H0.")
            batch = {k: v.to("cuda") for k, v in batch.items()}
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                result = model.model.model(**batch, use_cache=False, return_dict=True)
                hidden = result.last_hidden_state[0, : lengths[0]]
                last = hidden[-1].float().cpu().numpy().astype(np.float16)
                if not np.array_equal(last, vectors[split][index]):
                    raise ValueError("Re-extracted last token differs from H0; investigate before fitting H2.")
                memory = pool(hidden, spans)
            timing = dict(
                seconds=time.monotonic() - tick,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
            rec.update(
                tokens=lengths[0],
                rendered_sha256=checksum(text),
                input_ids_sha256=checksum(ids),
                spans=spans,
                global_vector_exact_match=True,
                **timing,
            )
            write_cache(target, memory, rec)
            del result, hidden, batch
            completed += 1
            progress = dict(
                stage="H2_extracting_memory",
                completed=completed,
                total=sum(map(len, rows.values())),
                elapsed_s=time.monotonic() - started,
                last=timing,
                blind_access=False,
            )
            save(output / "progress.json", progress)
            print(dumps(progress), flush=True)
    for split, items in rows.items():
        target = output / f"{split}.npy"
        array = np.lib.format.open_memmap(
            target.with_suffix(".tmp"), mode="w+", dtype=np.float16, shape=(len(items), MAX_MEMORY, 2048)
        )
        recs = []
        for index, row in enumerate(items):
            array[index], rec = read_cache(
                output / "rows" / split / f"{row['id']}.npz", identity(row, split, expected["config_sha256"])
            )
            recs.append(rec)
        array.flush()
        del array
        target.with_suffix(".tmp").replace(target)
        save(output / f"{split}_records.json", recs)
        manifest["splits"][split] = dict(
            samples=len(items), memory_sha256=digest(target), records_sha256=digest(output / f"{split}_records.json")
        )
    manifest.update(complete=True, elapsed_s=time.monotonic() - started)
    save(path, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--globals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    extract(args.dataset, args.globals, args.model, args.output)
