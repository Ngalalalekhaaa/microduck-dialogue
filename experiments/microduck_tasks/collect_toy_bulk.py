"""Bounded multi-process collection of independently checked toy curricula."""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from .toy_curriculum import build_curriculum


def _collection_specs(*, repeats, base_seed):
    """Namespace new recordings by batch seed while retaining semantic split groups."""
    specs = build_curriculum(repeats=repeats, base_seed=base_seed)
    for spec in specs:
        spec["episode_id"] = f"seed{base_seed}_{spec['episode_id']}"
    return specs


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def _available_memory():
    lines = Path("/proc/meminfo").read_text().splitlines()
    return int(next(line.split()[1] for line in lines if line.startswith("MemAvailable:"))) * 1024


def _size(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _run_phase(pool, specs, output, *, workers, summaries, started, budget_gib):
    from .toy_collection_worker import collect_one

    pending, remaining = {}, iter(specs)
    exhausted, refusal, last_report = False, None, 0.0
    while pending or not exhausted:
        if refusal is None:
            free = shutil.disk_usage(output).free
            if (output / "STOP").exists():
                refusal = "Operator STOP file: current episodes finish; no new episodes start."
            elif free < 8 * 2**30 or _available_memory() < 4 * 2**30:
                refusal = "Reserve reached: keep 8 GiB disk and 4 GiB available RAM."
            elif sum(item.get("bytes", 0) for item in summaries) > budget_gib * 2**30:
                refusal = "This collection reached its configured disk budget."
        while not exhausted and refusal is None and len(pending) < workers:
            spec = next(remaining, None)
            if spec is None:
                exhausted = True
                break
            pending[pool.submit(collect_one, spec, str(output), True)] = spec["episode_id"]
        if refusal:
            exhausted = True
        completed, _ = (
            concurrent.futures.wait(pending, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
            if pending
            else (set(), set())
        )
        for future in completed:
            episode_id = pending.pop(future)
            try:
                summary = future.result()
            except Exception as exc:
                summary = dict(episode_id=episode_id, valid=False, error=f"{type(exc).__name__}: {exc}")
                partial = output / "episodes" / episode_id
                if partial.exists():
                    shutil.move(str(partial), str(output / "quarantine" / episode_id))
            folder = Path(summary.get("output", output / "quarantine" / episode_id))
            summary["bytes"] = _size(folder) if folder.exists() else 0
            summaries.append(summary)
            with (output / "completed.jsonl").open("a") as stream:
                stream.write(json.dumps(summary, ensure_ascii=False, allow_nan=False) + "\n")
        now = time.monotonic()
        if completed or now - last_report >= 15:
            progress = dict(
                completed=len(summaries),
                valid=sum(bool(item.get("valid")) for item in summaries),
                quarantined=sum(not item.get("valid") for item in summaries),
                in_flight=len(pending),
                elapsed_s=round(now - started, 2),
                available_ram_gib=round(_available_memory() / 2**30, 2),
                disk_free_gib=round(shutil.disk_usage(output).free / 2**30, 2),
                stop_reason=refusal,
            )
            _save(output / "progress.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            last_report = now
    return refusal


def collect(output, *, workers=8, repeats=6, base_seed=410000, budget_gib=6.0, preflight_only=False):
    if not 1 <= workers <= 16 or not 0.1 <= budget_gib <= 8:
        raise ValueError("Use 1..16 measured workers and a 0.1..8 GiB collection budget.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    specs = _collection_specs(repeats=repeats, base_seed=base_seed)
    (output / "episodes").mkdir()
    (output / "quarantine").mkdir()
    _save(output / "curriculum.json", specs)
    _save(
        output / "collection_config.json",
        dict(
            workers=workers,
            repeats=repeats,
            base_seed=base_seed,
            episode_id_format="seed{base_seed}_{curriculum_episode_id}",
            episodes=len(specs),
            budget_gib=budget_gib,
            preflight_only=preflight_only,
            source_sha256={
                name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in ("toy_curriculum.py", "toy_teacher.py", "toy_collection_worker.py", "collect_toy_bulk.py")
            },
        ),
    )
    groups, preflight, rest = set(), [], []
    for spec in specs:
        if spec["split_group"] in groups:
            rest.append(spec)
        else:
            groups.add(spec["split_group"])
            preflight.append(spec)
    summaries, started = [], time.monotonic()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        stop_reason = _run_phase(
            pool, preflight, output, workers=workers, summaries=summaries, started=started, budget_gib=budget_gib
        )
        invalid = [item for item in summaries if not item.get("valid")]
        if invalid:
            stop_reason = "Preflight semantic or recording checks failed; remaining paraphrases were not started."
        if stop_reason is None and not preflight_only:
            stop_reason = _run_phase(
                pool, rest, output, workers=workers, summaries=summaries, started=started, budget_gib=budget_gib
            )
    report = dict(
        requested_episodes=len(preflight) if preflight_only else len(specs),
        completed_episodes=len(summaries),
        valid_episodes=sum(bool(item.get("valid")) for item in summaries),
        quarantined=[item for item in summaries if not item.get("valid")],
        elapsed_s=time.monotonic() - started,
        workers=workers,
        stop_reason=stop_reason,
        complete=stop_reason is None,
        bytes=_size(output),
    )
    valid = [item for item in summaries if item.get("valid")]
    report["goal_status"] = dict(sum((Counter(item.get("goal_status", {})) for item in valid), Counter()))
    _save(output / "collection_report.json", report)
    if valid:
        from .toy_collection_audit import audit
        from .toy_dataset import export

        report["dataset"] = export(output / "episodes", output / "dataset", three_way=True)
        _save(output / "audit.json", audit(output / "dataset", source_root=output / "episodes"))
        report["bytes"] = _size(output)
        _save(output / "collection_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("dataset", "quarantined")}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=410000)
    parser.add_argument("--budget-gib", type=float, default=6)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    report = collect(
        args.output,
        workers=args.workers,
        repeats=args.repeats,
        base_seed=args.base_seed,
        budget_gib=args.budget_gib,
        preflight_only=args.preflight_only,
    )
    raise SystemExit(0 if report["complete"] and not report["quarantined"] else 1)


if __name__ == "__main__":
    main()
