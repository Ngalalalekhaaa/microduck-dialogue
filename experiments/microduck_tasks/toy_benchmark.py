"""Measure bounded, independent toy collection processes on this host."""

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

COMMAND = "先踢蓝球，再踢红球，再原地转一圈"
GIB = 1024**3


def gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    used, total, utilization = (int(value.strip()) for value in result.stdout.splitlines()[0].split(","))
    return {"used_mib": used, "total_mib": total, "utilization_percent": utilization}


def process_rss(process):
    try:
        parent = psutil.Process(process.pid)
        members = [parent, *parent.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for member in members:
        try:
            total += member.memory_info().rss
        except psutil.Error:
            pass
    return total


def terminate_owned(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_wave(output, count, timeout):
    output.mkdir(parents=True)
    env = dict(os.environ, MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    processes, files, episode_dirs = [], [], []
    rss_peaks = [0] * count
    aggregate_peak = 0
    cpu_samples = []
    psutil.cpu_percent(interval=None)
    available_min = psutil.virtual_memory().available
    gpu_peak = gpu_utilization_peak = 0
    gpu_free_min = 1_000_000
    aborted = None
    start = time.monotonic()
    last_gpu = -100.0
    finish_times = {}
    try:
        for index in range(count):
            episode = output / f"episode_{index:02d}"
            episode_dirs.append(episode)
            log = (output / f"process_{index:02d}.log").open("w")
            files.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        str(Path(__file__).with_name("toy.sh")),
                        "--no-viewer",
                        "--command",
                        COMMAND,
                        "--output",
                        str(episode),
                    ],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        with (output / "resources.jsonl").open("w") as samples:
            while True:
                now = time.monotonic()
                rss = [process_rss(process) for process in processes]
                rss_peaks = [max(previous, current) for previous, current in zip(rss_peaks, rss, strict=True)]
                aggregate_peak = max(aggregate_peak, sum(rss))
                available = psutil.virtual_memory().available
                available_min = min(available_min, available)
                sample = {"elapsed_s": now - start, "process_rss_bytes": rss, "available_memory_bytes": available}
                sample["cpu_utilization_percent"] = psutil.cpu_percent(interval=None)
                cpu_samples.append(sample["cpu_utilization_percent"])
                if now - last_gpu >= 1:
                    gpu = gpu_memory()
                    last_gpu = now
                    gpu_peak = max(gpu_peak, gpu["used_mib"])
                    gpu_utilization_peak = max(gpu_utilization_peak, gpu["utilization_percent"])
                    gpu_free_min = min(gpu_free_min, gpu["total_mib"] - gpu["used_mib"])
                    sample["gpu"] = gpu
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                for index, process in enumerate(processes):
                    if process.poll() is not None and index not in finish_times:
                        finish_times[index] = now - start
                if all(process.poll() is not None for process in processes):
                    break
                if available < 4 * GIB:
                    aborted = "MemAvailable below 4 GiB"
                elif gpu_free_min < 300:
                    aborted = "GPU free memory below 300 MiB"
                elif now - start > timeout:
                    aborted = f"Wave exceeded {timeout:g} second timeout"
                if aborted:
                    terminate_owned(processes)
                    break
                time.sleep(0.2)
    finally:
        terminate_owned(processes)
        for log in files:
            log.close()
    elapsed = time.monotonic() - start
    episodes = []
    for index, episode in enumerate(episode_dirs):
        path = episode / "summary.json"
        summary = json.loads(path.read_text()) if path.exists() else {}
        results = summary.get("results", [])
        verified = (
            processes[index].returncode == 0
            and len(results) == 3
            and all(row["status"] == "succeeded" and row.get("verified") for row in results)
            and not summary.get("errors")
            and not summary.get("runtime_failure")
        )
        episodes.append(
            {
                "output": str(episode),
                "returncode": processes[index].returncode,
                "wall_seconds": finish_times.get(index),
                "rss_peak_bytes": rss_peaks[index],
                "success": verified,
                "goal_statuses": [row["status"] for row in results],
                "simulation_seconds": summary.get("simulation_seconds"),
                "errors": summary.get("errors", []),
                "runtime_failure": summary.get("runtime_failure"),
            }
        )
    successes = sum(row["success"] for row in episodes)
    report = {
        "parallel_processes": count,
        "wall_seconds": elapsed,
        "successful_episodes": successes,
        "failed_episodes": count - successes,
        "episodes_per_hour": successes * 3600 / elapsed,
        "aggregate_rss_peak_bytes": aggregate_peak,
        "available_memory_min_bytes": available_min,
        "gpu_used_peak_mib": gpu_peak,
        "gpu_free_min_mib": gpu_free_min,
        "gpu_utilization_peak_percent": gpu_utilization_peak,
        "cpu_utilization_mean_percent": sum(cpu_samples) / max(1, len(cpu_samples)),
        "cpu_utilization_peak_percent": max(cpu_samples, default=0),
        "aborted": aborted,
        "episodes": episodes,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--parallel", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args(argv)
    if any(value < 1 or value > 16 for value in args.parallel):
        parser.error("Each --parallel value must be between 1 and 16.")
    output = args.output or Path(__file__).parent / "outputs" / f"toy_parallel_benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "command": COMMAND,
        "started_at": datetime.now().astimezone().isoformat(),
        "cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_bytes": psutil.virtual_memory().total,
        "baseline_available_memory_bytes": psutil.virtual_memory().available,
        "baseline_gpu": gpu_memory(),
        "scope": "Independent headless CPU MuJoCo episodes, EGL rendering, rule teacher, learned skill policies.",
        "sampling": {"rss_interval_s": 0.2, "gpu_interval_s": 1.0, "peak_kind": "sampled"},
        "limitations": "One complete wave per concurrency; fixed layout and command; no VLM inference or training.",
        "warmup": None,
        "waves": [],
    }
    if not args.skip_warmup:
        report["warmup"] = run_wave(output / "warmup", 1, args.timeout)
    if report["warmup"] is None or not report["warmup"]["aborted"]:
        for count in args.parallel:
            wave = run_wave(output / f"parallel_{count}", count, args.timeout)
            report["waves"].append(wave)
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            if wave["aborted"]:
                break
    report["finished_at"] = datetime.now().astimezone().isoformat()
    successful_waves = [wave for wave in report["waves"] if wave["failed_episodes"] == 0 and not wave["aborted"]]
    report["maximum_validated_parallel_processes"] = max(
        (wave["parallel_processes"] for wave in successful_waves), default=0
    )
    highest_throughput = max((wave["episodes_per_hour"] for wave in successful_waves), default=0)
    report["recommendation_rule"] = "Smallest measured concurrency achieving at least 90% of measured peak throughput."
    report["recommended_batch_parallel_processes"] = min(
        (
            wave["parallel_processes"]
            for wave in successful_waves
            if wave["episodes_per_hour"] >= 0.9 * highest_throughput
        ),
        default=None,
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"REPORT={output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
