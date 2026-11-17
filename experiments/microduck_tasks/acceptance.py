"""Run every predeclared seed/colour case and retain failed physical trials."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

COLORS = {"red": "红", "yellow": "黄", "blue": "蓝"}
WALL_TIMEOUT_SECONDS = 180
HERE = Path(__file__).resolve().parent


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _instructions(color, index):
    before = ("现在能看到什么？", "说说你眼前有什么", "你看到了哪些球？")
    kick = ("踢{color}球", "把{color}色的球踢走", "请踢一下{color}色的球")
    after = ("现在眼前有什么？", "你现在视野中有什么？", "能看到什么？")
    return [before[index % 3], kick[index % 3].format(color=COLORS[color]), after[index % 3]]


def _launch(command, log):
    """Bound a fresh process group, allowing interrupted demo cleanup to save data."""
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=HERE.parents[1], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=WALL_TIMEOUT_SECONDS), False
        except subprocess.TimeoutExpired:
            # demo handles KeyboardInterrupt and saves trajectory/contacts in finally.
            os.killpg(process.pid, signal.SIGINT)
            try:
                return process.wait(timeout=10), True
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return process.wait(timeout=5), True
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise


def _read_case(directory, color):
    diagnostics = []
    summary = {}
    path = directory / "summary.json"
    if path.is_file():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError("Expected the demo summary object.")
        except (ValueError, OSError) as exc:
            diagnostics.append(f"Cannot read summary.json: {exc}")
            summary = {}
    events = []
    path = directory / "events.jsonl"
    if path.is_file():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Expected an event object.")
                events.append(event)
            except ValueError as exc:
                diagnostics.append(f"events.jsonl line {number}: {exc}")
    results = summary.get("results", [])
    kick = next((row for row in results if row.get("intent", {}).get("kind") == "kick"), None)
    initial_visible = None
    if events:
        detected = events[0].get("observation", {}).get("detections", [])
        initial_visible = sorted({row["color"] for row in detected if row.get("color") in COLORS})
    search_events = [
        row for row in events if row.get("phase") == "search" and row.get("execution", {}).get("started") is True
    ]
    metadata = summary.get("metadata", {})
    if not metadata and (directory / "episode.json").is_file():
        try:
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8")).get("metadata", {})
        except (ValueError, OSError) as exc:
            diagnostics.append(f"Cannot read episode.json: {exc}")
    artifacts = {
        name: (directory / name).is_file()
        for name in ("scene.xml", "trajectory.npz", "contacts.json", "events.jsonl", "episode.json", "summary.json")
    }
    success = bool(
        kick
        and kick.get("status") == "success"
        and kick.get("intent", {}).get("color") == color
        and (kick.get("evaluation") or {}).get("success") is True
        and len(results) == 3
        and all(row.get("status") == "success" for row in results)
        and not summary.get("errors")
        and not summary.get("runtime_failure")
        and all(artifacts.values())
    )
    max_tilt = summary.get("max_tilt_deg")
    simulation_seconds = summary.get("simulation_seconds")
    if events and max_tilt is None:
        max_tilt = max((row.get("observation", {}).get("tilt_deg", 0) for row in events), default=None)
    if events and simulation_seconds is None:
        simulation_seconds = max(
            max(row.get("observation", {}).get("time_s", 0), row.get("execution", {}).get("end_time_s") or 0)
            for row in events
        )
    return {
        "success": success,
        "status": kick.get("status", "missing_kick_result") if kick else "missing_kick_result",
        "initial_visible_colors": initial_visible,
        "target_initially_visible": color in initial_visible if initial_visible is not None else None,
        "searched": bool(search_events),
        "search_actions": len(search_events),
        "simulation_seconds": simulation_seconds,
        "max_tilt_deg": max_tilt,
        "initial_layout_world_xy_m": metadata.get("initial_layout_world_xy_m"),
        "metadata": metadata,
        "results": results,
        "artifacts": artifacts,
        "errors": [*summary.get("errors", []), *diagnostics],
    }


def _summarize(manifest):
    cases = manifest["cases"]
    manifest.update(
        completed_cases=len(cases),
        success_count=sum(row["success"] for row in cases),
        failure_count=sum(not row["success"] for row in cases),
        failures=[
            {"case_id": row["case_id"], "status": row["status"], "output": row["output"]}
            for row in cases
            if not row["success"]
        ],
        search_cases=sum(row["searched"] for row in cases),
    )


def run(output, seeds, colors):
    """Run a fixed case matrix; the same seed always starts from the same layout."""
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    plan = [
        {
            "case_id": f"seed_{seed}_{color}",
            "seed": seed,
            "target_color": color,
            "instructions": _instructions(color, seed_index + color_index),
        }
        for seed_index, seed in enumerate(seeds)
        for color_index, color in enumerate(colors)
    ]
    manifest = {
        "format_version": "microduck-physical-acceptance-1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "hardware": False,
        "backend": "rgbd_rule_teacher",
        "is_trained_vlm": False,
        "wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
        "timeout_cleanup_grace_seconds": 10,
        "random_radius_m": [0.3, 0.6],
        "same_seed_same_layout": True,
        "planned_total": len(plan),
        "plan": plan,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(HERE.glob("*.py"))
        },
        "cases": [],
    }
    _summarize(manifest)
    _json(output / "summary.json", manifest)
    layouts = {}
    for specification in plan:
        directory = output / specification["case_id"]
        command = ["bash", str(HERE / "run.sh"), "--output", str(directory), "--seed", str(specification["seed"])]
        for instruction in specification["instructions"]:
            command.extend(["--command", instruction])
        started = time.monotonic()
        launch_error = None
        try:
            code, timeout = _launch(command, logs / f"{specification['case_id']}.log")
        except OSError as exc:
            code, timeout, launch_error = None, False, f"{type(exc).__name__}: {exc}"
        case = _read_case(directory, specification["target_color"])
        case.update(
            specification,
            output=str(directory),
            process_exit_code=code,
            wall_timeout=timeout,
            wall_seconds=time.monotonic() - started,
            command=command,
        )
        if timeout:
            case.update(success=False, status="wall_timeout")
        elif code != 0:
            case["success"] = False
            if case["status"] == "success":
                case["status"] = "process_failed"
        if launch_error:
            case.update(success=False, status="launch_failed")
            case["errors"].append(launch_error)
        layout = case["initial_layout_world_xy_m"]
        if layout is not None:
            seed = specification["seed"]
            if seed in layouts and layout != layouts[seed]:
                case.update(success=False, status="inconsistent_initial_layout")
                manifest["same_seed_same_layout"] = False
            layouts.setdefault(seed, layout)
        manifest["cases"].append(case)
        _summarize(manifest)
        _json(output / "summary.json", manifest)
        print(
            f"{specification['case_id']}: {case['status']} | success={case['success']}"
            f" | searched={case['searched']} | {case['wall_seconds']:.1f}s wall",
            flush=True,
        )
    print(
        f"Success: {manifest['success_count']}/{manifest['planned_total']}; results: {output / 'summary.json'}",
        flush=True,
    )
    return 0 if manifest["success_count"] == manifest["planned_total"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New acceptance directory; existing paths are refused."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seeds", nargs="+", help="Space- or comma-separated seeds; default: 0 1 2.")
    group.add_argument("--seed", type=int, help="One fixed seed, equivalent to --seeds N.")
    parser.add_argument("--colors", nargs="+", default=list(COLORS), help="Target colours; default: red yellow blue.")
    args = parser.parse_args(argv)
    try:
        seeds = (
            [args.seed]
            if args.seed is not None
            else [int(v) for item in (args.seeds or ["0", "1", "2"]) for v in item.split(",")]
        )
        colors = [value for item in args.colors for value in item.split(",")]
        if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
            raise ValueError("Use distinct nonnegative integer seeds.")
        if not colors or len(set(colors)) != len(colors) or any(color not in COLORS for color in colors):
            raise ValueError("Use distinct target colours from red, yellow and blue.")
        return run(args.output, seeds, colors)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
