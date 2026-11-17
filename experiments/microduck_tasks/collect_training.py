"""Collect a resumable, predeclared rule-teacher simulation training corpus.

The planner sees RGB-D measurements only. Ground-truth physics verifies finished
trials and remains in audit artifacts, never in the student input. Failed trials
are retained rather than silently resampled. Every scene seed has exactly one
predeclared split, including its three colour tasks and direct-command episode.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
FORMAT = "microduck-rule-training-collection-1"
COLORS = {"red": "红", "yellow": "黄", "blue": "蓝"}
KICK_PHRASES = {
    "train": (
        "踢{color}球",
        "把{color}色的球踢走",
        "请踢一下{color}色的球",
        "帮我把{color}球踢出去",
        "踢一脚{color}色的小球",
        "请把那个{color}球踢开",
        "别踢{other}球，踢{color}球",
        "不要踢{other}球，请把{color}球踢走",
    ),
    "val": ("小鸭子，帮忙踢一下{color}色的球", "请将{color}色小球踢出去"),
    "test": ("小鸭子，踢走那颗{color}色的球", "帮我踢开{color}球，别踢{other}球"),
}
VIEW_PHRASES = (
    "现在能看到什么？",
    "说说你眼前有什么",
    "你看到了哪些球？",
    "你现在视野中有什么？",
    "眼前有啥？",
    "描述一下你看到的球",
)
MOVE_PHRASES = {
    "move_forward": ("往前走", "向前挪一点", "前进", "朝前走一点"),
    "move_backward": ("往后走", "退后一点", "后退", "向后挪一下"),
    "move_left": ("往左走", "向左挪一点", "左移", "朝左走一点"),
    "move_right": ("往右走", "向右挪一点", "右移", "朝右走一点"),
    "turn_left": ("向左转", "左转一下", "朝左转一点", "转向左边"),
    "turn_right": ("向右转", "右转一下", "朝右转一点", "转向右边"),
}
HELD_OUT_VIEW_PHRASES = {
    "val": ("说说当前视野里有哪些球？", "请描述你眼前看见的东西"),
    "test": ("你能看到哪些颜色的球？", "帮我看看前面有什么？"),
}


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_plan(seed_start=1000, train_seeds=80, val_seeds=10, test_seeds=10):
    """Freeze split assignment before collecting, never by eventual success."""
    counts = {"train": train_seeds, "val": val_seeds, "test": test_seeds}
    if seed_start < 0 or any(isinstance(count, bool) or count < 1 for count in counts.values()):
        raise ValueError("Use a nonnegative seed start and at least one seed in each split.")
    seeds, cursor = {}, seed_start
    for split, count in counts.items():
        seeds[split] = list(range(cursor, cursor + count))
        cursor += count
    # Get held-out data early, without waiting until all training scenes finish.
    ordered = [(split, seed) for split in counts for seed in seeds[split][:5]]
    ordered.extend((split, seed) for split in counts for seed in seeds[split][5:])
    plan = []
    for split, seed in ordered:
        index = seed - seed_start
        kick_phrases = KICK_PHRASES[split]
        view_phrases = VIEW_PHRASES if split == "train" else HELD_OUT_VIEW_PHRASES[split]
        move_phrase_index = index % 2 if split == "train" else 2 if split == "val" else 3
        for color_index, (color, chinese) in enumerate(COLORS.items()):
            phrase = index + color_index
            other = list(COLORS.values())[(color_index + 1) % len(COLORS)]
            plan.append(
                {
                    "case_id": f"seed_{seed}_{color}",
                    "seed": seed,
                    "split": split,
                    "kind": "kick",
                    "target_color": color,
                    "instructions": [
                        view_phrases[phrase % len(view_phrases)],
                        kick_phrases[phrase % len(kick_phrases)].format(color=chinese, other=other),
                        view_phrases[(phrase + 2) % len(view_phrases)],
                    ],
                }
            )
        plan.append(
            {
                "case_id": f"seed_{seed}_direct",
                "seed": seed,
                "split": split,
                "kind": "direct_dialogue",
                "target_color": None,
                "instructions": [
                    view_phrases[index % len(view_phrases)],
                    *[phrases[move_phrase_index] for phrases in MOVE_PHRASES.values()],
                    ("停下", "停止", "先停一下", "不要动")[index % 4],
                    view_phrases[(index + 3) % len(view_phrases)],
                    ("踢球", "踢一下", "踢红球还是蓝球", "踢红球和蓝球")[index % 4],
                ],
            }
        )
    return seeds, plan


def _launch(command, log, timeout):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=HERE.parents[1], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
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


def _size(directory):
    return sum(path.stat().st_blocks * 512 for path in directory.rglob("*") if path.is_file())


def _executor_recovery(event):
    decision = event.get("decision") or {}
    return decision.get("skill") == "stand" and (
        (event.get("phase") == "settle" and decision.get("seconds") == 1.0)
        or (event.get("phase") == "recover" and decision.get("seconds") == 3.0)
    )


def _direction_acceptance(archive, events, expected_skill=None):
    import numpy as np

    rows = []
    expected = {
        "move_forward": (0, 1, 0.003),
        "move_backward": (0, -1, 0.003),
        "move_left": (1, 1, 0.003),
        "move_right": (1, -1, 0.003),
        "turn_left": (2, 1, 0.05),
        "turn_right": (2, -1, 0.05),
    }
    if expected_skill is not None:
        expected = {expected_skill: expected[expected_skill]}
    for event in events:
        if event.get("phase") != "direct":
            continue
        skill = event["decision"]["skill"]
        finish = next((row for row in events if row["task_id"] == event["task_id"] and row.get("done")), None)
        if skill not in expected or finish is None or not event["execution"].get("completed"):
            return {"success": False, "measurements": rows, "error": "missing_or_unexpected_direct_execution"}
        start = int(np.argmin(abs(archive["time"] - event["execution"]["start_time_s"])))
        end = int(np.argmin(abs(archive["time"] - finish["observation"]["time_s"])))
        yaw = float(archive["yaw"][start])
        world_delta = archive["trunk_xyz"][end, :2] - archive["trunk_xyz"][start, :2]
        cosine, sine = math.cos(yaw), math.sin(yaw)
        body_delta = np.array([[cosine, sine], [-sine, cosine]]) @ world_delta
        yaw_delta = float((archive["yaw"][end] - yaw + math.pi) % (2 * math.pi) - math.pi)
        axis, sign, minimum = expected[skill]
        measured = float(body_delta[axis]) if axis < 2 else yaw_delta
        rows.append(
            {
                "skill": skill,
                "body_displacement_xy_m": body_delta.tolist(),
                "yaw_delta_rad": yaw_delta,
                "signed_progress": sign * measured,
                "minimum_progress": minimum,
                "success": math.isfinite(measured) and sign * measured >= minimum,
            }
        )
    return {
        "success": len(rows) == len(expected)
        and {row["skill"] for row in rows} == set(expected)
        and all(row["success"] for row in rows),
        "measurements": rows,
    }


def _direct_acceptance(directory, summary, specification=None):
    import numpy as np

    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        qvel, tilts, root_xyz = archive["qvel"], archive["tilt_deg"], archive["trunk_xyz"]
        finite = all(np.isfinite(archive[key]).all() for key in ("qpos", "qvel", "tilt_deg", "trunk_xyz"))
        trace_valid = bool(
            len(tilts)
            and finite
            and float(tilts.max()) < 30
            and float(root_xyz[:, 2].min()) >= 0.065
            and float(tilts[-1]) < 10
            and float(np.linalg.norm(qvel[-1, :2])) < 0.025
            and float(np.linalg.norm(qvel[-1, 3:6])) < 0.3
            and float(root_xyz[-1, 2]) > 0.09
        )
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        expected_skill = (specification or {}).get("expected_skill")
        directions = _direction_acceptance(archive, events, expected_skill)
    contacts = json.loads((directory / "contacts.json").read_text(encoding="utf-8"))
    robot_contacts = [contact for contact in contacts if contact.get("other_color") is None]
    results = summary.get("results", [])
    expected_kinds = ["describe", "move", "move", "move", "move", "turn", "turn", "stop", "describe", "clarify"]
    expected_directions = [None, "forward", "backward", "left", "right", "left", "right", None, None, None]
    if expected_skill is not None:
        kind, direction = expected_skill.split("_", 1)
        expected_kinds, expected_directions = [kind], [direction]
    grammar_valid = len(results) == len(expected_kinds) and all(
        row.get("intent", {}).get("kind") == kind
        and row.get("intent", {}).get("direction") == direction
        and row.get("status") == ("clarification" if kind == "clarify" else "success")
        for row, kind, direction in zip(results, expected_kinds, expected_directions, strict=True)
    )
    return {
        "success": trace_valid and grammar_valid and not robot_contacts and directions["success"],
        "trace_valid": trace_valid,
        "grammar_valid": grammar_valid,
        "robot_ball_contacts": len(robot_contacts),
        "actual_directions": directions,
        "source": "independent_trace_checks_not_student_input",
    }


def _read_case(directory, specification):
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    event_path = directory / "events.jsonl"
    events = (
        [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if event_path.is_file()
        else []
    )
    artifacts = {
        name: (directory / name).is_file()
        for name in ("scene.xml", "trajectory.npz", "contacts.json", "events.jsonl", "episode.json", "summary.json")
    }
    results = summary.get("results", [])
    evaluation = None
    if specification["kind"] == "kick":
        kick = next((row for row in results if row.get("intent", {}).get("kind") == "kick"), None)
        status = kick.get("status") if kick else "missing_kick_result"
        evaluation = kick.get("evaluation") if kick else None
        success = bool(
            kick
            and kick.get("intent", {}).get("color") == specification["target_color"]
            and evaluation
            and evaluation.get("success") is True
            and len(results) == 3
            and all(row.get("status") == "success" for row in results)
        )
    else:
        evaluation = _direct_acceptance(directory, summary, specification) if all(artifacts.values()) else None
        success = bool(evaluation and evaluation["success"])
        status = "success" if success else "direct_verification_failed"
    success = bool(
        success and all(artifacts.values()) and not summary.get("errors") and not summary.get("runtime_failure")
    )
    actions = [event for event in events if event.get("execution", {}).get("completed") is True]
    initial_visible = sorted({d["color"] for d in events[0]["observation"]["detections"]}) if events else []
    return {
        **specification,
        "output": str(directory),
        "success": success,
        "status": status,
        "evaluation": evaluation,
        "artifacts": artifacts,
        "initial_visible_colors": initial_visible,
        "target_initially_visible": specification["target_color"] in initial_visible
        if specification["target_color"]
        else None,
        "searched": any(event.get("phase") == "search" for event in actions),
        "selected_kick": next((event["decision"]["skill"] for event in actions if event.get("phase") == "kick"), None),
        "event_count": len(events),
        "decision_count": sum(not _executor_recovery(event) for event in actions),
        "task_count": len(results),
        "simulation_seconds": summary.get("simulation_seconds"),
        "max_tilt_deg": summary.get("max_tilt_deg"),
        "bytes_on_disk": _size(directory),
        "errors": summary.get("errors", []),
    }


def _annotate(directory, specification, plan_sha256, verification=None):
    for name in ("episode.json", "summary.json"):
        path = directory / name
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            document.setdefault("metadata", {}).update(
                training_split=specification["split"],
                collection_case=specification["case_id"],
                collection_kind=specification["kind"],
                collection_plan_sha256=plan_sha256,
            )
            if "scenario" in specification:
                document["metadata"].update(
                    collection_scenario=specification["scenario"],
                    expected_foot_evaluation_only=specification.get("expected_foot"),
                    expected_skill_evaluation_only=specification.get("expected_skill"),
                )
            if verification is not None:
                document["metadata"].update(
                    collection_verified=verification["success"],
                    collection_evaluation=verification["evaluation"],
                    collection_status=verification["status"],
                )
            _json(path, document)


def _summarize(manifest):
    cases = manifest["cases"]
    kicks = [case for case in cases if case["kind"] == "kick"]
    successful_kicks = [case for case in kicks if case["success"]]
    manifest.update(
        completed_cases=len(cases),
        successful_cases=sum(case["success"] for case in cases),
        failed_cases=sum(not case["success"] for case in cases),
        completed_kicks=len(kicks),
        successful_kicks=len(successful_kicks),
        successful_search_kicks=sum(case["searched"] for case in successful_kicks),
        successful_kick_feet={
            foot: sum(case["selected_kick"] == foot for case in successful_kicks)
            for foot in ("kick_left", "kick_right")
        },
        bytes_on_disk=sum(case["bytes_on_disk"] for case in cases),
        mean_wall_seconds=sum(case.get("wall_seconds", 0) for case in cases) / len(cases) if cases else 0,
        by_split={
            split: {
                "complete": sum(case["split"] == split for case in cases),
                "success": sum(case["split"] == split and case["success"] for case in cases),
                "successful_seeds": sorted(
                    {case["seed"] for case in cases if case["split"] == split and case["success"]}
                ),
            }
            for split in ("train", "val", "test")
        },
        updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    )


def _recheck_completed(manifest):
    lookup = {case["case_id"]: case for case in manifest["plan"]}
    for index, previous in enumerate(manifest["cases"]):
        specification = lookup[previous["case_id"]]
        directory = Path(previous["output"])
        current = _read_case(directory, specification)
        for key in ("process_exit_code", "wall_timeout", "wall_seconds"):
            current[key] = previous[key]
        if current["wall_timeout"] or current["process_exit_code"] != 0:
            current["success"] = False
            current["status"] = previous["status"]
        _annotate(directory, specification, manifest["plan_sha256"], current)
        manifest["cases"][index] = current
    _summarize(manifest)


def _run_case(specification, output, plan_sha256, timeout):
    directory = output / specification["case_id"]
    command = [
        "bash",
        str(HERE / "run.sh"),
        "--output",
        str(directory),
        "--seed",
        str(specification["seed"]),
        "--teacher",
        "legacy",
        "--nominal-motion",
    ]
    if "layout_world_xy_m" in specification:
        layouts = output / "layouts"
        layouts.mkdir(exist_ok=True)
        layout_path = layouts / f"{specification['case_id']}.json"
        _json(layout_path, specification["layout_world_xy_m"])
        command.extend(["--layout-json", str(layout_path)])
    for instruction in specification["instructions"]:
        command.extend(["--command", instruction])
    started = time.monotonic()
    code, timed_out = _launch(command, output / "logs" / f"{specification['case_id']}.log", timeout)
    _annotate(directory, specification, plan_sha256)
    case = _read_case(directory, specification)
    case.update(process_exit_code=code, wall_timeout=timed_out, wall_seconds=time.monotonic() - started)
    if timed_out or code != 0:
        case["success"] = False
        if timed_out:
            case["status"] = "wall_timeout"
        elif case["status"] == "success":
            case["status"] = "process_failed"
    _annotate(directory, specification, plan_sha256, case)
    return case


def _collect_pending(args, output, manifest):
    """Each worker owns a case directory; only this thread writes the manifest."""
    completed = {case["case_id"] for case in manifest["cases"]}
    pending = [case for case in manifest["plan"] if case["case_id"] not in completed][: args.limit]
    stop = threading.Event()
    previous_handler = signal.signal(signal.SIGINT, lambda _signum, _frame: stop.set())
    summary_path = output / "summary.json"
    workers = getattr(args, "workers", 1)
    cursor, futures, error, stop_announced = 0, {}, None, False
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while futures or (cursor < len(pending) and not stop.is_set()):
                while cursor < len(pending) and len(futures) < workers and not stop.is_set():
                    free = shutil.disk_usage(output).free
                    if free < args.min_free_gib * 1024**3:
                        error = OSError(
                            f"Collection paused with {free / 1024**3:.2f} GiB free; preserve {args.min_free_gib} GiB."
                        )
                        stop.set()
                        break
                    specification = pending[cursor]
                    directory = output / specification["case_id"]
                    if directory.exists():
                        suffix = 1
                        while (output / f"{specification['case_id']}_interrupted_{suffix}").exists():
                            suffix += 1
                        archive = output / f"{specification['case_id']}_interrupted_{suffix}"
                        directory.rename(archive)
                        _annotate(
                            archive,
                            specification,
                            manifest["plan_sha256"],
                            {"success": False, "evaluation": None, "status": "interrupted_attempt"},
                        )
                        manifest.setdefault("interrupted_attempts", []).append(
                            {"case_id": specification["case_id"], "output": str(archive)}
                        )
                        _json(summary_path, manifest)
                    future = pool.submit(_run_case, specification, output, manifest["plan_sha256"], args.timeout)
                    futures[future] = specification
                    cursor += 1
                if stop.is_set() and not stop_announced:
                    print("Pausing after running cases finish; no additional cases will start.", flush=True)
                    stop_announced = True
                done, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    specification = futures.pop(future)
                    try:
                        case = future.result()
                    except Exception as exc:
                        error = exc
                        stop.set()
                        manifest.setdefault("collector_errors", []).append(
                            {"case_id": specification["case_id"], "error": f"{type(exc).__name__}: {exc}"}
                        )
                    else:
                        manifest["cases"].append(case)
                        _summarize(manifest)
                        print(
                            f"{len(manifest['cases'])}/{len(manifest['plan'])} {case['split']} {case['case_id']}:"
                            f" {case['status']} | verified={case['success']} | {case['wall_seconds']:.1f}s"
                            f" | {case['bytes_on_disk'] / 1024**2:.1f} MiB"
                            f" | kicks={manifest['successful_kicks']}/{manifest['completed_kicks']}",
                            flush=True,
                        )
                    _json(summary_path, manifest)
        if error:
            raise error
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def run(args, *, plan_factory=make_plan):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds, plan = plan_factory(args.seed_start, args.train_seeds, args.val_seeds, args.test_seeds)
    plan_json = json.dumps({"seed_splits": seeds, "plan": plan}, ensure_ascii=False, sort_keys=True)
    plan_sha256 = hashlib.sha256(plan_json.encode()).hexdigest()
    summary_path = output / "summary.json"
    if summary_path.is_file():
        manifest = json.loads(summary_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT or manifest.get("plan_sha256") != plan_sha256:
            raise ValueError(
                "This output contains another collection plan; choose a new directory or matching options."
            )
    else:
        if any(output.iterdir()):
            raise ValueError("A new collection needs an empty directory; preserve existing files elsewhere.")
        manifest = {
            "format_version": FORMAT,
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "hardware": False,
            "backend": "rgbd_rule_teacher",
            "seed_splits": seeds,
            "plan": plan,
            "plan_sha256": plan_sha256,
            "planned_total": len(plan),
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(HERE.glob("*.py"))
            },
            "cases": [],
        }
    (output / "logs").mkdir(exist_ok=True)
    manifest.setdefault("invocations", []).append(
        {
            "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "previously_completed": len(manifest["cases"]),
            "case_limit": args.limit,
            "recheck_only": bool(getattr(args, "recheck_only", False)),
            "min_free_gib": args.min_free_gib,
            "workers": getattr(args, "workers", 1),
            "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    _summarize(manifest)
    _json(summary_path, manifest)
    if getattr(args, "recheck_only", False):
        _recheck_completed(manifest)
        _json(summary_path, manifest)
        print(f"Rechecked {len(manifest['cases'])} saved cases without running physics: {summary_path}", flush=True)
        return 0
    _collect_pending(args, output, manifest)
    print(f"Collection saved: {summary_path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--train-seeds", type=int, default=80)
    parser.add_argument("--val-seeds", type=int, default=10)
    parser.add_argument("--test-seeds", type=int, default=10)
    parser.add_argument(
        "--limit", type=int, help="Run at most this many new cases, then save and exit; rerun to resume."
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--min-free-gib", type=float, default=8)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--recheck-only", action="store_true", help="Revalidate completed traces; do not run new cases."
    )
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive.")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be finite and positive.")
        if not math.isfinite(args.min_free_gib) or args.min_free_gib < 8:
            raise ValueError("Preserve at least 8 GiB free while collecting data.")
        return run(args)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
