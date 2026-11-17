"""Run offline Chinese robot tasks and prepare image/language demonstrations."""

import argparse
import codecs
import hashlib
import json
import math
import os
import queue
import select
import sys
import threading
import time
import uuid
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zoneinfo import ZoneInfo

from .bootstrap import configure


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _versions():
    result = {"python": sys.version.split()[0]}
    for name in ("mujoco", "onnxruntime", "numpy", "strands-robots", "bam"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed as distribution"
    return result


def _layout_file(path):
    layout = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(layout, dict) or set(layout) != {"red", "yellow", "blue"}:
        raise ValueError("Layout JSON must map exactly red, yellow and blue to [x, y] metre coordinates.")
    for color, point in layout.items():
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)
        ):
            raise ValueError(f"Layout {color} must contain two finite coordinates.")
        if not 0.08 <= math.hypot(*point) <= 2:
            raise ValueError(f"Layout {color} must be between 0.08 and 2 metres from the initial robot origin.")
    points = list(layout.values())
    if any(math.dist(a, b) < 0.08 for index, a in enumerate(points) for b in points[index + 1 :]):
        raise ValueError("Keep ball centres at least 0.08 metres apart in layout JSON.")
    return layout


def _stop_task(runtime, task):
    """Cancel promptly and wait until no task can race a subsequent command."""
    runtime.request_stop()
    if task is None:
        return
    deadline = time.monotonic() + 20
    while task.is_alive():
        # TaskSession.run clears cancellation at entry; reassert it if stop was
        # received during that startup window, before any new task is allowed.
        runtime.request_stop()
        task.join(timeout=0.05)
        if time.monotonic() >= deadline and task.is_alive():
            raise TimeoutError("Task cancellation did not finish; no new command was started.")


def interactive(session, *, viewer=True):
    """One background task, immediate stop, and continuous standing control."""
    from .language import parse_instruction

    runtime = session.runtime
    runtime.start_realtime()
    active = None
    errors = []
    window = None
    commands = queue.Queue()
    input_done = threading.Event()
    reader = None

    def read_commands():
        # input()/readline can invoke Tk's process-wide input hook from this
        # background thread. Read bytes instead, leaving every Tk call on main.
        decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")(errors="replace")
        pending = ""
        print("你 > ", end="", flush=True)
        while not input_done.is_set():
            try:
                ready, _, _ = select.select([sys.stdin.fileno()], [], [], 0.1)
                if not ready:
                    continue
                chunk = os.read(sys.stdin.fileno(), 4096)
            except OSError:
                chunk = b""
            pending += decoder.decode(chunk, final=not chunk)
            if not chunk:
                if pending.strip():
                    commands.put(pending.strip())
                commands.put(None)
                return
            while "\n" in pending:
                command, pending = pending.split("\n", 1)
                command = command.strip()
                commands.put(command)
                if command.lower() in {"exit", "quit", "退出", "结束"}:
                    return
                print("你 > ", end="", flush=True)

    def run_command(command):
        try:
            session.run(command)
        except Exception as exc:
            errors.append(exc)
            runtime.request_stop()
            print(f"任务异常：{type(exc).__name__}: {exc}", flush=True)

    print("中文仿真交互：输入一个任务；忙时可输入‘停下’。输入 exit 或‘退出’结束。", flush=True)
    try:
        if viewer:
            from .live_view import LiveView

            window = LiveView(session, on_stop=lambda: commands.put("停下"))
            print("实时窗口已打开：左侧为外部视角，右侧为 front / near 相机。", flush=True)
        reader = threading.Thread(target=read_commands, name="microduck-input", daemon=True)
        reader.start()
        while True:
            if errors:
                raise errors[0]
            if runtime.failure:
                raise RuntimeError(runtime.failure)
            if window is not None and not window.pump():
                break
            try:
                command = commands.get(timeout=0.02)
            except queue.Empty:
                continue
            if command is None:
                break
            if not command:
                continue
            if command.lower() in {"exit", "quit", "退出", "结束"}:
                break
            if parse_instruction(command).kind == "stop":
                _stop_task(runtime, active)
                active = None
                session.run(command)
            elif active is not None and active.is_alive():
                print("当前任务正在执行；请先说‘停下’，再输入新任务。", flush=True)
            else:
                if active is not None:
                    active.join()
                active = threading.Thread(target=run_command, args=(command,), name="microduck-task", daemon=True)
                active.start()
    finally:
        input_done.set()
        try:
            _stop_task(runtime, active)
        finally:
            try:
                if window is not None:
                    window.close()
            finally:
                if reader is not None:
                    reader.join(timeout=1)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New run directory; existing paths are refused.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher", choices=("legacy", "efficient"), default="efficient")
    parser.add_argument(
        "--nominal-motion", action="store_true", help="Disable measured heading compensation for comparison."
    )
    parser.add_argument("--radius-min", type=float, default=0.3)
    parser.add_argument("--radius-max", type=float, default=0.6)
    parser.add_argument("--layout-json", type=Path, help="Optional explicit red/yellow/blue [x,y] layout.")
    parser.add_argument("--command", action="append", help="Repeat to run commands in one continuous world.")
    parser.add_argument(
        "--interactive", action="store_true", help="Read commands while standing control keeps running."
    )
    parser.add_argument("--no-viewer", action="store_true", help="Disable the default live window in interactive mode.")
    parser.add_argument(
        "--backend",
        choices=("rule", "vlm", "vlm-v2", "head"),
        default="rule",
        help="rule baseline or a pretrained VLM choosing every skill.",
    )
    parser.add_argument("--vlm-endpoint", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--vlm-model", default="microduck-vlm")
    parser.add_argument("--head-checkpoint", type=Path, help="Trained frozen-feature skill head (.pt).")
    parser.add_argument("--head-model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--vlm-timeout", type=float, default=45)
    parser.add_argument("--vlm-max-steps", type=int, default=40)
    parser.add_argument(
        "--vlm-rgb-only",
        action="store_true",
        help="Omit RGB-D positions from model input; execution guard still checks them.",
    )
    parser.add_argument(
        "--expected-color",
        choices=("red", "yellow", "blue"),
        help="Independent kick test target, never supplied to the VLM; one --command only.",
    )
    parser.add_argument(
        "--video", action="store_true", help="Render the saved physical trajectory after closing simulation."
    )
    parser.add_argument("--export-dataset", type=Path, help="New VLM messages dataset directory.")
    parser.add_argument("--export-only", type=Path, help="Export existing episodes without constructing a simulation.")
    return parser


def run(args):
    from .dataset import Recorder, export_vlm_dataset

    if args.backend in ("vlm-v2", "head") and args.vlm_rgb_only:
        raise ValueError("The v2 navigation executor currently requires RGBD entities; omit --vlm-rgb-only.")

    if args.export_only is not None:
        if args.export_dataset is None:
            raise ValueError("--export-only requires --export-dataset with a new output directory.")
        result = export_vlm_dataset(args.export_only, args.export_dataset)
        print(json.dumps(result["counts"], ensure_ascii=False, indent=2), flush=True)
        return 0

    configure()
    from .runtime import Runtime
    from .scene import CAMERAS, sample_layout
    from .tasks import TaskSession

    backend = None
    if args.backend == "head":
        from .head_backend import SkillHeadBackend

        if args.head_checkpoint is None:
            raise ValueError("--backend head requires --head-checkpoint pointing to the trained head.pt.")
        backend = SkillHeadBackend(args.head_checkpoint, args.head_model)
    if args.backend in ("vlm", "vlm-v2"):
        from .skill_vlm import LocalVLMBackend

        backend_class = LocalVLMBackend
        if args.backend == "vlm-v2":
            from .plan_backend import LocalPlanBackend

            backend_class = LocalPlanBackend
        backend = backend_class(endpoint=args.vlm_endpoint, model=args.vlm_model, timeout=args.vlm_timeout)
        if args.vlm_model in {"microduck-qlora", "microduck-onnx", "microduck-plan-qlora"}:
            backend.refresh_model_metadata()
        deployment = Path(__file__).resolve().parents[2] / ".cache/microduck_vlm/deployment.json"
        if (
            backend.endpoint == "http://127.0.0.1:8081/v1/chat/completions"
            and args.vlm_model == "microduck-vlm"
            and deployment.is_file()
        ):
            backend.metadata["configured_local_deployment"] = json.loads(deployment.read_text(encoding="utf-8"))
    if args.expected_color and (
        args.backend not in ("vlm", "vlm-v2", "head") or args.interactive or len(args.command or []) != 1
    ):
        raise ValueError("--expected-color requires --backend vlm and exactly one --command without --interactive.")

    if not math.isfinite(args.radius_min) or not math.isfinite(args.radius_max):
        raise ValueError("Radius bounds must be finite metres.")
    layout = (
        _layout_file(args.layout_json)
        if args.layout_json
        else sample_layout(args.seed, args.radius_min, args.radius_max)
    )
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    episode_id = f"episode_{timestamp}_{uuid.uuid4().hex[:8]}"
    output = args.output or Path(__file__).parent / "outputs" / episode_id
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = (
        args.command
        if args.command is not None
        else ([] if args.interactive else ["现在能看到什么？", "踢红球", "现在眼前有什么？"])
    )
    metadata = {
        "backend": backend.name if backend else "rgbd_rule_teacher",
        "is_trained_vlm": backend is not None,
        "robot_finetuned": backend.metadata["robot_finetuned"] if backend else False,
        "vlm": backend.metadata if backend else None,
        "vlm_rgbd_assist": backend is not None and not args.vlm_rgb_only,
        "expected_color_evaluation_only": args.expected_color,
        "hardware": False,
        "teacher": args.teacher,
        "motion_control": "nominal" if args.nominal_motion else "calibrated_heading_hold",
        "seed": args.seed,
        "initial_layout_world_xy_m": layout,
        "layout_source": str(args.layout_json) if args.layout_json else "random_polar_coordinates",
        "radius_min_m": args.radius_min,
        "radius_max_m": args.radius_max,
        "camera": CAMERAS,
        "versions": _versions(),
        "interactive": args.interactive,
        "live_viewer": args.interactive and not args.no_viewer,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    runtime = session = None
    summary = {"episode_id": episode_id, "output": str(output), "metadata": metadata, "results": [], "errors": []}
    exit_code = 0
    started = time.monotonic()
    try:
        runtime = Runtime(output)
        runtime.policy.motion.enabled = not args.nominal_motion
        metadata["motion_commands"] = runtime.policy.motion.commands
        metadata["motion_calibration_sha256"] = hashlib.sha256(
            Path(__file__).with_name("motion_calibration.json").read_bytes()
        ).hexdigest()
        metadata.update(physics=runtime.physics, policy_sha256=runtime.policy.hashes)
        recorder = Recorder(output, episode_id, args.seed, metadata)
        runtime.reset(layout)
        if backend is None:
            session = TaskSession(runtime, recorder, teacher=args.teacher)
        else:
            from .skill_tasks import SkillTaskSession

            session_class = SkillTaskSession
            if args.backend == "vlm-v2":
                from .plan_tasks import PlanTaskSession

                session_class = PlanTaskSession
            elif args.backend == "head":
                from .head_backend import SkillHeadTaskSession

                session_class = SkillHeadTaskSession
            session = session_class(
                runtime,
                recorder,
                backend,
                max_steps=args.vlm_max_steps,
                timeout=args.vlm_timeout,
                rgbd_assist=not args.vlm_rgb_only,
                expected_color=args.expected_color,
            )
            runtime.start_realtime()
        for command in commands:
            print(f"指令：{command}", flush=True)
            session.run(command)
            if runtime.failure:
                break
        if args.interactive and not runtime.failure:
            interactive(session, viewer=not args.no_viewer)
    except KeyboardInterrupt:
        summary["errors"].append({"stage": "run", "type": "KeyboardInterrupt", "message": "Interrupted by operator."})
        exit_code = 130
    except Exception as exc:
        summary["errors"].append({"stage": "run", "type": type(exc).__name__, "message": str(exc)})
        exit_code = 2
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                summary["errors"].append({"stage": "close", "type": type(exc).__name__, "message": str(exc)})
                exit_code = 2
            for stage, operation in (
                ("trajectory", runtime.save_trace),
                ("contacts", lambda: _write_json(output / "contacts.json", runtime.contacts)),
            ):
                try:
                    operation()
                except Exception as exc:
                    summary["errors"].append({"stage": stage, "type": type(exc).__name__, "message": str(exc)})
                    exit_code = 2
            summary.update(
                physics=runtime.physics,
                max_target_excess_rad=getattr(runtime, "target_excess", []).tolist()
                if hasattr(runtime, "target_excess")
                else [],
                max_tilt_deg=max((row[4] for row in runtime.rows), default=None),
                simulation_seconds=runtime.rows[-1][0] if runtime.rows else 0,
                runtime_failure=runtime.failure,
            )
        summary["results"] = session.results if session is not None else []
        failures = [
            row
            for row in summary["results"]
            if row["status"] not in {"success", "clarification", "cancelled", "answered", "model_reported"}
            or (args.expected_color is not None and row["status"] != "success")
        ]
        if (failures or summary.get("runtime_failure")) and not exit_code:
            exit_code = 2
        summary["wall_seconds"] = time.monotonic() - started
        summary["exit_code"] = exit_code
        if backend is not None:
            metadata.update(
                backend=backend.name, robot_finetuned=backend.metadata["robot_finetuned"], vlm=backend.metadata
            )
            episode_path = output / "episode.json"
            if episode_path.is_file():
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
                episode["metadata"] = metadata
                _write_json(episode_path, episode)
        _write_json(output / "summary.json", summary)

    if args.video and (output / "trajectory.npz").is_file():
        try:
            from .recording import render_video

            summary["video"] = str(render_video(output))
        except Exception as exc:
            summary["errors"].append({"stage": "video", "type": type(exc).__name__, "message": str(exc)})
            exit_code = exit_code or 3
    if args.export_dataset is not None and (output / "episode.json").is_file():
        try:
            manifest = export_vlm_dataset(output, args.export_dataset)
            summary["dataset"] = {"path": str(args.export_dataset.resolve()), "counts": manifest["counts"]}
        except Exception as exc:
            summary["errors"].append({"stage": "export", "type": type(exc).__name__, "message": str(exc)})
            exit_code = exit_code or 3
    summary.update(exit_code=exit_code, wall_seconds=time.monotonic() - started)
    _write_json(output / "summary.json", summary)
    print(f"记录目录：{output}", flush=True)
    if summary["errors"]:
        print(json.dumps(summary["errors"], ensure_ascii=False), file=sys.stderr, flush=True)
    return exit_code


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.export_only is not None and (
        args.interactive or args.video or args.command or args.output or args.layout_json
    ):
        parser.error("--export-only cannot be combined with simulation or recording options.")
    try:
        return run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
