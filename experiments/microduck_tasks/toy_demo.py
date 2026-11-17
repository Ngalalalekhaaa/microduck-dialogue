"""Visual, recorded toy dialogue and continuous multi-goal demonstrations."""

import argparse
import codecs
import copy
import json
import os
import select
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from .bootstrap import configure
from .toy_dispatch import ToyInputDispatcher

DEFAULT_COMMAND = "先踢蓝球，再踢红球，再原地转一圈"


def _backend_label(backend):
    return {
        "frozen_vlm_toy_selector": "模型技能选择 / 模板回复",
        "qwen35_toy_zero_shot": "原始 Qwen 生成技能计划和回复",
        "toy_rule_teacher": "规则教师",
    }.get(backend.name, backend.name)


def _goal_label(goal):
    if goal is None:
        return "无"
    from .action_catalog import LABELS

    kind = goal["type"]
    if kind == "perform":
        return f"{LABELS[goal['action']]} × {goal['repeat']}"
    if kind in ("kick", "navigate"):
        color = next((a["value"] for a in goal["target"]["attributes"] if a["name"] == "color"), "指定")
        name = {"red": "红", "yellow": "黄", "blue": "蓝"}.get(color, color)
        return f"踢{name}球" if kind == "kick" else f"走到{name}球旁边"
    if kind == "posture":
        return "坐下保持" if goal["pose"] == "sitting" else "站起来"
    direction = {"forward": "前", "backward": "后", "left": "左", "right": "右"}[goal["direction"]]
    if kind == "turn":
        return f"向{direction}转 {goal['angle_deg']:g} 度"
    limit = goal["limit"]
    duration = "直到喊停" if limit["kind"] == "until_cancelled" else f"{limit['seconds']:g} 秒"
    return f"向{direction}走，{duration}"


def _pump_without_waiting(view, paint):
    """Keep Tk stop callbacks responsive while inference owns session/world locks."""
    if view.closed:
        return False
    view.root.update()
    if view.closed:
        return False
    session = view.session
    if not session.state_lock.acquire(blocking=False):
        return True
    try:
        if not session.runtime.lock.acquire(blocking=False):
            return True
        try:
            return paint()
        finally:
            session.runtime.lock.release()
    finally:
        session.state_lock.release()


def _window(session, submit, shutdown):
    import tkinter as tk

    from .live_view import LiveView

    class ToyView(LiveView):
        def __init__(self):
            super().__init__(session)
            source = _backend_label(session.backend)
            self.root.title(f"小鸭多任务演示 — 真实仿真 / {source}")
            self.reply = tk.StringVar(value="小鸭：准备好了。")
            self.state = tk.StringVar(value="任务：空闲")
            panel = tk.Frame(self.root)
            panel.pack(side=tk.BOTTOM, fill=tk.X, before=self.label)
            tk.Label(panel, textvariable=self.reply, anchor="w", wraplength=1050).pack(fill=tk.X)
            tk.Label(panel, textvariable=self.state, anchor="w", wraplength=1050).pack(fill=tk.X)
            self.entry = tk.Entry(panel)
            self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=6)
            self.entry.bind("<Return>", lambda _event: self.send())
            tk.Button(panel, text="发送", command=self.send).pack(side=tk.RIGHT, padx=6)
            self.renderer.camera.distance = 0.85

        def send(self):
            command = self.entry.get().strip()
            if command:
                self.entry.delete(0, tk.END)
                submit(command)

        def stop_task(self):
            submit("停下")

        def request_close(self):
            submit("停下")
            self.closed = True
            shutdown.set()

        def pump(self):
            self.reply.set(f"小鸭：{session.last_reply or '准备好了。'}")
            return _pump_without_waiting(self, self.paint)

        def paint(self):
            execution = copy.deepcopy(session.execution)
            labels = {"idle": "空闲", "running": "执行中", "paused": "已暂停", "failed": "未完成"}
            active = _goal_label(execution["active"])
            self.state.set(
                f"任务：{labels.get(execution['state'], execution['state'])}；当前 {active}；"
                f"后续 {len(execution['pending'])} 步"
            )
            return super().pump()

    return ToyView()


def _read_terminal(submit, shutdown, *, close_on_eof):
    decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")(errors="replace")
    pending = ""
    print("可在终端或窗口输入：你好 / 你现在能看到什么 / 停下；输入 exit 退出。", flush=True)
    while not shutdown.is_set():
        try:
            if not select.select([sys.stdin.fileno()], [], [], 0.1)[0]:
                continue
            chunk = os.read(sys.stdin.fileno(), 4096)
        except (OSError, ValueError):
            chunk = b""
        pending += decoder.decode(chunk, final=not chunk)
        if not chunk:
            if pending.strip():
                submit(pending.strip())
            if close_on_eof:
                shutdown.set()
            return
        while "\n" in pending:
            command, pending = pending.split("\n", 1)
            command = command.strip()
            if command.lower() in ("exit", "quit", "退出", "结束"):
                shutdown.set()
                return
            if command:
                submit(command)


def _wait_task(session, shutdown, *, bound_continuous):
    continuous_start = None
    deadline = time.monotonic() + 600
    while session.worker is not None and session.worker.is_alive() and not shutdown.is_set():
        with session.state_lock:
            active = copy.deepcopy(session.execution["active"])
        if bound_continuous and active and active["type"] == "move" and active["limit"]["kind"] == "until_cancelled":
            now = session.runtime.snapshot()["time_s"]
            if continuous_start is None:
                continuous_start = now
            if now - continuous_start >= 1.0:
                print("无窗口演示的持续运动已达到约 1 秒，记录自动‘停下’。", flush=True)
                session.run("停下", wait=False)
                session.wait(timeout=1)
                return
        if time.monotonic() > deadline:
            session.run("停下", wait=False)
            raise TimeoutError("Dialogue task exceeded the 600 second demonstration budget.")
        time.sleep(0.01)
    if not shutdown.is_set():
        session.wait(timeout=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", action="append", help="Repeat for sequential utterances in one world.")
    parser.add_argument("--interactive", action="store_true", help="Start without the default task; accept live chat.")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=("default", "random"), default="default")
    parser.add_argument("--backend", choices=("teacher", "selector", "vlm"), default="teacher")
    parser.add_argument(
        "--vlm-context",
        choices=("full", "compact", "rules"),
        default="compact",
        help="Prompt variant for --backend vlm.",
    )
    parser.add_argument("--head-checkpoint", type=Path, help="Trained selector.pt, required for --backend selector.")
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args(argv)
    if args.backend == "selector" and args.head_checkpoint is None:
        parser.error("--backend selector requires --head-checkpoint pointing to a trained selector.pt.")
    if args.backend != "selector" and args.head_checkpoint is not None:
        parser.error("--head-checkpoint requires --backend selector.")
    configure()
    from .action_runtime import LAYOUT
    from .scene import sample_layout
    from .toy_motion import ToyRuntime
    from .toy_session import ToySession

    backend = None
    if args.backend == "selector":
        from .toy_selector_backend import ToySelectorBackend

        backend = ToySelectorBackend(args.head_checkpoint, args.model_dir)
        print("实验技能选择器：目标由模型预测，回复为模板；仅锁步仿真。", flush=True)
    elif args.backend == "vlm":
        from .toy_vlm_backend import ToyVLMBackend

        backend = ToyVLMBackend(args.model_dir, prompt_variant=args.vlm_context)
        print(f"原始 Qwen（未针对本任务微调）：{args.vlm_context} 上下文，直接生成计划和回复；仅锁步仿真。", flush=True)
    commands = args.command if args.command is not None else ([] if args.interactive else [DEFAULT_COMMAND])
    output = (
        args.output
        or Path(__file__).parent / "outputs" / f"toy_demo_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    layout = copy.deepcopy(LAYOUT) if args.layout == "default" else sample_layout(args.seed)
    runtime = ToyRuntime(output, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
    runtime.reset(layout)
    session = ToySession(
        runtime, seed=args.seed, split_group=f"manual_layout_{args.layout}_{args.seed}", backend=backend
    )
    session.recorder.metadata.update(
        initial_layout_world_xy_m=layout,
        live_viewer=not args.no_viewer,
        interactive=args.interactive,
        simulation_clock=(
            "lockstep" if getattr(session.backend, "requires_lockstep", False) else "realtime_when_interactive"
        ),
    )
    session.recorder.save_metadata()
    shutdown, manual_override, sequence_done = threading.Event(), threading.Event(), threading.Event()
    errors = []
    window = reader = sequence = None
    inputs = ToyInputDispatcher(session, shutdown, manual_override, errors)
    submit = inputs.submit

    def save():
        runtime.save_trace()
        with session.state_lock:
            results, execution = copy.deepcopy(session.results), copy.deepcopy(session.execution)
        session.recorder.metadata.update(physics=runtime.physics, policy_sha256=runtime.policy.hashes)
        session.recorder.save_metadata()
        value = dict(
            metadata=session.recorder.metadata,
            results=results,
            execution=execution,
            errors=list(errors),
            runtime_failure=runtime.failure,
            simulation_seconds=runtime.snapshot()["time_s"],
        )
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
        temporary.replace(output / "summary.json")

    def run_sequence():
        try:
            for command in commands:
                if shutdown.is_set() or manual_override.is_set():
                    break
                completed, _ = submit(command, scripted=True)
                while not completed.wait(0.05):
                    if shutdown.is_set():
                        return
                _wait_task(session, shutdown, bound_continuous=args.no_viewer)
                with session.state_lock:
                    failed = any(row["status"] == "failed" for row in session.results)
                if runtime.failure or failed:
                    print("本轮任务未通过验收，停止后续预设指令。", flush=True)
                    break
            save()
            print(f"本轮记录已保存：{output}", flush=True)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            shutdown.set()
        finally:
            sequence_done.set()

    dispatcher = threading.Thread(target=inputs.dispatch, name="toy-dialogue-input", daemon=True)
    dispatcher.start()
    try:
        if not args.no_viewer:
            window = _window(session, submit, shutdown)
        if not args.no_viewer or args.interactive:
            if not getattr(session.backend, "requires_lockstep", False):
                runtime.start_realtime()
            reader = threading.Thread(
                target=_read_terminal,
                args=(submit, shutdown),
                kwargs={"close_on_eof": args.no_viewer},
                name="toy-terminal",
                daemon=True,
            )
            reader.start()
        sequence = threading.Thread(target=run_sequence, name="toy-demo-sequence", daemon=True)
        sequence.start()
        while not shutdown.is_set():
            if runtime.failure:
                errors.append(runtime.failure)
                break
            if window is not None and not window.pump():
                break
            if args.no_viewer and not args.interactive and sequence_done.is_set():
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        shutdown.set()
        if not inputs.stop_pending.is_set():
            submit("停下")
        inputs.close()
        dispatcher.join(timeout=20)
        session.close()
        if sequence is not None:
            sequence.join(timeout=20)
        if reader is not None:
            reader.join(timeout=1)
        if window is not None:
            window.close()
        runtime.close()
        save()
    print(f"记录目录：{output}", flush=True)
    if errors:
        print(json.dumps(errors, ensure_ascii=False), file=sys.stderr)
    return int(bool(errors or runtime.failure or any(row["status"] == "failed" for row in session.results)))


if __name__ == "__main__":
    raise SystemExit(main())
