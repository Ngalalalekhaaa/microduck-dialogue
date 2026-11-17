"""Native experiment lifecycle, with nonblocking telemetry and execution feedback."""

import copy
import json
import threading
from pathlib import Path

from .toy_dispatch import ToyInputDispatcher
from .toy_teacher import _goal_label


class ToyDesktopState:
    def __init__(self, output, backend_factory, *, seed=0):
        self.output, self.backend_factory, self.seed = Path(output), backend_factory, seed
        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.session = self.inputs = self.backend = self.viewer = None
        self.dispatch_thread = self.load_thread = self.reset_thread = None
        self.messages, self.errors = [], []
        self.execution = dict(state="idle", active=None, pending=[])
        self.phase, self.detail = "loading", "正在加载模型和仿真…"
        self.busy = self.episode = self.event_index = 0
        self.placement = None

    @staticmethod
    def _thread(target, name):
        worker = threading.Thread(target=target, name=name, daemon=True)
        worker.start()
        return worker

    def start(self):
        self.load_thread = self._thread(self._initialize, "toy-desktop-load")

    def _initialize(self):
        try:
            self.backend = self.backend_factory()
            if not self.shutdown.is_set():
                self._new_world()
        except Exception as exc:
            self._error(exc)

    def _new_world(self):
        from .action_runtime import LAYOUT
        from .toy_motion import ToyRuntime
        from .toy_native_viewer import NativeViewer
        from .toy_session import ToySession

        self.episode += 1
        folder = self.output / f"episode_{self.episode:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        runtime = ToyRuntime(folder, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
        viewer = session = None
        try:
            runtime.reset(copy.deepcopy(LAYOUT))
            runtime.enable_interactive_pacing()
            session = ToySession(runtime, seed=self.seed, split_group="manual_desktop", backend=self.backend)
            session.recorder.metadata.update(live_desktop=True, simulation_clock="paced_lockstep", max_goals=15)
            session.recorder.save_metadata()
            viewer = NativeViewer(runtime.scene, runtime.display_snapshot)
            viewer.start()
            inputs = ToyInputDispatcher(session, threading.Event(), threading.Event(), self.errors)
            with self.lock:
                self.session, self.inputs, self.viewer = session, inputs, viewer
                self.messages, self.event_index, self.busy = [], 0, 0
                self.errors.clear()
                self.execution = copy.deepcopy(session.execution)
                if self.placement:
                    viewer.place(*self.placement)
                self.phase, self.detail = "ready", "准备好了"
                self.dispatch_thread = self._thread(inputs.dispatch, "toy-desktop-input")
        except BaseException:
            if viewer is not None:
                viewer.close()
            if session is not None:
                session.close()
            runtime.close()
            raise

    def _error(self, exc):
        with self.lock:
            self.phase, self.detail = "error", f"{type(exc).__name__}: {exc}"
            self.errors.append(self.detail)
            self.messages.append(dict(role="execution", text=self.detail))

    def submit(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ValueError("请输入 1–1000 个字符。")
        with self.lock:
            if self.phase != "ready":
                raise ValueError("仿真正在准备或重置，请稍后发送。")
            self._collect_events()
            self.messages.append(dict(role="user", text=text.strip()))
            self.busy += 1
            episode = self.episode
            completed, box = self.inputs.submit(text.strip())

        def receive():
            while not completed.wait(0.1):
                if self.shutdown.is_set():
                    return
            with self.lock:
                if episode == self.episode:
                    self._collect_events()
                    self.busy = max(0, self.busy - 1)
                    if not box:
                        self.messages.append(dict(role="execution", text="此条指令已取消或处理失败，未提交执行。"))

        self._thread(receive, "toy-desktop-reply")

    def _collect_events(self):
        session = self.session
        if session is None or not session.recorder.lock.acquire(blocking=False):
            return
        try:
            events = session.recorder.events[self.event_index :]
            self.event_index += len(events)
        finally:
            session.recorder.lock.release()
        for row in events:
            if row["event"] == "dialogue":
                if row["request"]["event"]["kind"] != "feedback":
                    self.messages.append(dict(role="assistant", text=row["response"]["reply"]))
            elif row["event"] == "goal_start":
                self.messages.append(dict(role="execution", text=f"开始执行：{_goal_label(row['goal'])}。"))
            elif row["event"] == "goal_result":
                status = {"succeeded": "完成", "failed": "失败", "cancelled": "已取消", "blocked": "已暂停"}
                label = status.get(row["status"], row["status"])
                reasons = {
                    "motion_and_recovery_verified": "动作完成，已恢复稳定。",
                    "target_contact_and_displacement_verified": "已确认触球并产生位移。",
                    "safe_pause_boundary": "已站稳，任务进度保留。",
                    "user_or_local_stop": "任务已停止。",
                    "cancelled": "任务已停止。",
                    "body_not_upright": "身体姿态不稳定。",
                    "pause_stability_unverified": "未能确认站稳，请检查画面后重置。",
                    "guard_rejected": "当前观测或身体状态不满足执行条件。",
                    "execution_exception": "执行发生异常，请查看本次记录。",
                }
                detail = reasons.get(row["reason"], "" if row["status"] == "succeeded" else "详情已保存到本次记录。")
                self.messages.append(dict(role="execution", text=f"{_goal_label(row['goal'])}：{label}。{detail}"))
            elif row["event"] == "queue_control" and row.get("phase") == "safe_boundary_reached":
                self.messages.append(dict(role="execution", text="已在稳定动作边界暂停，剩余任务保留。"))

    def snapshot(self):
        with self.lock:
            self._collect_events()
            session, viewer = self.session, self.viewer
            sim = {}
            if session is not None:
                snap = session.runtime.display_snapshot()
                sim = {key: snap[key] for key in ("time_s", "skill", "tilt_deg", "stationary", "display_frame_id")}
                if session.state_lock.acquire(blocking=False):
                    try:
                        self.execution = copy.deepcopy(session.execution)
                    finally:
                        session.state_lock.release()
            return dict(
                ready=self.phase == "ready",
                loading=self.phase in ("loading", "resetting"),
                detail=self.detail,
                error="; ".join(self.errors[-3:]) or (viewer.error if viewer else None),
                backend_name=getattr(self.backend, "name", "loading"),
                output=str(self.output),
                episode=self.episode,
                viewer_running=bool(viewer and viewer.is_running()),
                viewer_geometry=viewer.geometry() if viewer else None,
                messages=copy.deepcopy(self.messages),
                execution=copy.deepcopy(self.execution),
                queue_capacity=15,
                inference_pending=self.busy,
                sim=sim,
                pause_requested=bool(session and session.pause_requested),
            )

    def place_viewer(self, x, y, width, height):
        values = (x, y, width, height)
        if any(type(value) is not int for value in values) or not (200 <= width <= 8192 and 200 <= height <= 8192):
            raise ValueError("请提供有效的窗口位置和尺寸。")
        with self.lock:
            self.placement = values
            if self.viewer is not None:
                self.viewer.place(*values)

    def reset(self):
        with self.lock:
            if self.phase not in ("ready", "error") or self.session is None:
                raise ValueError("当前无法重置，请等待初始化结束。")
            if self.reset_thread and self.reset_thread.is_alive():
                return
            self.phase, self.detail = "resetting", "正在结束旧任务并重置场景…"
            self.session.request_cancel()
            self.inputs.close()

        def work():
            try:
                self._dispose_world()
                if not self.shutdown.is_set():
                    self._new_world()
            except Exception as exc:
                self._error(exc)

        self.reset_thread = self._thread(work, "toy-desktop-reset")

    def _dispose_world(self):
        session = self.session
        if session is None:
            return
        session.request_cancel()
        self.inputs.close()
        if self.dispatch_thread:
            self.dispatch_thread.join(180)
            if self.dispatch_thread.is_alive():
                raise TimeoutError("模型尚未结束，停止状态已保留。")
        session.close()
        try:
            session.runtime.save_trace()
            (session.runtime.output / "summary.json").write_text(
                json.dumps(
                    dict(results=session.results, execution=session.execution, errors=self.errors),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            with self.lock:
                viewer, self.viewer = self.viewer, None
                self.session = self.inputs = None
            try:
                if viewer:
                    viewer.close()
            finally:
                session.runtime.close()

    def request_shutdown(self):
        self.shutdown.set()
        if self.session:
            self.session.request_cancel()

    def close(self):
        self.request_shutdown()
        with self.lock:
            viewer, self.viewer = self.viewer, None
        if viewer:
            viewer.close()
        for worker in (self.load_thread, self.reset_thread):
            if worker is not None:
                worker.join(60)
                if worker.is_alive():
                    raise TimeoutError("桌面已关闭，但模型线程尚未返回；记录目录已保留。")
        self._dispose_world()
