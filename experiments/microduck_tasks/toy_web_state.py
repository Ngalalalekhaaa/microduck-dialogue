"""Local experiment lifecycle and cached snapshots for a responsive browser UI."""

import copy
import io
import json
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from .toy_dispatch import ToyInputDispatcher


class ToyWebState:
    def __init__(self, output, backend_factory, *, seed=0):
        self.output, self.backend_factory, self.seed = Path(output), backend_factory, seed
        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.session = self.inputs = self.backend = None
        self.dispatch_thread = None
        self.messages, self.errors, self.frames = [], [], {}
        self.telemetry, self.execution = {}, dict(state="idle", active=None, pending=[])
        self.phase, self.detail = "loading", "正在准备仿真和模型…"
        self.busy, self.episode, self.frame_id = 0, 0, 0
        self.renderer = None
        self.render_lock = threading.Lock()
        self.threads = []

    def start(self):
        self._thread(self._initialize, "toy-web-load")
        self._thread(self._render_loop, "toy-web-render")

    def _thread(self, target, name):
        thread = threading.Thread(target=target, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread

    def _initialize(self):
        try:
            self.backend = self.backend_factory()
            if not self.shutdown.is_set():
                self._new_world()
        except Exception as exc:
            self._error(exc)

    def _new_world(self):
        from .action_runtime import LAYOUT
        from .recording import _ReplayRenderer
        from .toy_motion import ToyRuntime
        from .toy_session import ToySession

        self.episode += 1
        folder = self.output / f"episode_{self.episode:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        runtime = ToyRuntime(folder, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
        runtime.reset(copy.deepcopy(LAYOUT))
        session = ToySession(runtime, seed=self.seed, split_group="manual_web", backend=self.backend)
        session.recorder.metadata.update(live_web=True, simulation_clock="lockstep", max_goals=15)
        session.recorder.save_metadata()
        renderer = _ReplayRenderer(runtime.scene)
        renderer.camera.distance = 0.85
        inputs = ToyInputDispatcher(session, threading.Event(), threading.Event(), self.errors)
        with self.render_lock:
            self.renderer = renderer
            with self.lock:
                self.session, self.inputs = session, inputs
                self.messages, self.frames = [], {}
                self.errors.clear()
                self.telemetry, self.execution = {}, copy.deepcopy(session.execution)
                self.busy, self.phase, self.detail = 0, "ready", "准备好了"
        self.dispatch_thread = self._thread(inputs.dispatch, "toy-web-input")

    def _error(self, exc):
        with self.lock:
            self.phase, self.detail = "error", f"{type(exc).__name__}: {exc}"
            self.errors.append(self.detail)

    def submit(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ValueError("请输入 1–1000 个字符。")
        text = text.strip()
        with self.lock:
            if self.phase != "ready":
                raise ValueError("仿真正在准备或重置，请稍后发送。")
            message_id, episode = uuid.uuid4().hex, self.episode
            self.messages.append(dict(id=message_id, user=text, reply=None, decision=None))
            self.busy += 1
            completed, box = self.inputs.submit(text)

        def receive():
            while not completed.wait(0.1):
                if self.shutdown.is_set():
                    return
            response = box[0] if box else dict(reply="这条旧指令已取消，未提交执行。", decision="cancel", goals=[])
            with self.lock:
                if episode != self.episode:
                    return
                for item in self.messages:
                    if item["id"] == message_id:
                        item.update(
                            reply=response["reply"], decision=response["decision"], goals=response.get("goals", [])
                        )
                self.busy = max(0, self.busy - 1)

        self._thread(receive, "toy-web-reply")
        return message_id

    def snapshot(self):
        with self.lock:
            session = self.session
            return dict(
                phase=self.phase,
                detail=self.detail,
                episode=self.episode,
                busy=self.busy,
                max_goals=15,
                backend=getattr(self.backend, "name", "loading"),
                execution=copy.deepcopy(self.execution),
                telemetry=copy.deepcopy(self.telemetry),
                messages=copy.deepcopy(self.messages),
                last_reply=session.last_reply if session else "",
                pause_requested=bool(getattr(session, "pause_requested", False)),
                errors=list(self.errors[-5:]),
                frame_id=self.frame_id,
                output=str(self.output),
            )

    def _render_loop(self):
        while not self.shutdown.is_set():
            started = time.monotonic()
            try:
                with self.render_lock:
                    session, renderer = self.session, self.renderer
                    if session is not None and renderer is not None:
                        self._capture(session, renderer)
            except Exception as exc:
                self._error(exc)
                return
            self.shutdown.wait(max(0.01, 0.12 - (time.monotonic() - started)))

    def _capture(self, session, renderer):
        if not session.runtime.lock.acquire(blocking=False):
            return
        try:
            snap = session.runtime.snapshot()
        finally:
            session.runtime.lock.release()
        if session.state_lock.acquire(blocking=False):
            try:
                execution = copy.deepcopy(session.execution)
            finally:
                session.state_lock.release()
            with self.lock:
                self.execution = execution
        with self.lock:
            if self.frames and self.telemetry.get("time_s") == snap["time_s"]:
                self.telemetry["failure"] = session.runtime.failure
                return
        views = renderer.frame(snap["qpos"], snap["qvel"], snap["time_s"])
        frames = {}
        for name, view in zip(("world", "front", "near"), views, strict=True):
            stream = io.BytesIO()
            Image.fromarray(view).save(stream, format="JPEG", quality=85)
            frames[name] = stream.getvalue()
        with self.lock:
            self.frames = frames
            self.frame_id += 1
            self.telemetry = {key: snap[key] for key in ("time_s", "skill", "tilt_deg", "stationary")}
            self.telemetry["failure"] = session.runtime.failure

    def reset(self):
        with self.lock:
            if self.phase not in ("ready", "error") or self.session is None:
                raise ValueError("当前无法重置，请等待初始化结束。")
            self.phase, self.detail = "resetting", "正在结束旧任务并重置场景…"
            self.inputs.submit("停下")
            self.inputs.close()

        def work():
            try:
                self.dispatch_thread.join(180)
                if self.dispatch_thread.is_alive():
                    raise TimeoutError("模型尚未返回，保留停止状态；请稍后重试重置。")
                self.session.close()
                with self.render_lock:
                    self._save()
                    self.renderer.close()
                    self.renderer = None
                    self.session.runtime.close()
                    self.session = None
                if not self.shutdown.is_set():
                    self._new_world()
            except Exception as exc:
                self._error(exc)

        self._thread(work, "toy-web-reset")

    def _save(self):
        session = self.session
        if session is None:
            return
        session.runtime.save_trace()
        value = dict(results=session.results, execution=session.execution, errors=self.errors)
        (session.runtime.output / "summary.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def close(self):
        self.shutdown.set()
        if self.session is not None:
            self.session.request_cancel()
            self.inputs.close()
            if self.dispatch_thread is not None:
                self.dispatch_thread.join(60)
            self.session.close()
            with self.render_lock:
                self._save()
                if self.renderer is not None:
                    self.renderer.close()
                self.session.runtime.close()
