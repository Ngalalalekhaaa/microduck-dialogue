"""Desktop display of live physics snapshots; Tk stays on the main thread."""

import threading
import time

from PIL import Image, ImageTk

from .recording import _compose, _font, _ReplayRenderer


class LiveView:
    def __init__(self, session, *, on_stop=None, fps=15):
        import tkinter as tk

        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Create the live display on the main thread.")
        self.session = session
        self.on_stop = on_stop
        self.closed = False
        self.renderer = self.photo = self.root = None
        self.next_frame = 0.0
        self.interval = 1 / fps
        try:
            self.root = tk.Tk()
            self.root.title("HD1910 仿真 — 实时外部视角 / front / near")
            self.root.protocol("WM_DELETE_WINDOW", self.request_close)
            self.root.bind("<Escape>", lambda _event: self.request_close())
            self.root.minsize(640, 460)
            width = min(1280, max(640, self.root.winfo_screenwidth() - 80))
            height = min(900, max(460, self.root.winfo_screenheight() - 100))
            self.root.geometry(f"{width}x{height}")
            controls = tk.Frame(self.root)
            controls.pack(side=tk.BOTTOM, fill=tk.X)
            tk.Label(controls, text="在终端输入指令 · front：前视相机 · near：额外的虚拟近场相机").pack(side=tk.LEFT)
            tk.Button(controls, text="停止任务", command=self.stop_task).pack(side=tk.RIGHT, padx=8, pady=5)
            self.label = tk.Label(self.root, background="#121820")
            self.label.pack(fill=tk.BOTH, expand=True)
            self.renderer = _ReplayRenderer(session.runtime.scene)
            if getattr(session.backend, "metadata", {}).get("backend") == "action_playground":
                self.renderer.camera.distance = 0.75
            self.root.update()
        except Exception as exc:
            self.close()
            raise RuntimeError(
                "无法打开实时窗口。请在本机桌面终端运行并检查 DISPLAY；无桌面环境可加 --no-viewer 使用纯终端模式。"
            ) from exc

    def stop_task(self):
        self.session.runtime.request_stop()
        if self.on_stop is not None:
            self.on_stop()
        print("已请求停止当前任务，恢复站立。", flush=True)

    def request_close(self):
        self.closed = True
        self.session.runtime.request_stop()

    def pump(self):
        if self.closed:
            return False
        self.root.update()
        if self.closed:
            return False
        now = time.monotonic()
        if now < self.next_frame:
            return True
        snapshot = self.session.runtime.snapshot()
        views = self.renderer.frame(snapshot["qpos"], snapshot["qvel"], snapshot["time_s"])
        backend = self.session.backend
        metadata = getattr(backend, "metadata", {"backend": getattr(backend, "name", "rgbd_rule_teacher")})
        frame = _compose(
            views,
            snapshot["time_s"],
            snapshot["skill"],
            snapshot["tilt_deg"],
            {"instruction": getattr(self.session, "instruction", "在终端输入指令")},
            _font(20),
            live=True,
            model_metadata=metadata,
        )
        frame.thumbnail(
            (max(1, self.label.winfo_width()), max(1, self.label.winfo_height())), Image.Resampling.BILINEAR
        )
        self.photo = ImageTk.PhotoImage(frame, master=self.root)
        self.label.configure(image=self.photo)
        self.next_frame = time.monotonic() + self.interval
        return True

    def close(self):
        self.closed = True
        try:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        finally:
            if self.root is not None:
                self.photo = None
                self.root.destroy()
                self.root = None
