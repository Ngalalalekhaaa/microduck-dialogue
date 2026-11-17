"""Native MuJoCo UI over isolated display state, beside a separate Chinese Qt UI.

The official passive viewer owns its GLFW render thread. This experiment-only
exception leaves all existing RGB-D/offscreen rendering on ``strands-gl``.
Mouse/camera/menu edits affect the display copy, never the controlled world.
"""

import ctypes as C
import ctypes.util
import os
import threading
import time
from pathlib import Path

import numpy as np


class _X11Window:
    """Locate only this process's MuJoCo window, without private GLFW pointers."""

    def __init__(self):
        self.x = C.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        ptr, ulong, integer = C.c_void_p, C.c_ulong, C.c_int
        signatures = {
            "XOpenDisplay": (ptr, [C.c_char_p]),
            "XDefaultRootWindow": (ulong, [ptr]),
            "XInternAtom": (ulong, [ptr, C.c_char_p, integer]),
            "XGetWindowProperty": (
                integer,
                [ptr, ulong, ulong, C.c_long, C.c_long, integer, ulong]
                + [C.POINTER(ulong), C.POINTER(integer), C.POINTER(ulong), C.POINTER(ulong), C.POINTER(ptr)],
            ),
            "XFetchName": (integer, [ptr, ulong, C.POINTER(ptr)]),
            "XMoveResizeWindow": (integer, [ptr, ulong, integer, integer, C.c_uint, C.c_uint]),
            "XGetGeometry": (
                integer,
                [ptr, ulong, C.POINTER(ulong)] + [C.POINTER(integer)] * 2 + [C.POINTER(C.c_uint)] * 4,
            ),
            "XTranslateCoordinates": (
                integer,
                [ptr, ulong, ulong, integer, integer, C.POINTER(integer), C.POINTER(integer), C.POINTER(ulong)],
            ),
            "XFlush": (integer, [ptr]),
            "XFree": (integer, [ptr]),
            "XCloseDisplay": (integer, [ptr]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.x, name)
            function.restype, function.argtypes = result, arguments
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("无法连接 X11 桌面；请在本机桌面终端运行原生界面。")
        self.root = self.x.XDefaultRootWindow(self.display)
        self.window_id = None
        self.lock = threading.Lock()

    def _property(self, window, name):
        atom = self.x.XInternAtom(self.display, name.encode(), 0)
        actual, size, count, remaining, value = C.c_ulong(), C.c_int(), C.c_ulong(), C.c_ulong(), C.c_void_p()
        status = self.x.XGetWindowProperty(
            self.display,
            window,
            atom,
            0,
            4096,
            0,
            0,
            C.byref(actual),
            C.byref(size),
            C.byref(count),
            C.byref(remaining),
            C.byref(value),
        )
        try:
            if status or not value or size.value != 32:
                return []
            return list(C.cast(value, C.POINTER(C.c_ulong))[: count.value])
        finally:
            if value:
                self.x.XFree(value)

    def find(self):
        with self.lock:
            for window in self._property(self.root, "_NET_CLIENT_LIST"):
                if self._property(window, "_NET_WM_PID") != [os.getpid()]:
                    continue
                name = C.c_void_p()
                self.x.XFetchName(self.display, window, C.byref(name))
                try:
                    if name and C.string_at(name).startswith(b"MuJoCo"):
                        self.window_id = window
                        return window
                finally:
                    if name:
                        self.x.XFree(name)
        return None

    def place(self, x, y, width, height):
        if width < 200 or height < 200:
            raise ValueError("MuJoCo 窗口宽高至少为 200 像素。")
        with self.lock:
            if self.window_id:
                self.x.XMoveResizeWindow(self.display, self.window_id, x, y, width, height)
                self.x.XFlush(self.display)

    def geometry(self):
        with self.lock:
            if not self.window_id:
                return None
            root, child, x, y = C.c_ulong(), C.c_ulong(), C.c_int(), C.c_int()
            width, height, border, depth = (C.c_uint() for _ in range(4))
            if not self.x.XGetGeometry(
                self.display,
                self.window_id,
                C.byref(root),
                C.byref(x),
                C.byref(y),
                C.byref(width),
                C.byref(height),
                C.byref(border),
                C.byref(depth),
            ):
                return None
            self.x.XTranslateCoordinates(
                self.display, self.window_id, self.root, 0, 0, C.byref(x), C.byref(y), C.byref(child)
            )
            return x.value, y.value, width.value, height.value

    def close(self):
        with self.lock:
            if self.display:
                self.x.XCloseDisplay(self.display)
                self.display = None


class NativeViewer:
    """A native interactive viewer; ``snapshot_getter`` must use a display cache."""

    def __init__(self, scene, snapshot_getter, *, show_left_ui=False, show_right_ui=False, hz=30):
        import mujoco
        import mujoco.viewer

        if not np.isfinite(hz) or not 1 <= hz <= 120:
            raise ValueError("Viewer rate must be finite and between 1 and 120 Hz.")
        self.mujoco, self.snapshot_getter, self.hz = mujoco, snapshot_getter, hz
        self.model = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        self.data = mujoco.MjData(self.model)
        self.shutdown = threading.Event()
        self.error = self.worker = self.window = None
        self.frame_id = None
        initial = snapshot_getter()
        if initial is not None:
            self._copy_snapshot(initial)
        previous_threads = set(threading.enumerate())
        self.handle = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=show_left_ui, show_right_ui=show_right_ui
        )
        # MuJoCo 3.10 close() signals asynchronously; join its GLFW owner before atexit.
        self.native_threads = [
            thread
            for thread in threading.enumerate()
            if thread not in previous_threads and thread.name.endswith("(_launch_internal)")
        ]
        with self.handle.lock():
            self.handle.opt.geomgroup[3:] = 0
            self.handle.cam.distance = 0.9
            self.handle.cam.azimuth, self.handle.cam.elevation = 140, -22
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
            if body >= 0:
                self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.handle.cam.trackbodyid = body
        try:
            self.window = _X11Window()
            deadline = time.monotonic() + 2
            while not self.window.find() and time.monotonic() < deadline:
                time.sleep(0.02)
        except (OSError, RuntimeError):
            # Native viewing still works if the desktop has no X11 placement API.
            self.window = None

    @property
    def window_id(self):
        return self.window.window_id if self.window else None

    def geometry(self):
        return self.window.geometry() if self.window and self.handle.is_running() else None

    def place(self, x, y, width, height):
        if self.window and self.handle.is_running():
            self.window.place(int(x), int(y), int(width), int(height))

    def _copy_snapshot(self, snapshot):
        qpos, qvel = np.asarray(snapshot["qpos"]), np.asarray(snapshot["qvel"])
        timestamp = snapshot["time_s"]
        if qpos.shape != self.data.qpos.shape or qvel.shape != self.data.qvel.shape:
            raise ValueError("Display snapshot shape does not match the scene model.")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all() or not np.isfinite(timestamp):
            raise ValueError("Display snapshot must contain finite state and time.")
        self.data.qpos[:], self.data.qvel[:], self.data.time = qpos, qvel, timestamp
        self.mujoco.mj_forward(self.model, self.data)
        self.frame_id = snapshot.get("display_frame_id")

    def update(self):
        """Copy cache only; native menu changes cannot write to the robot world."""
        snapshot = self.snapshot_getter()
        if snapshot is not None:
            with self.handle.lock():
                self._copy_snapshot(snapshot)
        self.handle.sync()

    def start(self):
        if self.worker is not None:
            return

        def run():
            try:
                while not self.shutdown.is_set() and self.handle.is_running():
                    started = time.monotonic()
                    self.update()
                    self.shutdown.wait(max(0, 1 / self.hz - (time.monotonic() - started)))
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.shutdown.set()

        self.worker = threading.Thread(target=run, name="toy-native-sync", daemon=True)
        self.worker.start()

    def is_running(self):
        return not self.shutdown.is_set() and self.handle.is_running()

    def close(self):
        self.shutdown.set()
        if self.worker is not None:
            self.worker.join(3)
        self.handle.close()
        for thread in self.native_threads:
            thread.join(3)
        if self.window:
            self.window.close()
