"""Standalone system-Python Qt side panel for the local native MuJoCo desktop."""

import argparse
import itertools
import json
import queue
import socket
import threading
import time
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets


def exchange(socket_path, request, timeout=0.6):
    """One newline-delimited JSON request/response, only from the IPC worker."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        deadline = time.monotonic() + timeout
        connection.settimeout(timeout)
        connection.connect(str(socket_path))
        connection.sendall(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
        received = bytearray()
        while b"\n" not in received:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("本地仿真响应超时。")
            connection.settimeout(remaining)
            part = connection.recv(65536)
            if not part:
                raise ConnectionError("本地仿真连接已关闭。")
            received.extend(part)
            if len(received) > 2_000_000:
                raise ValueError("本地仿真返回的信息过长。")
        response = json.loads(received.partition(b"\n")[0])
        if not isinstance(response, dict):
            raise ValueError("本地仿真未返回有效状态。")
        return response


class IPCClient(QtCore.QObject):
    response = QtCore.pyqtSignal(int, object, object)
    failure = QtCore.pyqtSignal(int, object, str)

    def __init__(self, socket_path, parent=None):
        super().__init__(parent)
        self.socket_path = socket_path
        self.requests = queue.PriorityQueue()
        self.sequence = itertools.count()
        self.stopped = threading.Event()
        self.worker = threading.Thread(target=self._work, name="toy-qt-ipc", daemon=True)
        self.worker.start()

    def submit(self, request):
        number = next(self.sequence)
        priority = -10 if request["command"] == "shutdown" else 20 if request["command"] == "state" else 0
        self.requests.put((priority, number, dict(request)))
        return number

    def _work(self):
        while not self.stopped.is_set():
            _, number, request = self.requests.get()
            if request is None:
                return
            try:
                result = exchange(self.socket_path, request)
                self.response.emit(number, request, result)
            except (OSError, ValueError) as exc:
                self.failure.emit(number, request, str(exc))

    def stop(self):
        self.stopped.set()
        self.requests.put((-100, next(self.sequence), None))
        self.worker.join(timeout=0.1)


class MessageInput(QtWidgets.QPlainTextEdit):
    send_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.preedit_active = False
        self.ime_return_until = 0.0
        self.setAttribute(QtCore.Qt.WA_InputMethodEnabled, True)
        self.setPlaceholderText("输入指令或聊天内容…\nEnter 发送 · Shift+Enter 换行")

    def inputMethodEvent(self, event):
        self.preedit_active = bool(event.preeditString())
        if event.commitString():
            # Some IMEs deliver the Return key immediately after committing the candidate.
            self.ime_return_until = time.monotonic() + 0.12
        super().inputMethodEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if self.preedit_active or time.monotonic() < self.ime_return_until or event.isAutoRepeat():
                event.accept()
                return
            if not event.modifiers() & QtCore.Qt.ShiftModifier:
                if self.toPlainText().strip():
                    self.send_requested.emit()
                event.accept()
                return
        super().keyPressEvent(event)


def desktop_layout(available):
    """Qt logical coordinates; the native X11 placement request is converted separately."""
    margin, gap = 12, 8
    width, height = max(600, available.width() - 2 * margin), max(400, available.height() - 2 * margin)
    sidebar = min(440, max(320, round(width * 0.32)))
    viewer = QtCore.QRect(available.x() + margin, available.y() + margin, width - sidebar - gap, height)
    chat = QtCore.QRect(viewer.right() + gap + 1, viewer.y(), sidebar, height)
    return viewer, chat


def goal_label(goal):
    if not goal:
        return "无"
    kind = goal.get("type")
    if kind == "perform":
        labels = {
            "roll": "前滚翻",
            "nod": "点头",
            "look_around": "左右看",
            "sit_stand": "坐下再站起",
            "ground_pick": "俯身再起",
            "flamingo_left": "抬左脚",
            "flamingo_right": "抬右脚",
            "moonwalk": "后退滑步",
            "body_sway": "轻晃身体",
            "open_beak": "张嘴",
            "close_beak": "闭嘴",
            "groove": "节奏舞步",
            "caught_dancing": "害羞舞步",
        }
        action = goal.get("action", "动作")
        return f"{labels.get(action, action)} × {goal.get('repeat', 1)}"
    if kind == "posture":
        return "坐下保持" if goal.get("pose") == "sitting" else "站起来"
    if kind in ("kick", "navigate"):
        color = next(
            (a.get("value") for a in goal.get("target", {}).get("attributes", []) if a.get("name") == "color"), "指定"
        )
        name = {"red": "红", "yellow": "黄", "blue": "蓝"}.get(color, color)
        return f"踢{name}球" if kind == "kick" else f"走到{name}球旁边"
    direction = {"forward": "前", "backward": "后", "left": "左", "right": "右"}.get(goal.get("direction"), "")
    if kind == "turn":
        return f"向{direction}转 {goal.get('angle_deg', '')}°"
    if kind == "move":
        limit = goal.get("limit", {})
        duration = "持续" if limit.get("kind") == "until_cancelled" else f"{limit.get('seconds', '')} 秒"
        return f"向{direction}走 · {duration}"
    return str(kind or "任务")


class ChatWindow(QtWidgets.QWidget):
    def __init__(self, socket_path, *, smoke=False, smoke_output=None):
        super().__init__()
        self.client = IPCClient(socket_path, self)
        self.client.response.connect(self._response)
        self.client.failure.connect(self._failure)
        self.state = {}
        self.poll_pending = self.closing = self.allow_close = False
        self.message_signature = None
        self.sent_messages = []
        self.accepted_messages = []
        self.smoke, self.smoke_output, self.smoke_result = smoke, smoke_output, None
        self.smoke_started = time.monotonic()
        self.setWindowTitle("小鸭 · 对话")
        self.setMinimumSize(320, 420)
        self.setFont(QtGui.QFont("Noto Sans CJK SC", 10))
        self.setStyleSheet(
            "QWidget { background: #282828; color: #e5e5e5; }"
            "QPlainTextEdit { background: #202020; border: 1px solid #505050; border-radius: 5px; padding: 8px; }"
            "QPushButton { background: #404040; border: 1px solid #606060; border-radius: 4px; padding: 8px; }"
            "QPushButton:hover { background: #505050; } QPushButton:disabled { color: #858585; border-color: #404040; }"
            "QPushButton#stop { background: #633737; } QLabel#muted { color: #b5b5b5; }"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(9)
        title = QtWidgets.QLabel("小鸭仿真 · 对话")
        title.setFont(QtGui.QFont("Noto Sans CJK SC", 14, QtGui.QFont.Bold))
        layout.addWidget(title)
        self.status = QtWidgets.QLabel("正在连接本地仿真…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.task = QtWidgets.QLabel("队列：0 / 15 · 当前：无")
        self.task.setWordWrap(True)
        self.task.setObjectName("muted")
        layout.addWidget(self.task)
        self.transcript = QtWidgets.QPlainTextEdit()
        self.transcript.setReadOnly(True)
        layout.addWidget(self.transcript, 1)
        self.editor = MessageInput()
        self.editor.setFixedHeight(100)
        self.editor.send_requested.connect(self.send_input)
        layout.addWidget(self.editor)
        controls = QtWidgets.QHBoxLayout()
        self.pause = self._button("安全暂停", lambda: self.send_message("暂停"), controls)
        self.resume = self._button("继续", lambda: self.send_message("继续"), controls)
        self.stop = self._button("停止取消", lambda: self.send_message("停"), controls)
        self.stop.setObjectName("stop")
        layout.addLayout(controls)
        bottom = QtWidgets.QHBoxLayout()
        self.reset = self._button("新场景", lambda: self.client.submit(dict(command="reset")), bottom)
        self.send = self._button("发送", self.send_input, bottom)
        layout.addLayout(bottom)
        self.footer = QtWidgets.QLabel("安全暂停会等待完整动作结束并站稳；停止取消会立即中断。")
        self.footer.setObjectName("muted")
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)
        screen = QtWidgets.QApplication.primaryScreen()
        self.available = screen.availableGeometry()
        self.viewer_rect, chat_rect = desktop_layout(self.available)
        self.setGeometry(chat_rect)
        self.scale = screen.devicePixelRatio()
        self.placement = dict(
            command="place_viewer",
            **{
                key: round(value * self.scale)
                for key, value in zip(("x", "y", "width", "height"), self.viewer_rect.getRect(), strict=True)
            },
        )
        self.client.submit(self.placement)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(200)
        self._apply_state({})
        self._poll()
        if smoke:
            QtCore.QTimer.singleShot(800, self._start_smoke)

    def _button(self, text, callback, layout):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _poll(self):
        if not self.poll_pending and not self.closing:
            self.poll_pending = True
            self.client.submit(dict(command="state"))

    def send_input(self):
        text = self.editor.toPlainText().strip()
        if text and self.state.get("ready"):
            self.send_message(text)
            self.editor.clear()

    def send_message(self, text):
        if not self.closing:
            self.sent_messages.append(text)
            self.client.submit(dict(command="message", text=text))

    def _response(self, number, request, result):
        command = request["command"]
        if command == "state":
            self.poll_pending = False
            self._apply_state(result)
        elif command == "shutdown":
            self._finish_close()
        elif result.get("accepted") is False or result.get("error"):
            self.status.setText(str(result.get("error") or "请求未被接受，请稍后重试。"))
        elif command == "message" and result.get("accepted") is True:
            self.accepted_messages.append(request["text"])

    def _failure(self, number, request, message):
        if request["command"] == "state":
            self.poll_pending = False
        if request["command"] == "shutdown":
            self._finish_close()
        else:
            self.status.setText(f"连接暂不可用：{message}")
            if request["command"] == "message" and not self.editor.toPlainText():
                self.editor.setPlainText(request["text"])

    def _apply_state(self, state):
        self.state = state
        ready, loading = bool(state.get("ready")), bool(state.get("loading"))
        execution = state.get("execution") or {}
        phase = execution.get("state", "idle")
        names = {"idle": "空闲", "running": "执行中", "paused": "已安全暂停", "failed": "任务未完成"}
        model = state.get("backend_name") or "本地模型"
        inference = int(state.get("inference_pending") or 0)
        status = (
            "正在加载模型…" if loading else f"{model} · {names.get(phase, phase)}" if ready else "正在连接本地仿真…"
        )
        if inference:
            status += f" · 正在思考 / 解析指令（{inference}），仿真暂时保持当前姿态"
        if state.get("pause_pending") or (state.get("pause_requested") and phase != "paused"):
            status += " · 等待当前动作完成并站稳后暂停"
        if state.get("error"):
            status = f"错误：{state['error']}"
        self.status.setText(status)
        active, pending = execution.get("active"), execution.get("pending") or []
        count = int(active is not None) + len(pending)
        self.task.setText(f"队列：{count} / {state.get('queue_capacity', 15)} · 当前：{goal_label(active)}")
        self.send.setEnabled(ready and not self.closing)
        self.pause.setEnabled(ready and (phase == "running" or inference > 0))
        self.resume.setEnabled(ready and phase in ("running", "paused"))
        self.stop.setEnabled(ready)
        self.reset.setEnabled(ready)
        messages = state.get("messages") or []
        signature = tuple((row.get("role"), str(row.get("text", ""))) for row in messages)
        if signature != self.message_signature:
            bar = self.transcript.verticalScrollBar()
            follow, previous = bar.value() >= bar.maximum() - 5, bar.value()
            names = {"user": "你", "assistant": "小鸭", "execution": "执行结果"}
            self.transcript.setPlainText(
                "\n\n".join(f"{names.get(role, role or '状态')}：\n{text}" for role, text in signature)
            )
            bar.setValue(bar.maximum() if follow else previous)
            self.message_signature = signature

    def closeEvent(self, event):
        if self.allow_close:
            event.accept()
            return
        event.ignore()
        if not self.closing:
            self.closing = True
            self.timer.stop()
            self.status.setText("正在关闭本地仿真…")
            self.client.submit(dict(command="shutdown"))
            QtCore.QTimer.singleShot(1500, self._finish_close)

    def _finish_close(self):
        self.allow_close = True
        self.close()

    def _start_smoke(self):
        if not self.state.get("ready") and time.monotonic() - self.smoke_started < 10:
            QtCore.QTimer.singleShot(200, self._start_smoke)
            return
        self.editor.setFocus()
        self.editor.clear()
        before = len(self.sent_messages)
        QtWidgets.QApplication.sendEvent(self.editor, QtGui.QInputMethodEvent("xiaoya", []))
        self._smoke_enter()
        composition_safe = len(self.sent_messages) == before
        commit = QtGui.QInputMethodEvent("", [])
        commit.setCommitString("小鸭你好")
        QtWidgets.QApplication.sendEvent(self.editor, commit)
        self._smoke_enter()
        self.smoke_result = dict(
            ready=bool(self.state.get("ready")),
            ime_preedit_return_did_not_send=composition_safe,
            ime_commit_return_did_not_send=len(self.sent_messages) == before,
            committed_chinese=self.editor.toPlainText(),
            logical_available_geometry=list(self.available.getRect()),
            logical_chat_geometry=list(self.geometry().getRect()),
            device_pixel_ratio=self.scale,
            requested_viewer_geometry=self.placement,
        )
        QtCore.QTimer.singleShot(180, self._smoke_send)

    def _smoke_enter(self):
        event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Return, QtCore.Qt.NoModifier)
        QtWidgets.QApplication.sendEvent(self.editor, event)

    def _smoke_send(self):
        self._smoke_enter()
        QtCore.QTimer.singleShot(500, self._finish_smoke)

    def _finish_smoke(self):
        result = self.smoke_result
        result.update(
            enter_sent_chinese="小鸭你好" in self.sent_messages,
            ipc_accepted_chinese="小鸭你好" in self.accepted_messages,
            viewer_geometry=self.state.get("viewer_geometry"),
        )
        result["success"] = (
            all(
                result[key]
                for key in (
                    "ready",
                    "ime_preedit_return_did_not_send",
                    "ime_commit_return_did_not_send",
                    "enter_sent_chinese",
                    "ipc_accepted_chinese",
                )
            )
            and result["committed_chinese"] == "小鸭你好"
        )
        if self.smoke_output:
            path = Path(self.smoke_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            screenshot = path.with_suffix(".png")
            result["screenshot_saved"] = self.grab().save(str(screenshot))
            result["screenshot"] = str(screenshot)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    args = parser.parse_args(argv)
    if args.smoke_output and not args.smoke:
        parser.error("--smoke-output requires --smoke; normal desktop windows never close automatically.")
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication([])
    window = ChatWindow(args.socket, smoke=args.smoke, smoke_output=args.smoke_output)
    window.show()
    result = app.exec_()
    window.client.stop()
    if args.smoke and not (window.smoke_result or {}).get("success"):
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
