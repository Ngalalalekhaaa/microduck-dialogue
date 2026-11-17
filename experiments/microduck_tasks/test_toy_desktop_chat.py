"""System-Python Qt checks for Chinese IME, asynchronous local IPC, and desktop layout."""

import copy
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from .toy_desktop_chat import ChatWindow, MessageInput, desktop_layout, exchange


class Server:
    def __init__(self, path):
        self.path, self.requests, self.delay = path, [], {}
        self.state = dict(
            ready=True,
            loading=False,
            error=None,
            backend_name="Q1 实测",
            output="test-output",
            viewer_running=True,
            viewer_geometry=None,
            messages=[],
            queue_capacity=15,
            inference_pending=0,
            execution=dict(state="idle", active=None, pending=[]),
            sim=dict(time_s=2.0, display_frame_id=1, skill="stand", stationary=True, tilt_deg=0.5),
        )
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(path))
        self.socket.listen(8)
        self.socket.settimeout(0.1)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped.is_set():
            try:
                connection, _ = self.socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection:
                try:
                    received = bytearray()
                    while b"\n" not in received:
                        received.extend(connection.recv(65536))
                    request = json.loads(received.partition(b"\n")[0])
                    self.requests.append(request)
                    command = request["command"]
                    time.sleep(self.delay.get(command, 0))
                    if command == "message":
                        self.state["messages"].append(dict(role="user", text=request["text"]))
                    elif command == "place_viewer":
                        self.state["viewer_geometry"] = [request[key] for key in ("x", "y", "width", "height")]
                    response = copy.deepcopy(self.state) if command == "state" else dict(accepted=True)
                    connection.sendall(json.dumps(response, ensure_ascii=False).encode() + b"\n")
                except OSError:
                    pass

    def close(self):
        self.stopped.set()
        self.socket.close()
        self.thread.join(2)


@pytest.fixture(scope="module")
def app():
    value = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    value.setQuitOnLastWindowClosed(False)
    return value


@pytest.fixture
def server():
    with tempfile.TemporaryDirectory(prefix="duck-qt-") as directory:
        value = Server(Path(directory) / "ipc.sock")
        yield value
        value.close()


def spin(app, condition, timeout=2):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "Qt operation did not finish within the test budget."
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


@pytest.fixture
def window(app, server):
    value = ChatWindow(server.path)
    value.show()
    spin(app, lambda: value.state.get("ready"))
    yield value
    value.timer.stop()
    value.allow_close = True
    value.close()
    value.client.stop()
    app.processEvents()


def enter(widget, modifiers=QtCore.Qt.NoModifier):
    QtWidgets.QApplication.sendEvent(widget, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Return, modifiers))


def test_chinese_preedit_and_commit_enter_do_not_send_until_a_new_return(app):
    editor, sent = MessageInput(), []
    editor.send_requested.connect(lambda: sent.append(editor.toPlainText()))
    QtWidgets.QApplication.sendEvent(editor, QtGui.QInputMethodEvent("xiaoya", []))
    enter(editor)
    assert sent == []
    commit = QtGui.QInputMethodEvent("", [])
    commit.setCommitString("小鸭先点头，再向左转九十度")
    QtWidgets.QApplication.sendEvent(editor, commit)
    enter(editor)
    assert sent == [] and editor.toPlainText() == "小鸭先点头，再向左转九十度"
    QtTest.QTest.qWait(150)
    enter(editor)
    assert sent == ["小鸭先点头，再向左转九十度"]


def test_shift_enter_is_a_newline_and_plain_enter_sends(app):
    editor, sent = MessageInput(), []
    editor.send_requested.connect(lambda: sent.append(editor.toPlainText()))
    editor.setPlainText("第一行")
    editor.moveCursor(QtGui.QTextCursor.End)
    enter(editor, QtCore.Qt.ShiftModifier)
    editor.insertPlainText("第二行")
    assert sent == [] and editor.toPlainText() == "第一行\n第二行"
    enter(editor)
    assert sent == ["第一行\n第二行"]


@pytest.mark.parametrize("screen", [(0, 0, 800, 600), (0, 25, 1920, 1055), (2560, 0, 1920, 1080)])
def test_sidebar_is_right_of_viewer_and_fits_available_logical_screen(screen):
    available = QtCore.QRect(*screen)
    viewer, chat = desktop_layout(available)
    assert available.contains(viewer) and available.contains(chat)
    assert chat.x() - viewer.right() == 9
    assert viewer.y() == chat.y() and viewer.height() == chat.height()
    assert 320 <= chat.width() <= 440 and viewer.width() >= 400


def test_utf8_ipc_is_single_request_single_response(server):
    assert exchange(server.path, dict(command="message", text="踢蓝球")) == dict(accepted=True)
    state = exchange(server.path, dict(command="state"))
    assert state["messages"] == [dict(role="user", text="踢蓝球")]


def test_controls_send_exact_distinct_commands_and_reset_uses_its_own_message(app, server, window):
    window._apply_state(dict(server.state, execution=dict(state="running", active={"type": "turn"}, pending=[])))
    for button in (window.pause, window.resume, window.stop, window.reset):
        button.click()
    spin(app, lambda: len([row for row in server.requests if row["command"] in ("message", "reset")]) == 4)
    actions = [row for row in server.requests if row["command"] in ("message", "reset")]
    assert actions == [
        dict(command="message", text="暂停"),
        dict(command="message", text="继续"),
        dict(command="message", text="停"),
        dict(command="reset"),
    ]


def test_state_uses_plain_chinese_text_and_does_not_claim_pending_pause_is_complete(server, window):
    state = copy.deepcopy(server.state)
    state.update(inference_pending=1, pause_requested=True)
    state["messages"] = [dict(role="execution", text="前滚翻已完成 <b>真实结果</b>")]
    state["execution"] = dict(state="running", active=dict(type="perform", action="roll", repeat=1), pending=[{}])
    window._apply_state(state)
    assert "正在思考" in window.status.text() and "保持当前姿态" in window.status.text()
    assert "等待当前动作完成并站稳" in window.status.text() and "已安全暂停" not in window.status.text()
    assert "2 / 15" in window.task.text() and "前滚翻" in window.task.text()
    assert "<b>真实结果</b>" in window.transcript.toPlainText()
    state["execution"]["state"] = "paused"
    window._apply_state(state)
    assert "已安全暂停" in window.status.text() and "等待当前动作" not in window.status.text()


def test_slow_ipc_never_blocks_gui_timer_or_input(app, server, window):
    window.timer.stop()
    spin(app, lambda: not window.poll_pending)
    server.delay["state"] = 0.9
    ticks = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(15)
    window._poll()
    editor = window.editor
    editor.setPlainText("仍然可以输入汉字")
    QtTest.QTest.qWait(300)
    assert len(ticks) >= 10 and editor.toPlainText() == "仍然可以输入汉字"
    spin(app, lambda: not window.poll_pending, timeout=1)
    assert "连接暂不可用" in window.status.text()
    timer.stop()


def test_closing_sends_shutdown_asynchronously(app, server, window):
    started = time.monotonic()
    window.close()
    assert time.monotonic() - started < 0.1
    spin(app, lambda: window.allow_close)
    assert any(row["command"] == "shutdown" for row in server.requests)


def test_smoke_records_ime_layout_and_widget_screenshot(app, server, tmp_path):
    output = tmp_path / "smoke.json"
    window = ChatWindow(server.path, smoke=True, smoke_output=output)
    window.show()
    try:
        spin(app, lambda: window.allow_close, timeout=5)
        result = json.loads(output.read_text())
        assert result["success"] and result["screenshot_saved"]
        assert result["committed_chinese"] == "小鸭你好" and result["ipc_accepted_chinese"]
        assert output.with_suffix(".png").is_file()
        assert result["logical_chat_geometry"][2] <= 440
        assert any(row["command"] == "place_viewer" for row in server.requests)
    finally:
        window.allow_close = True
        window.close()
        window.client.stop()


def test_standalone_system_python_smoke_cli_exits_after_shutdown(server, tmp_path):
    output = tmp_path / "standalone_smoke.json"
    completed = subprocess.run(
        [
            "/usr/bin/python3",
            str(Path(__file__).with_name("toy_desktop_chat.py")),
            "--socket",
            str(server.path),
            "--smoke",
            "--smoke-output",
            str(output),
        ],
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["success"]
    assert any(row == dict(command="message", text="小鸭你好") for row in server.requests)
    assert any(row["command"] == "shutdown" for row in server.requests)
