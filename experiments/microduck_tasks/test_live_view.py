"""Interactive shutdown and terminal decoding regressions without a desktop."""

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from experiments.microduck_tasks.demo import interactive


class _Runtime:
    failure = None

    def __init__(self):
        self.cancel = threading.Event()

    def start_realtime(self):
        pass

    def request_stop(self):
        self.cancel.set()


@pytest.fixture
def terminal(monkeypatch):
    reader, writer = os.pipe()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(encoding="utf-8", fileno=lambda: reader))
    try:
        yield writer
    finally:
        os.close(writer)
        os.close(reader)


def _assert_workers_stopped():
    assert not any(
        thread.is_alive() and thread.name in {"microduck-input", "microduck-task"} for thread in threading.enumerate()
    )


@pytest.mark.parametrize("interrupt", [False, True])
def test_window_exit_cancels_starting_task_and_joins_terminal(monkeypatch, terminal, interrupt):
    """A task clearing cancellation during startup must not outlive the window."""
    runtime = _Runtime()
    entered, release_startup, stopped = (threading.Event() for _ in range(3))
    calls = []

    def run(command):
        assert command == "踢红球"
        entered.set()
        assert release_startup.wait(2)
        time.sleep(0.03)
        runtime.cancel.clear()
        assert runtime.cancel.wait(2), "Window shutdown lost the cancellation during task startup."
        stopped.set()

    class Window:
        def __init__(self, session, *, on_stop):
            self.deadline = time.monotonic() + 3

        def pump(self):
            assert time.monotonic() < self.deadline, "Terminal input did not start a task."
            if not entered.is_set():
                return True
            runtime.request_stop()
            release_startup.set()
            if interrupt:
                raise KeyboardInterrupt
            return False

        def close(self):
            assert stopped.is_set(), "The window closed before the task stopped."
            assert threading.current_thread() is threading.main_thread()
            calls.append("closed")

    monkeypatch.setitem(sys.modules, "experiments.microduck_tasks.live_view", SimpleNamespace(LiveView=Window))
    os.write(terminal, "踢红球\n".encode())
    session = SimpleNamespace(runtime=runtime, run=run)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            interactive(session)
    else:
        interactive(session)
    assert calls == ["closed"]
    _assert_workers_stopped()


def test_headless_terminal_decodes_chinese_across_read_boundaries(monkeypatch, terminal):
    runtime = _Runtime()
    received = []
    completed = threading.Event()
    original_read = os.read

    def read_one_byte(fd, count):
        return original_read(fd, min(count, 1))

    monkeypatch.setattr(os, "read", read_one_byte)

    def run(command):
        received.append(command)
        completed.set()

    def write_commands():
        os.write(terminal, "你现在看到了什么？\n".encode())
        completed.wait(2)
        os.write(terminal, b"exit\n")

    writer = threading.Thread(target=write_commands)
    writer.start()
    try:
        interactive(SimpleNamespace(runtime=runtime, run=run), viewer=False)
    finally:
        writer.join(timeout=3)
    assert not writer.is_alive()
    assert received == ["你现在看到了什么？"]
    _assert_workers_stopped()


@pytest.mark.parametrize(
    "metadata,title",
    [
        ({"backend": "rgbd_rule_teacher"}, "RGB-D rule teacher"),
        ({"backend": "local_pretrained_vlm", "robot_finetuned": False}, "Local pretrained VLM"),
        (
            {"backend": "qwen35-frozen-skill-head", "robot_finetuned": False, "head_trained": True},
            "Frozen VLM + trained skill head",
        ),
        (
            {
                "backend": "local_finetuned_vlm",
                "robot_finetuned": True,
                "server_metadata": {"adapter": "/models/final_adapter", "robot_finetuned": True},
            },
            "QLoRA VLM",
        ),
    ],
)
def test_live_and_replay_frames_display_actual_controller_during_recovery(monkeypatch, tmp_path, metadata, title):
    import numpy as np
    from PIL import ImageDraw

    from experiments.microduck_tasks.recording import _compose, _font, _replay_metadata

    written = []
    original = ImageDraw.ImageDraw.text

    def capture(draw, position, text, *args, **kwargs):
        written.append(text)
        return original(draw, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    views = [np.zeros((height, width, 3), dtype=np.uint8) for height, width in ((720, 960), (240, 320), (240, 320))]
    event = {"instruction": "踢红球", "controller_source": "executor"}
    _compose(views, 1, "stand", 0, event, _font(20), live=True, model_metadata=metadata)
    assert title in written
    if title != "RGB-D rule teacher":
        assert "RGB-D rule teacher" not in written
        assert "Local inference / simulation only" in written
    if title == "QLoRA VLM":
        assert "Adapter: final_adapter" in written
    (tmp_path / "episode.json").write_text(json.dumps({"metadata": {"backend": "rgbd_rule_teacher"}}))
    (tmp_path / "summary.json").write_text(json.dumps({"metadata": {"backend": metadata["backend"], "vlm": metadata}}))
    written.clear()
    _compose(views, 1, "stand", 0, event, _font(20), model_metadata=_replay_metadata(tmp_path))
    assert title in written
