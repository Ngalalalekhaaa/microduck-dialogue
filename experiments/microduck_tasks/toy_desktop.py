"""Open the native MuJoCo viewer with a Chinese Qt chat window on its right."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from .bootstrap import REPO, configure


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("onnx", "q1", "q1r", "teacher"), default="q1r", help="默认：Q1R PyTorch")
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path(__file__).parent / "outputs/q1r_onnx_20261117/onnx_int8_fp16_head",
    )
    parser.add_argument("--model-dir", type=Path, default=REPO / ".cache/microduck_head/model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qt-python", default="/usr/bin/python3")
    args = parser.parse_args(argv)
    if args.backend == "onnx" and args.checkpoint is not None:
        parser.error("ONNX 模型已经合并检查点，请用 --onnx-model 指定导出目录。")
    if args.checkpoint is None and args.backend in ("q1", "q1r"):
        if args.backend == "q1r":
            selected = Path(__file__).parent / "outputs/q1r_20261116/selected_model.json"
            if not selected.exists():
                parser.error("Q1R 仍在训练或验证，请等待 selected_model.json 生成后再测试。")
            args.checkpoint = Path(json.loads(selected.read_text())["checkpoint"])
        else:
            args.checkpoint = Path(__file__).parent / "outputs/staged_plan_20261116/q1/train1200/checkpoint_100"
    return args


def make_backend(args):
    if args.backend == "teacher":
        from .toy_teacher import ToyTeacher

        return ToyTeacher()
    if args.backend == "onnx":
        from .q1r_onnx_backend import ToyQ1ROnnxBackend

        return ToyQ1ROnnxBackend(model_dir=args.onnx_model)
    if args.backend == "q1r":
        from .q1r_backend import ToyQ1RBackend

        return ToyQ1RBackend(model_dir=args.model_dir, checkpoint=args.checkpoint)
    from .toy_q1_backend import ToyQ1Backend

    return ToyQ1Backend(model_dir=args.model_dir, checkpoint=args.checkpoint)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = arguments(argv)
    if not os.environ.get("DISPLAY"):
        raise SystemExit("请在有图形桌面的本机终端运行（需要 DISPLAY）。")
    if args.backend == "onnx":
        from .onnx_bootstrap import ensure_desktop_environment

        ensure_desktop_environment(argv)
    subprocess.run([args.qt_python, "-c", "from PyQt5 import QtWidgets, QtGui, QtCore"], check=True)
    os.environ["MUJOCO_GL"] = "egl"
    vendor = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if vendor.is_file():
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)
        os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
    dependencies = REPO / ".cache/microduck_head/qlora_deps"
    if args.backend in ("q1", "q1r") and dependencies.is_dir():
        sys.path.insert(0, str(dependencies))
    configure()
    from .toy_desktop_ipc import DesktopServer
    from .toy_desktop_state import ToyDesktopState

    output = args.output or Path(__file__).parent / "outputs" / f"toy_desktop_{datetime.now():%Y%m%d_%H%M%S}"
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    app = ToyDesktopState(output, lambda: make_backend(args), seed=args.seed)
    signal.signal(signal.SIGTERM, lambda *_: app.request_shutdown())
    with tempfile.TemporaryDirectory(prefix="microduck-desktop-") as temporary:
        socket_path = Path(temporary) / "chat.sock"
        server = DesktopServer(socket_path, app)
        server_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        server_thread.start()
        child = None
        try:
            child = subprocess.Popen(
                [args.qt_python, str(Path(__file__).with_name("toy_desktop_chat.py")), "--socket", str(socket_path)]
            )
            (output / "desktop.json").write_text(
                json.dumps(
                    dict(
                        pid=os.getpid(),
                        qt_pid=child.pid,
                        socket=str(socket_path),
                        output=str(output),
                        backend=args.backend,
                        onnx_model=str(args.onnx_model.resolve()) if args.backend == "onnx" else None,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"MicroDuck 原生仿真与中文对话框；记录目录：{output}", flush=True)
            app.start()
            while not app.shutdown.wait(0.2):
                if child.poll() is not None:
                    break
                with app.lock:
                    if app.phase == "ready" and app.viewer and not app.viewer.is_running():
                        break
        except KeyboardInterrupt:
            pass
        finally:
            app.request_shutdown()
            server.shutdown()
            server.server_close()
            try:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(2)
            finally:
                app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
