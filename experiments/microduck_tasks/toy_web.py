"""Serve the local rendered MicroDuck experiment with a Chinese browser chat panel."""

import argparse
import json
import os
import secrets
import signal
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .bootstrap import REPO, configure

STATIC = Path(__file__).with_name("toy_web_static")


def handler_for(app, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, content, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy", "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
            )
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, status, value):
            self.send(
                status,
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def allowed(self, *, write=False):
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            host = self.headers.get("Host", "")
            if host not in hosts:
                self.json(403, dict(error="请从本机地址打开测试页。"))
                return False
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{host}":
                self.json(403, dict(error="请求来源不匹配。"))
                return False
            if write and not secrets.compare_digest(self.headers.get("X-MicroDuck-Session", ""), token):
                self.json(403, dict(error="请刷新本机测试页后重试。"))
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            path = urlparse(self.path).path
            if path == "/api/state":
                self.json(200, app.snapshot())
            elif path in ("/frame/world.jpg", "/frame/front.jpg", "/frame/near.jpg"):
                name = Path(path).stem
                with app.lock:
                    frame = app.frames.get(name)
                self.send(200 if frame else 503, frame or b"", "image/jpeg")
            elif path in ("/", "/app.js", "/style.css"):
                name = "index.html" if path == "/" else path[1:]
                data = (STATIC / name).read_bytes()
                if path == "/":
                    data = data.replace(b"__SESSION_TOKEN__", token.encode())
                mime = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[name]
                self.send(200, data, mime + "; charset=utf-8")
            elif path == "/favicon.ico":
                self.send(204, b"", "image/x-icon")
            else:
                self.json(404, dict(error="页面不存在。"))

        def do_POST(self):
            if not self.allowed(write=True):
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536 or self.headers.get_content_type() != "application/json":
                    raise ValueError("请提交有效的 JSON 请求。")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("请求必须是 JSON 对象。")
                if self.path == "/api/message":
                    message_id = app.submit(value.get("text"))
                    self.json(202, dict(id=message_id))
                elif self.path == "/api/reset":
                    app.reset()
                    self.json(202, dict(ok=True))
                else:
                    self.json(404, dict(error="接口不存在。"))
            except (ValueError, UnicodeDecodeError) as exc:
                self.json(400, dict(error=str(exc)))

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", choices=("q1", "teacher"), default="q1")
    parser.add_argument("--model-dir", type=Path, default=REPO / ".cache/microduck_head/model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "outputs/staged_plan_20261116/q1/train1200/checkpoint_100",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be 1024..65535.")
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("LP_NUM_THREADS", "2")
    dependencies = REPO / ".cache/microduck_head/qlora_deps"
    if args.backend == "q1" and dependencies.is_dir():
        sys.path.insert(0, str(dependencies))
    configure()
    from .toy_web_state import ToyWebState

    def backend_factory():
        if args.backend == "teacher":
            from .toy_teacher import ToyTeacher

            return ToyTeacher()
        from .toy_q1_backend import ToyQ1Backend

        kwargs = {}
        if args.model_dir is not None:
            kwargs["model_dir"] = args.model_dir
        if args.checkpoint is not None:
            kwargs["checkpoint"] = args.checkpoint
        return ToyQ1Backend(**kwargs)

    output = args.output or Path(__file__).parent / "outputs" / f"toy_web_{datetime.now():%Y%m%d_%H%M%S}"
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    app = ToyWebState(output, backend_factory, seed=args.seed)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(app, secrets.token_urlsafe(32)))
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    app.start()
    print(f"MicroDuck 测试页：http://127.0.0.1:{args.port}/", flush=True)
    print(f"记录目录：{output}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
