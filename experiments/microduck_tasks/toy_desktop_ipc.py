"""Owner-only local IPC between the physics/model process and native Qt sidebar."""

import json
import os
import socketserver


class DesktopServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path, app):
        self.app = app
        super().__init__(str(path), DesktopHandler)
        os.chmod(path, 0o600)


class DesktopHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(1)
        try:
            line = self.rfile.readline(65537)
            if len(line) > 65536 or not line.endswith(b"\n"):
                raise ValueError("请求过长或不完整。")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("请求必须是 JSON 对象。")
            app, command = self.server.app, request.get("command")
            if command == "state":
                response = app.snapshot()
            elif command == "message":
                app.submit(request.get("text"))
                response = dict(accepted=True)
            elif command == "reset":
                app.reset()
                response = dict(accepted=True)
            elif command == "place_viewer":
                app.place_viewer(*(request.get(key) for key in ("x", "y", "width", "height")))
                response = dict(accepted=True)
            elif command == "shutdown":
                app.request_shutdown()
                response = dict(accepted=True)
            else:
                raise ValueError("未知桌面命令。")
        except Exception as exc:
            response = dict(error=str(exc))
        try:
            self.wfile.write(json.dumps(response, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
