"""Browser admission must preserve Chinese and never submit malformed controls."""

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from .toy_web import handler_for
from .toy_web_state import ToyWebState


@pytest.fixture
def endpoint():
    received = []
    app = SimpleNamespace(
        lock=threading.RLock(),
        frames={"world": b"jpeg-data"},
        snapshot=lambda: dict(phase="ready", messages=[]),
        submit=lambda text: received.append(text) or "message-id",
        reset=lambda: received.append("reset"),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app, "test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], received
    server.shutdown()
    thread.join(2)
    server.server_close()


def request(port, path, *, body=None, headers=None):
    conn = HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        conn.request("GET" if body is None else "POST", path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        conn.close()


def test_page_and_frames_and_utf8_command_roundtrip(endpoint):
    port, received = endpoint
    status, page, headers = request(port, "/")
    assert status == 200 and "中文" in page.decode()
    assert "test-token" in page.decode() and "__SESSION_TOKEN__" not in page.decode()
    assert headers["Cache-Control"] == "no-store"
    assert request(port, "/frame/world.jpg")[:2] == (200, b"jpeg-data")
    text = "先点头，再向左转九十度"
    status, body, _ = request(
        port,
        "/api/message",
        body=json.dumps(dict(text=text), ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "X-MicroDuck-Session": "test-token"},
    )
    assert status == 202 and json.loads(body)["id"] == "message-id"
    assert received == [text]


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"X-MicroDuck-Session": "wrong"},
        {"X-MicroDuck-Session": "test-token", "Origin": "https://other.example"},
        {"X-MicroDuck-Session": "test-token", "Host": "other.example"},
    ],
)
def test_external_or_missing_session_requests_never_reach_robot(endpoint, extra):
    port, received = endpoint
    status, _, _ = request(
        port,
        "/api/message",
        body=b'{"text":"pause"}',
        headers={"Content-Type": "application/json", **extra},
    )
    assert status == 403 and received == []


@pytest.mark.parametrize("text", [None, "", "   ", "中" * 1001, 123])
def test_invalid_input_does_not_reach_dispatcher_or_break_later_input(tmp_path, text):
    app = ToyWebState(tmp_path, lambda: None)
    app.phase = "ready"
    completed = threading.Event()
    completed.set()
    accepted = []
    app.inputs = SimpleNamespace(submit=lambda value: (accepted.append(value) or completed, []))
    with pytest.raises(ValueError, match="1000"):
        app.submit(text)
    assert app.messages == [] and accepted == []
    app.submit("点头")
    for thread in app.threads:
        thread.join(2)
    assert accepted == ["点头"]


def test_close_during_model_loading_does_not_create_late_world(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def factory():
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(name="loaded")

    app = ToyWebState(tmp_path, factory)
    app._new_world = lambda: pytest.fail("Closed service must not create a new world.")
    thread = threading.Thread(target=app._initialize)
    thread.start()
    assert entered.wait(2)
    app.close()
    release.set()
    thread.join(2)
    assert not thread.is_alive() and app.session is None
