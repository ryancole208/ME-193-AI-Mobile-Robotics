"""HTTP API for the laptop (contract 9.1). Stdlib http.server in a background thread.

Every request needs the header X-Token: <net.token from setup.yaml>, else 401.

Threading: the API thread never touches the counting state directly. It
  * reads snapshots the main loop publishes (status, stream) under a lock,
  * reads the thread-safe EventLog,
  * reads files from disk,
  * and sends commands (start/stop/mode/reset/...) to the main loop through a
    queue, waiting up to 5 s for the reply.
So a slow laptop or a big download can never stall step counting.

GET  /api/status                      GET  /api/events?since=<id>
GET  /api/stream?since=<t>            GET  /api/files
GET  /api/files/<session_id>/<file>   GET  /api/diag   (extra: the diag snapshot)
POST /api/session/start | /api/session/stop | /api/session/truth
POST /api/mode | /api/display | /api/reset
POST /api/model   (body: model.npz bytes; checked before it replaces the active model)
"""

from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from recorder import list_sessions
from settings import SESSIONS_DIR

log = logging.getLogger("step.api")

MAX_MODEL_BYTES = 50 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _clean(o):
    """NaN/inf are not valid JSON: send null."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class Shared:
    """What the main loop publishes for the API thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.status: dict = {}
        self.stream: dict = {"t": [], "ankle_gz": [], "hip_vert": [], "p_right": [], "p_left": []}
        self.diag: dict = {}
        self.current_session = None
        self.commands: queue.Queue = queue.Queue()

    def command(self, name: str, payload=None, timeout: float = 5.0) -> dict:
        reply: queue.Queue = queue.Queue(maxsize=1)
        self.commands.put((name, payload, reply))
        try:
            return reply.get(timeout=timeout)
        except queue.Empty:
            return {"ok": False, "error": "the counting loop did not answer within 5 s"}


class ApiServer:
    def __init__(self, port: int, token: str, shared: Shared, evlog, model_check):
        self.port, self.token, self.shared, self.evlog = port, token, shared, evlog
        self.model_check = model_check     # callable(bytes) -> dict (runs in the API thread)
        self.bound = False
        self.error = ""
        self.requests = 0
        self.httpd = None

    def start(self) -> bool:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):     # keep the app log readable
                pass

            def _send(self, code: int, obj=None, body: bytes | None = None, ctype="application/json"):
                if body is None:
                    body = json.dumps(_clean(obj)).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _auth(self) -> bool:
                server.requests += 1
                if self.headers.get("X-Token") != server.token:
                    self._send(401, {"ok": False, "error": "missing or wrong X-Token header"})
                    return False
                return True

            def _body(self, limit=1024 * 1024) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                if n > limit:
                    raise ValueError(f"body of {n} bytes is larger than {limit}")
                return self.rfile.read(n) if n else b""

            def _json(self) -> dict:
                b = self._body()
                return json.loads(b) if b else {}

            def do_GET(self):
                if not self._auth():
                    return
                try:
                    server.get(self, urlparse(self.path))
                except Exception as e:
                    log.error(f"While serving GET {self.path}: {type(e).__name__}: {e}")
                    self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

            def do_POST(self):
                if not self._auth():
                    return
                try:
                    server.post(self, urlparse(self.path))
                except (ValueError, json.JSONDecodeError) as e:
                    self._send(400, {"ok": False, "error": str(e)})
                except Exception as e:
                    log.error(f"While serving POST {self.path}: {type(e).__name__}: {e}")
                    self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        try:
            self.httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
            self.httpd.daemon_threads = True
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"
            log.error(f"While binding the HTTP API to port {self.port}: {self.error}. Likely fix: another "
                      "process uses the port; change net.port in setup.yaml AND ports: in app.yaml")
            return False
        threading.Thread(target=self.httpd.serve_forever, name="api", daemon=True).start()
        self.bound = True
        log.info(f"HTTP API listening on port {self.port}")
        return True

    # ------------------------------------------------------------ GET
    def get(self, h, url) -> None:
        q = parse_qs(url.query)
        path = url.path.rstrip("/")
        sh = self.shared
        if path == "/api/status":
            with sh.lock:
                h._send(200, dict(sh.status))
        elif path == "/api/events":
            since = int(q.get("since", ["0"])[0] or 0)
            h._send(200, self.evlog.since(since))
        elif path == "/api/stream":
            since = float(q.get("since", ["-1e18"])[0] or -1e18)
            with sh.lock:
                s = sh.stream
                keep = [i for i, t in enumerate(s["t"]) if t > since]
                h._send(200, {k: [v[i] for i in keep] for k, v in s.items()})
        elif path == "/api/diag":
            with sh.lock:
                h._send(200, dict(sh.diag))
        elif path == "/api/files":
            with sh.lock:
                cur = sh.current_session
            h._send(200, {"sessions": list_sessions(cur)})
        elif path.startswith("/api/files/"):
            parts = path.split("/")[3:]
            if len(parts) != 2 or not all(SAFE_NAME.match(p) for p in parts):
                h._send(400, {"ok": False, "error": "use /api/files/<session_id>/<file>"})
                return
            f = SESSIONS_DIR / parts[0] / parts[1]
            if not f.is_file():
                h._send(404, {"ok": False, "error": f"{parts[0]}/{parts[1]} not found"})
                return
            h._send(200, body=f.read_bytes(), ctype="application/octet-stream")
        else:
            h._send(404, {"ok": False, "error": f"unknown endpoint {path}"})

    # ------------------------------------------------------------ POST
    def post(self, h, url) -> None:
        path = url.path.rstrip("/")
        sh = self.shared
        if path == "/api/model":
            body = h._body(MAX_MODEL_BYTES)
            h._send(200, self.model_check(body))
            return
        simple = {"/api/session/start": "session_start", "/api/session/stop": "session_stop",
                  "/api/reset": "reset"}
        if path in simple:
            h._send(200, sh.command(simple[path]))
        elif path == "/api/session/truth":
            p = h._json()
            if not isinstance(p.get("steps"), int) or p["steps"] < 0:
                raise ValueError('"steps" must be a whole number >= 0')
            if p.get("distance_in") is not None and not isinstance(p["distance_in"], (int, float)):
                raise ValueError('"distance_in" must be a number or null')
            h._send(200, sh.command("truth", p))
        elif path == "/api/mode":
            p = h._json()
            if p.get("mode") not in ("live", "record"):
                raise ValueError('"mode" must be "live" or "record"')
            h._send(200, sh.command("mode", p["mode"]))
        elif path == "/api/display":
            p = h._json()
            if p.get("show") not in ("steps", "distance"):
                raise ValueError('"show" must be "steps" or "distance"')
            h._send(200, sh.command("display", p["show"]))
        else:
            h._send(404, {"ok": False, "error": f"unknown endpoint {path}"})
