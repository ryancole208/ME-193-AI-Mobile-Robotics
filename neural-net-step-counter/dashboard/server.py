"""Live dashboard server (brief section 5.1).

    UNO Q  <--poll every 0.5 s / 1 s--  this server  <--every 0.4 s--  browser page

The browser never talks to the UNO Q directly: this server polls it, keeps the
last few minutes in memory, logs everything to data/live_logs/<date>/, and
passes the page's button presses through. That keeps the token off the page
and means a Wi-Fi drop only affects one poller, which simply keeps retrying.

The UNO Q counts on its own; nothing here is needed for counting. If the UNO Q
can't reach the laptop's polls (e.g. different network), set `net_mode: push`:
then the UNO Q POSTs its status/events/stream to /ingest and gets any queued
commands back in the reply.

For debugging without a laptop cable, the page also shows why the UNO Q is
unreachable (refused / timeout / bad name / wrong token), a connection log,
app restarts (uptime going backwards) and the UNO Q's own /api/diag snapshot.

Run with `python stepnet.py dash` (or `python dashboard/server.py`).
`--demo` swaps the UNO Q for dashboard/demo.py's simulated one.
"""
import csv
import logging
import socket
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from unoq import UnoQError, cfg_client  # noqa: E402

STATUS_PERIOD = 0.5
FAST_PERIOD = 1.0
DIAG_PERIOD = 2.0
CONN_LOG_KEEP = 60
STREAM_KEEP_S = 12.0          # page shows 10 s; keep a little extra
EVENT_KEEP = 5000
COMMANDS = {"session/start", "session/stop", "session/truth", "mode", "display", "reset"}
STREAM_KEYS = ("t", "ankle_gz", "hip_vert", "p_right", "p_left")
STATUS_COLS = ["mode", "display", "steps", "steps_sigma", "steps_ci", "dist_m", "dist_sigma_m", "units",
               "cadence_spm", "ankle_link", "ankle_batt_mv", "hip_ok", "model_version", "using", "infer_ms",
               "uptime_s", "session_id", "var_split"]


def classify(err):
    """(kind, what it most likely means) for a failed request, from the error text."""
    s = str(err)
    if "401" in s:
        return "token", "The UNO Q rejected the token: `token` in config.yaml must equal net.token in the UNO Q's setup.yaml."
    if "refused" in s or "10061" in s:
        return "refused", ("The UNO Q is on the network but nothing listens on the port: "
                           "the step-counter app is stopped, crashed, or still starting.")
    if "NameResolution" in s or "getaddrinfo" in s or "Name or service" in s:
        return "dns", "The name in unoq_url doesn't resolve on this network: use the UNO Q's IP address instead."
    if "timed out" in s or "Timeout" in s:
        return "timeout", ("No answer at all: the UNO Q is off, out of Wi-Fi range, or on a different "
                           "network than this laptop.")
    if "10065" in s or "10051" in s or "unreachable" in s.lower():
        return "network", "No route to the UNO Q: check that the laptop and the UNO Q are on the same network."
    return "error", "Unexpected error; see the message."


class CsvLog:
    """Appends rows to data/live_logs/<today>/<name>.csv, writing a header for new files."""

    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()

    def write(self, name, header, row):
        with self.lock:
            d = self.root / datetime.now().strftime("%Y-%m-%d")   # new folder at midnight
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{name}.csv"
            new = not path.exists()
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(header)
                w.writerow(row)


class Hub:
    """Everything the page sees, filled by the poller (or by /ingest in push mode)."""

    def __init__(self, cfg, demo=False):
        self.cfg = cfg
        self.demo = demo
        self.push = cfg.get("net_mode", "poll") == "push" and not demo
        if demo:
            from dashboard.demo import FakeUnoQ
            self.client = FakeUnoQ(cfg)
            self.log = CsvLog(cfg["paths"]["live_logs"] / "demo")   # keep pretend data out of the real logs
        else:
            # (connect, read) timeouts; the loop itself is the retry. Windows takes ~2 s to report
            # "refused", so a shorter connect timeout would mislabel a stopped app as a timeout.
            self.client = cfg_client(cfg, timeout=(3.0, 1.5), retries=0)
            self.log = CsvLog(cfg["paths"]["live_logs"])
        self.url = "demo (simulated UNO Q)" if demo else cfg["unoq_url"]
        self.lock = threading.Lock()
        self.started = time.time()
        self.status = None
        self.connected = False
        self.last_ok = None          # wall time of the last good contact
        self.error = "not contacted yet"
        self.kind, self.hint = "starting", "The dashboard has not tried the UNO Q yet."
        self.fails = 0               # failed status polls in a row
        self.rtt_ms = None
        self.uptime = None           # UNO Q app uptime; going backwards means it restarted
        self.restarts = 0
        self.conn_log = deque(maxlen=CONN_LOG_KEEP)
        self.diag = None             # last /api/diag snapshot
        self.diag_at = None
        self.diag_error = ""
        self.events = {}             # id -> event
        self.last_event_id = 0
        self.stream = {k: deque() for k in STREAM_KEYS}
        self.stream_t = 0.0          # latest UNO Q monotonic time we have stream data for
        self.queue = []              # commands waiting for the next /ingest (push mode)
        self.stop = threading.Event()

    # ------------------------------------------------------------ ingest (shared by poll + push)

    def note(self, kind, text):
        """One line in the page's connection log (and connection.csv)."""
        with self.lock:
            self.conn_log.append({"at": time.time(), "kind": kind, "text": text})
        self.log.write("connection", ["logged_at", "kind", "text"],
                       [datetime.now().isoformat(timespec="milliseconds"), kind, text])

    def take_status(self, st, rtt_ms=None):
        now = time.time()
        up = st.get("uptime_s")
        with self.lock:
            was, down_for = self.connected, (None if self.last_ok is None else now - self.last_ok)
            restarted = up is not None and self.uptime is not None and up < self.uptime - 1
            if restarted:
                self.restarts += 1
            self.uptime = up if up is not None else self.uptime
            self.status, self.connected, self.last_ok, self.error = st, True, now, ""
            self.kind, self.hint, self.fails, self.rtt_ms = "ok", "", 0, rtt_ms
        if restarted:
            self.note("restart", f"UNO Q app restarted (uptime went back to {up:.0f} s)")
        if not was:
            gap = f" after {down_for:.0f} s without contact" if down_for else ""
            self.note("connected", f"reachable{gap}" + (f", {rtt_ms:.0f} ms" if rtt_ms else "")
                      + (f", app uptime {up:.0f} s" if up is not None else ""))
            print(f"[dash] UNO Q reachable ({self.url}); backfilling events since id {self.last_event_id}")
        self.log.write("status", ["logged_at", "reachable", "rtt_ms"] + STATUS_COLS,
                       [datetime.now().isoformat(timespec="milliseconds"), 1,
                        "" if rtt_ms is None else f"{rtt_ms:.0f}"] + [st.get(k, "") for k in STATUS_COLS])

    def take_error(self, err):
        kind, hint = classify(err)
        with self.lock:
            was, old_kind = self.connected, self.kind
            self.connected, self.error, self.kind, self.hint = False, str(err), kind, hint
            self.fails += 1
        if was or kind != old_kind:         # log changes, not every failed poll
            self.note(kind, f"{hint}  ({err})")
        if was:
            print(f"[dash] UNO Q unreachable, retrying: {err}")
        self.log.write("status", ["logged_at", "reachable", "rtt_ms"] + STATUS_COLS,
                       [datetime.now().isoformat(timespec="milliseconds"), 0, ""] + [""] * len(STATUS_COLS))

    def take_events(self, resp):
        """Merge events. Ids are unique, so a backfill after a drop can't double-count;
        a deletion arrives as an id we already have with deleted=true."""
        rows = []
        with self.lock:
            for ev in resp.get("events", []):
                eid = ev.get("id")
                if eid is None:
                    continue
                prev = self.events.get(eid)
                if prev is None:
                    self.events[eid] = dict(ev)
                    rows.append(ev)
                elif ev.get("deleted") and not prev.get("deleted"):
                    prev["deleted"] = True
                    rows.append(ev)
                self.last_event_id = max(self.last_event_id, int(eid))
            if resp.get("last_id") is not None:
                self.last_event_id = max(self.last_event_id, int(resp["last_id"]))
            if len(self.events) > EVENT_KEEP:
                for eid in sorted(self.events)[: len(self.events) - EVENT_KEEP]:
                    del self.events[eid]
        for ev in rows:
            self.log.write("events", ["logged_at", "id", "t", "foot", "conf", "src", "session_id", "deleted"],
                           [datetime.now().isoformat(timespec="milliseconds"), ev.get("id"), ev.get("t"),
                            ev.get("foot"), ev.get("conf"), ev.get("src"), ev.get("session_id"),
                            1 if ev.get("deleted") else 0])

    def take_stream(self, s):
        ts = s.get("t") or []
        if not ts:
            return
        with self.lock:
            if ts[-1] < self.stream_t - 5:      # the UNO Q restarted (its clock went backwards)
                for q in self.stream.values():
                    q.clear()
                self.stream_t = 0.0
            for i, t in enumerate(ts):
                if t <= self.stream_t:
                    continue
                for k in STREAM_KEYS:
                    vals = s.get(k) or []
                    self.stream[k].append(vals[i] if i < len(vals) else None)
            self.stream_t = max(self.stream_t, ts[-1])
            while self.stream["t"] and self.stream["t"][0] < self.stream_t - STREAM_KEEP_S:
                for q in self.stream.values():
                    q.popleft()

    # ------------------------------------------------------------ poller

    def poll_diag(self):
        """The UNO Q's own health snapshot. Optional: a failure here never marks the UNO Q unreachable."""
        try:
            d = self.client.get_json("/api/diag")
            with self.lock:
                self.diag, self.diag_at, self.diag_error = d, time.time(), ""
        except Exception as e:
            with self.lock:
                self.diag_error = ("this UNO Q build has no /api/diag" if "404" in str(e) else str(e))

    def poll_forever(self):
        next_status = next_fast = next_diag = 0.0
        while not self.stop.is_set():
            now = time.monotonic()
            if now >= next_status:
                next_status = now + STATUS_PERIOD
                try:
                    t0 = time.perf_counter()
                    st = self.client.status()
                    self.take_status(st, (time.perf_counter() - t0) * 1000)
                except Exception as e:           # never let the poller die
                    self.take_error(e)
            if now >= next_fast and self.connected:
                next_fast = now + FAST_PERIOD
                try:
                    self.take_events(self.client.events(self.last_event_id))
                    self.take_stream(self.client.stream(self.stream_t))
                except Exception as e:
                    self.take_error(e)
            if now >= next_diag and self.connected:
                next_diag = now + DIAG_PERIOD
                self.poll_diag()
            self.stop.wait(0.05)

    # ------------------------------------------------------------ views for the page

    def state(self):
        with self.lock:
            age = None if self.last_ok is None else time.time() - self.last_ok
            stream = {k: list(q) for k, q in self.stream.items()}
            t_now = self.stream_t
            recent = [e for e in self.events.values() if e.get("t") is not None and e["t"] >= t_now - STREAM_KEEP_S]
            return {"connected": self.connected and (age is not None and age < 3),
                    "error": self.error, "age_s": age,
                    "net_mode": "demo" if self.demo else "push" if self.push else "poll",
                    "unoq_url": self.url, "units_pref": self.cfg.get("units", "ft"),
                    "status": self.status, "stream": stream, "now_t": t_now, "events": recent,
                    "conn": {"kind": self.kind, "hint": self.hint, "fails": self.fails, "rtt_ms": self.rtt_ms,
                             "uptime_s": self.uptime, "restarts": self.restarts,
                             "dash_up_s": time.time() - self.started, "log": list(self.conn_log)[::-1]},
                    "diag": self.diag, "diag_error": self.diag_error,
                    "diag_age_s": None if self.diag_at is None else time.time() - self.diag_at}

    def history(self, minutes=2.0, bin_s=5.0):
        with self.lock:
            evs = [e for e in self.events.values() if not e.get("deleted") and e.get("t") is not None]
            t_now = max([self.stream_t] + [e["t"] for e in evs]) if evs or self.stream_t else 0.0
        n_bins = int(round(minutes * 60 / bin_s))
        t0 = t_now - n_bins * bin_s
        nn, bl = [0] * n_bins, [0] * n_bins
        for e in evs:
            k = int((e["t"] - t0) // bin_s)
            if 0 <= k < n_bins:
                (nn if e.get("src") == "nn" else bl)[k] += 1
        return {"bin_s": bin_s, "t0": t0, "t_now": t_now, "nn": nn, "baseline": bl}


def create_app(cfg, demo=False):
    hub = Hub(cfg, demo)
    app = Flask(__name__, static_folder=None)
    static = HERE / "static"

    @app.get("/")
    def index():
        return send_from_directory(static, "index.html")

    @app.get("/local/state")
    def local_state():
        return jsonify(hub.state())

    @app.get("/local/history")
    def local_history():
        return jsonify(hub.history(float(request.args.get("minutes", 2))))

    @app.post("/local/cmd/<path:cmd>")
    def local_cmd(cmd):
        """Pass-through for every UNO Q command (contract 8.1)."""
        if cmd not in COMMANDS:
            abort(404)
        body = request.get_json(silent=True) or {}
        if hub.push:
            with hub.lock:
                hub.queue.append({"path": "/api/" + cmd, "body": body})
            return jsonify({"queued": True, "note": "push mode: sent with the UNO Q's next /ingest"})
        try:
            res = hub.client.post_json("/api/" + cmd, body)
        except UnoQError as e:
            hub.note("command", f"{cmd} failed: {e}")
            return jsonify({"ok": False, "error": str(e)}), 502
        hub.note("command", f"{cmd} {body or ''} -> {'ok' if res.get('ok', True) else res.get('error', 'failed')}")
        return jsonify(res)

    @app.post("/local/mark")
    def local_mark():
        note = (request.get_json(silent=True) or {}).get("note", "")
        st = hub.status or {}
        hub.log.write("marks", ["logged_at", "unoq_t", "steps", "dist_m", "session_id", "note"],
                      [datetime.now().isoformat(timespec="milliseconds"), hub.stream_t, st.get("steps"),
                       st.get("dist_m"), st.get("session_id"), note])
        return jsonify({"ok": True})

    @app.post("/ingest")
    def ingest():
        """Push mode: the UNO Q sends {status?, events?, stream?} in the 8.1 shapes."""
        if request.headers.get("X-Token") != cfg["token"]:
            abort(401)
        body = request.get_json(silent=True) or {}
        if body.get("status"):
            hub.take_status(body["status"])
        ev = body.get("events")
        if ev:
            hub.take_events(ev if isinstance(ev, dict) else {"events": ev})
        if body.get("stream"):
            hub.take_stream(body["stream"])
        with hub.lock:
            cmds, hub.queue = hub.queue, []
        return jsonify({"commands": cmds, "last_event_id": hub.last_event_id})

    app.hub = hub
    return app


def run(cfg, demo=False, browser=True):
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # don't print every 0.4 s page poll
    app = create_app(cfg, demo)
    hub = app.hub
    if not hub.push:
        threading.Thread(target=hub.poll_forever, daemon=True).start()
    # Push mode must accept connections from the UNO Q, so it can't be localhost-only.
    host = cfg.get("dash_host") or ("0.0.0.0" if hub.push else "127.0.0.1")
    port = int(cfg.get("dash_port", 8080))
    page = f"http://localhost:{port}"
    # Windows can let a second server bind the same port silently; then the browser talks to the old one.
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"[dash] port {port} is already in use, probably by another `stepnet.py dash`. "
                             f"Stop that one (Ctrl+C in its terminal) or set dash_port in config.yaml.")
    print(f"[dash] {page}  ({'push: waiting for /ingest' if hub.push else 'polling ' + hub.url})")
    print(f"[dash] logging to {hub.log.root}")
    print("[dash] the page is only up while this runs; Ctrl+C stops it")
    if browser:
        threading.Timer(1.0, webbrowser.open, [page]).start()
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        hub.stop.set()


if __name__ == "__main__":
    from stepnet import load_config
    run(load_config())
