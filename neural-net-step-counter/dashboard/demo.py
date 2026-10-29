"""A pretend UNO Q for `python stepnet.py dash --demo`.

Same methods as unoq.UnoQ (status / events / stream / get_json / post_json) and
the same JSON shapes as the real API, so the dashboard runs unchanged with no
hardware. Everything is a function of the time since start:

    0-40 s of every 50 s   walking, 1.1 s stride (about 109 steps/min)
    40-50 s                standing still
    every 120 s, 70-76 s   the ankle link drops ("lost"), counting carries on
    every 180 s, 100-108 s the "Wi-Fi" drops: requests time out, so the page
                           shows the unreachable banner and then backfills

Buttons work too: mode, display, session start/stop, truth and reset.
"""
import math
import random
import threading
import time
from datetime import datetime

from unoq import UnoQError

STRIDE_S = 1.1
WALK_S, CYCLE_S = 40.0, 50.0
LATENCY_S = 0.4          # an event is published this long after its heel strike
STREAM_HZ = 25
L0_M = 0.7303


def _walking(t):
    return (t % CYCLE_S) < WALK_S


def _strikes(t_from, t_to):
    """(t, foot) heel strikes in [t_from, t_to): right at 0.5 s + n*stride into each walk, left half a stride later."""
    out = []
    c = math.floor(t_from / CYCLE_S) * CYCLE_S
    while c < t_to:
        for n in range(int(WALK_S / STRIDE_S)):
            for foot, off in (("R", 0.5), ("L", 0.5 + STRIDE_S / 2)):
                ts = c + off + n * STRIDE_S
                if t_from <= ts < t_to and ts < c + WALK_S:
                    out.append((ts, foot))
        c += CYCLE_S
    return sorted(out)


def _bump(x, mu, w):
    return math.exp(-((x - mu) / w) ** 2)


class FakeUnoQ:
    def __init__(self, cfg):
        self.url = "demo"
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.rng = random.Random(1)
        self.mode, self.display, self.session_id = "live", "steps", None
        self.next_id, self.changes = 1, []     # (id, event dict), like the UNO Q's change log
        self.done_t = 0.0                      # strikes up to here are already published
        self.count_from = 0.0                  # reset moves this forward

    def now(self):
        return time.monotonic() - self.t0

    def _check_wifi(self):
        if 100 <= self.now() % 180 < 108:
            raise UnoQError("UNO Q unreachable at demo: Read timed out (simulated Wi-Fi drop)")

    def _advance(self):
        """Publish every strike older than LATENCY_S as an NN event plus (usually) a baseline event."""
        t_to = self.now() - LATENCY_S
        if t_to <= self.done_t:
            return
        for ts, foot in _strikes(self.done_t, t_to):
            ev = {"id": self.next_id, "t": round(ts, 3), "foot": foot, "conf": round(self.rng.uniform(0.8, 0.99), 4),
                  "src": "nn", "session_id": self.session_id, "deleted": False}
            self.changes.append((self.next_id, ev))
            self.next_id += 1
            if self.rng.random() > 0.04:      # the baseline misses the odd step
                bl = dict(ev, id=self.next_id, t=round(ts + self.rng.uniform(-0.03, 0.03), 3), conf=1.0, src="baseline")
                self.changes.append((self.next_id, bl))
                self.next_id += 1
        self.done_t = t_to

    def _nn_events(self):
        return [e for _, e in self.changes if e["src"] == "nn" and not e["deleted"] and e["t"] >= self.count_from]

    # ------------------------------------------------------------ the API

    def status(self, **kw):
        self._check_wifi()
        with self.lock:
            self._advance()
            t = self.now()
            evs = self._nn_events()
            n = len(evs)
            v_event = sum(e["conf"] * (1 - e["conf"]) for e in evs)
            v_model, v_sys = 0.02 * n, (0.01 * n) ** 2
            sig = math.sqrt(v_event + v_model + v_sys)
            lost = 70 <= t % 120 < 76
            return {"mode": self.mode, "display": self.display, "steps": n, "steps_sigma": round(sig, 3),
                    "steps_ci": [max(0, round(n - 1.96 * sig)), round(n + 1.96 * sig)],
                    "dist_m": round(n * L0_M, 3), "dist_sigma_m": round(0.04 * n * L0_M + 0.1, 3), "units": "ft",
                    "cadence_spm": round(6.0 * sum(1 for e in evs if e["t"] >= t - 10), 1),
                    "var_split": {"event": round(v_event, 3), "model": round(v_model, 3), "systematic": round(v_sys, 3)},
                    "ankle_link": "lost" if lost else "ok", "ankle_batt_mv": int(2750 - t / 60),
                    "hip_ok": True, "model_version": "demo-0001", "using": "nn",
                    "infer_ms": round(4.0 + self.rng.random(), 1), "uptime_s": round(t, 1),
                    "session_id": self.session_id}

    def events(self, since, **kw):
        self._check_wifi()
        with self.lock:
            self._advance()
            out = [e for i, e in self.changes if i > since][:5000]
            return {"events": out, "last_id": out[-1]["id"] if out else since}

    def stream(self, since, **kw):
        self._check_wifi()
        t_now = self.now()
        k1 = int(t_now * STREAM_HZ)
        k0 = max(k1 - 10 * STREAM_HZ, int(since * STREAM_HZ) + 1)
        strikes = _strikes(k0 / STREAM_HZ - 2, k1 / STREAM_HZ + 2)
        right = [ts for ts, f in strikes if f == "R"]
        s = {"t": [], "ankle_gz": [], "hip_vert": [], "p_right": [], "p_left": []}
        for k in range(k0, k1 + 1):
            t = k / STREAM_HZ
            noise = self.rng.gauss(0, 1)
            if _walking(t) and right:
                last_r = max([r for r in right if r <= t], default=right[0] - STRIDE_S)
                ph = ((t - last_r) / STRIDE_S) % 1.0
                gz = 320 * _bump(ph, 0.72, 0.09) - 130 * _bump(ph, 0.03, 0.04) - 70 * _bump(ph, 0.93, 0.04) + 4 * noise
                hip = 1.0 + 0.22 * math.cos(4 * math.pi * ph) + 0.015 * noise
            else:
                gz, hip = 3 * noise, 1.0 + 0.008 * noise
            pr = min(1.0, sum(0.95 * _bump(t, ts, 0.05) for ts, f in strikes if f == "R"))
            pl = min(1.0, sum(0.93 * _bump(t, ts, 0.05) for ts, f in strikes if f == "L"))
            for key, v in (("t", round(t, 2)), ("ankle_gz", round(gz, 1)), ("hip_vert", round(hip, 4)),
                           ("p_right", round(pr, 4)), ("p_left", round(pl, 4))):
                s[key].append(v)
        return s

    def get_json(self, path, params=None, **kw):
        self._check_wifi()
        if path == "/api/diag":
            return self._diag()
        raise UnoQError(f"GET {path} -> HTTP 404 (not simulated in demo mode)")

    def post_json(self, path, body=None, **kw):
        self._check_wifi()
        body = body or {}
        with self.lock:
            if path == "/api/mode":
                self.mode = body.get("mode", self.mode)
                return {"ok": True, "mode": self.mode}
            if path == "/api/display":
                self.display = body.get("show", self.display)
                return {"ok": True, "show": self.display}
            if path == "/api/session/start":
                self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                return {"ok": True, "session_id": self.session_id}
            if path == "/api/session/stop":
                sid, self.session_id = self.session_id, None
                return {"ok": True, "session_id": sid}
            if path == "/api/session/truth":
                return {"ok": True, "note": "demo mode: nothing was saved"}
            if path == "/api/reset":
                self.count_from = self.now()
                return {"ok": True}
        raise UnoQError(f"POST {path} -> HTTP 404 (not simulated in demo mode)")

    def _diag(self):
        t = self.now()
        lost = 70 <= t % 120 < 76
        return {"ankle": {"state": "lost" if lost else "ok", "relay": "connected", "rate_hz": 0.0 if lost else 100.0,
                          "packets": int(t * 10), "seq_gaps": int(t // 120), "source_gaps": 0, "resyncs": 0,
                          "loss_pct": 0.4, "mtu": 247, "n_per_packet": [10], "battery_mv": int(2750 - t / 60),
                          "bad_packets": 0},
                "hip": {"ok": True, "reason": "mean |a| 1.00 g", "rate_hz": 100.0, "late": 0, "skipped": 0,
                        "adc_errors": 0, "adc_bits": 14, "gaps": 0, "clipped_pct": 0.0, "rail_pct": 0.0,
                        "amag_g": 1.0, "cal": "6face", "last_raw": [9600, 7450, 8770]},
                "sync": {"ankle": {"slope": 1.0013, "ppm": 1300.0, "offset_s": 42.8, "residual_ms": 3.1,
                                   "fitted": True, "buckets": 12},
                         "hip": {"slope": 1.0011, "ppm": 1100.0, "offset_s": 39.8, "residual_ms": 0.3,
                                 "fitted": True, "buckets": 13}},
                "present_10s": {"ankle": 0.4 if lost else 1.0, "hip": 1.0},
                "nn": {"model": "demo-0001", "member_ms": [0.8, 0.8, 0.9, 0.8, 0.8], "last_ms": 4.3, "max_ms": 9.8,
                       "p_right": None, "p_left": None},
                "gestures": [], "grid_rows": int(t * 100)}
