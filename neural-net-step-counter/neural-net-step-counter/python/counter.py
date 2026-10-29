"""Turning probability streams (and raw signals) into counted steps.

NNCounter - peak picking on the ensemble-mean streams p_right / p_left:
  * a step is a local maximum >= thr (from model.npz), at least 0.30 s after
    that foot's previous step;
  * it is COMMITTED only when the stream is final past it (StreamInfer.final_k),
    and `committed_until` only moves forward - so overlapping 3 s windows can
    never count the same heel strike twice;
  * every event stores its confidence c = peak height and which ensemble
    members also had a peak within +-60 ms.

Uncertainty (per session), from three independent sources:
  var_event  = sum c_i (1 - c_i)   each event is a coin flip "real step or not"
                                   with probability c_i: the variance of a sum of
                                   independent Bernoullis (Poisson-binomial)
  var_model  = variance across ensemble members of each member's own count -
               how much the count depends on the particular trained network
  systematic = (r * N)^2           r = RMS relative count error on held-out
                                   sessions (from the laptop): errors that don't
                                   average out, e.g. a consistent bias
  sigma_N = sqrt(kappa^2 (var_event + var_model) + (r N)^2), kappa calibrated on
  the laptop so the 95 % interval N +- 1.96 sigma_N really covers 95 % of sessions.

BaselineCounter - the classical detector of contract 9.5, always running so
the report can compare it with the network (and it is what is shown when no
model is loaded).

EventLog - ids for the API. One counter hands out both event ids and
"change" numbers: a new event's change number IS its id; a gesture deletion
re-sends the same event id with deleted=true under a NEW change number. So
GET /api/events?since=<last_id> returns new events and deletions alike, ids are
monotonic and never reused, and state.json carries the counter across restarts.
"""

from __future__ import annotations

import bisect
import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from settings import DT, STATE_JSON, read_json, write_json_atomic

log = logging.getLogger("step.count")

REFRACTORY = 30            # samples = 0.30 s between same-foot events
MEMBER_TOL = 6             # samples = +-60 ms for "member agreed"
FEET = ("R", "L")


@dataclass
class Event:
    id: int
    k: int                 # grid index
    foot: str              # "R" | "L"
    conf: float
    src: str               # "nn" | "baseline"
    session_id: str
    members: list = field(default_factory=list)
    deleted: bool = False
    length_m: float = 0.0  # step length assigned by distance.py

    @property
    def t(self) -> float:
        return self.k * DT

    def api(self) -> dict:
        return {"id": self.id, "t": round(self.t, 3), "foot": self.foot, "conf": round(self.conf, 4),
                "src": self.src, "session_id": self.session_id, "deleted": self.deleted}


class EventLog:
    """Thread-safe change log behind GET /api/events."""

    MAX_CHANGES = 200_000
    ID_MARGIN = 10_000    # added on every start: covers ids handed out after the last state save

    def __init__(self):
        st = read_json(STATE_JSON, {}) or {}
        self._next = int(st.get("next_event_id", 1)) + self.ID_MARGIN
        self._saved = None
        self._lock = threading.Lock()
        self._changes: list = []           # (change_no, api dict)
        self.save()

    def new_id(self) -> int:
        with self._lock:
            i = self._next
            self._next += 1
            return i

    def publish(self, ev: Event) -> None:
        with self._lock:
            # keep the log sorted by change number even if events of one tick are published out of id order
            bisect.insort(self._changes, (ev.id, ev.api()), key=lambda c: c[0])
            self._trim()

    def publish_deletion(self, ev: Event) -> None:
        with self._lock:
            n = self._next
            self._next += 1
            self._changes.append((n, ev.api()))
            self._trim()

    def _trim(self) -> None:
        if len(self._changes) > self.MAX_CHANGES:
            del self._changes[: len(self._changes) - self.MAX_CHANGES]

    def since(self, since: int, limit: int = 5000) -> dict:
        with self._lock:
            # changes are appended in increasing change_no order (ids come from one counter)
            lo, hi = 0, len(self._changes)
            while lo < hi:
                mid = (lo + hi) // 2
                if self._changes[mid][0] <= since:
                    lo = mid + 1
                else:
                    hi = mid
            out = self._changes[lo: lo + limit]
            last = out[-1][0] if out else since
            return {"events": [d for _, d in out], "last_id": last}

    def save(self) -> None:
        if self._saved != self._next:
            write_json_atomic(STATE_JSON, {"next_event_id": self._next})
            self._saved = self._next


# ---------------------------------------------------------------- uncertainty
def summarize(events: list, member_counts, kappa: float, r: float) -> dict:
    N = len(events)
    c = np.array([e.conf for e in events], dtype=np.float64)
    var_event = float(np.sum(c * (1 - c))) if N else 0.0
    mc = np.asarray(member_counts, dtype=np.float64)
    var_model = float(np.var(mc)) if mc.size > 1 else 0.0     # population variance over members
    systematic = (r * N) ** 2
    sigma = float(np.sqrt(kappa ** 2 * (var_event + var_model) + systematic))
    return {"N": N, "sigma": sigma,
            "ci": [max(0, int(round(N - 1.96 * sigma))), max(0, int(round(N + 1.96 * sigma)))],
            "var_split": {"event": kappa ** 2 * var_event, "model": kappa ** 2 * var_model,
                          "systematic": systematic}}


def _local_max(p: np.ndarray, i: int) -> bool:
    """>= on the left, > on the right: a flat-topped peak is found exactly once."""
    return p[i] >= p[i - 1] and p[i] > p[i + 1]


class _Picker:
    """Peak picker for one stream, scanning forward only."""

    def __init__(self):
        self.scan_k = None
        self.last_k = -10 ** 9

    def scan(self, p: np.ndarray, base: int, k_to: int, thr: float) -> list:
        """p: stream values for grid indices base.. ; returns peak indices in [scan_k, k_to)."""
        if self.scan_k is None:
            self.scan_k = base + 1
        k = max(self.scan_k, base + 1)
        out = []
        while k < k_to:
            i = k - base
            v = p[i]
            if v >= thr and np.isfinite(v) and np.isfinite(p[i - 1]) and np.isfinite(p[i + 1]) \
                    and _local_max(p, i) and k - self.last_k >= REFRACTORY:
                out.append(k)
                self.last_k = k
            k += 1
        self.scan_k = max(self.scan_k, k_to)
        return out


class NNCounter:
    def __init__(self, model, evlog: EventLog):
        self.model = model
        self.evlog = evlog
        M = model.n_members
        self.mean_pick = {f: _Picker() for f in FEET}
        self.member_pick = [{f: _Picker() for f in FEET} for _ in range(M)]
        self.member_peaks = [[] for _ in range(M)]      # (k, foot) per member, current session
        self.committed_until = None
        self.events: list = []                          # current session, not deleted
        self.session_id = ""
        self.session_k0 = 0

    def new_session(self, session_id: str, k0: int) -> None:
        self.session_id, self.session_k0 = session_id, k0
        self.events = []
        self.member_peaks = [[] for _ in self.member_peaks]

    def update(self, grid, final_k, excluded) -> list:
        """Commit everything that is final. excluded(t) -> True inside a gesture window."""
        if final_k is None or grid.base is None:
            return []
        thr = self.model.thr
        base = grid.base
        # members first: they must be known up to k + 6 before an event at k is committed
        for m, picks in enumerate(self.member_pick):
            for fi, f in enumerate(FEET):
                for k in picks[f].scan(grid.pm[:, m, fi], base, final_k - 1, thr):
                    if k >= self.session_k0 and not excluded(k * DT):
                        self.member_peaks[m].append((k, f))
        limit = final_k - 1 - MEMBER_TOL
        new = []
        for fi, f in enumerate(FEET):
            for k in self.mean_pick[f].scan(grid.p[:, fi], base, limit, thr):
                if k < self.session_k0:
                    continue
                if excluded(k * DT):
                    log.info(f"NN {f} step at t={k * DT:.2f} not counted: inside a tap-gesture window")
                    continue
                agree = [any(ff == f and abs(kk - k) <= MEMBER_TOL for kk, ff in peaks[-20:])
                         for peaks in self.member_peaks]
                ev = Event(self.evlog.new_id(), k, f, float(grid.p[k - base, fi]), "nn", self.session_id, agree)
                new.append(ev)
        new.sort(key=lambda e: e.k)
        self.events.extend(new)
        self.committed_until = limit
        return new

    def delete_window(self, t0: float, t1: float) -> list:
        gone = [e for e in self.events if t0 <= e.t <= t1]
        self.events = [e for e in self.events if not (t0 <= e.t <= t1)]
        self.member_peaks = [[(k, f) for k, f in peaks if not t0 <= k * DT <= t1] for peaks in self.member_peaks]
        return gone

    def member_counts(self) -> list:
        return [len(p) for p in self.member_peaks]

    def summary(self) -> dict:
        return summarize(self.events, self.member_counts(), self.model.kappa, self.model.r)


# ---------------------------------------------------------------- baseline (contract 9.5)
class BaselineCounter:
    """Classical detector, streaming over the fused grid (raw channels).

    Right heel strike: the first local minimum of ankle gz after a mid-swing gz
    peak above gz_thr (100 deg/s), within 0.35 s of that peak. (In mid-swing the
    shank rotates fast forward; at heel strike that rotation stops abruptly,
    which is the first dip after the swing peak.) The gz sign is auto-detected:
    if the dominant walking peaks are negative, the signal is flipped.
    Left heel strike: the largest hip-vertical peak with prominence >= hip_prom
    (0.15 g) between two consecutive right strikes (each heel strike jolts the
    pelvis upward); if there is none, the midpoint. 0.30 s minimum between
    same-foot events.

    Beyond the contract (so the hip-only and ankle-only acceptance cases also
    work for the baseline): while the ankle is absent, every hip-vertical peak
    with that prominence counts as a step, alternating R/L.

    Baseline events carry a heuristic confidence (they only matter for sigma
    when the baseline is the shown count): 0.95 gyro strike, 0.9 hip-peak left
    strike, 0.8 midpoint left strike, 0.7 hip-only step.
    """

    LOOKAHEAD = 41            # samples needed after a candidate before deciding (> 0.35 s and > HIP_ONLY_HALF)
    SWING_WINDOW = 35         # 0.35 s
    SIGN_EVERY = 100          # re-evaluate the gz sign every 1 s ...
    SIGN_SPAN = 1000          # ... over the last 10 s
    HIP_ONLY_HALF = 40        # +-0.4 s prominence window for hip-only peaks

    def __init__(self, evlog: EventLog, gz_thr: float = 100.0, hip_prom: float = 0.15):
        self.evlog = evlog
        self.gz_thr, self.hip_prom = gz_thr, hip_prom
        self.sign = 1.0
        self.sign_votes = 0
        self.scan_k = None
        self.peak_k = None          # current mid-swing peak awaiting its heel strike
        self.last_R = None          # grid index of the previous right strike
        self.last_L = -10 ** 9
        self.last_any_hip = -10 ** 9
        self.hip_only_foot = 0
        self.events: list = []
        self.session_id = ""
        self.session_k0 = 0

    def new_session(self, session_id: str, k0: int) -> None:
        self.session_id, self.session_k0 = session_id, k0
        self.events = []

    def _emit(self, k: int, foot: str, conf: float, out: list, excluded) -> None:
        if k < self.session_k0:
            return
        if excluded(k * DT):
            log.info(f"baseline {foot} step at t={k * DT:.2f} not counted: inside a tap-gesture window")
            return
        out.append(Event(self.evlog.new_id(), k, foot, conf, "baseline", self.session_id))

    def _update_sign(self, grid, k: int) -> None:
        k0, ch, _, _ = grid.rows(k - self.SIGN_SPAN, k)
        present = ch[:, 10] > 0.5
        if present.sum() < 300:
            return
        gz = ch[present, 5] * 500.0
        hi, lo = float(np.percentile(gz, 99)), -float(np.percentile(gz, 1))
        want = self.sign
        if lo > 1.2 * hi and lo > self.gz_thr:
            want = -1.0
        elif hi > 1.2 * lo and hi > self.gz_thr:
            want = 1.0
        if want != self.sign:
            log.info(f"baseline: gz sign flipped to {want:+.0f} (99th pct {hi:.0f}, 1st pct {-lo:.0f} deg/s)")
            self.sign = want

    def update(self, grid, excluded) -> list:
        if grid.base is None:
            return []
        end = grid.end - self.LOOKAHEAD
        if self.scan_k is None:
            self.scan_k = grid.base + 1
        k = max(self.scan_k, grid.base + 1)
        if k >= end:
            return []
        k0, ch, _, _ = grid.rows(k - 1, end + self.LOOKAHEAD)
        k = max(k, k0 + 1)                     # every index needs a left neighbour inside the slice
        gz = ch[:, 5] * 500.0 * self.sign
        hv = ch[:, 7]
        ap = ch[:, 10] > 0.5
        hp = ch[:, 11] > 0.5
        out: list = []
        while k < end:
            if k % self.SIGN_EVERY == 0:
                self._update_sign(grid, k)
                gz = ch[:, 5] * 500.0 * self.sign
            i = k - k0
            if ap[i]:
                if self.peak_k is None:
                    if gz[i] > self.gz_thr and gz[i] >= gz[i - 1] and gz[i] > gz[i + 1]:
                        self.peak_k = k
                else:
                    if k - self.peak_k > self.SWING_WINDOW:
                        self.peak_k = None             # no heel-strike dip in time: not a stride
                    elif gz[i] <= gz[i - 1] and gz[i] < gz[i + 1]:
                        self._right_strike(grid, k, out, excluded)
                        self.peak_k = None
            elif hp[i]:
                self.peak_k = None
                self._hip_only(hv, hp, i, k, out, excluded)
            k += 1
        self.scan_k = end
        out.sort(key=lambda e: e.k)
        self.events.extend(out)
        return out

    def _right_strike(self, grid, k: int, out: list, excluded) -> None:
        if self.last_R is not None and k - self.last_R < REFRACTORY:
            return
        prev = self.last_R
        self.last_R = k
        self._emit(k, "R", 0.95, out, excluded)
        if prev is None:
            return
        # left strike between the two right strikes
        s0, ch, _, _ = grid.rows(prev + 1, k)
        if len(ch) < 3 or s0 != prev + 1:
            kl, conf = (prev + k) // 2, 0.8
        else:
            kl, conf = self._best_hip_peak(ch[:, 7], ch[:, 11] > 0.5, s0)
        if kl - self.last_L >= REFRACTORY:
            self.last_L = kl
            self._emit(kl, "L", conf, out, excluded)

    def _best_hip_peak(self, hv: np.ndarray, hp: np.ndarray, s0: int):
        n = len(hv)
        mid = s0 + n // 2
        if hp.mean() < 0.8:
            return mid, 0.8
        best, best_v = None, -np.inf
        for j in range(1, n - 1):
            if hv[j] >= hv[j - 1] and hv[j] > hv[j + 1]:
                prom = hv[j] - max(hv[:j].min(), hv[j + 1:].min())   # prominence within the R-R interval
                if prom >= self.hip_prom and hv[j] > best_v:
                    best, best_v = j, hv[j]
        return (s0 + best, 0.9) if best is not None else (mid, 0.8)

    def _hip_only(self, hv, hp, i, k, out, excluded) -> None:
        lo, hi = max(i - self.HIP_ONLY_HALF, 0), min(i + self.HIP_ONLY_HALF + 1, len(hv))
        if i < 1 or i + 1 >= len(hv) or not hp[lo:hi].all():
            return
        if not (hv[i] >= hv[i - 1] and hv[i] > hv[i + 1]) or hv[i] < hv[lo:hi].max():
            return
        prom = hv[i] - max(hv[lo:i].min(), hv[i + 1:hi].min())
        if prom >= self.hip_prom and k - self.last_any_hip >= REFRACTORY:
            self.last_any_hip = k
            self.hip_only_foot ^= 1
            self._emit(k, FEET[self.hip_only_foot], 0.7, out, excluded)

    def delete_window(self, t0: float, t1: float) -> list:
        gone = [e for e in self.events if t0 <= e.t <= t1]
        self.events = [e for e in self.events if not (t0 <= e.t <= t1)]
        return gone

    def summary(self, r: float = 0.10, kappa: float = 1.0) -> dict:
        """No ensemble: var_model = 0. r = 0.10 is an assumed systematic error
        for the classical detector (replace it with the measured one from the
        laptop's report if you have it)."""
        return summarize(self.events, [], kappa, r)
