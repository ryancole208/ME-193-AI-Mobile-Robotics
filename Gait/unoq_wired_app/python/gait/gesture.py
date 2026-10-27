"""Foot-tap gesture detector. Runs on the raw 100 Hz stream, not on windows.

A TAP is a short spike in |a| above `tap_g` that
  * lasts less than `max_tap_ms` above the threshold (spikes closer than
    DEBOUNCE_MS are merged, so the ringing of one impact counts once), and
  * is followed by a quick return to about 1 g (|a| within RETURN_TOL_G of 1 g
    within RETURN_MS).

A GESTURE is a train of taps with every gap in [min_gap_ms, max_gap_ms].
The train ends when no tap follows within max_gap_ms. Then it is judged:

  * 4 taps (taps_switch) -> "switch"      8 taps (taps_long) -> "long"
  * The wearer must be STILL before and after: in the 1.5 s before the first
    tap and the 1 s after the last, the std of |a| and of |g| must be below
    still_noise_mult x the noise measured in setup's stand-still step. This is
    the main protection against false switches while running or biking.
    (Both windows keep a 0.3 s margin from the taps, because the heel lifts a
    little before the first tap and settles after the last.)
  * A gesture within lockout_s of the previous one is flagged `locked_out`;
    live mode ignores those (the redo-setup confirmation is the exception).

Anything else with 2+ taps is logged as a "near_miss" with the reason, so the
thresholds can be tuned from the logs.

Why the rhythm makes this safe: 150-450 ms between taps of the SAME foot is far
faster than walking or running, where the same foot lands about once a second.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

FS = 100
DEBOUNCE_MS = 30
RETURN_MS = 120
RETURN_TOL_G = 0.35
MARGIN_S = 0.3        # gap between the taps and the stillness windows
PRE_S = 1.5
POST_S = 1.0
MIN_PRE_S = 0.5       # shorter pre-window (e.g. right after another gesture) -> skip that check
NOISE_FLOOR_A = 0.01  # g; stops a super-quiet setup from making "still" impossible
NOISE_FLOOR_G = 1.0   # deg/s


@dataclass
class GestureParams:
    tap_g: float = 2.0
    max_tap_ms: float = 60
    min_gap_ms: float = 150
    max_gap_ms: float = 450
    taps_switch: int = 4
    taps_long: int = 8
    still_noise_mult: float = 3
    lockout_s: float = 3
    amag_noise_g: float = 0.02      # from config.signal.noise
    gmag_noise_dps: float = 2.0

    @classmethod
    def from_config(cls, cfg: dict) -> "GestureParams":
        g = cfg["gesture"]
        noise = cfg["signal"]["noise"]
        return cls(
            tap_g=g["tap_g"], max_tap_ms=g["max_tap_ms"], min_gap_ms=g["min_gap_ms"],
            max_gap_ms=g["max_gap_ms"], taps_switch=int(g["taps_switch"]), taps_long=int(g["taps_long"]),
            still_noise_mult=g["still_noise_mult"], lockout_s=g["lockout_s"],
            amag_noise_g=noise["amag_std_g"], gmag_noise_dps=noise["gmag_std_dps"],
        )

    @property
    def still_thr_a(self) -> float:
        return self.still_noise_mult * max(self.amag_noise_g, NOISE_FLOOR_A)

    @property
    def still_thr_g(self) -> float:
        return self.still_noise_mult * max(self.gmag_noise_dps, NOISE_FLOOR_G)


@dataclass
class Tap:
    index: int          # sample index of the peak
    start: int
    end: int            # first sample back below threshold
    peak_g: float

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0 / FS


@dataclass
class GestureEvent:
    kind: str                      # "switch" | "long" | "near_miss"
    taps: list = field(default_factory=list)
    reason: str = ""
    start_index: int = 0
    end_index: int = 0
    pre_std: tuple = (None, None)  # (|a| std, |g| std) before
    post_std: tuple = (None, None)
    locked_out: bool = False       # valid gesture, but within lockout_s of the previous one

    @property
    def n_taps(self) -> int:
        return len(self.taps)

    @property
    def gaps_ms(self) -> list:
        return [(b.index - a.index) * 1000.0 / FS for a, b in zip(self.taps, self.taps[1:])]

    def describe(self) -> str:
        gaps = ", ".join(f"{g:.0f}" for g in self.gaps_ms)
        peaks = ", ".join(f"{t.peak_g:.1f}" for t in self.taps)
        pre = "skipped" if self.pre_std[0] is None else f"|a| {self.pre_std[0]:.3f} g, |g| {self.pre_std[1]:.2f} dps"
        post = "n/a" if self.post_std[0] is None else f"|a| {self.post_std[0]:.3f} g, |g| {self.post_std[1]:.2f} dps"
        return (f"{self.kind} ({self.reason or 'ok'}): {self.n_taps} taps, gaps ms [{gaps}], "
                f"peaks g [{peaks}], still before: {pre}, after: {post}")


class GestureDetector:
    def __init__(self, params: GestureParams):
        self.p = params
        hist = int((PRE_S + MARGIN_S + 1.0) * FS) + int(params.max_gap_ms * FS / 1000) * 10
        self._amag = deque(maxlen=hist)
        self._gmag = deque(maxlen=hist)
        self._idx = deque(maxlen=hist)
        self._spike = None           # [start, end_or_None, peak_index, peak]
        self._pending = None         # Tap waiting for the quick return to ~1 g
        self._train: list[Tap] = []
        self._train_bad = ""
        self._judging: list = []     # closed trains waiting for their post-stillness window
        self._last_accept_end = -10**9
        self._history_start = None   # first index after the last reset
        self.recent_spans: deque = deque(maxlen=20)  # (start, end) of rhythmic trains, to taint windows

    # ------------------------------------------------------------------ input
    def reset(self) -> None:
        """Forget everything in flight (called on a gap in the stream)."""
        self._amag.clear()
        self._gmag.clear()
        self._idx.clear()
        self._spike = None
        self._pending = None
        self._train = []
        self._train_bad = ""
        self._judging = []
        self._history_start = None

    def push(self, start_index: int, samples: np.ndarray) -> list[GestureEvent]:
        """samples: (n, 6) in g and deg/s. Returns events completed by these samples."""
        if self._idx and start_index != self._idx[-1] + 1:
            self.reset()
        if self._history_start is None:
            self._history_start = start_index
        amag = np.linalg.norm(samples[:, 0:3], axis=1)
        gmag = np.linalg.norm(samples[:, 3:6], axis=1)
        events: list[GestureEvent] = []
        for k in range(len(samples)):
            events += self._step(start_index + k, float(amag[k]), float(gmag[k]))
        return events

    # ------------------------------------------------------------------ per sample
    def _step(self, i: int, a: float, g: float) -> list[GestureEvent]:
        p = self.p
        self._amag.append(a)
        self._gmag.append(g)
        self._idx.append(i)
        out: list[GestureEvent] = []

        # 1) spike tracking with debounce
        if a > p.tap_g:
            if self._spike is None:
                self._spike = [i, None, i, a]
            else:
                if self._spike[1] is not None:      # re-crossing within the debounce: same impact
                    self._spike[1] = None
                if a > self._spike[3]:
                    self._spike[2], self._spike[3] = i, a
        elif self._spike is not None:
            if self._spike[1] is None:
                self._spike[1] = i
            elif (i - self._spike[1]) * 1000 / FS >= DEBOUNCE_MS:
                start, end, pk_i, pk = self._spike
                self._spike = None
                dur = (end - start) * 1000 / FS
                if dur <= p.max_tap_ms:
                    self._pending = Tap(pk_i, start, end, pk)
                else:
                    self._break_train(f"spike of {dur:.0f} ms is longer than max_tap_ms={p.max_tap_ms}")

        # 2) a tap counts once |a| is back near 1 g
        if self._pending is not None:
            if abs(a - 1.0) < RETURN_TOL_G:
                self._add_tap(self._pending)
                self._pending = None
            elif (i - self._pending.end) * 1000 / FS > RETURN_MS:
                self._break_train(f"|a| did not return to ~1 g within {RETURN_MS} ms after a spike")
                self._pending = None

        # 3) close the train when no tap follows within max_gap
        if self._train and self._spike is None and self._pending is None:
            if (i - self._train[-1].index) * 1000 / FS > p.max_gap_ms:
                self._close_train()

        # 4) judge closed trains once their post-stillness window is complete
        still_pending = []
        for train, bad in self._judging:
            if i >= train[-1].index + int((MARGIN_S + POST_S) * FS):
                ev = self._judge(train, bad)
                if ev is not None:
                    out.append(ev)
            else:
                still_pending.append((train, bad))
        self._judging = still_pending
        return out

    def _add_tap(self, tap: Tap) -> None:
        if self._train:
            gap = (tap.index - self._train[-1].index) * 1000 / FS
            if gap < self.p.min_gap_ms and not self._train_bad:
                self._train_bad = f"gap of {gap:.0f} ms is shorter than min_gap_ms={self.p.min_gap_ms}"
        self._train.append(tap)
        if len(self._train) >= 2:
            self._mark_span(self._train)

    def _break_train(self, reason: str) -> None:
        if self._train:
            self._train_bad = self._train_bad or reason

    def _close_train(self) -> None:
        self._judging.append((self._train, self._train_bad))
        self._train, self._train_bad = [], ""

    def _mark_span(self, train: list[Tap]) -> None:
        span = (train[0].start - int(MARGIN_S * FS), train[-1].end + int(MARGIN_S * FS))
        if self.recent_spans and self.recent_spans[-1][0] == span[0]:
            self.recent_spans[-1] = span
        else:
            self.recent_spans.append(span)

    # ------------------------------------------------------------------ judging
    def _window_std(self, lo: int, hi: int):
        """(std |a|, std |g|) over sample indices [lo, hi), or None if not in history."""
        if not self._idx or lo < self._idx[0] or hi > self._idx[-1] + 1 or hi - lo < 2:
            return None
        off = self._idx[0]
        a = np.array(self._amag)[lo - off: hi - off]
        g = np.array(self._gmag)[lo - off: hi - off]
        return float(a.std()), float(g.std())

    def _judge(self, train: list[Tap], bad: str) -> GestureEvent | None:
        p = self.p
        n = len(train)
        if n < 2:
            return None  # a single bump is not worth logging
        first, last = train[0].index, train[-1].index
        ev = GestureEvent("near_miss", list(train), "", train[0].start, train[-1].end)

        pre_hi = first - int(MARGIN_S * FS)
        pre_lo = max(pre_hi - int(PRE_S * FS), self._last_accept_end, self._history_start or 0)
        pre = self._window_std(pre_lo, pre_hi) if pre_hi - pre_lo >= int(MIN_PRE_S * FS) else None
        post_lo = last + int(MARGIN_S * FS)
        post = self._window_std(post_lo, post_lo + int(POST_S * FS))
        if pre is not None:
            ev.pre_std = pre
        if post is not None:
            ev.post_std = post

        reasons = []
        if bad:
            reasons.append(bad)
        if n not in (p.taps_switch, p.taps_long):
            reasons.append(f"{n} taps (need {p.taps_switch} to switch or {p.taps_long} for the long gesture)")
        if pre is not None and (pre[0] > p.still_thr_a or pre[1] > p.still_thr_g):
            reasons.append(f"not still before (|a| std {pre[0]:.3f} > {p.still_thr_a:.3f} g "
                           f"or |g| std {pre[1]:.2f} > {p.still_thr_g:.2f} dps)")
        if post is None:
            reasons.append("stream ended before the 'still after' check")
        elif post[0] > p.still_thr_a or post[1] > p.still_thr_g:
            reasons.append(f"not still after (|a| std {post[0]:.3f} > {p.still_thr_a:.3f} g "
                           f"or |g| std {post[1]:.2f} > {p.still_thr_g:.2f} dps)")
        if reasons:
            ev.reason = "; ".join(reasons)
            return ev
        ev.kind = "switch" if n == p.taps_switch else "long"
        # Lockout is only MARKED here: live mode ignores locked-out gestures, but the
        # redo-setup confirmation (4 taps within 5 s of the 8-tap gesture) needs them.
        ev.locked_out = (first - self._last_accept_end) < p.lockout_s * FS
        if ev.locked_out:
            ev.reason = f"within the {p.lockout_s:.0f} s lockout after the previous gesture"
        self._last_accept_end = post_lo + int(POST_S * FS)
        return ev

    # ------------------------------------------------------------------ helpers
    def overlaps_gesture(self, start: int, end: int) -> bool:
        """True if [start, end) overlaps a rhythmic tap train (used to keep
        gesture windows out of the display smoothing)."""
        return any(s < end and start < e for s, e in self.recent_spans)

    @property
    def in_progress(self) -> int:
        """Taps in the train currently being collected."""
        return len(self._train)
