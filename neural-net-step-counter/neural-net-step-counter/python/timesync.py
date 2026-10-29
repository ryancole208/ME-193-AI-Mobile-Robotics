"""Sensor clock -> host (Linux monotonic) time, one ClockSync per source.

Why this is needed: the two sensors run on different clocks.
  * ankle: the MPU-6050's internal oscillator sets its sample rate; t0_us is
    "samples so far x 10 ms" on THAT clock, which can be off by a few percent.
  * hip:   the STM32's micros() schedule (crystal, much better, but still not
           the Linux clock).
To put both on one 100 Hz grid we need host_time = a * sensor_time + b.

Every packet gives one pair (sensor time of its LAST sample, host receive
time). receive - sensor = transport latency + clock offset. Latency is always
>= some minimum and only ever adds delay (BLE connection events, retries,
scheduling), so the MINIMUM of (receive - sensor) over a few seconds is the
packet that had the least latency - the lower envelope. We:
  * keep the pairs of the last 120 s,
  * take the minimum offset per 5 s bucket (robust to latency jitter),
  * fit host = a * sensor + b by least squares on those bucket minima
    (the slope a captures oscillator drift), refit every 5 s,
  * before 20 s of pairs exist: slope 1, b = running minimum offset over the
    LAST 2 s, updated on every packet. Not the minimum since the start: with
    the MPU-6050 clock 2 % slow, the offset grows by 20 ms every second, so
    the first packet's offset would map later samples up to ~0.4 s too early.
The mapped time is "when the sample would have arrived with minimum latency".
That minimum differs between BLE and Bridge by a few tens of ms; the network
is trained with a +-50 ms hip time shift to absorb exactly this.

Sensor time is never compared with host time anywhere else in the app.
"""

from __future__ import annotations

from collections import deque

import numpy as np

WINDOW_S = 120.0
BUCKET_S = 5.0
REFIT_S = 5.0
MIN_SPAN_S = 20.0
WARMUP_MIN_S = 2.0          # warm-up offset = min over this many recent seconds


class ClockSync:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Called when the source restarts (its sample index is no longer comparable)."""
        self.pairs: deque = deque()          # (sensor_s, host_s)
        self.a = 1.0
        self.b = None                        # host = a * sensor + b
        self.residual_ms = None
        self.n_buckets = 0
        self.fitted = False
        self._last_fit = -1e9
        self.resets = getattr(self, "resets", -1) + 1

    def add(self, sensor_s: float, host_s: float) -> None:
        self.pairs.append((sensor_s, host_s))
        while self.pairs and self.pairs[0][1] < host_s - WINDOW_S:
            self.pairs.popleft()
        if not self.fitted:
            self.b = self._recent_min(host_s)
        if host_s - self._last_fit >= REFIT_S:
            self._fit(host_s)

    def _recent_min(self, now: float) -> float:
        """Warm-up offset: the lowest (receive - sensor) of the last WARMUP_MIN_S seconds
        (~20 packets: enough to catch a minimum-latency one; short enough that 2 % drift
        adds at most ~40 ms)."""
        return min(h - s for s, h in reversed(self.pairs) if h >= now - WARMUP_MIN_S)

    def _fit(self, now: float) -> None:
        self._last_fit = now
        p = np.asarray(self.pairs, dtype=np.float64)
        span = p[-1, 1] - p[0, 1]
        # lower envelope: per 5 s bucket (by receive time), the pair with the smallest offset
        bucket = np.floor((p[:, 1] - p[0, 1]) / BUCKET_S).astype(int)
        off = p[:, 1] - p[:, 0]
        mins = []
        for k in np.unique(bucket):
            idx = np.flatnonzero(bucket == k)
            j = idx[np.argmin(off[idx])]
            mins.append(p[j])
        m = np.asarray(mins)
        self.n_buckets = len(m)
        if span < MIN_SPAN_S or len(m) < 3:
            # not enough history for a slope: slope 1 and the recent running minimum
            self.a, self.b, self.fitted = 1.0, self._recent_min(now), False
            self.residual_ms = None
            return
        s0 = m[:, 0].mean()                  # centre for numerical stability
        A = np.column_stack([m[:, 0] - s0, np.ones(len(m))])
        coef, *_ = np.linalg.lstsq(A, m[:, 1], rcond=None)
        a, c = float(coef[0]), float(coef[1])
        if not 0.9 < a < 1.1:                # > 10 % drift is not a clock, it is a bug or a restart
            return
        self.a, self.b, self.fitted = a, c - a * s0, True
        self.residual_ms = float(np.sqrt(np.mean((A @ coef - m[:, 1]) ** 2)) * 1000)

    def ready(self) -> bool:
        return self.b is not None

    def to_host(self, sensor_s):
        return self.a * np.asarray(sensor_s, dtype=np.float64) + self.b

    def describe(self) -> str:
        if self.b is None:
            return f"{self.name}: no data"
        res = "n/a" if self.residual_ms is None else f"{self.residual_ms:.1f} ms"
        kind = "fit" if self.fitted else "slope 1 (warming up)"
        return (f"{self.name}: slope {self.a:.6f} ({(self.a - 1) * 1e6:+.0f} ppm), offset {self.b:.3f} s, "
                f"residual {res}, buckets {self.n_buckets}, {kind}")
