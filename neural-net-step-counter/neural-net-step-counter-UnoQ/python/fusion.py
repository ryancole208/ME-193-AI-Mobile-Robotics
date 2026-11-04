"""Fusion: both sensors on one 100 Hz grid in host time, and the 12 channels.

Pipeline per source:  samples (sensor sample index) --timesync--> host times
                      --linear interpolation--> grid times t_k = k * 10 ms

* The grid runs LAG_S = 0.5 s behind real time, so packets still in flight
  (BLE sends 10 samples per ~100 ms) have arrived before their grid row is built.
* Missing samples: a hole of <= 0.2 s between two received samples is bridged by
  linear interpolation. Anything longer (or no sample after t yet) sets that
  source's PRESENT flag to 0 and its channels to 0 for those rows. The network
  is trained with whole-source dropout, so it keeps counting from the other one.

build_channels() is THE preprocessing (contract section 9.3). The laptop
trains on fused.npy written from it, and model.npz carries prep_hash = SHA-256
of the channel definition text below; the self-check compares hashes so a
model trained on different channels is refused.
"""

from __future__ import annotations

import hashlib

import numpy as np

from settings import DT, FS

LAG_S = 0.5
MAX_GAP_SAMPLES = 21          # index step of 21 = 20 missing samples = 0.2 s hole -> still interpolated
GRAVITY_TAU_S = 3.0
N_CH = 12
CH_NAMES = ["ank_ax", "ank_ay", "ank_az", "ank_gx", "ank_gy", "ank_gz", "ank_amag",
            "hip_vert", "hip_horiz", "hip_amag_m1", "ankle_present", "hip_present"]
GYRO_DIV = 500.0              # deg/s / 500 keeps the gyro channels O(1) like the accel channels

# Contract 9.3, verbatim (both sides hard-code this text; whitespace is collapsed before hashing).
PREP_TEXT = """1–6. ankle ax, ay, az (g); gx, gy, gz (°/s ÷ 500). Right-leg frame: mirror left-leg data with the existing `leg:` logic.
7. ankle |a| (g)
8. hip vertical accel (g, gravity removed): projection onto a running gravity estimate (low-pass of hip accel, τ = 3 s, started from the setup gravity)
9. hip horizontal accel magnitude (g)
10. hip |a| − 1 (g)
11. ankle_present (0/1)
12. hip_present (0/1)"""


def prep_hash() -> str:
    return hashlib.sha256(" ".join(PREP_TEXT.split()).encode("utf-8")).hexdigest()


def build_channels(ankle: np.ndarray, ankle_present: np.ndarray, hip: np.ndarray, hip_present: np.ndarray,
                   gravity: np.ndarray):
    """The 12 network channels for n consecutive grid rows.

    ankle (n, 6): ax ay az [g], gx gy gz [deg/s], already in the right-leg frame
    hip   (n, 3): calibrated hip accel [g], in whatever frame the board sits
    *_present (n,): bool
    gravity (3,): running low-pass of hip accel (state, carried between calls;
                  starts from the setup's still-standing hip gravity)
    Returns (channels (n, 12) float32, new gravity).

    Hip channels are orientation-free on purpose: the case can sit at any angle
    on the belt. With g_hat = gravity / |gravity|:
      vertical   = a . g_hat - |gravity|     (positive = accelerating upward)
      horizontal = |a - (a . g_hat) g_hat|
      amag - 1   = |a| - 1
    The low-pass (time constant 3 s) updates BEFORE the projection, and only
    while the hip is present (its estimate is held through dropouts).
    """
    n = len(ankle)
    out = np.zeros((n, N_CH), dtype=np.float32)
    ap = np.asarray(ankle_present, dtype=bool)
    hp = np.asarray(hip_present, dtype=bool)
    out[ap, 0:3] = ankle[ap, 0:3]
    out[ap, 3:6] = ankle[ap, 3:6] / GYRO_DIV
    out[ap, 6] = np.linalg.norm(ankle[ap, 0:3], axis=1)
    alpha = 1.0 - np.exp(-DT / GRAVITY_TAU_S)
    g = np.asarray(gravity, dtype=np.float64).copy()
    for i in np.flatnonzero(hp):           # sequential: the low-pass is a recursion
        a = hip[i].astype(np.float64)
        g += alpha * (a - g)
        gn = np.linalg.norm(g)
        gh = g / gn if gn > 1e-6 else np.array([0.0, 0.0, 1.0])
        v = float(a @ gh)
        out[i, 7] = v - gn
        out[i, 8] = np.linalg.norm(a - v * gh)
        out[i, 9] = np.linalg.norm(a) - 1.0
    out[:, 10] = ap
    out[:, 11] = hp
    return out, g


class SourceBuffer:
    """Recent samples of one source, indexed by its sensor sample index."""

    KEEP_SAMPLES = 10 * FS

    def __init__(self, nch: int):
        self.nch = nch
        self.clear()

    def clear(self) -> None:
        self.idx = np.zeros(0, dtype=np.int64)
        self.val = np.zeros((0, self.nch), dtype=np.float64)

    def add(self, start_index: int, values: np.ndarray) -> None:
        n = len(values)
        new_idx = np.arange(start_index, start_index + n, dtype=np.int64)
        if len(self.idx) and new_idx[0] <= self.idx[-1]:
            keep = new_idx > self.idx[-1]          # never go backwards in sensor time
            new_idx, values = new_idx[keep], values[keep]
        self.idx = np.concatenate([self.idx, new_idx])[-self.KEEP_SAMPLES:]
        self.val = np.concatenate([self.val, values])[-self.KEEP_SAMPLES:]

    def sample(self, sync, T: np.ndarray):
        """Interpolate at host times T. Returns (values (len(T), nch), present (len(T),) bool)."""
        out = np.zeros((len(T), self.nch))
        present = np.zeros(len(T), dtype=bool)
        if len(self.idx) < 2 or not sync.ready():
            return out, present
        H = sync.to_host(self.idx * DT)
        j = np.searchsorted(H, T, side="right") - 1
        ok = (j >= 0) & (j < len(H) - 1)
        jj = np.clip(j, 0, len(H) - 2)
        ok &= (self.idx[jj + 1] - self.idx[jj]) <= MAX_GAP_SAMPLES
        w = np.clip((T - H[jj]) / np.maximum(H[jj + 1] - H[jj], 1e-9), 0.0, 1.0)
        out[ok] = (1 - w[ok, None]) * self.val[jj[ok]] + w[ok, None] * self.val[jj[ok] + 1]
        present[ok] = True
        return out, present


class GridBuffer:
    """The fused grid (raw channels) + NN probability streams, by global grid index k
    (t_k = k * DT, host monotonic seconds). Keeps the last KEEP_S seconds."""

    KEEP_S = 60

    def __init__(self, n_members: int = 0):
        self.base = None            # grid index of row 0
        self.ch = np.zeros((0, N_CH), dtype=np.float32)
        self.reset_streams(n_members)

    def reset_streams(self, n_members: int) -> None:
        self.n_members = n_members
        n = len(self.ch)
        self.p = np.full((n, 2), np.nan, dtype=np.float32)              # ensemble mean
        self.pm = np.full((n, max(n_members, 1), 2), np.nan, dtype=np.float32)

    @property
    def end(self) -> int:
        """Grid index one past the newest row."""
        return (self.base or 0) + len(self.ch)

    def append(self, k0: int, rows: np.ndarray) -> None:
        if self.base is None:
            self.base = k0
        assert k0 == self.end, "grid rows must be contiguous"
        n = len(rows)
        self.ch = np.concatenate([self.ch, rows])
        self.p = np.concatenate([self.p, np.full((n, 2), np.nan, dtype=np.float32)])
        self.pm = np.concatenate([self.pm, np.full((n,) + self.pm.shape[1:], np.nan, dtype=np.float32)])
        extra = len(self.ch) - self.KEEP_S * FS
        if extra > 0:
            self.ch, self.p, self.pm = self.ch[extra:], self.p[extra:], self.pm[extra:]
            self.base += extra

    def rows(self, k0: int, k1: int):
        """Slice [k0, k1) clipped to what is kept: returns (k0_clipped, ch, p, pm)."""
        k0 = max(k0, self.base or 0)
        a, b = k0 - (self.base or 0), max(k1 - (self.base or 0), k0 - (self.base or 0))
        return k0, self.ch[a:b], self.p[a:b], self.pm[a:b]

    def time(self, k) -> np.ndarray:
        return np.asarray(k) * DT


class Fusion:
    """Builds grid rows from the two source buffers."""

    def __init__(self, gravity0):
        self.ankle = SourceBuffer(6)
        self.hip = SourceBuffer(3)
        self.gravity = np.asarray(gravity0, dtype=np.float64)
        self.next_k = None
        self.rows_built = 0

    def set_gravity(self, g) -> None:
        self.gravity = np.asarray(g, dtype=np.float64)

    def advance(self, now: float, ankle_sync, hip_sync, hip_enabled: bool = True):
        """Build all rows with t_k <= now - LAG_S. Returns (k0, channels) or None."""
        k_end = int(np.floor((now - LAG_S) / DT)) + 1
        if self.next_k is None:
            if not (ankle_sync.ready() or hip_sync.ready()):
                return None
            self.next_k = k_end - 1
        if k_end <= self.next_k:
            return None
        ks = np.arange(self.next_k, k_end)
        T = ks * DT
        ank, ap = self.ankle.sample(ankle_sync, T)
        hip, hp = self.hip.sample(hip_sync, T)
        if not hip_enabled:                    # hip judged implausible (unplugged / dead)
            hp[:] = False
        ch, self.gravity = build_channels(ank, ap, hip, hp, self.gravity)
        k0 = self.next_k
        self.next_k = k_end
        self.rows_built += len(ks)
        return k0, ch
