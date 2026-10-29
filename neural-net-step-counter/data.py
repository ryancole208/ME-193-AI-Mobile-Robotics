"""Sessions, training windows, augmentation, and the shared channel definition.

Contract references (see BRIEF_LAPTOP_STEPCOUNTER.md section 8):
  8.2  session folder layout  -> Session
  8.3  the 12 input channels  -> CHANNEL_TEXT / PREP_HASH and the index constants

Why a hash of a *text*? The UNO Q builds the 12 channels live, the laptop trains
on the same 12 channels recorded in fused.npy. If either side ever changes what a
channel means, the network silently gets garbage. Both sides hard-code the same
human-readable description, hash it, and refuse a model whose prep_hash differs.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

FS = 100            # Hz, fused.npy sample rate
WIN = 400           # 4 s training window
N_CH = 12

# 0-based channel indices into the 12-channel input (fused.npy column = index + 1).
AX, AY, AZ, GX, GY, GZ = range(6)
ANK_AMAG = 6
HIP_VERT = 7
HIP_HORZ = 8
HIP_AMAG1 = 9
ANK_PRESENT = 10
HIP_PRESENT = 11
ANKLE_CH = [AX, AY, AZ, GX, GY, GZ, ANK_AMAG]
HIP_CH = [HIP_VERT, HIP_HORZ, HIP_AMAG1]
GYRO_SCALE = 500.0  # gyro channels are stored as (deg/s) / 500

# Contract 8.3, verbatim. Do NOT edit: the UNO Q hashes the identical text.
CHANNEL_TEXT = """1–6. ankle ax, ay, az (g); gx, gy, gz (°/s ÷ 500). Right-leg frame: mirror left-leg data with the existing `leg:` logic.
7. ankle |a| (g)
8. hip vertical accel (g, gravity removed): projection onto a running gravity estimate (low-pass of hip accel, τ = 3 s, started from the setup gravity)
9. hip horizontal accel magnitude (g)
10. hip |a| − 1 (g)
11. ankle_present (0/1)
12. hip_present (0/1)"""


def prep_hash(text=CHANNEL_TEXT):
    """SHA-256 of the channel text with all whitespace runs collapsed to one space."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


PREP_HASH = prep_hash()


# ---------------------------------------------------------------- sessions

class Session:
    """One recorded session folder (contract 8.2), plus labels.npz if `label` has run."""

    def __init__(self, folder):
        self.dir = Path(folder)
        self.id = self.dir.name
        meta_path = self.dir / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        fused = np.load(self.dir / "fused.npy")
        # Column 0 is host time. It is float32 per the contract, so it is only good to
        # ~10 ms for large clock values; we work in sample indices wherever we can.
        self.t = fused[:, 0].astype(np.float64)
        self.x = np.ascontiguousarray(fused[:, 1:13], dtype=np.float32)
        self.gestures = load_gestures(self.dir / "gestures.csv")
        self.labels = None
        lab = self.dir / "labels.npz"
        if lab.exists():
            with np.load(lab, allow_pickle=False) as z:
                self.labels = {k: z[k] for k in z.files}

    def __len__(self):
        return len(self.x)

    @property
    def truth(self):
        return self.meta.get("truth") or None

    @property
    def true_steps(self):
        tr = self.truth
        return None if tr is None or tr.get("steps") is None else int(tr["steps"])

    @property
    def distance_m(self):
        tr = self.truth
        if tr is None or tr.get("distance_in") in (None, ""):
            return None
        return float(tr["distance_in"]) * 0.0254

    @property
    def is_calib_walk(self):
        tr = self.truth
        return bool(tr and tr.get("is_calib_walk"))

    @property
    def duration_s(self):
        return len(self.x) / FS

    @property
    def category(self):
        """walk / run / negative, used to stratify the held-out split.

        Negatives are sessions with a true count of 0. Runs are recognised from the
        notes ("run", "jog") or, failing that, from cadence: above ~140 steps/min
        (ignoring the 10 s of standing in the protocol) is almost certainly running.
        """
        n = self.true_steps
        if n == 0:
            return "negative"
        notes = ((self.truth or {}).get("notes") or "").lower()
        if "run" in notes or "jog" in notes:
            return "run"
        active_min = max(self.duration_s - 10.0, 1.0) / 60.0
        if n is not None and n / active_min > 140:
            return "run"
        return "walk"

    def label_idx(self):
        """(right_idx, left_idx) labelled strikes as int sample indices."""
        if self.labels is None:
            return None
        return self.labels["right_idx"].astype(int), self.labels["left_idx"].astype(int)


def load_gestures(path):
    """gestures.csv -> list of (t_first_tap, t_last_tap, kind). Missing file = no gestures."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append((float(row["t_first_tap"]), float(row["t_last_tap"]), row.get("kind", "")))
            except (KeyError, ValueError):
                continue
    return out


def list_sessions(sessions_dir):
    """All complete session folders (those with a fused.npy), oldest first."""
    d = Path(sessions_dir)
    if not d.exists():
        return []
    return [Session(p) for p in sorted(d.iterdir()) if (p / "fused.npy").exists()]


# ---------------------------------------------------------------- normalisation

def compute_norm(sessions):
    """Per-channel mean/std over every sample of the given (training) sessions.

    The present flags are normalised too, so the network sees one consistent
    input format; a flag that never changes has std 0, which we replace by 1.
    """
    x = np.concatenate([s.x for s in sessions], axis=0).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalise(x, mean, std):
    """(N, 12) raw -> (12, N) normalised, the network's input layout."""
    return ((x - mean) / std).T.astype(np.float32)


# ---------------------------------------------------------------- targets

def make_targets(n, right_pos, left_pos, sigma=3.0):
    """Gaussian bump (peak 1.0, sigma in samples) at each strike, one row per foot.

    Soft targets instead of single 1-samples: a strike labelled 20 ms off is
    still mostly right, and the network learns a smooth peak we can pick.
    Positions may be fractional (time-stretch moves them between samples).
    """
    y = np.zeros((2, n), np.float32)
    idx = np.arange(n)
    reach = 4 * sigma
    for row, positions in ((0, right_pos), (1, left_pos)):
        positions = np.asarray(positions, float)
        positions = positions[(positions > -reach) & (positions < n + reach)]  # only strikes near the window
        for p in positions:
            lo = max(0, int(np.floor(p - reach)))
            hi = min(n, int(np.ceil(p + reach)) + 1)
            if lo >= hi:
                continue
            bump = np.exp(-0.5 * ((idx[lo:hi] - p) / sigma) ** 2)
            y[row, lo:hi] = np.maximum(y[row, lo:hi], bump)
    return y


# ---------------------------------------------------------------- windows + augmentation

def _interp_rows(x, pos):
    """Linear interpolation of rows of x at fractional positions (edge-clamped)."""
    pos = np.clip(pos, 0, len(x) - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    f = (pos - i0)[:, None]
    return x[i0] * (1 - f) + x[i1] * f


def _rotation(rng, max_deg):
    """Random rotation, up to +-max_deg about each axis (models the pod sitting differently)."""
    ax, ay, az = np.deg2rad(rng.uniform(-max_deg, max_deg, 3))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


class WindowSampler:
    """Draws random 4 s windows (with targets) from labelled sessions.

    Each augmentation exists to cover a real-world variation the recorded
    sessions under-represent:
      gain x[0.9,1.1]      sensor scale differences, strap tightness
      ankle rotation +-10  pod mounted at a slightly different angle
      hip shift +-50 ms    ankle/hip clock-sync error over BLE vs wired
      Gaussian noise       sensor noise
      drop a source (15%)  BLE drop-outs / hip unplugged: the net must cope with one sensor
      time-stretch +-10%   cadence variety beyond what we walked
    """

    def __init__(self, sessions, ch_mean, ch_std, rng, augment=True, p_drop=0.15):
        self.sessions = [s for s in sessions if s.labels is not None and len(s) > 0]
        if not self.sessions:
            raise ValueError("no labelled sessions to sample from")
        w = np.array([len(s) for s in self.sessions], float)
        self.weights = w / w.sum()   # long sessions are sampled proportionally more
        self.mean, self.std = ch_mean, ch_std
        self.rng = rng
        self.augment = augment
        self.p_drop = p_drop

    def window(self):
        rng = self.rng
        s = self.sessions[rng.choice(len(self.sessions), p=self.weights)]
        n = len(s)
        stretch = rng.uniform(0.9, 1.1) if self.augment else 1.0
        span = WIN * stretch                       # source samples covered by the window
        start = rng.uniform(0, max(0.0, n - span))
        pos = start + np.arange(WIN) * (span / WIN)
        hip_shift = rng.uniform(-5, 5) if self.augment else 0.0   # +-50 ms at 100 Hz

        x = np.empty((WIN, N_CH), np.float32)
        ank_cols = ANKLE_CH + [ANK_PRESENT]
        hip_cols = HIP_CH + [HIP_PRESENT]
        x[:, ank_cols] = _interp_rows(s.x[:, ank_cols], pos)
        x[:, hip_cols] = _interp_rows(s.x[:, hip_cols], pos + hip_shift)
        x[:, [ANK_PRESENT, HIP_PRESENT]] = np.round(x[:, [ANK_PRESENT, HIP_PRESENT]])

        # Labels are true event times, so they follow the stretch but NOT the hip
        # shift (the shift simulates a clock error, not a change in when feet land).
        r_idx, l_idx = s.label_idx()
        scale = WIN / span
        y = make_targets(WIN, (r_idx - start) * scale, (l_idx - start) * scale)

        if self.augment:
            x = self._augment(x)
        return normalise(x, self.mean, self.std), y

    def _augment(self, x):
        rng = self.rng
        x = x.copy()
        ank_on = x[:, ANK_PRESENT:ANK_PRESENT + 1]
        hip_on = x[:, HIP_PRESENT:HIP_PRESENT + 1]

        # Rotation acts on the raw vectors; |a| is recomputed so it stays consistent.
        R = _rotation(rng, 10.0).astype(np.float32)
        x[:, AX:AZ + 1] = x[:, AX:AZ + 1] @ R.T
        x[:, GX:GZ + 1] = x[:, GX:GZ + 1] @ R.T

        # Gain per source. For hip |a|-1 this is an approximation (scaling the
        # deviation from 1 g rather than |a| itself), which is what we want anyway.
        x[:, AX:GZ + 1] *= rng.uniform(0.9, 1.1)
        x[:, HIP_CH] *= rng.uniform(0.9, 1.1)

        noise = np.zeros_like(x)
        noise[:, AX:AZ + 1] = rng.normal(0, 0.01, (WIN, 3))          # g
        noise[:, GX:GZ + 1] = rng.normal(0, 2.0 / GYRO_SCALE, (WIN, 3))  # 2 deg/s
        noise[:, HIP_CH] = rng.normal(0, 0.01, (WIN, 3))
        x[:, ANKLE_CH] += noise[:, ANKLE_CH] * ank_on
        x[:, HIP_CH] += noise[:, HIP_CH] * hip_on
        x[:, ANK_AMAG] = np.linalg.norm(x[:, AX:AZ + 1], axis=1)

        # Drop a whole source: zero its channels and its present flag, exactly what
        # the UNO Q writes into the fused stream when a sensor is missing.
        if rng.random() < self.p_drop:
            if rng.random() < 0.5:
                x[:, ANKLE_CH + [ANK_PRESENT]] = 0.0
            else:
                x[:, HIP_CH + [HIP_PRESENT]] = 0.0
        return x

    def batch(self, size):
        xs, ys = zip(*(self.window() for _ in range(size)))
        return np.stack(xs), np.stack(ys)


def tile_windows(sessions, ch_mean, ch_std):
    """Non-overlapping, un-augmented 4 s windows: a fixed held-out loss set."""
    xs, ys = [], []
    for s in sessions:
        xn = normalise(s.x, ch_mean, ch_std)
        r_idx, l_idx = s.label_idx()
        y = make_targets(len(s), r_idx, l_idx)
        for a in range(0, len(s) - WIN + 1, WIN):
            xs.append(xn[:, a:a + WIN])
            ys.append(y[:, a:a + WIN])
    if not xs:
        return np.zeros((0, N_CH, WIN), np.float32), np.zeros((0, 2, WIN), np.float32)
    return np.stack(xs), np.stack(ys)
