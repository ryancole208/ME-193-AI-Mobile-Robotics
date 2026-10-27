"""Window features for both models.

All inputs are (256, 6) windows in the right-leg pod frame:
columns ax, ay, az (g) and gx, gy, gz (deg/s). Missing samples are
interpolated first.

"Calibrate once" robustness rules (see README, Design decisions):
* No raw gyro means anywhere: every gyro feature is de-meaned or
  variance/frequency based, so the MPU-6050's drifting gyro offset cancels.
* Accel features are magnitudes, dynamics and spectra. The only orientation
  feature is shank pitch, measured relative to the gravity direction stored
  in config_wired.json during setup.
"""

from __future__ import annotations

import re

import numpy as np

from .windowing import fill_missing

FS = 100.0
BAND = (0.3, 6.0)          # Hz, frequency features use only this band
AC_LAG_S = (0.2, 1.5)      # autocorrelation peak search range (stride periods)
PITCH_SMOOTH = 25          # samples (0.25 s) of low-pass before pitch
NFFT_DOMINANT = 1024       # zero-padding for a finer dominant-frequency estimate

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "amag", "gmag"]
PER_CHANNEL = ["std", "range", "rms", "skew", "kurt", "jerk", "domf", "domshare", "spent", "acpeak", "aclag"]

MODEL_B_FEATURES = (
    [f"{c}_{f}" for c in CHANNELS for f in PER_CHANNEL]
    + ["amag_mean", "pitch_mean", "pitch_range", "corr_ax_ay", "corr_ay_az", "corr_ax_az", "corr_gx_gz", "sma"]
)

_EPS = 1e-12


# ---------------------------------------------------------------- Model A

_MODEL_A_RE = re.compile(r"^max_(abs_)?a([xyz])_g$")


def model_a_feature_name(axis: str = "y", use_abs: bool = False) -> str:
    if axis not in ("x", "y", "z"):
        raise ValueError(f"model_a axis must be x, y or z, got {axis!r}")
    return f"max_{'abs_' if use_abs else ''}a{axis}_g"


def model_a_feature(window: np.ndarray, name: str) -> float:
    """The single Model A feature, e.g. 'max_ay_g' = max(a_y) in g."""
    m = _MODEL_A_RE.match(name)
    if not m:
        raise ValueError(f"unknown Model A feature {name!r} (expected max_ay_g, max_abs_ay_g, ...)")
    col = "xyz".index(m.group(2))
    x = window[:, col]
    x = x[~np.isnan(x)]
    return float(np.max(np.abs(x)) if m.group(1) else np.max(x))


# ---------------------------------------------------------------- building blocks


def skewness(x: np.ndarray) -> float:
    d = x - x.mean()
    s = np.sqrt(np.mean(d ** 2))
    return 0.0 if s < 1e-9 else float(np.mean(d ** 3) / s ** 3)


def kurtosis_excess(x: np.ndarray) -> float:
    d = x - x.mean()
    s2 = np.mean(d ** 2)
    return 0.0 if s2 < 1e-18 else float(np.mean(d ** 4) / s2 ** 2 - 3.0)


def mean_abs_jerk(x: np.ndarray, fs: float = FS) -> float:
    """Mean |dx/dt| in units per second."""
    return float(np.mean(np.abs(np.diff(x))) * fs)


def spectral_features(x: np.ndarray, fs: float = FS, band=BAND) -> tuple[float, float, float]:
    """(dominant frequency Hz, share of band power near it, normalised spectral entropy).

    The signal is de-meaned and Hann-windowed. The dominant frequency uses a
    zero-padded FFT plus parabolic peak interpolation (256 samples alone give
    only 0.39 Hz resolution). The share is the power in the peak bin and its
    two neighbours divided by all power in 0.3-6 Hz. Entropy is the Shannon
    entropy of the in-band power distribution divided by log(#bins), so 0 is
    a pure tone and 1 is flat noise.
    """
    d = (x - x.mean()) * np.hanning(len(x))
    # share and entropy on the plain FFT
    p = np.abs(np.fft.rfft(d)) ** 2
    f = np.fft.rfftfreq(len(d), 1.0 / fs)
    m = (f >= band[0]) & (f <= band[1])
    pb = p[m]
    total = pb.sum()
    if total < _EPS:
        return 0.0, 0.0, 0.0
    k = int(np.argmax(pb))
    share = float(pb[max(k - 1, 0): k + 2].sum() / total)
    q = pb / total
    q = q[q > 0]
    entropy = float(-(q * np.log(q)).sum() / np.log(len(pb))) if len(pb) > 1 else 0.0
    # dominant frequency on the zero-padded FFT
    nfft = max(NFFT_DOMINANT, len(d))
    pz = np.abs(np.fft.rfft(d, n=nfft)) ** 2
    fz = np.fft.rfftfreq(nfft, 1.0 / fs)
    mz = np.where((fz >= band[0]) & (fz <= band[1]))[0]
    j = mz[int(np.argmax(pz[mz]))]
    dom = fz[j]
    if 0 < j < len(pz) - 1:
        a, b, c = np.log(pz[j - 1] + _EPS), np.log(pz[j] + _EPS), np.log(pz[j + 1] + _EPS)
        denom = a - 2 * b + c
        if abs(denom) > _EPS:
            dom = fz[j] + 0.5 * (a - c) / denom * (fz[1] - fz[0])
    return float(dom), share, entropy


def autocorr_peak(x: np.ndarray, fs: float = FS, lag_s=AC_LAG_S) -> tuple[float, float]:
    """(height, lag in s) of the highest autocorrelation peak within lag_s.

    Uses the biased, normalised autocorrelation (r[0] = 1). Returns (0, 0)
    for a flat signal or when no local maximum exists in the range.
    """
    d = x - x.mean()
    var = np.dot(d, d)
    if var < _EPS:
        return 0.0, 0.0
    n = len(d)
    spec = np.fft.rfft(d, n=2 * n)
    r = np.fft.irfft(spec * np.conj(spec))[:n] / var
    lo, hi = int(lag_s[0] * fs), min(int(lag_s[1] * fs), n - 2)
    best_h, best_l = 0.0, 0
    for lag in range(max(lo, 1), hi + 1):
        if r[lag] > r[lag - 1] and r[lag] >= r[lag + 1] and r[lag] > best_h:
            best_h, best_l = float(r[lag]), lag
    return best_h, best_l / fs


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def shank_pitch_deg(acc: np.ndarray, gravity_unit_vector) -> np.ndarray:
    """Per-sample shank pitch relative to the stored standing gravity direction.

    Pitch is the rotation in the sagittal (x-y) plane of the pod, i.e. about
    the lateral z axis. The accel is low-passed (0.25 s moving average) so the
    estimate follows the gravity direction rather than impacts.
    """
    k = np.ones(PITCH_SMOOTH) / PITCH_SMOOTH
    pad = PITCH_SMOOTH // 2
    lp = np.empty_like(acc[:, :2])
    for c in range(2):
        xp = np.pad(acc[:, c], (pad, PITCH_SMOOTH - 1 - pad), mode="edge")
        lp[:, c] = np.convolve(xp, k, mode="valid")
    g = np.asarray(gravity_unit_vector, dtype=float)
    ref = np.arctan2(g[0], g[1])
    ang = np.arctan2(lp[:, 0], lp[:, 1]) - ref
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(ang)


# ---------------------------------------------------------------- Model B


def model_b_features(window: np.ndarray, gravity_unit_vector, fs: float = FS) -> np.ndarray:
    """All Model B candidate features, in MODEL_B_FEATURES order."""
    w = fill_missing(window)
    acc, gyr = w[:, 0:3], w[:, 3:6]
    signals = {
        "ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2],
        "gx": gyr[:, 0], "gy": gyr[:, 1], "gz": gyr[:, 2],
        "amag": np.linalg.norm(acc, axis=1),
        "gmag": np.linalg.norm(gyr, axis=1),
    }
    out = []
    for c in CHANNELS:
        x = signals[c]
        d = x - x.mean()
        dom, share, ent = spectral_features(x, fs)
        ach, acl = autocorr_peak(x, fs)
        out += [
            float(x.std(ddof=1)),
            float(x.max() - x.min()),
            float(np.sqrt(np.mean(d ** 2))),
            skewness(x),
            kurtosis_excess(x),
            mean_abs_jerk(x, fs),
            dom, share, ent, ach, acl,
        ]
    pitch = shank_pitch_deg(acc, gravity_unit_vector)
    da = acc - acc.mean(axis=0)
    out += [
        float(signals["amag"].mean()),
        float(pitch.mean()),
        float(pitch.max() - pitch.min()),
        _corr(acc[:, 0], acc[:, 1]),
        _corr(acc[:, 1], acc[:, 2]),
        _corr(acc[:, 0], acc[:, 2]),
        _corr(gyr[:, 0], gyr[:, 2]),
        float(np.mean(np.abs(da).sum(axis=1))),  # signal magnitude area of the dynamic accel
    ]
    return np.array(out, dtype=float)


def model_b_subset(full: np.ndarray, names: list[str]) -> np.ndarray:
    idx = [MODEL_B_FEATURES.index(n) for n in names]
    return np.asarray(full)[..., idx]
