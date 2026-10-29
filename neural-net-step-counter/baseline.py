"""Baseline heel-strike detector (contract 8.5). Same algorithm on the UNO Q.

Why a hand-written detector next to a neural network?
  1. It proposes the training labels (labeling.py sweeps its two thresholds until
     its count matches the hand-tallied truth).
  2. The UNO Q falls back to it when no model is loaded or the model fails its checks.
  3. It is the yardstick the NN has to beat in the report.

Algorithm
  Right (instrumented ankle): during swing the shank rotates fast forward, giving a
    big gz peak; at heel strike that rotation stops abruptly, giving the first
    local minimum after the peak. So: peak above gz_thr, then the first local
    minimum within 0.35 s.
  Left (other foot, no sensor): every footfall jolts the hip, so the left strike
    is the biggest hip-vertical bump (prominence >= hip_prom) between two right
    strikes; if there is none, the midpoint (gait is roughly symmetric).
  Both feet: at least 0.30 s between events of the same foot.

Pure numpy (no scipy), so it ports line-for-line to the UNO Q.
"""
import numpy as np

from data import FS, GZ, GYRO_SCALE, HIP_VERT, ANK_PRESENT

DEFAULT_GZ_THR = 100.0    # deg/s
DEFAULT_HIP_PROM = 0.15   # g
SEARCH_S = 0.35
MIN_GAP_S = 0.30


def gz_dps(x):
    """Ankle gz in deg/s with absent samples zeroed."""
    return x[:, GZ] * GYRO_SCALE * (x[:, ANK_PRESENT] > 0.5)


def gz_sign(gz):
    """+1 or -1 so that mid-swing peaks are positive.

    The swing peak is the largest rotation of the gait cycle, so whichever tail
    of the distribution is bigger (99.5th percentile vs 0.5th) is the swing
    direction. This handles a pod mounted upside-down or mirrored.
    """
    if len(gz) == 0:
        return 1.0
    hi, lo = np.percentile(gz, 99.5), np.percentile(gz, 0.5)
    return -1.0 if abs(lo) > abs(hi) else 1.0


def _apply_min_gap(idx, gap):
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            out.append(i)
            last = i
    return np.array(out, int)


def swing_peaks(gz, gz_thr, fs=FS):
    """Local maxima above gz_thr; within 0.30 s of each other keep only the higher."""
    gap = int(round(MIN_GAP_S * fs))
    if len(gz) < 3:
        return np.zeros(0, int)
    cand = np.flatnonzero((gz[1:-1] > gz_thr) & (gz[1:-1] >= gz[:-2]) & (gz[1:-1] > gz[2:])) + 1
    peaks = []
    for i in cand:
        if peaks and i - peaks[-1] < gap:
            if gz[i] > gz[peaks[-1]]:
                peaks[-1] = i
        else:
            peaks.append(i)
    return np.array(peaks, int)


def detect_right(gz, gz_thr=DEFAULT_GZ_THR, fs=FS):
    """gz (deg/s, already sign-corrected) -> right heel-strike sample indices."""
    win = int(round(SEARCH_S * fs))
    strikes = []
    for pk in swing_peaks(gz, gz_thr, fs):
        for i in range(pk + 1, min(len(gz) - 1, pk + win + 1)):
            if gz[i] < gz[i - 1] and gz[i] <= gz[i + 1]:
                strikes.append(i)
                break
    return _apply_min_gap(strikes, int(round(MIN_GAP_S * fs)))


def peak_prominences(sig, peaks):
    """Topographic prominence (same definition as scipy.signal.peak_prominences):
    height above the higher of the two lowest points you must descend to on each
    side before reaching higher ground (or the edge of the segment)."""
    prom = np.zeros(len(peaks))
    for n, p in enumerate(peaks):
        h = sig[p]
        i = p
        left_min = h
        while i > 0 and sig[i - 1] <= h:
            i -= 1
            left_min = min(left_min, sig[i])
        i = p
        right_min = h
        while i < len(sig) - 1 and sig[i + 1] <= h:
            i += 1
            right_min = min(right_min, sig[i])
        prom[n] = h - max(left_min, right_min)
    return prom


def detect_left(hip_vert, rights, hip_prom=DEFAULT_HIP_PROM, fs=FS):
    """Left strikes between consecutive right strikes (see module docstring)."""
    lefts = []
    for a, b in zip(rights[:-1], rights[1:]):
        seg = hip_vert[a + 1:b]
        choice = (a + b) // 2
        if len(seg) >= 3:
            pk = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])) + 1
            if len(pk):
                pk = pk[peak_prominences(seg, pk) >= hip_prom]
            if len(pk):
                choice = a + 1 + pk[np.argmax(seg[pk])]
        lefts.append(choice)
    return _apply_min_gap(lefts, int(round(MIN_GAP_S * fs)))


def detect(x, gz_thr=DEFAULT_GZ_THR, hip_prom=DEFAULT_HIP_PROM, fs=FS):
    """(N, 12) raw channels -> (right_idx, left_idx)."""
    gz = gz_dps(x)
    gz = gz * gz_sign(gz)
    rights = detect_right(gz, gz_thr, fs)
    return rights, detect_left(x[:, HIP_VERT], rights, hip_prom, fs)


def self_check():
    """Synthetic gait: 20 strides of a clean gz swing peak + heel-strike dip and a
    hip bounce per step. The detector must find ~20 right and ~19 left strikes,
    and still do so when the pod is flipped (gz negated)."""
    from data import N_CH
    stride_s, n_strides = 1.1, 20
    t = np.arange(int((n_strides + 2) * stride_s * FS)) / FS
    gz = np.zeros_like(t)
    for k in range(n_strides):
        tp = (k + 1) * stride_s
        gz += 250 * np.exp(-0.5 * ((t - tp) / 0.06) ** 2) - 80 * np.exp(-0.5 * ((t - tp - 0.15) / 0.04) ** 2)
    x = np.zeros((len(t), N_CH), np.float32)
    x[:, GZ] = gz / GYRO_SCALE
    x[:, HIP_VERT] = 0.3 * np.cos(2 * np.pi * (t - 0.15) / (stride_s / 2))
    x[:, ANK_PRESENT] = 1
    results = []
    for name, flip in (("upright", 1), ("flipped", -1)):
        xx = x.copy()
        xx[:, GZ] *= flip
        r, l = detect(xx)
        ok = abs(len(r) - n_strides) <= 1 and abs(len(l) - (n_strides - 1)) <= 1
        results.append((f"baseline synthetic gait ({name})", ok, f"R={len(r)} L={len(l)}, expected ~{n_strides}/{n_strides - 1}"))
    return results


if __name__ == "__main__":
    for name, ok, detail in self_check():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")
