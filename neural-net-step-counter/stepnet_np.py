"""StepTCN forward pass in plain numpy, plus the peak picker.

This mirrors what the UNO Q runs (it has no PyTorch). `export` runs this on
ref_x and refuses to save a model.npz unless it matches PyTorch to < 1e-4, so a
layout mistake (e.g. transposed weights, wrong padding) is caught on the laptop
instead of showing up as a wrong step count on the device.

Run directly to check a model file:  python stepnet_np.py models/model.npz
"""
import sys

import numpy as np

KERNEL = 5
DILATIONS = (2, 4, 8, 16)
MIN_GAP = 30          # 0.30 s at 100 Hz between same-foot events
REQUIRED = ["version", "prep_hash", "n_members", "ch_mean", "ch_std", "thr", "kappa", "r", "L0_m",
            "ref_x", "ref_y"]


def conv1d(x, w, b, d=1):
    """'same'-padded dilated conv. x (C_in, T), w (C_out, C_in, K) PyTorch layout, b (C_out,)."""
    c_out, _, k = w.shape
    pad = d * (k - 1) // 2
    t = x.shape[1]
    xp = np.pad(x, ((0, 0), (pad, pad)))
    y = np.repeat(b[:, None], t, axis=1).astype(np.float32)
    for j in range(k):                      # one matrix product per kernel tap
        y += w[:, :, j] @ xp[:, j * d: j * d + t]
    return y


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))   # numerically stable for large |z|


def member_logits(m, i, x):
    h = np.maximum(conv1d(x, m[f"m{i}_c0_w"], m[f"m{i}_c0_b"]), 0)
    for j, d in enumerate(DILATIONS):
        h = h + np.maximum(conv1d(h, m[f"m{i}_r{j}_w"], m[f"m{i}_r{j}_b"], d), 0)
    return conv1d(h, m[f"m{i}_out_w"], m[f"m{i}_out_b"])


def ensemble_probs(m, x):
    """(12, T) normalised input -> (n_members, 2, T) probabilities."""
    x = np.asarray(x, np.float32)
    return np.stack([sigmoid(member_logits(m, i, x)) for i in range(int(m["n_members"]))])


def pick_peaks(p, thr, min_gap=MIN_GAP):
    """Event detection on one foot's probability trace.

    A sample is an event if it is a local maximum (>= left neighbour, > right
    neighbour) with p >= thr, and at least min_gap samples after the previous
    accepted event (a refractory period, scanned in time order). This must stay
    identical to the UNO Q's live peak picker, because `thr` is tuned with it.
    """
    p = np.asarray(p)
    if len(p) < 3:
        return np.zeros(0, int)
    cand = np.flatnonzero((p[1:-1] >= thr) & (p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:])) + 1
    out, last = [], -10 ** 9
    for i in cand:
        if i - last >= min_gap:
            out.append(i)
            last = i
    return np.array(out, int)


def load_model(path):
    """model.npz -> dict of numpy arrays (no pickle, so a file can't run code)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_model(m, expected_hash=None, tol=1e-4):
    """Self-check a loaded model.npz. Returns (ok, list of (name, ok, detail))."""
    res = []
    missing = [k for k in REQUIRED if k not in m]
    res.append(("required keys", not missing, "missing: " + ", ".join(missing) if missing else "all present"))
    if missing:
        return False, res
    n = int(m["n_members"])
    shapes_ok = True
    for i in range(n):
        want = {f"m{i}_c0_w": (32, 12, 5), f"m{i}_c0_b": (32,), f"m{i}_out_w": (2, 32, 1), f"m{i}_out_b": (2,)}
        for j in range(4):
            want[f"m{i}_r{j}_w"] = (32, 32, 5)
            want[f"m{i}_r{j}_b"] = (32,)
        for k, shp in want.items():
            if k not in m or tuple(m[k].shape) != shp:
                shapes_ok = False
    res.append(("weight shapes", shapes_ok, f"{n} members"))
    if expected_hash is not None:
        res.append(("prep_hash", str(m["prep_hash"]) == expected_hash, str(m["prep_hash"])[:16] + "..."))
    if shapes_ok:
        y = ensemble_probs(m, m["ref_x"])
        diff = float(np.max(np.abs(y - m["ref_y"])))
        res.append(("numpy vs PyTorch ref_y", diff < tol, f"max abs diff {diff:.2e} (limit {tol:g})"))
    return all(r[1] for r in res), res


if __name__ == "__main__":
    from data import PREP_HASH
    path = sys.argv[1] if len(sys.argv) > 1 else "models/model.npz"
    ok, results = check_model(load_model(path), PREP_HASH)
    for name, good, detail in results:
        print(f"  [{'OK' if good else 'FAIL'}] {name}: {detail}")
    sys.exit(0 if ok else 1)
