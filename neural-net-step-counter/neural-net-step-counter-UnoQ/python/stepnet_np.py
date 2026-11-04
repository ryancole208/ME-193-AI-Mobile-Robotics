"""StepTCN forward pass in numpy (PyTorch is NOT installed on the UNO Q).

Architecture (contract 9.4), per ensemble member:
    h = ReLU(Conv1d(12 -> 32, k=5, d=1)(x))
    for d in (2, 4, 8, 16):  h = h + ReLU(Conv1d(32 -> 32, k=5, d)(h))     # residual blocks
    logits = Conv1d(32 -> 2, k=1)(h);  p = sigmoid(logits)                   # (right, left)
"same" zero padding (pad = d * (k - 1) / 2 on each side), no batch-norm, dropout
only in training (so nothing to do here). Weights use PyTorch's Conv1d layout
(out, in, k). Receptive field: 1 + 4 * (1 + 2 + 4 + 8 + 16) = 125 samples, i.e.
an output depends on +-62 samples (+-0.62 s) around it.

How a dilated conv is computed here: for each of the 5 taps, multiply the
weight slice W[:, :, j] (out x in) with the input shifted by (j - 2) * d, and
sum. That is 5 small matrix products per layer instead of a Python loop over
time, so one member on a 3 s window takes a few ms.

Streaming (StreamInfer): every 0.5 s each member runs on the last 3.0 s of
the grid. Only outputs at least 0.7 s (70 samples, > the 62-sample receptive
half-width) from both window edges are TRUSTED: there the zero padding cannot
reach, so the value is exactly what the network would give on an endless
recording. Trusted outputs are written into the grid's p streams once and are
then FINAL - which is what lets the counter commit events without ever
recounting them in the next overlapping window.

The model file is checked before use (check_model): keys and shapes,
prep_hash (same preprocessing as training), and the stored PyTorch reference
output reproduced to < 1e-4.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from fusion import N_CH, prep_hash

log = logging.getLogger("step.nn")

HIDDEN = 32
K = 5
DILATIONS = (2, 4, 8, 16)
REF_TOL = 1e-4
WIN = 300          # 3.0 s window
TRUST = 70         # 0.7 s from either edge


def conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray, d: int) -> np.ndarray:
    """x (C_in, T), w (C_out, C_in, k), b (C_out,) -> (C_out, T); 'same' zero padding, dilation d."""
    c_out, _, k = w.shape
    T = x.shape[1]
    pad = d * (k - 1) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    y = np.repeat(b[:, None], T, axis=1).astype(x.dtype)
    for j in range(k):
        y += w[:, :, j] @ xp[:, j * d: j * d + T]
    return y


def forward_member(m: dict, x: np.ndarray) -> np.ndarray:
    """x (12, T) normalised -> probabilities (2, T)."""
    h = np.maximum(conv1d(x, m["c0_w"], m["c0_b"], 1), 0)
    for j, d in enumerate(DILATIONS):
        h = h + np.maximum(conv1d(h, m[f"r{j}_w"], m[f"r{j}_b"], d), 0)
    logits = conv1d(h, m["out_w"], m["out_b"], 1)
    return 0.5 * (1.0 + np.tanh(0.5 * logits))      # = sigmoid, but cannot overflow exp() for large |logits|


class Model:
    def __init__(self, z):
        self.version = str(z["version"])
        self.prep_hash = str(z["prep_hash"])
        self.n_members = int(z["n_members"])
        self.members = []
        for i in range(self.n_members):
            m = {"c0_w": z[f"m{i}_c0_w"], "c0_b": z[f"m{i}_c0_b"],
                 "out_w": z[f"m{i}_out_w"], "out_b": z[f"m{i}_out_b"]}
            for j in range(len(DILATIONS)):
                m[f"r{j}_w"], m[f"r{j}_b"] = z[f"m{i}_r{j}_w"], z[f"m{i}_r{j}_b"]
            self.members.append({k: np.asarray(v, dtype=np.float32) for k, v in m.items()})
        self.ch_mean = np.asarray(z["ch_mean"], dtype=np.float32)
        std = np.asarray(z["ch_std"], dtype=np.float32)
        self.ch_std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        self.thr = float(z["thr"])
        self.kappa = float(z["kappa"])
        self.r = float(z["r"])
        self.weinberg_K = float(z["weinberg_K"]) if "weinberg_K" in z else None
        self.r_L = float(z["r_L"]) if "r_L" in z else None
        self.L0_m = float(z["L0_m"]) if "L0_m" in z else None
        self.ref_x = np.asarray(z["ref_x"], dtype=np.float32)
        self.ref_y = np.asarray(z["ref_y"], dtype=np.float32)
        self.n_params = sum(v.size for v in self.members[0].values()) if self.members else 0
        self.last_ms: list = []

    def normalise(self, ch: np.ndarray) -> np.ndarray:
        """grid rows (T, 12) -> network input (12, T). All 12 channels, flags included,
        are normalised with the training mean/std from model.npz."""
        return ((ch - self.ch_mean) / self.ch_std).T.astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (12, T) -> (n_members, 2, T); records the time per member in last_ms."""
        out, ms = [], []
        for m in self.members:
            t0 = time.perf_counter()
            out.append(forward_member(m, x))
            ms.append((time.perf_counter() - t0) * 1000)
        self.last_ms = ms
        return np.stack(out)


def _shape_problems(z) -> list:
    req = {"version": (), "prep_hash": (), "n_members": (), "ch_mean": (N_CH,), "ch_std": (N_CH,),
           "thr": (), "kappa": (), "r": (), "ref_x": (N_CH, 400)}
    probs = [f"missing key {k}" for k in req if k not in z]
    probs += [f"{k} has shape {z[k].shape}, expected {s}" for k, s in req.items()
              if k in z and s and z[k].shape != s]
    if probs:
        return probs
    M = int(z["n_members"])
    if M < 1:
        return [f"n_members is {M}"]
    per = {"c0_w": (HIDDEN, N_CH, K), "c0_b": (HIDDEN,), "out_w": (2, HIDDEN, 1), "out_b": (2,)}
    for j in range(len(DILATIONS)):
        per[f"r{j}_w"], per[f"r{j}_b"] = (HIDDEN, HIDDEN, K), (HIDDEN,)
    for i in range(M):
        for k, s in per.items():
            key = f"m{i}_{k}"
            if key not in z:
                probs.append(f"missing key {key}")
            elif z[key].shape != s:
                probs.append(f"{key} has shape {z[key].shape}, expected {s}")
    if "ref_y" not in z:
        probs.append("missing key ref_y")
    elif z["ref_y"].shape != (M, 2, 400):
        probs.append(f"ref_y has shape {z['ref_y'].shape}, expected {(M, 2, 400)}")
    return probs


def check_model(path) -> tuple:
    """Load and verify a model file. Returns (Model or None, checks dict).
    checks: {name: {"ok": bool, "detail": str}} - also returned by POST /api/model."""
    checks = {}

    def put(name, ok, detail):
        checks[name] = {"ok": bool(ok), "detail": detail}
        return ok

    try:
        with np.load(path, allow_pickle=False) as f:
            z = {k: f[k] for k in f.files}
    except FileNotFoundError:
        put("load", False, f"{path} not found")
        return None, checks
    except Exception as e:
        put("load", False, f"{type(e).__name__}: {e}")
        return None, checks
    probs = _shape_problems(z)
    if not put("load", not probs, "keys and shapes OK" if not probs else "; ".join(probs[:6])):
        return None, checks
    model = Model(z)
    want = prep_hash()
    put("prep_hash", model.prep_hash == want,
        "matches" if model.prep_hash == want else
        f"model {model.prep_hash[:12]}... != app {want[:12]}... (trained with different channels)")
    sane = 0 < model.thr < 1 and model.kappa > 0 and model.r >= 0 and np.all(np.isfinite(model.ch_mean))
    put("params", sane, f"thr {model.thr:.3f}, kappa {model.kappa:.2f}, r {model.r:.3f}, "
        f"{model.n_members} members x {model.n_params} params")
    try:
        t0 = time.perf_counter()
        y = model.forward(model.ref_x)
        ms = (time.perf_counter() - t0) * 1000
        diff = float(np.max(np.abs(y - model.ref_y)))
        put("reference", diff < REF_TOL and np.isfinite(diff),
            f"max |numpy - PyTorch| = {diff:.2e} (limit {REF_TOL:.0e}), {ms:.1f} ms for 4 s x {model.n_members}")
    except Exception as e:
        put("reference", False, f"forward pass failed: {type(e).__name__}: {e}")
    ok = all(c["ok"] for c in checks.values())
    return (model if ok else None), checks


class StreamInfer:
    """Runs the ensemble on the grid every 0.5 s and writes FINAL probabilities."""

    def __init__(self, model: Model):
        self.model = model
        self.final_k = None        # p streams are final for grid indices < final_k
        self.last_ms = 0.0
        self.max_ms = 0.0
        self.calls = 0

    def step(self, grid) -> None:
        end = grid.end
        if grid.base is None:
            return
        start = end - WIN if self.final_k is None else min(end - WIN, self.final_k - TRUST)
        start = max(start, grid.base)
        if end - start < 2 * TRUST + 1:
            return
        k0, ch, _, _ = grid.rows(start, end)
        t0 = time.perf_counter()
        y = self.model.forward(self.model.normalise(ch))       # (M, 2, T)
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.max_ms = max(self.max_ms, self.last_ms)
        self.calls += 1
        if self.last_ms > 100:
            log.warning(f"inference took {self.last_ms:.0f} ms (> 100 ms budget) for {ch.shape[0]} samples")
        a, b = k0 + TRUST, end - TRUST
        if self.final_k is not None:
            a = max(a, self.final_k)
        if b > a:
            i0, i1 = a - grid.base, b - grid.base
            seg = y[:, :, a - k0: b - k0]                       # (M, 2, n)
            grid.pm[i0:i1] = np.transpose(seg, (2, 0, 1))
            grid.p[i0:i1] = seg.mean(axis=0).T
            self.final_k = b
