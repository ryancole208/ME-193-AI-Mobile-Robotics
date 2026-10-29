r"""Uncertainty calibration for the step count and the distance.

The UNO Q reports N +- 1.96 sigma_N, with

    sigma_N^2 = kappa^2 * (sum c(1-c) + var across members) + (r * N)^2
                \_____ event ____/   \____ model ____/      \ systematic /

  event:      each detected event with confidence c is a coin flip that is
              "real" with probability c, so it contributes Bernoulli variance c(1-c).
  model:      the 5 ensemble members disagree on the count; their spread is the
              model's own uncertainty.
  systematic: errors that grow with the count (e.g. a consistently missed
              shuffle step), measured as the RMS relative error r.
  kappa >= 1: an honesty factor. The first two terms are usually over-confident,
              so we inflate them until 95% of held-out true counts really do fall
              inside the 95% interval.

All of this is fitted on the held-out sessions only. With ~5 held-out sessions
the numbers are rough; we say so in the output rather than pretend otherwise.
"""
import numpy as np

from stepnet_np import pick_peaks

# Two-sided normal quantiles (avoids needing scipy just for norm.ppf).
Z = {0.50: 0.674490, 0.80: 1.281552, 0.90: 1.644854, 0.95: 1.959964}
LEVELS = (0.50, 0.80, 0.90, 0.95)


def count_stats(member_probs, thr):
    """Ensemble count and its variance terms for one session.

    member_probs: (M, 2, T). Events come from the ensemble mean (as on the UNO Q);
    member counts give the model variance.
    """
    mean = member_probs.mean(axis=0)
    n, ev = 0, 0.0
    for foot in range(2):
        pk = pick_peaks(mean[foot], thr)
        c = mean[foot, pk]
        n += len(pk)
        ev += float(np.sum(c * (1 - c)))
    member_counts = [sum(len(pick_peaks(m[f], thr)) for f in range(2)) for m in member_probs]
    mv = float(np.var(member_counts, ddof=1)) if len(member_counts) > 1 else 0.0
    return n, ev, mv


def sigma_n(n, ev, mv, kappa, r):
    return np.sqrt(kappa ** 2 * (np.asarray(ev) + np.asarray(mv)) + (r * np.asarray(n)) ** 2)


def var_split(n, ev, mv, kappa, r):
    """The three variance terms, as the UNO Q shows them in status.var_split."""
    return {"event": kappa ** 2 * ev, "model": kappa ** 2 * mv, "systematic": (r * n) ** 2}


def fit_r(n_pred, n_true):
    """RMS relative count error over held-out sessions with steps (negatives have no relative error)."""
    n_pred, n_true = np.asarray(n_pred, float), np.asarray(n_true, float)
    m = n_true > 0
    if not m.any():
        print("  WARNING: no held-out sessions with steps; using r = 0.05")
        return 0.05
    return float(np.sqrt(np.mean(((n_pred[m] - n_true[m]) / n_true[m]) ** 2)))


def fit_kappa(n_pred, n_true, ev, mv, r, target=0.95, kappa_max=20.0):
    """Smallest kappa >= 1 with >= target of sessions inside N +- 1.96 sigma_N.

    Solved exactly: session i is covered once kappa >= kappa_i, where
    kappa_i^2 (ev+mv) = (|N-T|/1.96)^2 - (rN)^2. Sort the kappa_i and take the
    one that brings coverage to the target.
    """
    n_pred, n_true = np.asarray(n_pred, float), np.asarray(n_true, float)
    ev, mv = np.asarray(ev, float), np.asarray(mv, float)
    need = (np.abs(n_pred - n_true) / Z[0.95]) ** 2 - (r * n_pred) ** 2
    base = ev + mv
    with np.errstate(divide="ignore", invalid="ignore"):
        k_i = np.where(need <= 0, 0.0, np.where(base > 0, np.sqrt(np.maximum(need, 0) / base), np.inf))
    k_sorted = np.sort(k_i)
    idx = int(np.ceil(target * len(k_sorted))) - 1
    kappa = max(1.0, float(k_sorted[max(idx, 0)]))
    if not np.isfinite(kappa) or kappa > kappa_max:
        print(f"  WARNING: kappa would exceed {kappa_max}; capping. The NN error is not explained by "
              "its own uncertainty on some sessions, check those labels.")
        kappa = kappa_max
    return kappa


def coverage(n_pred, n_true, sig):
    """Observed fraction inside the interval at each nominal level."""
    err = np.abs(np.asarray(n_pred, float) - np.asarray(n_true, float))
    sig = np.asarray(sig, float)
    return {lvl: float(np.mean(err <= Z[lvl] * sig)) for lvl in LEVELS}


def plot_coverage(cov, n_sessions, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0.4, 1], [0.4, 1], "--", color="#999", label="perfect calibration")
    ax.plot(list(cov), list(cov.values()), "o-", color="#3366aa", label="observed")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("observed coverage (held-out sessions)")
    ax.set_title(f"Count interval coverage (n = {n_sessions} sessions)")
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- distance

def weinberg_sum(hip_vert, step_idx):
    """sum over consecutive steps of (a_max - a_min)^(1/4) of hip vertical accel.

    Weinberg (2002): stride length ~ K * (vertical bounce)^(1/4). Bigger, faster
    steps bounce the hip more. Summing over steps gives distance / K.
    """
    s = np.sort(np.asarray(step_idx, int))
    total = 0.0
    for a, b in zip(s[:-1], s[1:]):
        seg = hip_vert[a:b + 1]
        total += float(seg.max() - seg.min()) ** 0.25
    return total


def fit_weinberg(walks):
    """walks: list of (distance_m, weinberg_sum). Returns (K, r_L).

    K is pooled over all calibration walks. r_L is the leave-one-out relative
    distance error (fit K without walk i, predict walk i), or 0.05 with < 2 walks.
    """
    walks = [(d, s) for d, s in walks if d and s > 0]
    if not walks:
        return None, 0.05
    D = np.array([w[0] for w in walks])
    S = np.array([w[1] for w in walks])
    K = float(D.sum() / S.sum())
    if len(walks) < 2:
        print("  NOTE: only one calibration walk, r_L defaults to 0.05")
        return K, 0.05
    rel = []
    for i in range(len(walks)):
        k_i = (D.sum() - D[i]) / (S.sum() - S[i])
        rel.append((k_i * S[i] - D[i]) / D[i])
    return K, float(np.sqrt(np.mean(np.square(rel))))
