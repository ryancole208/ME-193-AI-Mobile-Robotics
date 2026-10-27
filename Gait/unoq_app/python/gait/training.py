"""Turn setup recordings into config.json: windows -> features -> two kNN models.

Steps (all explained in the README, "Design decisions"):
 1. Stand-still recording -> gravity direction (for shank pitch) and the noise
    level of |a| and |g| (for the gesture "still" checks).
 2. Each activity recording: convert units, mirror a left leg into the right-leg
    frame, drop the first and last trim_s seconds, cut 2.56 s windows every
    1.28 s, skip windows with > 5 % missing samples, compute features.
 3. Cap at max_windows_per_class, subsampling evenly through each recording.
 4. Cross-validation that respects time: every recording is cut into
    `cv_folds` consecutive blocks and block b of every recording forms fold b
    (so every fold has every class = stratified, and windows are grouped by
    where they came from). Training windows that overlap a test window in time
    are dropped ("purged"), because 50 %-overlapping neighbours are nearly
    copies of each other and would inflate the score.
 5. Choose k from {1,3,...,11} by CV balanced accuracy (ties -> larger k).
 6. Model B: drop near-constant features, rank the rest by ANOVA F-score and
    keep the top N. Selection is redone inside every CV fold so the score is
    honest.
 7. Fit the final models on all (capped) windows and write config.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .config import DEFAULT_GESTURE, K_CHOICES
from .constants import CLASSES, DEVICE_NAME, SCHEMA_VERSION
from .errors import ConfigError
from .features import MODEL_B_FEATURES, model_a_feature, model_a_feature_name, model_b_features
from .gesture import GestureDetector, GestureParams
from .knn import KNNClassifier, StandardScaler
from .metrics import balanced_accuracy, confusion_matrix, accuracy
from .recording import Recording
from .sensor import SignalConverter
from .windowing import samples_to_dense, window_array

FS = 100


# ---------------------------------------------------------------- signal block


def base_signal(settings: dict) -> dict:
    return {
        "sample_rate_hz": FS,
        "window_s": 2.56,
        "hop_s": 1.28,
        "max_missing_frac": 0.05,
        "accel_lsb_per_g": 2048,
        "gyro_lsb_per_dps": 16.4,
        "leg": settings["leg"],
        "axis_map": {"x": "+x", "y": "+y", "z": "+z"},
    }


def converter_for(signal: dict, leg: str | None = None) -> SignalConverter:
    return SignalConverter(signal["accel_lsb_per_g"], signal["gyro_lsb_per_dps"], signal["axis_map"], leg or signal["leg"])


def dense_physical(rec: Recording, conv: SignalConverter) -> tuple[int, np.ndarray]:
    """Recording -> (start index, (n, 6) array in g / deg/s with NaN rows for gaps)."""
    start, dense = samples_to_dense(rec.index, rec.raw.astype(float))
    ok = ~np.isnan(dense[:, 0])
    out = np.full_like(dense, np.nan)
    if ok.any():
        out[ok] = conv.convert(dense[ok])
    return start, out


def stand_stats(rec: Recording, conv: SignalConverter) -> tuple[list, dict]:
    """Mean gravity direction and noise of |a|, |g| from the stand-still step."""
    _, x = dense_physical(rec, conv)
    x = x[~np.isnan(x[:, 0])]
    if len(x) < 2 * FS:
        raise ConfigError("computing the stand-still statistics", f"only {len(x)} samples were recorded",
                          "keep the ankle node connected for the whole 10 s stand-still step and redo setup")
    g = x[:, 0:3].mean(axis=0)
    norm = float(np.linalg.norm(g))
    if not 0.8 < norm < 1.2:
        raise ConfigError("computing the gravity direction", f"mean |a| while standing is {norm:.2f} g, expected about 1.00",
                          "run diag: a value far from 1 g usually means a byte-order or scaling bug on the Pico W")
    amag = np.linalg.norm(x[:, 0:3], axis=1)
    gmag = np.linalg.norm(x[:, 3:6], axis=1)
    noise = {"amag_std_g": round(float(amag.std()), 5), "gmag_std_dps": round(float(gmag.std()), 4)}
    return [round(float(v), 5) for v in g / norm], noise


# ---------------------------------------------------------------- windows + features


@dataclass
class WindowSet:
    fa: np.ndarray            # (n,) Model A feature
    fb: np.ndarray            # (n, len(MODEL_B_FEATURES))
    y: np.ndarray             # (n,) class index
    rec: np.ndarray           # (n,) recording number
    start: np.ndarray         # (n,) window start (stream sample index)
    rec_names: list = field(default_factory=list)
    skipped_missing: int = 0

    def subset(self, idx) -> "WindowSet":
        return WindowSet(self.fa[idx], self.fb[idx], self.y[idx], self.rec[idx], self.start[idx],
                         self.rec_names, self.skipped_missing)

    def __len__(self) -> int:
        return len(self.y)


def recording_windows(rec: Recording, conv: SignalConverter, signal: dict, trim_s: float):
    """Valid windows (after trimming) of one recording, plus the count skipped for gaps."""
    start, x = dense_physical(rec, conv)
    trim = int(round(trim_s * FS))
    if len(x) <= 2 * trim:
        return [], 0
    x = x[trim: len(x) - trim]
    win = int(round(signal["window_s"] * FS))
    hop = int(round(signal["hop_s"] * FS))
    wins = window_array(x, win, hop, signal["max_missing_frac"], start_index=start + trim)
    valid = [w for w in wins if w.valid]
    return valid, len(wins) - len(valid)


def extract_windows(recordings: list[tuple[int, Recording]], conv: SignalConverter, signal: dict,
                    gravity, feature_a: str, trim_s: float) -> WindowSet:
    fa, fb, y, rid, st, names = [], [], [], [], [], []
    skipped = 0
    for r, (cls, rec) in enumerate(recordings):
        names.append(rec.path.name if rec.path else f"recording{r}")
        wins, sk = recording_windows(rec, conv, signal, trim_s)
        skipped += sk
        for w in wins:
            fa.append(model_a_feature(w.data, feature_a))
            fb.append(model_b_features(w.data, gravity))
            y.append(cls)
            rid.append(r)
            st.append(w.start_index)
    nb = len(MODEL_B_FEATURES)
    return WindowSet(np.array(fa, float), np.array(fb, float).reshape(-1, nb), np.array(y, int),
                     np.array(rid, int), np.array(st, np.int64), names, skipped)


def cap_per_class(ws: WindowSet, cap: int) -> np.ndarray:
    """Indices that keep at most `cap` windows per class, spread evenly through
    each recording (proportional to how many windows each recording has)."""
    keep = []
    for c in np.unique(ws.y):
        idx_c = np.where(ws.y == c)[0]
        if len(idx_c) <= cap:
            keep.extend(idx_c.tolist())
            continue
        recs = np.unique(ws.rec[idx_c])
        counts = np.array([np.sum(ws.rec[idx_c] == r) for r in recs])
        alloc = np.floor(counts / counts.sum() * cap).astype(int)
        for j in np.argsort(-(counts / counts.sum() * cap - alloc))[: cap - alloc.sum()]:
            alloc[j] += 1
        for r, a in zip(recs, alloc):
            idx_r = idx_c[ws.rec[idx_c] == r]
            idx_r = idx_r[np.argsort(ws.start[idx_r])]
            pick = np.unique(np.round(np.linspace(0, len(idx_r) - 1, a)).astype(int)) if a > 0 else []
            keep.extend(idx_r[pick].tolist())
    return np.array(sorted(keep), dtype=int)


# ---------------------------------------------------------------- CV machinery


def assign_folds(ws: WindowSet, n_folds: int) -> np.ndarray:
    """Block b (in time) of every recording -> fold b."""
    fold = np.zeros(len(ws), dtype=int)
    for r in np.unique(ws.rec):
        idx = np.where(ws.rec == r)[0]
        idx = idx[np.argsort(ws.start[idx])]
        for b, part in enumerate(np.array_split(idx, n_folds)):
            fold[part] = b
    return fold


def train_mask_for(ws: WindowSet, fold: np.ndarray, f: int, win: int) -> np.ndarray:
    """Training windows for test fold f: not in f and not overlapping any test window in time."""
    test = fold == f
    train = ~test
    for r in np.unique(ws.rec[test]):
        ts = ws.start[test & (ws.rec == r)]
        cand = np.where(train & (ws.rec == r))[0]
        for i in cand:
            if np.any(np.abs(ws.start[i] - ts) < win):
                train[i] = False
    return train


def near_constant(X: np.ndarray) -> np.ndarray:
    """Boolean mask of features that barely vary (or are not finite)."""
    sd = X.std(axis=0)
    scale = np.abs(X.mean(axis=0)) + 1e-12
    return (~np.all(np.isfinite(X), axis=0)) | (sd < 1e-9) | (sd / scale < 1e-6)


def anova_f(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One-way ANOVA F-score per feature (between-class / within-class variance)."""
    classes = np.unique(y)
    n, k = len(y), len(classes)
    grand = X.mean(axis=0)
    ssb = np.zeros(X.shape[1])
    ssw = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        ssb += len(Xc) * (Xc.mean(axis=0) - grand) ** 2
        ssw += ((Xc - Xc.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ssb / max(k - 1, 1)) / (ssw / max(n - k, 1))
    return np.where(np.isfinite(f), f, np.where(ssb > 0, np.inf, 0.0))


def select_features(X: np.ndarray, y: np.ndarray, top_n: int | None):
    """(kept column indices in rank order, dropped-as-constant indices, F scores)."""
    const = near_constant(X)
    f = anova_f(np.where(np.isfinite(X), X, 0.0), y)
    f[const] = -1
    order = [int(i) for i in np.argsort(-f, kind="stable") if not const[i]]
    if top_n is not None:
        order = order[:top_n]
    return order, [int(i) for i in np.where(const)[0]], f


def predict_all_k(Xtr, ytr, Xte, ks=K_CHOICES) -> tuple[dict, np.ndarray]:
    """Fit a scaler on the training part, then predict the test part for every k at once."""
    knn = KNNClassifier.fit(Xtr, ytr, max(ks), len(CLASSES))
    dist, idx = knn.kneighbors(Xte)
    out = {}
    for k in ks:
        kk = min(k, dist.shape[1])
        out[k] = knn.vote(dist[:, :kk], idx[:, :kk]).argmax(axis=1)
    return out, dist[:, 0]


def cross_validate(ws: WindowSet, which: str, n_folds: int, win: int, top_n: int | None):
    """Out-of-fold predictions for every k. Returns ({k: preds}, fold ids)."""
    fold = assign_folds(ws, n_folds)
    preds = {k: np.full(len(ws), -1) for k in K_CHOICES}
    for f in range(n_folds):
        test = fold == f
        if not test.any():
            continue
        train = train_mask_for(ws, fold, f, win)
        if len(np.unique(ws.y[train])) < 2:
            continue
        if which == "A":
            Xtr, Xte = ws.fa[train, None], ws.fa[test, None]
        else:
            cols, _, _ = select_features(ws.fb[train], ws.y[train], top_n)
            Xtr, Xte = ws.fb[train][:, cols], ws.fb[test][:, cols]
        p, _ = predict_all_k(Xtr, ws.y[train], Xte)
        for k in K_CHOICES:
            preds[k][test] = p[k]
    return preds, fold


def choose_k(y: np.ndarray, preds: dict) -> tuple[int, dict]:
    scores = {k: balanced_accuracy(y, preds[k]) for k in K_CHOICES}
    best = max(scores.values())
    k = max(kk for kk, s in scores.items() if s >= best - 1e-12)  # ties -> larger k (smoother)
    return k, scores


def unsure_distance(knn: KNNClassifier, ws: WindowSet, fold: np.ndarray, win: int, pct: float = 99.0) -> float:
    """99th percentile of each window's distance to its nearest neighbour from
    ANOTHER time block (and not overlapping it) - what a new window from a new
    session typically sees."""
    Xs = knn._Xs
    sq = (Xs ** 2).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] - 2 * Xs @ Xs.T + sq[None, :], 0))  # (n, n) without an (n, n, dims) temp
    same = (ws.rec[:, None] == ws.rec[None, :]) & ((fold[:, None] == fold[None, :]) |
                                                   (np.abs(ws.start[:, None] - ws.start[None, :]) < win))
    d[same] = np.inf
    nn = d.min(axis=1)
    nn = nn[np.isfinite(nn)]
    return round(float(np.percentile(nn, pct)), 4) if len(nn) else None


def sklearn_crosscheck(X, y, k) -> str:
    """Optional: compare our numpy kNN with scikit-learn on the final data."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler as SkScaler
    except ImportError:
        return "scikit-learn not installed in the app container - cross-check skipped (it runs in the tests)"
    Xs = SkScaler().fit_transform(X)
    sk = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(Xs, y)
    ours = KNNClassifier.fit(X, y, k, len(CLASSES))
    rng = np.random.default_rng(0)
    q = X[rng.choice(len(X), size=min(200, len(X)), replace=False)] + rng.normal(0, 1e-3, (min(200, len(X)), X.shape[1]))
    agree = np.mean(sk.predict(SkScaler().fit(X).transform(q)) == ours.predict(q)[0])
    return f"numpy kNN agrees with scikit-learn on {agree * 100:.1f} % of 200 perturbed queries"


# ---------------------------------------------------------------- gesture tuning


def tune_gestures(gesture_rec: Recording | None, still_rec: Recording | None, conv: SignalConverter,
                  noise: dict) -> tuple[dict, list[str]]:
    """Fit tap thresholds to the 5 recorded switch gestures (see README)."""
    g = dict(DEFAULT_GESTURE)
    notes = []
    if gesture_rec is None or gesture_rec.n == 0:
        return g, ["no gesture-check recording: default gesture thresholds kept"]
    lo = 1.3
    if still_rec is not None and still_rec.n:
        _, xs = dense_physical(still_rec, conv)
        amag = np.linalg.norm(xs[~np.isnan(xs[:, 0])][:, 0:3], axis=1)
        lo = max(lo, float(np.percentile(amag, 99.9)))
        notes.append(f"still |a| 99.9th percentile: {lo:.2f} g")

    start, xg = dense_physical(gesture_rec, conv)
    # These timing bounds must stay as loose as the interactive gesture_check's own detector
    # (GestureParams' dataclass defaults) - otherwise taps that were accepted live, and that the
    # wearer now believes work, get re-judged as near-misses here and tuned into a *stricter*
    # final config than what setup actually confirmed.
    permissive = GestureParams(tap_g=max(1.3, lo + 0.1), max_tap_ms=250, min_gap_ms=80, max_gap_ms=500,
                               amag_noise_g=noise["amag_std_g"], gmag_noise_dps=noise["gmag_std_dps"])
    events = run_detector(permissive, start, xg)
    found = [e for e in events if e.kind == "switch"]
    notes.append(f"gesture check: {len(found)} four-tap gestures found with a permissive detector "
                 f"(tap_g {permissive.tap_g:.2f} g); {len(events) - len(found)} near-misses")
    if len(found) < 3:
        notes.append("fewer than 3 clean gestures: default thresholds kept")
        return g, notes
    peaks = np.array([t.peak_g for e in found for t in e.taps])
    gaps = np.array([gp for e in found for gp in e.gaps_ms])
    hi = float(np.percentile(peaks, 20))
    if hi <= lo + 0.2:
        notes.append(f"taps ({hi:.2f} g) are barely above still noise ({lo:.2f} g): default thresholds kept; tap harder")
        return g, notes
    g["tap_g"] = round(float(np.clip(lo + 0.5 * (hi - lo), 1.3, 4.0)), 2)
    # durations at the new threshold
    retuned = GestureParams(tap_g=g["tap_g"], max_tap_ms=250, min_gap_ms=80, max_gap_ms=500,
                            amag_noise_g=noise["amag_std_g"], gmag_noise_dps=noise["gmag_std_dps"])
    durs = [t.duration_ms for e in run_detector(retuned, start, xg) if e.kind == "switch" for t in e.taps]
    if durs:
        g["max_tap_ms"] = int(np.clip(1.5 * max(durs) + 10, 40, 250))
    g["min_gap_ms"] = int(np.clip(0.7 * gaps.min(), 50, 250))
    g["max_gap_ms"] = int(np.clip(1.3 * gaps.max(), 350, 500))
    notes.append(f"tap peaks: 20th pct {hi:.2f} g, median {np.median(peaks):.2f} g; gaps {gaps.min():.0f}-{gaps.max():.0f} ms")
    notes.append(f"tuned: tap_g {g['tap_g']} g, max_tap_ms {g['max_tap_ms']}, gaps {g['min_gap_ms']}-{g['max_gap_ms']} ms")
    return g, notes


def run_detector(params: GestureParams, start: int, x: np.ndarray) -> list:
    """Run a detector over a dense array (NaN rows split the stream)."""
    det = GestureDetector(params)
    events = []
    ok = ~np.isnan(x[:, 0])
    i = 0
    while i < len(x):
        if not ok[i]:
            det.reset()
            i += 1
            continue
        j = i
        while j < len(x) and ok[j]:
            j += 1
        events += det.push(start + i, x[i:j])
        i = j
    return events


# ---------------------------------------------------------------- the whole build


@dataclass
class BuildResult:
    config: dict
    report: dict


def build_config(settings: dict, stand: Recording, activity: dict[int, list[Recording]],
                 gestures: Recording | None, progress=lambda msg: None) -> BuildResult:
    """activity: class index -> recordings. Returns the config and data for the report."""
    signal = base_signal(settings)
    conv = converter_for(signal)
    gravity, noise = stand_stats(stand, conv)
    signal["gravity_unit_vector"] = gravity
    signal["noise"] = noise
    win = int(round(signal["window_s"] * FS))

    missing = [CLASSES[c] for c in range(len(CLASSES)) if not activity.get(c)]
    if missing:
        raise ConfigError("building the models", f"no recording for {missing}", "finish setup for those activities")

    feature_a = model_a_feature_name(settings["model_a"]["axis"], settings["model_a"]["use_abs"])
    recs = [(c, r) for c in sorted(activity) for r in activity[c]]
    progress("computing features")
    ws_all = extract_windows(recs, conv, signal, gravity, feature_a, settings["trim_s"])
    for c in range(len(CLASSES)):
        if np.sum(ws_all.y == c) < settings["cv_folds"]:
            raise ConfigError("building the models",
                              f"only {int(np.sum(ws_all.y == c))} usable windows for {CLASSES[c]}",
                              "record that activity again with the node connected the whole time "
                              "(windows with > 5 % missing samples are skipped)")
    ws = ws_all.subset(cap_per_class(ws_all, settings["max_windows_per_class"]))

    report = {"windows_total": {CLASSES[c]: int(np.sum(ws_all.y == c)) for c in range(len(CLASSES))},
              "windows_used": {CLASSES[c]: int(np.sum(ws.y == c)) for c in range(len(CLASSES))},
              "windows_skipped_missing": ws_all.skipped_missing,
              "recordings": ws.rec_names, "gravity": gravity, "noise": noise}

    models = {}
    for which in ("A", "B"):
        progress(f"cross-validating model {which}")
        top_n = settings["model_b"]["top_n"]
        preds, fold = cross_validate(ws, which, settings["cv_folds"], win, top_n)
        k, scores = choose_k(ws.y, preds)
        if which == "A":
            X = ws.fa[:, None]
            names = [feature_a]
            sel = None
        else:
            cols, const, f = select_features(ws.fb, ws.y, top_n)
            X = ws.fb[:, cols]
            names = [MODEL_B_FEATURES[i] for i in cols]
            sel = {"ranking": "anova_f", "top_n": top_n,
                   "dropped_near_constant": [MODEL_B_FEATURES[i] for i in const],
                   "f_scores": {MODEL_B_FEATURES[i]: round(float(f[i]), 2) for i in cols}}
        knn = KNNClassifier.fit(X, ws.y, k, len(CLASSES))
        ud = unsure_distance(knn, ws, fold, win)
        cm = confusion_matrix(ws.y, preds[k])
        models[which] = {"knn": knn, "names": names, "k": k, "scores": scores, "cm": cm,
                         "bal_acc": balanced_accuracy(ws.y, preds[k]), "acc": accuracy(ws.y, preds[k]),
                         "unsure": ud, "selection": sel, "crosscheck": sklearn_crosscheck(X, ws.y, k)}

    a = models["A"]
    class_ranges = {}
    for c, name in enumerate(CLASSES):
        v = ws.fa[ws.y == c]
        class_ranges[name] = {"min": round(float(v.min()), 3), "median": round(float(np.median(v)), 3),
                              "max": round(float(v.max()), 3)}

    progress("tuning gestures")
    gesture, gnotes = tune_gestures(gestures, (activity.get(0) or [None])[0], conv, noise)
    report["gesture_notes"] = gnotes

    def model_block(m, extra):
        knn = m["knn"]
        return {"type": "knn", **extra, "k": m["k"], "weights": "distance",
                "scaler": {"mean": [round(v, 6) for v in knn.scaler.mean.tolist()],
                           "std": [round(v, 6) for v in knn.scaler.std.tolist()]},
                "X": [[round(v, 6) for v in row] for row in knn.X.tolist()],
                "y": knn.y.tolist(),
                "cv_balanced_accuracy": round(m["bal_acc"], 4),
                "cv_accuracy": round(m["acc"], 4),
                "cv_scores_by_k": {str(k): round(s, 4) for k, s in m["scores"].items()},
                "unsure_distance": m["unsure"]}

    cfg = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "classes": list(CLASSES),
        "signal": signal,
        "model_a": {**model_block(a, {"feature": feature_a}), "class_ranges": class_ranges},
        "model_b": {**model_block(models["B"], {"features": models["B"]["names"]}),
                    "selection": models["B"]["selection"]},
        "display": {"default_model": settings["default_model"], "switch_mode": settings["switch_mode"],
                    "unsure_enabled": settings["unsure_enabled"]},
        "gesture": gesture,
        "link": {"device_name": DEVICE_NAME, "low_battery_mV": settings["low_battery_mV"],
                 "battery_empty_mV": settings["battery_empty_mV"], "battery_full_mV": settings["battery_full_mV"]},
    }
    report["models"] = models
    return BuildResult(cfg, report)


def models_from_config(cfg: dict) -> tuple[KNNClassifier, KNNClassifier]:
    n = len(cfg["classes"])
    return KNNClassifier.from_config(cfg["model_a"], n), KNNClassifier.from_config(cfg["model_b"], n)


__all__ = ["build_config", "models_from_config", "StandardScaler", "tune_gestures", "stand_stats",
           "extract_windows", "cap_per_class", "assign_folds", "train_mask_for", "select_features", "anova_f"]
