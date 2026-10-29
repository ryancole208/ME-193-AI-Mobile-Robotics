"""`train` and `export`: the StepTCN ensemble, threshold tuning, calibration, reports.

Pipeline (brief section 4):
  1. Split labelled sessions into train / held-out BY SESSION (stratified by
     walk / run / negative). Windows from one session never land on both sides,
     otherwise the held-out score would just measure memorisation of that walk.
  2. Train 5 members, each on a bootstrap resample of the training sessions with
     its own seed. Their disagreement is the "model" part of the uncertainty.
  3. Tune the peak threshold on held-out sessions to minimise count error.
  4. Calibrate r and kappa on held-out sessions (calibration.py).
  5. Fit the Weinberg distance constant on calibration walks.
  6. Write reports/ for the write-up and models/ensemble.pt for `export`.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np

import baseline
import calibration
import data
import labeling
import stepnet_np
from data import FS, WIN

N_MEMBERS = 5
BATCH = 64
LR = 2e-3
POS_WEIGHT = 10.0          # strikes are rare (a few % of samples); weight the positives up
HELDOUT_FRAC = 0.2
PATIENCE = 6               # epochs without held-out improvement before stopping
MATCH_TOL = int(0.060 * FS)  # +-60 ms event matching
THR_GRID = np.round(np.arange(0.20, 0.90 + 1e-9, 0.02), 2)


# ---------------------------------------------------------------- data selection

def load_labelled(sessions_dir, include_review=False):
    used, skipped = [], []
    for s in data.list_sessions(sessions_dir):
        if s.labels is None or s.true_steps is None:
            continue
        if bool(s.labels["use_for_training"]) or include_review:
            used.append(s)
        else:
            skipped.append(s.id)
    return used, skipped


def split_sessions(sessions, frac=HELDOUT_FRAC, seed=0):
    """Stratified session-level split: ~frac of each category held out, at least
    one per category when the category has >= 2 sessions."""
    rng = np.random.default_rng(seed)
    by_cat = {}
    for s in sessions:
        by_cat.setdefault(s.category, []).append(s)
    train, held = [], []
    for cat, ss in sorted(by_cat.items()):
        order = rng.permutation(len(ss))
        if len(ss) >= 2:
            n_hold = min(max(1, int(round(frac * len(ss)))), len(ss) - 1)
        else:
            n_hold = 0
            print(f"  WARNING: only one '{cat}' session; it goes to training and '{cat}' is not evaluated")
        held += [ss[i] for i in order[:n_hold]]
        train += [ss[i] for i in order[n_hold:]]
    return train, held


# ---------------------------------------------------------------- training

def train_member(i, train_sessions, Xh, Yh, mean, std, seed, max_epochs, steps_per_epoch):
    import torch
    from model import StepTCN

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    boot = [train_sessions[j] for j in rng.integers(0, len(train_sessions), len(train_sessions))]
    sampler = data.WindowSampler(boot, mean, std, rng, augment=True)

    model = StepTCN()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT))
    xh, yh = torch.from_numpy(Xh), torch.from_numpy(Yh)

    def held_loss():
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for a in range(0, len(xh), 256):
                tot += lossf(model(xh[a:a + 256]), yh[a:a + 256]).item() * len(xh[a:a + 256])
        return tot / max(len(xh), 1)

    hist = {"train": [], "held": []}
    best, best_state, bad = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            xb, yb = sampler.batch(BATCH)
            loss = lossf(model(torch.from_numpy(xb)), torch.from_numpy(yb))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        hl = held_loss()
        hist["train"].append(float(np.mean(losses)))
        hist["held"].append(hl)
        print(f"    member {i} epoch {epoch + 1:3d}  train {hist['train'][-1]:.4f}  held-out {hl:.4f}")
        if hl < best - 1e-5:
            best, best_state, bad = hl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"    member {i}: early stop (best held-out {best:.4f})")
                break
    model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["held"]) + 1)
    return model.eval(), hist


def session_probs(models, s, mean, std):
    """(M, 2, N) probabilities over a whole session (the net is fully convolutional)."""
    from model import probs
    return probs(models, data.normalise(s.x, mean, std))


def ensemble_events(p, thr):
    """Right/left event indices from the ensemble-mean probabilities."""
    mean = p.mean(axis=0)
    return stepnet_np.pick_peaks(mean[0], thr), stepnet_np.pick_peaks(mean[1], thr)


def tune_thr(held_probs, held_true):
    errs = []
    for thr in THR_GRID:
        e = 0
        for p, t in zip(held_probs, held_true):
            r, l = ensemble_events(p, thr)
            e += abs(len(r) + len(l) - t)
        errs.append(e)
    errs = np.array(errs)
    best = np.flatnonzero(errs == errs.min())
    thr = float(THR_GRID[best[np.argmin(np.abs(THR_GRID[best] - 0.5))]])  # ties -> nearest 0.5
    return thr, errs


def baseline_default(s):
    r, l = baseline.detect(s.x)
    return labeling.drop_gestures(r, s.t, s.gestures), labeling.drop_gestures(l, s.t, s.gestures)


def match_events(pred, true, tol=MATCH_TOL):
    """Greedy one-to-one matching within +-tol samples -> (TP, FP, FN)."""
    pred, true = np.sort(pred), np.sort(true)
    used = np.zeros(len(true), bool)
    tp = 0
    for p in pred:
        j = np.searchsorted(true, p - tol)
        best, bd = -1, tol + 1
        while j < len(true) and true[j] <= p + tol:
            if not used[j] and abs(true[j] - p) < bd:
                best, bd = j, abs(true[j] - p)
            j += 1
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, len(pred) - tp, len(true) - tp


def run_train(cfg, include_review=False, max_epochs=30, steps_per_epoch=50, seed=0):
    import torch

    paths = cfg["paths"]
    models_dir, reports_dir = Path(paths["models"]), Path(paths["reports"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sessions, skipped = load_labelled(paths["sessions"], include_review)
    if skipped:
        print(f"  excluded (needs_review): {', '.join(skipped)}  (use --include-review to keep them)")
    if len(sessions) < 2:
        raise SystemExit("Need at least 2 labelled sessions (run `pull`, `truth`, `label` first).")
    train, held = split_sessions(sessions, seed=seed)
    if not held:
        raise SystemExit("No held-out sessions after the split: record more sessions per category.")
    print(f"  {len(train)} training sessions, {len(held)} held-out: {', '.join(s.id for s in held)}")

    mean, std = data.compute_norm(train)
    Xh, Yh = data.tile_windows(held, mean, std)
    if len(Xh) == 0:
        raise SystemExit("Held-out sessions are shorter than one 4 s window.")

    t0 = time.time()
    models, hists = [], []
    for i in range(N_MEMBERS):
        print(f"  training member {i + 1}/{N_MEMBERS}")
        m, h = train_member(i, train, Xh, Yh, mean, std, seed=1000 * (seed + 1) + i,
                            max_epochs=max_epochs, steps_per_epoch=steps_per_epoch)
        models.append(m)
        hists.append(h)
    print(f"  training took {(time.time() - t0) / 60:.1f} min")

    # ---- threshold + calibration on held-out sessions only
    held_p = [session_probs(models, s, mean, std) for s in held]
    held_true = [s.true_steps for s in held]
    thr, thr_errs = tune_thr(held_p, held_true)
    stats = [calibration.count_stats(p, thr) for p in held_p]
    n_pred = [st[0] for st in stats]
    r = calibration.fit_r(n_pred, held_true)
    kappa = calibration.fit_kappa(n_pred, held_true, [st[1] for st in stats], [st[2] for st in stats], r)
    sig = calibration.sigma_n(n_pred, [st[1] for st in stats], [st[2] for st in stats], kappa, r)
    cov = calibration.coverage(n_pred, held_true, sig)
    calibration.plot_coverage(cov, len(held), reports_dir / "coverage.png")
    print(f"  thr = {thr:.2f}   r = {r:.4f}   kappa = {kappa:.2f}")
    print("  coverage (nominal -> observed): " + ", ".join(f"{k:.0%}->{v:.0%}" for k, v in cov.items()))
    if len(held) < 10:
        print(f"  NOTE: calibrated on only {len(held)} held-out sessions; the intervals are approximate.")

    # ---- distance (uses the labelled steps, not the NN, so all calibration walks count)
    walks = []
    for s in sessions:
        if s.is_calib_walk and s.distance_m:
            if np.mean(s.x[:, data.HIP_PRESENT]) < 0.9:
                print(f"  calibration walk {s.id} skipped: hip sensor missing for >10% of it")
                continue
            r_idx, l_idx = s.label_idx()
            walks.append((s.distance_m, calibration.weinberg_sum(s.x[:, data.HIP_VERT], np.r_[r_idx, l_idx])))
    K, r_L = calibration.fit_weinberg(walks)
    L0_m = float(cfg.get("L0_m", 0.7303))
    if K is None:
        print(f"  no calibration walks: distance will use fixed step length L0 = {L0_m} m, r_L = {r_L}")
    else:
        print(f"  Weinberg K = {K:.4f} from {len(walks)} walk(s), r_L = {r_L:.3f}")

    # ---- reference input for the export check: a busy held-out window
    ref_x = _reference_window(held, mean, std)

    version = "steptcn-" + time.strftime("%Y%m%d-%H%M%S")
    ckpt = {
        "version": version, "prep_hash": data.PREP_HASH,
        "state_dicts": [m.state_dict() for m in models],
        "ch_mean": mean, "ch_std": std, "thr": thr, "kappa": kappa, "r": r,
        "weinberg_K": K, "r_L": r_L, "L0_m": L0_m, "ref_x": ref_x,
        "split": {"train": [s.id for s in train], "held": [s.id for s in held]},
        "histories": hists,
    }
    torch.save(ckpt, models_dir / "ensemble.pt")
    (models_dir / "train_summary.json").write_text(json.dumps({
        "version": version, "thr": thr, "kappa": kappa, "r": r, "weinberg_K": K, "r_L": r_L,
        "coverage": {str(k): v for k, v in cov.items()}, "split": ckpt["split"]}, indent=2))
    print(f"  saved {models_dir / 'ensemble.pt'} ({version}); run `export` next")

    write_reports(cfg, models, hists, train, held, mean, std, thr, thr_errs, kappa, r, cov, K, r_L)


def _reference_window(held, mean, std):
    best, best_n = None, -1
    for s in held:
        if len(s) < WIN:
            continue
        r_idx, l_idx = s.label_idx()
        ev = np.sort(np.r_[r_idx, l_idx])
        a = int(np.clip((ev[len(ev) // 2] if len(ev) else len(s) // 2) - WIN // 2, 0, len(s) - WIN))
        n = int(np.sum((ev >= a) & (ev < a + WIN)))
        if n > best_n:
            best, best_n = data.normalise(s.x[a:a + WIN], mean, std), n
    if best is None:  # no session is long enough; any input works for a numerical check
        best = np.random.default_rng(0).normal(size=(data.N_CH, WIN)).astype(np.float32)
    return best


# ---------------------------------------------------------------- reports

def write_reports(cfg, models, hists, train, held, mean, std, thr, thr_errs, kappa, r, cov, K, r_L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from model import param_count, receptive_field

    rep = Path(cfg["paths"]["reports"])

    # Training curves
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, h in enumerate(hists):
        c = f"C{i}"
        ax.plot(range(1, len(h["train"]) + 1), h["train"], "--", color=c, lw=1)
        ax.plot(range(1, len(h["held"]) + 1), h["held"], "-", color=c, lw=1.5, label=f"member {i}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss (pos_weight 10)")
    ax.set_title("Training (dashed) and held-out (solid) loss")
    ax.legend()
    fig.savefig(rep / "training_curves.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Threshold sweep
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(THR_GRID, thr_errs, "o-", ms=3)
    ax.axvline(thr, color="#d62728", ls="--")
    ax.set_xlabel("peak threshold")
    ax.set_ylabel("total |count error| (held-out)")
    fig.savefig(rep / "threshold.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Per-session counts + event matching
    rows, match = [], {"nn": [0, 0, 0], "baseline": [0, 0, 0]}
    held_ids = {s.id for s in held}
    cat_err = {}
    for s in train + held:
        p = session_probs(models, s, mean, std)
        nr, nl = ensemble_events(p, thr)
        br, bl = baseline_default(s)
        n_nn, n_bl, n_true = len(nr) + len(nl), len(br) + len(bl), s.true_steps
        split = "held-out" if s.id in held_ids else "train"
        rows.append((s.id, split, s.category, n_true, n_nn, n_bl))
        if split == "held-out":
            lr, ll = s.label_idx()
            for key, (pr, pl) in (("nn", (nr, nl)), ("baseline", (br, bl))):
                for pred, lab in ((pr, lr), (pl, ll)):
                    tp, fp, fn = match_events(pred, lab)
                    match[key][0] += tp; match[key][1] += fp; match[key][2] += fn
            cat_err.setdefault(s.category, []).append((n_nn, n_true))

    def pct(n, t):
        return f"{100 * (n - t) / t:+.1f}%" if t else ("0 (ok)" if n == 0 else f"{n} false")

    table = "".join(
        f"<tr class='{sp}'><td>{sid}</td><td>{sp}</td><td>{cat}</td><td>{t}</td><td>{nn}</td><td>{pct(nn, t)}</td>"
        f"<td>{bl}</td><td>{pct(bl, t)}</td></tr>" for sid, sp, cat, t, nn, bl in rows)

    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        return prec, rec

    match_rows = ""
    for key, (tp, fp, fn) in match.items():
        prec, rec = prf(tp, fp, fn)
        match_rows += f"<tr><td>{key}</td><td>{tp}</td><td>{fp}</td><td>{fn}</td><td>{prec:.3f}</td><td>{rec:.3f}</td></tr>"

    # Example probability traces (10 s around the middle of up to 3 held-out sessions with steps)
    trace_imgs = []
    for s in [s for s in held if s.true_steps][:3]:
        p = session_probs(models, s, mean, std).mean(axis=0)
        lr, ll = s.label_idx()
        c = int(np.median(np.r_[lr, ll])) if len(lr) + len(ll) else len(s) // 2
        a, b = max(0, c - 500), min(len(s), c + 500)
        t = np.arange(a, b) / FS
        fig, ax = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
        gz = baseline.gz_dps(s.x)
        ax[0].plot(t, gz[a:b] * baseline.gz_sign(gz), lw=0.8, color="#555")
        ax[0].set_ylabel("ankle gz (deg/s)")
        ax[1].plot(t, p[0, a:b], color="#1f77b4", label="p_right")
        ax[1].plot(t, p[1, a:b], color="#ff7f0e", label="p_left")
        ax[1].axhline(thr, color="#999", ls="--", lw=0.8, label="thr")
        for idx, col in ((lr, "#1f77b4"), (ll, "#ff7f0e")):
            sel = idx[(idx >= a) & (idx < b)]
            ax[1].plot(sel / FS, np.full(len(sel), 1.04), "v", color=col, ms=5)
        ax[1].set_ylim(0, 1.1)
        ax[1].set_ylabel("probability")
        ax[1].set_xlabel(f"time in session {s.id} (s)   (triangles = labels)")
        ax[1].legend(loc="upper right", ncol=3)
        name = f"probs_{s.id}.png"
        fig.savefig(rep / name, dpi=100, bbox_inches="tight")
        plt.close(fig)
        trace_imgs.append(name)

    _network_diagram(rep / "network.png", param_count(), receptive_field())

    # Console summary against the targets in the brief
    print("\n  Held-out count error (target: walking <= 2%, running <= 4%)")
    summary_html = ""
    for cat, target in (("walk", 2.0), ("run", 4.0), ("negative", None)):
        pairs = cat_err.get(cat, [])
        if not pairs:
            line = f"    {cat:8s}: no held-out sessions"
        elif cat == "negative":
            false = sum(n for n, _ in pairs)
            line = f"    negative: {false} false steps over {len(pairs)} session(s)"
        else:
            per = [abs(n - t) / t * 100 for n, t in pairs]
            pooled = abs(sum(n for n, _ in pairs) - sum(t for _, t in pairs)) / sum(t for _, t in pairs) * 100
            verdict = "MET" if np.mean(per) <= target else "not met"
            line = (f"    {cat:8s}: mean |error| {np.mean(per):.2f}% per session, pooled {pooled:.2f}% "
                    f"over {len(pairs)} session(s)  (target {target}%: {verdict})")
        print(line)
        summary_html += f"<li>{line.strip()}</li>"

    cov_txt = ", ".join(f"{k:.0%} &rarr; {v:.0%}" for k, v in cov.items())
    imgs = "".join(f'<img src="{n}">' for n in trace_imgs)
    html = f"""<!doctype html><meta charset="utf-8"><title>StepTCN training report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;line-height:1.45}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 10px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}tr.train{{color:#888}}img{{max-width:100%;display:block;margin:10px 0}}
</style>
<h1>StepTCN training report</h1>
<p>{len(train)} training sessions, {len(held)} held-out sessions (split by session, stratified by activity).
Ensemble of {len(models)} members, {param_count():,} parameters each.</p>
<h2>Held-out results</h2><ul>{summary_html}</ul>
<p>Peak threshold <b>{thr:.2f}</b>, RMS relative error r = <b>{r:.4f}</b>, kappa = <b>{kappa:.2f}</b>.
Coverage (nominal &rarr; observed): {cov_txt}.
{'<b>Calibrated on fewer than 10 sessions: treat the intervals as approximate.</b>' if len(held) < 10 else ''}</p>
<p>Distance: {'Weinberg K = %.4f, r_L = %.3f' % (K, r_L) if K is not None else 'no calibration walks; fixed step length L0, r_L = %.2f' % r_L}.</p>
<h2>Per-session counts</h2>
<p>Grey rows are training sessions (optimistic, shown for completeness). Baseline uses the default thresholds.</p>
<table><tr><th>session</th><th>split</th><th>activity</th><th>true</th><th>NN</th><th>NN err</th><th>baseline</th><th>baseline err</th></tr>{table}</table>
<h2>Event timing on held-out sessions (&plusmn;60 ms)</h2>
<p>Labels came from a tuned baseline, so the baseline's score here is flattering; the NN being close to or above it while
also matching the counts is the meaningful result.</p>
<table><tr><th>detector</th><th>TP</th><th>FP</th><th>FN</th><th>precision</th><th>recall</th></tr>{match_rows}</table>
<h2>Training curves</h2><img src="training_curves.png">
<h2>Threshold sweep</h2><img src="threshold.png">
<h2>Uncertainty calibration</h2><img src="coverage.png" style="max-width:420px">
<h2>Example probability traces</h2>{imgs or '<p>(no held-out walking sessions)</p>'}
<h2>Network</h2><img src="network.png" style="max-width:520px">
"""
    (rep / "index.html").write_text(html, encoding="utf-8")
    print(f"\n  report: {rep / 'index.html'}")


def _network_diagram(path, n_params, rf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = [("input  12 ch x T  (100 Hz)", 0),
              ("Conv1d 12->32, k=5, d=1 + ReLU", 12 * 32 * 5 + 32)]
    for d in (2, 4, 8, 16):
        layers.append((f"x + ReLU(Conv1d 32->32, k=5, d={d})", 32 * 32 * 5 + 32))
    layers.append(("Conv1d 32->2, k=1  -> logits (R, L)", 32 * 2 + 2))
    layers.append(("sigmoid -> p_right, p_left", 0))
    fig, ax = plt.subplots(figsize=(6, 0.7 * len(layers) + 0.8))
    for k, (name, n) in enumerate(layers):
        y = len(layers) - k
        ax.add_patch(plt.Rectangle((0.05, y - 0.35), 0.9, 0.6, fc="#e8eef7" if n else "#f4f4f4", ec="#3366aa"))
        ax.text(0.5, y - 0.05, name + (f"   [{n:,} params]" if n else ""), ha="center", va="center", fontsize=9)
        if k:
            ax.annotate("", (0.5, y + 0.25), (0.5, y + 0.65), arrowprops=dict(arrowstyle="<-", color="#3366aa"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, len(layers) + 0.6)
    ax.axis("off")
    ax.set_title(f"StepTCN: {n_params:,} parameters, receptive field {rf} samples ({rf / FS:.2f} s)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- export

def export(cfg):
    """models/ensemble.pt -> models/model.npz (contract 8.4), verified in numpy first."""
    import torch
    from model import StepTCN, probs, to_npz_dict

    models_dir = Path(cfg["paths"]["models"])
    ck_path = models_dir / "ensemble.pt"
    if not ck_path.exists():
        raise SystemExit(f"{ck_path} not found: run `train` first.")
    ck = torch.load(ck_path, weights_only=False)   # our own file, contains numpy arrays
    if ck["prep_hash"] != data.PREP_HASH:
        raise SystemExit("The checkpoint was trained with a different channel definition (prep_hash). Retrain.")

    models = []
    for sd in ck["state_dicts"]:
        m = StepTCN()
        m.load_state_dict(sd)
        models.append(m.eval())

    d = {"version": np.array(ck["version"]), "prep_hash": np.array(data.PREP_HASH),
         "n_members": np.array(len(models), np.int64)}
    for i, m in enumerate(models):
        d.update(to_npz_dict(m, i))
    d.update(ch_mean=np.asarray(ck["ch_mean"], np.float32), ch_std=np.asarray(ck["ch_std"], np.float32),
             thr=np.float64(ck["thr"]), kappa=np.float64(ck["kappa"]), r=np.float64(ck["r"]),
             L0_m=np.float64(ck["L0_m"]), r_L=np.float64(ck["r_L"]))
    if ck.get("weinberg_K") is not None:
        d["weinberg_K"] = np.float64(ck["weinberg_K"])
    ref_x = np.asarray(ck["ref_x"], np.float32)
    d["ref_x"] = ref_x
    d["ref_y"] = probs(models, ref_x).astype(np.float32)

    # The gate: the numpy forward pass (what the UNO Q runs) must reproduce PyTorch.
    diff = float(np.max(np.abs(stepnet_np.ensemble_probs(d, ref_x) - d["ref_y"])))
    print(f"  numpy vs PyTorch on ref_x: max abs diff {diff:.2e}")
    if not diff < 1e-4:
        raise SystemExit("  FAIL: numpy forward pass does not match PyTorch; model.npz NOT written.")

    out = models_dir / "model.npz"
    np.savez(out, **d)
    ok, results = stepnet_np.check_model(stepnet_np.load_model(out), data.PREP_HASH)  # re-read from disk
    for name, good, detail in results:
        print(f"  [{'OK' if good else 'FAIL'}] {name}: {detail}")
    if not ok:
        raise SystemExit("  model.npz failed its re-load check.")
    print(f"  wrote {out} ({out.stat().st_size / 1024:.0f} KB), version {ck['version']}; run `upload` next")
