"""Count-constrained auto-labelling.

Hand-labelling 2,500 heel strikes is not realistic, but counting them with a
tally app while walking is. So we use the count as a constraint: run the
baseline detector over a grid of its two thresholds and keep the setting whose
right + left total is closest to the true count. If even the best setting is
more than 2% (or 3 steps) off, the timing labels can't be trusted, so the
session is marked needs_review and kept out of training.

The NN then learns to reproduce the baseline's timing where the baseline was
right, and (through augmentation and the extra channels) to generalise past it.
"""
import base64
import io
import json

import numpy as np

import baseline
from data import FS, HIP_VERT

GZ_SWEEP = np.arange(60.0, 200.0 + 1e-9, 10.0)           # deg/s
PROM_SWEEP = np.round(np.arange(0.05, 0.40 + 1e-9, 0.05), 3)  # g


def allowed_error(true_steps):
    """More than 2% or 3 steps (whichever is larger) off -> needs review."""
    return max(0.02 * true_steps, 3)


def drop_gestures(idx, t, gestures):
    """Remove proposals that fall inside a tap-gesture window (the taps look like steps)."""
    if not gestures or len(idx) == 0:
        return idx
    keep = np.ones(len(idx), bool)
    for t0, t1, _ in gestures:
        keep &= ~((t[idx] >= t0) & (t[idx] <= t1))
    return idx[keep]


def label_session(s, include_review=False):
    """Returns a dict ready to save as labels.npz (or None if the session has no truth)."""
    true_n = s.true_steps
    if true_n is None:
        return None
    if true_n == 0:
        # Pure negative: biking, sitting, fidgeting, gestures. No strikes at all.
        r = l = np.zeros(0, int)
        gz_thr, prom = baseline.DEFAULT_GZ_THR, baseline.DEFAULT_HIP_PROM
    else:
        gz = baseline.gz_dps(s.x)
        gz = gz * baseline.gz_sign(gz)
        hip = s.x[:, HIP_VERT]
        best = None
        for gz_thr in GZ_SWEEP:
            rights_all = baseline.detect_right(gz, gz_thr)      # independent of hip_prom
            rights = drop_gestures(rights_all, s.t, s.gestures)
            for prom in PROM_SWEEP:
                lefts = drop_gestures(baseline.detect_left(hip, rights_all, prom), s.t, s.gestures)
                err = abs(len(rights) + len(lefts) - true_n)
                # Ties go to the setting closest to the defaults (the most "typical" one).
                tie = abs(gz_thr - baseline.DEFAULT_GZ_THR) / 100 + abs(prom - baseline.DEFAULT_HIP_PROM) / 0.15
                key = (err, tie)
                if best is None or key < best[0]:
                    best = (key, gz_thr, prom, rights, lefts)
        _, gz_thr, prom, r, l = best
    labelled = len(r) + len(l)
    needs_review = abs(labelled - true_n) > allowed_error(true_n) if true_n > 0 else False
    return {
        "right_idx": r.astype(np.int64), "left_idx": l.astype(np.int64),
        "right_t": s.t[r] if len(r) else np.zeros(0), "left_t": s.t[l] if len(l) else np.zeros(0),
        "gz_thr": np.float64(gz_thr), "hip_prom": np.float64(prom),
        "true_steps": np.int64(true_n), "labelled_steps": np.int64(labelled),
        "needs_review": np.bool_(needs_review),
        "use_for_training": np.bool_((not needs_review) or include_review),
    }


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def write_report(s, lab, reports_dir):
    """reports/label_<id>.html: whole-session signals with labels, plus a 10 s zoom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gz = baseline.gz_dps(s.x)
    gz = gz * baseline.gz_sign(gz)
    hip = s.x[:, HIP_VERT]
    t = np.arange(len(s)) / FS
    r, l = lab["right_idx"], lab["left_idx"]
    imgs = []
    spans = [(0, len(s))]
    if len(r) > 2:  # zoom on 10 s around the middle right strike
        c = int(r[len(r) // 2])
        spans.append((max(0, c - 500), min(len(s), c + 500)))
    for a, b in spans:
        fig, ax = plt.subplots(2, 1, figsize=(16 if b - a > 2000 else 10, 5), sharex=True)
        ax[0].plot(t[a:b], gz[a:b], lw=0.6, color="#3366aa")
        ax[0].axhline(lab["gz_thr"], color="#999", ls="--", lw=0.8)
        rr = r[(r >= a) & (r < b)]
        ax[0].plot(t[rr], gz[rr], "o", ms=3, color="#d62728", label="right strike")
        ax[0].set_ylabel("ankle gz (deg/s)")
        ax[0].legend(loc="upper right")
        ax[1].plot(t[a:b], hip[a:b], lw=0.6, color="#555")
        ll = l[(l >= a) & (l < b)]
        ax[1].plot(t[ll], hip[ll], "o", ms=3, color="#ff7f0e", label="left strike")
        ax[1].set_ylabel("hip vertical (g)")
        ax[1].set_xlabel("time in session (s)")
        ax[1].legend(loc="upper right")
        for g0, g1, kind in s.gestures:
            for axx in ax:
                axx.axvspan(g0 - s.t[0], g1 - s.t[0], color="#aaa", alpha=0.3)
        imgs.append(_fig_to_b64(fig))
        plt.close(fig)

    status = "NEEDS REVIEW (excluded from training)" if lab["needs_review"] else "ok"
    if lab["needs_review"] and lab["use_for_training"]:
        status = "NEEDS REVIEW (included by --include-review)"
    html = f"""<!doctype html><meta charset="utf-8"><title>Labels {s.id}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1400px}}td{{padding:2px 12px}}img{{max-width:100%}}</style>
<h1>Session {s.id}</h1>
<table>
<tr><td>True count (tally)</td><td><b>{int(lab['true_steps'])}</b></td></tr>
<tr><td>Labelled count</td><td><b>{int(lab['labelled_steps'])}</b> (R {len(r)}, L {len(l)})</td></tr>
<tr><td>Chosen gz_thr / hip_prom</td><td>{float(lab['gz_thr']):.0f} deg/s / {float(lab['hip_prom']):.2f} g</td></tr>
<tr><td>Status</td><td>{status}</td></tr>
<tr><td>Category / notes</td><td>{s.category} / {(s.truth or {}).get('notes', '')}</td></tr>
</table>
<p>Grey bands are tap-gesture windows (proposals inside them were dropped).</p>
{''.join(f'<img src="data:image/png;base64,{b}"><br>' for b in imgs)}
"""
    out = reports_dir / f"label_{s.id}.html"
    out.write_text(html, encoding="utf-8")
    return out


def run(sessions, reports_dir, include_review=False):
    """Label every session that has a true count. Returns summary rows."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in sessions:
        lab = label_session(s, include_review)
        if lab is None:
            print(f"  {s.id}: no true count yet, skipped (enter it with `truth` or on the dashboard)")
            continue
        np.savez(s.dir / "labels.npz", **lab)
        (s.dir / "labels.json").write_text(json.dumps({
            "true_steps": int(lab["true_steps"]), "labelled_steps": int(lab["labelled_steps"]),
            "gz_thr": float(lab["gz_thr"]), "hip_prom": float(lab["hip_prom"]),
            "needs_review": bool(lab["needs_review"]), "use_for_training": bool(lab["use_for_training"]),
        }, indent=2))
        write_report(s, lab, reports_dir)
        flag = "  NEEDS REVIEW" if lab["needs_review"] else ""
        print(f"  {s.id} [{s.category:8s}] true {int(lab['true_steps']):4d}  labelled {int(lab['labelled_steps']):4d}  "
              f"gz_thr {float(lab['gz_thr']):.0f}  hip_prom {float(lab['hip_prom']):.2f}{flag}")
        rows.append((s.id, lab))
    return rows
