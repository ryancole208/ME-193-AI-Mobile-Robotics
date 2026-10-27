"""Evaluate both models from config.json on held-out recordings.

Writes reports/evaluate.md (confusion matrices, per-class precision / recall /
F1, accuracy, balanced accuracy) and PNG plots if matplotlib is available,
including a per-class histogram of Model A's feature with config's class_ranges.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .constants import CLASSES, UNSURE
from .errors import ConfigError
from .features import model_a_feature
from .metrics import accuracy, balanced_accuracy, confusion_markdown, confusion_matrix, per_class_markdown
from .pipeline import Classifier
from .recording import Recording
from .training import recording_windows


def recording_class(rec: Recording) -> int:
    name = rec.meta.get("class") or (rec.path.stem.split("-")[0] if rec.path else "")
    if name not in CLASSES:
        raise ConfigError(f"evaluating {rec.path}", f"cannot tell which activity it is (class={name!r})",
                          "name the file <class>-*.csv (e.g. walking-2.csv) or record it with RUN_MODE holdout")
    return CLASSES.index(name)


def evaluate(cfg: dict, recordings: list[Recording], leg: str | None = None, trim_s: float = 5.0) -> dict:
    clf = Classifier(cfg, leg=leg)
    y, pa, pb, fa = [], [], [], []
    per_rec = []
    for rec in recordings:
        c = recording_class(rec)
        wins, skipped = recording_windows(rec, clf.conv, cfg["signal"], trim_s)
        n0 = len(y)
        for w in wins:
            r = clf.classify(w)
            y.append(c)
            pa.append(r.pred_a)
            pb.append(r.pred_b)
            fa.append(model_a_feature(w.data, cfg["model_a"]["feature"]))
        per_rec.append({"file": rec.path.name if rec.path else "?", "class": CLASSES[c], "windows": len(y) - n0,
                        "skipped": skipped})
    y, pa, pb, fa = (np.array(v) for v in (y, pa, pb, fa))
    out = {"y": y, "feature_a": fa, "recordings": per_rec}
    for name, p in (("A", pa), ("B", pb)):
        out[name] = {"pred": p, "cm": confusion_matrix(y, p), "acc": accuracy(y, p),
                     "bal_acc": balanced_accuracy(y, p), "unsure": int(np.sum(p == UNSURE))}
    return out


def write_report(cfg: dict, res: dict, out_dir: Path, title: str = "Held-out evaluation") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = make_plots(cfg, res, out_dir)
    L = [f"# {title}", "", f"{datetime.now().isoformat(timespec='seconds')}, config.json created {cfg['created']}.", "",
         "| model | accuracy | balanced accuracy | CV balanced accuracy (setup) | unsure windows |",
         "|---|---:|---:|---:|---:|"]
    for name, key in (("A \"1D\"", "A"), ("B \"Full IMU\"", "B")):
        m = res[key]
        cv = cfg["model_a" if key == "A" else "model_b"]["cv_balanced_accuracy"]
        L.append(f"| {name} | {m['acc']:.3f} | {m['bal_acc']:.3f} | {cv:.3f} | {m['unsure']} |")
    L += ["", "## Recordings", "", "| file | class | windows | skipped (gaps) |", "|---|---|---:|---:|"]
    L += [f"| {r['file']} | {r['class']} | {r['windows']} | {r['skipped']} |" for r in res["recordings"]]
    for key in ("A", "B"):
        L += ["", f"## Model {key}", "", confusion_markdown(res[key]["cm"]), "", per_class_markdown(res[key]["cm"]), ""]
        if f"cm_{key}" in pngs:
            L += [f"![Model {key} confusion matrix]({pngs[f'cm_{key}'].name})", ""]
    L += ["## Model A feature by class", "", f"Feature `{cfg['model_a']['feature']}`; the dashed lines are the "
          "setup class_ranges (min / median / max).", ""]
    if "hist" in pngs:
        L += [f"![Model A feature histogram]({pngs['hist'].name})", ""]
    if not pngs:
        L += ["(matplotlib not installed here, so no PNG plots; run `python3 tools/dev.py evaluate` for plots)", ""]
    path = out_dir / "evaluate.md"
    path.write_text("\n".join(L))
    return path


def make_plots(cfg: dict, res: dict, out_dir: Path) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    pngs = {}
    for key in ("A", "B"):
        cm = res[key]["cm"]
        fig, ax = plt.subplots(figsize=(4.6, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Model {key}: balanced acc {res[key]['bal_acc']:.2f}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        p = out_dir / f"evaluate_cm_{key}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        pngs[f"cm_{key}"] = p
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fa, y = res["feature_a"], res["y"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    bins = np.linspace(min(fa.min(), 0.8) if len(fa) else 0, max(fa.max(), 1.5) if len(fa) else 2, 40)
    for c, name in enumerate(CLASSES):
        v = fa[y == c]
        if len(v):
            ax.hist(v, bins=bins, alpha=0.55, label=name, color=colors[c])
        r = cfg["model_a"].get("class_ranges", {}).get(name)
        if r:
            for k in ("min", "median", "max"):
                ax.axvline(r[k], color=colors[c], ls="--", lw=1)
    ax.set_xlabel(f"{cfg['model_a']['feature']} (g)")
    ax.set_ylabel("windows")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "evaluate_model_a_hist.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    pngs["hist"] = p
    return pngs
