"""reports/setup_report.md - what setup measured and how well both models did in CV."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .constants import CLASSES
from .metrics import confusion_markdown, per_class_markdown


def setup_report(cfg: dict, report: dict, path: Path) -> Path:
    m = report["models"]
    a, b = m["A"], m["B"]
    L = [f"# Setup report", "", f"Generated {datetime.now().isoformat(timespec='seconds')} "
         f"(config.json created {cfg['created']}), leg: **{cfg['signal']['leg']}**.", ""]
    L += ["## Results", "",
          "| model | features | k | CV balanced accuracy | CV accuracy |",
          "|---|---|---:|---:|---:|",
          f"| A \"1D\" | `{cfg['model_a']['feature']}` | {a['k']} | {a['bal_acc']:.3f} | {a['acc']:.3f} |",
          f"| B \"Full IMU\" | {len(b['names'])} selected | {b['k']} | {b['bal_acc']:.3f} | {b['acc']:.3f} |", "",
          "Cross-validation: each recording is cut into consecutive time blocks; block *b* of every recording "
          "is fold *b*, and training windows overlapping a test window are dropped. See the README.", ""]
    for name, mm in (("A", a), ("B", b)):
        L += [f"## Model {name}", "",
              "CV balanced accuracy by k: " + ", ".join(f"k={k}: {s:.3f}" for k, s in mm["scores"].items()), "",
              f"Chosen k = **{mm['k']}**. Unsure distance (99th pct nearest neighbour, used only if "
              f"unsure_enabled): {mm['unsure']}.", "", "Confusion matrix (CV, windows):", "",
              confusion_markdown(mm["cm"]), "", per_class_markdown(mm["cm"]), "",
              f"Cross-check: {mm['crosscheck']}", ""]
        if name == "A":
            L += ["Feature range per class (g):", "", "| class | min | median | max |", "|---|---:|---:|---:|"]
            for c in CLASSES:
                r = cfg["model_a"]["class_ranges"][c]
                L.append(f"| {c} | {r['min']:.2f} | {r['median']:.2f} | {r['max']:.2f} |")
            L.append("")
        else:
            sel = mm["selection"]
            L += [f"Kept features ({len(mm['names'])}, ranked by ANOVA F-score, top_n = {sel['top_n']}):", ""]
            L += [f"{i + 1}. `{n}` (F = {sel['f_scores'][n]})" for i, n in enumerate(mm["names"])]
            L += ["", f"Dropped as near-constant: {', '.join('`' + n + '`' for n in sel['dropped_near_constant']) or 'none'}", ""]
    L += ["## Data", "",
          "| class | windows recorded | windows used (cap) |", "|---|---:|---:|"]
    L += [f"| {c} | {report['windows_total'][c]} | {report['windows_used'][c]} |" for c in CLASSES]
    L += ["", f"Windows skipped for > 5 % missing samples: {report['windows_skipped_missing']}.",
          f"Recordings: {', '.join(report['recordings'])}.", "",
          f"Gravity direction (stand still): {report['gravity']}; noise: |a| std {report['noise']['amag_std_g']} g, "
          f"|g| std {report['noise']['gmag_std_dps']} deg/s.", "",
          "## Gesture tuning", ""] + [f"- {n}" for n in report["gesture_notes"]] + [
          "", "Final thresholds: " + ", ".join(f"{k} = {v}" for k, v in cfg["gesture"].items()), ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    return path
