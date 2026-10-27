"""Evaluate both models in config.json on held-out recordings.

    python3 tools/dev.py evaluate                      # every CSV in data/holdout/
    python3 tools/dev.py evaluate data/holdout/walking-*.csv --config config.json --out reports

Each recording's class comes from its header (RUN_MODE holdout writes it) or
its file name (<class>-*.csv). Writes reports/evaluate.md plus PNGs
(confusion matrices and a Model A feature histogram with the class_ranges).
Runs on the UNO Q (via tools/dev.py) or on a laptop with numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "python"))

from gait.config import load_config  # noqa: E402
from gait.errors import GaitError  # noqa: E402
from gait.evaluate import evaluate, write_report  # noqa: E402
from gait.recording import read_recording  # noqa: E402
from gait.settings import load_settings  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*")
    ap.add_argument("--config", default=str(APP / "config.json"))
    ap.add_argument("--out", default=str(APP / "reports"))
    ap.add_argument("--title", default="Held-out evaluation")
    a = ap.parse_args(argv)
    files = a.csv or sorted(glob.glob(str(APP / "data" / "holdout" / "*.csv")))
    if not files:
        print("While looking for held-out recordings: none found in data/holdout/. Likely fix: record some with "
              "`python3 tools/dev.py mode holdout`, or pass CSV files on the command line.")
        return 1
    try:
        cfg = load_config(Path(a.config))
        settings = load_settings(APP / "setup.yaml")
        recs = [read_recording(Path(f)) for f in files]
        res = evaluate(cfg, recs, leg=settings["leg"], trim_s=settings["trim_s"])
    except GaitError as e:
        print(e)
        return 1
    path = write_report(cfg, res, Path(a.out), a.title)
    print(f"model A: accuracy {res['A']['acc']:.3f}, balanced {res['A']['bal_acc']:.3f}")
    print(f"model B: accuracy {res['B']['acc']:.3f}, balanced {res['B']['bal_acc']:.3f}")
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
