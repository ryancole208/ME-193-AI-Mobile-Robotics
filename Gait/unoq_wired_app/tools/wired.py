#!/usr/bin/env python3
"""Helper commands for the wired app. Run on the UNO Q:

  python3 tools/wired.py diag                    RUN_MODE=diag + restart the app (asks first), then follow the log
  python3 tools/wired.py mode setup|holdout      same, for the other one-shot modes
  python3 tools/wired.py mode "replay data/setup/walking.csv"    replay on the real matrix
  python3 tools/wired.py replay FILE.csv [...]   replay offline, print the display commands + a summary
  python3 tools/wired.py evaluate [FILES]        held-out recordings (default data/holdout/*.csv) -> reports/evaluate.md

The board's own Python has no numpy, so replay/evaluate re-run this script inside
the app's own container image (a throwaway `docker run --rm`; nothing is installed).
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
IMAGE = "ghcr.io/arduino/app-bricks/python-apps-base:0.12.0"


def one_shot(mode: str) -> int:
    print(f"This writes RUN_MODE='{mode}' and restarts the wired app (any running App is stopped).")
    if input("Continue? [y/N] ").strip().lower() != "y":
        return 1
    (APP / "RUN_MODE").write_text(mode + "\n")
    rc = subprocess.call(["arduino-app-cli", "app", "restart", str(APP)])
    if rc == 0:
        print("Following the app log (Ctrl+C stops following; the app keeps running).")
        subprocess.call(["arduino-app-cli", "app", "logs", str(APP), "--follow"])
    return rc


def in_container(args: list[str]) -> int:
    """Re-run this script where numpy exists (locally if possible, else in the app image)."""
    try:
        import numpy  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        if not shutil.which("docker"):
            sys.exit("While starting replay/evaluate: no numpy and no docker. Likely fix: run this on the UNO Q, "
                     "or `pip install numpy pyyaml` (+ matplotlib for plots) on a laptop")
        return subprocess.call(["docker", "run", "--rm", "-i", "--user", f"{os.getuid()}:{os.getgid()}",
                                "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp", "-v", f"{APP}:/app", "-w", "/app",
                                "--entrypoint", "python", IMAGE, "tools/wired.py", "--inside", *args])
    return run_inside(args)


def run_inside(args: list[str]) -> int:
    sys.path.insert(0, str(APP / "python"))
    from gait.config import load_config
    from gait.errors import GaitError
    from gait.paths import Paths
    from gait.recording import read_recording
    from gait.settings import load_settings

    try:
        cfg = load_config(APP / "config_wired.json")
        settings = load_settings(APP / "setup.yaml")
    except GaitError as e:
        print(e)
        return 1
    cmd, files = args[0], args[1:]
    if cmd == "replay":
        from gait.display import Display, RecordingSender
        from gait.pipeline import label_name
        from gait.replay import SampleClock, replay_offline

        for f in files:
            rec = read_recording(Path(f))
            clock = SampleClock()
            sender = RecordingSender(clock=clock, echo=lambda m: print(f"  {clock.t:8.2f}s {m}"))
            print(f"== {f} ({rec.duration_s:.0f} s, class={rec.meta.get('class', '?')})")
            ctrl = replay_offline(rec, cfg, settings, Paths(APP), Display(sender, clock=clock))
            for m, key in (("A", "pred_a"), ("B", "pred_b")):
                print(f"   model {m}: {dict(Counter(label_name(getattr(r, key)) for r in ctrl.results if r.valid))}")
            print(f"   windows {ctrl.counts['windows']} (invalid {ctrl.counts['invalid']}); "
                  f"switch gestures {ctrl.counts['switches']}, near-misses {ctrl.counts['near_miss']}")
        return 0
    from gait.evaluate import evaluate, write_report

    files = files or sorted(glob.glob(str(APP / "data" / "holdout" / "*.csv")))
    if not files:
        print("No held-out recordings in data/holdout/. Record some with: python3 tools/wired.py mode holdout")
        return 1
    try:
        res = evaluate(cfg, [read_recording(Path(f)) for f in files], leg=settings["leg"], trim_s=settings["trim_s"])
    except GaitError as e:
        print(e)
        return 1
    path = write_report(cfg, res, APP / "reports")
    for m in ("A", "B"):
        print(f"model {m}: accuracy {res[m]['acc']:.3f}, balanced accuracy {res[m]['bal_acc']:.3f}")
    print(f"report: {path}")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a and a[0] == "--inside":
        return run_inside(a[1:])
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if a[0] == "diag":
        return one_shot("diag")
    if a[0] == "mode" and len(a) > 1:
        return one_shot(" ".join(a[1:]))
    if a[0] in ("replay", "evaluate"):
        if a[0] == "replay" and len(a) < 2:
            print("usage: python3 tools/wired.py replay FILE.csv")
            return 2
        return in_container(a)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
