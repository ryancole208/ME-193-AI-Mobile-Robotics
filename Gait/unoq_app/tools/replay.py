"""Replay a recorded CSV through the live pipeline (no hardware needed).

    python3 tools/dev.py replay data/setup/walking.csv [--config config.json] [--quiet]

Prints every display command the sketch would receive, then a summary:
windows, smoothed labels, gesture switches. Works with setup recordings,
holdout recordings and the raw-*.csv logs that live mode writes.

To watch it on the real matrix instead, use the app itself:
    python3 tools/dev.py mode "replay data/setup/walking.csv"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "python"))

from gait.config import load_config  # noqa: E402
from gait.display import Display, RecordingSender  # noqa: E402
from gait.errors import GaitError  # noqa: E402
from gait.paths import Paths  # noqa: E402
from gait.pipeline import label_name  # noqa: E402
from gait.recording import read_recording  # noqa: E402
from gait.replay import SampleClock, replay_offline  # noqa: E402
from gait.settings import load_settings  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--config", default=str(APP / "config.json"))
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(Path(a.config))
        settings = load_settings(APP / "setup.yaml")
    except GaitError as e:
        print(e)
        return 1
    for path in a.csv:
        rec = read_recording(Path(path))
        clock = SampleClock()
        sender = RecordingSender(clock=clock, echo=None if a.quiet else (lambda m: print(f"  {clock.t:8.2f}s {m}")))
        print(f"== {path} ({rec.duration_s:.0f} s, class={rec.meta.get('class', '?')})")
        ctrl = replay_offline(rec, cfg, settings, Paths(APP), Display(sender, clock=clock))
        sa = Counter(label_name(r.pred_a) for r in ctrl.results if r.valid)
        sb = Counter(label_name(r.pred_b) for r in ctrl.results if r.valid)
        print(f"   windows {ctrl.counts['windows']} (invalid {ctrl.counts['invalid']}, gesture {ctrl.counts['gesture_windows']})")
        print(f"   model A raw predictions: {dict(sa)}")
        print(f"   model B raw predictions: {dict(sb)}")
        print(f"   gestures: switches {ctrl.counts['switches']}, long {ctrl.counts['long']}, near-misses {ctrl.counts['near_miss']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
