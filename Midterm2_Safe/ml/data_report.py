"""Step 3: check the collected data before training.

Prints images per class x session (train vs. validation), flags bad images
(unreadable, wrong size, too dark, too bright) and short bursts, and saves:
    reports/data_report.png       random PREPROCESSED samples (what the model sees), one row per class
    reports/sheet_<class>.jpg     one row per burst, to spot mislabeled bursts by eye

Run from Midterm2_Safe/:  ..\\my_env\\Scripts\\python.exe -m ml.data_report
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from gestures import CLASS_TO_COUNT, CLASSES  # noqa: E402
from preprocess import raw_to_unit  # noqa: E402

DARK_MEAN = 15      # mean gray (whole letterboxed image) below this = lens covered / lights off
BRIGHT_MEAN = 200   # above this = blown out
SHEET_COLS = 8
THUMB = 128


def scan(data_dir=None):
    """{class: {session: {burst: [paths]}}} for every PNG on disk."""
    data_dir = Path(data_dir or config.DATA_DIR)
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(data_dir.glob("*/*/*.png")):
        cls, session = path.parent.parent.name, path.parent.name
        tree[cls][session][path.stem.rsplit("_", 1)[0]].append(path)
    return tree


def is_val(session):
    return session.lower().startswith(config.VAL_SESSION_PREFIX)


def read(path):
    # imdecode works with any path characters on Windows (imread doesn't).
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def write_img(path, img, quality=88):
    ext = Path(path).suffix
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    Path(path).write_bytes(buf.tobytes())


def _caption(img, text):
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def burst_sheet(cls, sessions, out_dir):
    rows = []
    for session, bursts in sorted(sessions.items()):
        for burst, paths in sorted(bursts.items()):
            idx = np.linspace(0, len(paths) - 1, min(SHEET_COLS, len(paths))).astype(int)
            thumbs = [cv2.resize(read(paths[i]), (THUMB, THUMB), interpolation=cv2.INTER_AREA)
                      for i in idx]
            thumbs += [np.zeros((THUMB, THUMB, 3), np.uint8)] * (SHEET_COLS - len(thumbs))
            row = np.hstack(thumbs)
            _caption(row, f"{session}/{burst} ({len(paths)})")
            rows.append(row)
    out = out_dir / f"sheet_{cls}.jpg"
    write_img(out, np.vstack(rows))
    return out


def sample_grid(tree, out_dir, per_class=10, seed=0):
    """Random preprocessed 96x96 samples, one row per class."""
    rng = random.Random(seed)
    s = config.INPUT_SIZE * 2
    rows = []
    for cls in CLASSES:
        paths = [p for b in tree.get(cls, {}).values() for ps in b.values() for p in ps]
        pick = rng.sample(paths, min(per_class, len(paths)))
        tiles = []
        for p in pick:
            unit = raw_to_unit(read(p))
            tile = cv2.resize((unit * 255).astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST)
            tiles.append(tile if tile.ndim == 3 else cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
        tiles += [np.zeros((s, s, 3), np.uint8)] * (per_class - len(tiles))
        row = np.hstack(tiles)
        _caption(row, f"{cls} ({CLASS_TO_COUNT[cls]})")
        rows.append(row)
    out = out_dir / "data_report.png"
    write_img(out, np.vstack(rows))
    return out


def main():
    ap = argparse.ArgumentParser(description="Check collected finger-count data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tree = scan()
    out_dir = config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted({s for c in tree.values() for s in c})
    expected_burst = config.BURST_FPS * config.BURST_SECONDS
    want = (config.RAW_SAVE_SIZE, config.RAW_SAVE_SIZE)
    problems = []

    head = f"{'class':<7}" + "".join(f"{s + (' (val)' if is_val(s) else ''):>14}" for s in sessions)
    print(head + f"{'train':>8}{'val':>6}{'bursts':>8}  status")
    print("-" * len(head + " " * 30))
    for cls in CLASSES:
        sess = tree.get(cls, {})
        n_by = {s: sum(len(p) for p in sess.get(s, {}).values()) for s in sessions}
        n_tr = sum(n for s, n in n_by.items() if not is_val(s))
        n_va = sum(n for s, n in n_by.items() if is_val(s))
        n_bursts = sum(len(b) for b in sess.values())
        for s, bursts in sess.items():
            for burst, paths in bursts.items():
                if len(paths) < expected_burst * 0.75:
                    problems.append(f"{cls}/{s}/{burst}: only {len(paths)} frames")
                for p in paths:
                    img = read(p)
                    if img is None:
                        problems.append(f"{p}: unreadable")
                        continue
                    if img.shape[:2] != want:
                        problems.append(f"{p}: size {img.shape[1]}x{img.shape[0]} (crop changed?)")
                    m = float(img.mean())
                    if m < DARK_MEAN:
                        problems.append(f"{p}: very dark (mean {m:.0f})")
                    elif m > BRIGHT_MEAN:
                        problems.append(f"{p}: very bright (mean {m:.0f})")
        status = []
        if n_tr < config.TARGET_PER_CLASS:
            status.append(f"train needs {config.TARGET_PER_CLASS - n_tr} more")
        if n_va == 0:
            status.append("NO validation data")
        print(f"{cls:<7}" + "".join(f"{n_by[s] or '':>14}" for s in sessions)
              + f"{n_tr:>8}{n_va:>6}{n_bursts:>8}  {', '.join(status) or 'OK'}")
        if sess:
            burst_sheet(cls, sess, out_dir)

    unknown = sorted(set(tree) - set(CLASSES))
    if unknown:
        problems.append(f"folders that are not classes: {', '.join(unknown)}")
    if not any(is_val(s) for s in sessions):
        problems.append(f"no validation session (an ID starting with '{config.VAL_SESSION_PREFIX}'); "
                        "training needs one for the split-by-session rule")
    grid = sample_grid(tree, out_dir, seed=args.seed)
    print(f"\nsaved {grid.relative_to(config.ROOT)} and reports/sheet_<class>.jpg")
    print(f"{len(problems)} problem(s)")
    for p in problems[:40]:
        print("  " + p)
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more")


if __name__ == "__main__":
    main()
