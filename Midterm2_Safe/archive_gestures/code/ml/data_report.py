"""Step 2: check the collected data before training.

Prints images per class / session / burst, flags bad images (wrong size, too dark,
too bright, unreadable) and bursts that are suspiciously short, and saves one
contact sheet per class (one row per burst) to reports/ so labels can be checked
by eye.

Run from Midterm2_Safe/:  ..\\my_env\\Scripts\\python.exe -m ml.data_report
"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from gestures import CLASSES  # noqa: E402

DARK_MEAN = 25      # mean gray below this = probably lens covered / lights off
BRIGHT_MEAN = 235   # above this = probably blown out
SHEET_COLS = 8      # frames shown per burst on the contact sheet
THUMB_W = 160


def scan(data_dir=config.DATA_DIR):
    """Return {class: {session: {burst: [paths]}}} for every PNG on disk."""
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(Path(data_dir).glob("*/*/*.png")):
        cls, session = path.parent.parent.name, path.parent.name
        burst = path.stem.rsplit("_", 1)[0]
        tree[cls][session][burst].append(path)
    return tree


def read(path):
    # imdecode works with any path characters on Windows (imread doesn't).
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def contact_sheet(cls, sessions, out_dir):
    """One row per burst, SHEET_COLS evenly spaced frames per row."""
    w, h = config.RAW_SAVE_SIZE
    th = int(THUMB_W * h / w)
    rows = []
    for session, bursts in sorted(sessions.items()):
        for burst, paths in sorted(bursts.items()):
            idx = np.linspace(0, len(paths) - 1, min(SHEET_COLS, len(paths))).astype(int)
            thumbs = []
            for i in idx:
                img = read(paths[i])
                img = np.zeros((h, w, 3), np.uint8) if img is None else img
                thumbs.append(cv2.resize(img, (THUMB_W, th), interpolation=cv2.INTER_AREA))
            thumbs += [np.zeros((th, THUMB_W, 3), np.uint8)] * (SHEET_COLS - len(thumbs))
            row = np.hstack(thumbs)
            label = f"{session}/{burst} ({len(paths)})"
            cv2.putText(row, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(row, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            rows.append(row)
    if not rows:
        return None
    sheet = np.vstack(rows)
    out = Path(out_dir) / f"sheet_{cls}.jpg"
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    out.write_bytes(buf.tobytes())
    return out


def main():
    tree = scan()
    out_dir = config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_burst = config.BURST_FPS * config.BURST_SECONDS
    want_hw = (config.RAW_SAVE_SIZE[1], config.RAW_SAVE_SIZE[0])

    unknown = sorted(set(tree) - set(CLASSES))
    problems = []
    total = 0
    print(f"{'class':<15}{'images':>7}{'bursts':>7}{'sessions':>9}{'mean gray':>10}  status")
    print("-" * 62)
    for cls in CLASSES:
        sessions = tree.get(cls, {})
        n = sum(len(p) for b in sessions.values() for p in b.values())
        n_bursts = sum(len(b) for b in sessions.values())
        means = []
        for session, bursts in sessions.items():
            for burst, paths in bursts.items():
                if len(paths) < expected_burst * 0.75:
                    problems.append(f"{cls}/{session}/{burst}: only {len(paths)} frames "
                                    f"(expected {expected_burst})")
                for p in paths:
                    img = read(p)
                    if img is None:
                        problems.append(f"{p}: unreadable")
                        continue
                    if img.shape[:2] != want_hw:
                        problems.append(f"{p}: size {img.shape[1]}x{img.shape[0]}, expected "
                                        f"{want_hw[1]}x{want_hw[0]} (crop changed?)")
                    m = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
                    means.append(m)
                    if m < DARK_MEAN:
                        problems.append(f"{p}: very dark (mean {m:.0f})")
                    elif m > BRIGHT_MEAN:
                        problems.append(f"{p}: very bright (mean {m:.0f})")
        total += n
        status = "OK" if n >= config.TARGET_PER_CLASS else f"need {config.TARGET_PER_CLASS - n} more"
        mean_s = f"{np.mean(means):.0f}" if means else "-"
        print(f"{cls:<15}{n:>7}{n_bursts:>7}{len(sessions):>9}{mean_s:>10}  {status}")
        sheet = contact_sheet(cls, sessions, out_dir)
        if sheet:
            print(f"{'':<15}contact sheet: {sheet.relative_to(config.ROOT)}")
    print("-" * 62)
    print(f"total images: {total}")

    if unknown:
        problems.append(f"folders that are not classes: {', '.join(unknown)}")
    n_sessions = len({s for c in tree.values() for s in c})
    if n_sessions < 2:
        print("\nNote: only 1 session. Validation will be split by burst, which tends to "
              "overestimate accuracy;\na 2nd session (different day / lighting) makes a "
              "more honest test set.")
    print(f"\n{len(problems)} problem(s)")
    for p in problems[:50]:
        print("  " + p)
    if len(problems) > 50:
        print(f"  ... and {len(problems) - 50} more")
    print("\nOpen the contact sheets and check every row shows its class.")


if __name__ == "__main__":
    main()
