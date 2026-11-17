"""Build a 4x4 montage of the 16 silly-walk simulation videos.

Reads sim01.mp4 ... sim16.mp4 from the sim_videos folder next to this script
and writes montage.mp4 into the same folder as this script.
"""
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE / "sim_videos"
OUTPUT = HERE / "montage.mp4"

COLS, ROWS = 4, 4
TILE_W, TILE_H = 480, 320          # each tile; full montage = 1920 x 1280
FPS = 30


def fit_tile(frame):
    """Scale a frame to fit inside one tile, keeping aspect ratio (black bars)."""
    h, w = frame.shape[:2]
    scale = min(TILE_W / w, TILE_H / h)
    nw, nh = int(w * scale), int(h * scale)
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    x, y = (TILE_W - nw) // 2, (TILE_H - nh) // 2
    tile[y:y + nh, x:x + nw] = small
    return tile


def main():
    paths = sorted(VIDEO_DIR.glob("*.mp4"))[:COLS * ROWS]
    if not paths:
        raise SystemExit(f"No .mp4 files found in {VIDEO_DIR}")
    print(f"Found {len(paths)} videos in {VIDEO_DIR}")

    caps = [cv2.VideoCapture(str(p)) for p in paths]
    last = [np.zeros((TILE_H, TILE_W, 3), np.uint8) for _ in caps]
    done = [False] * len(caps)

    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (COLS * TILE_W, ROWS * TILE_H))

    frame_count = 0
    while not all(done):
        for i, cap in enumerate(caps):
            if done[i]:
                continue                     # shorter clips freeze on last frame
            ok, frame = cap.read()
            if ok:
                last[i] = fit_tile(frame)
            else:
                done[i] = True

        tiles = []
        for i in range(COLS * ROWS):
            tile = last[i].copy() if i < len(last) else np.zeros((TILE_H, TILE_W, 3), np.uint8)
            if i < len(paths):
                cv2.putText(tile, paths[i].stem, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)

        grid = np.vstack([np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)])
        writer.write(grid)
        frame_count += 1

    for cap in caps:
        cap.release()
    writer.release()
    print(f"Wrote {frame_count} frames ({frame_count / FPS:.1f} s) to {OUTPUT}")


if __name__ == "__main__":
    main()
