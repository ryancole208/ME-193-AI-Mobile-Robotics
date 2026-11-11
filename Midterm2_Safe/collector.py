"""Burst capture for the /collect page.

Frames are taken from the camera thread's raw (overlay-free) frame on the server,
so the timing doesn't depend on the browser or the network. Each burst gets an
ID that prefixes its filenames, which lets undo and the data report find the
frames of one burst again:
    data/raw/<class_id>/<session_id>/<burst_id>_<nnn>.png
"""
import logging
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import config
from gestures import CLASSES
from preprocess import crop_box, frame_to_raw

log = logging.getLogger(__name__)

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,48}$")


class Collector:
    def __init__(self, camera, data_dir=config.DATA_DIR):
        self.camera = camera
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._state = {"active": False, "captured": 0, "expected": 0}
        # Bursts saved since the server started, newest last. Undo only walks this,
        # so it can never delete data from an earlier run by accident.
        self._history = []

    # ---- public API -------------------------------------------------------

    def start_burst(self, class_id, session_id, seconds, fps):
        """Start a burst in a background thread. Returns (ok, message)."""
        if class_id not in CLASSES:
            return False, f"unknown class {class_id!r}"
        if not SESSION_RE.match(session_id or ""):
            return False, "session ID may only use letters, digits, _ and - (max 48)"
        seconds = float(seconds)
        fps = float(fps)
        if not (0.5 <= seconds <= 60 and 1 <= fps <= 30):
            return False, "duration must be 0.5-60 s and rate 1-30 fps"
        if not self.camera.connected:
            return False, "camera not connected"
        with self._lock:
            if self._state["active"]:
                return False, "a burst is already running"
            burst_id = "b" + time.strftime("%Y%m%d-%H%M%S")
            self._state = {"active": True, "captured": 0,
                           "expected": int(round(seconds * fps)),
                           "class_id": class_id, "session_id": session_id,
                           "burst_id": burst_id, "error": None}
        threading.Thread(target=self._run_burst, name="burst", daemon=True,
                         args=(class_id, session_id, burst_id, seconds, fps)).start()
        return True, burst_id

    def status(self):
        with self._lock:
            s = dict(self._state)
        s["can_undo"] = bool(self._history)
        if self._history:
            last = self._history[-1]
            s["last_burst"] = {k: last[k] for k in ("class_id", "session_id", "burst_id")} | {
                "count": len(last["files"])}
        return s

    def undo_last(self):
        with self._lock:
            if self._state["active"]:
                return False, "wait for the burst to finish"
            if not self._history:
                return False, "nothing to undo (only bursts from this server run can be undone)"
            last = self._history.pop()
        for f in last["files"]:
            f.unlink(missing_ok=True)
        folder = self.data_dir / last["class_id"] / last["session_id"]
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
        log.info("Undo: deleted %d frames of %s (%s / %s)", len(last["files"]),
                 last["burst_id"], last["class_id"], last["session_id"])
        return True, f"deleted {len(last['files'])} frames ({last['class_id']}, {last['session_id']})"

    def counts(self):
        """{class: {session: n}} for everything on disk, plus the session list."""
        table = {c: {} for c in CLASSES}
        sessions = set()
        for c in CLASSES:
            cdir = self.data_dir / c
            if not cdir.is_dir():
                continue
            for sdir in cdir.iterdir():
                if sdir.is_dir():
                    n = sum(1 for _ in sdir.glob("*.png"))
                    if n:
                        table[c][sdir.name] = n
                        sessions.add(sdir.name)
        return {"classes": CLASSES, "sessions": sorted(sessions), "counts": table,
                "target": config.TARGET_PER_CLASS}

    # ---- burst thread -----------------------------------------------------

    def _run_burst(self, class_id, session_id, burst_id, seconds, fps):
        out = self.data_dir / class_id / session_id
        expected = int(round(seconds * fps))
        files = []
        error = None
        try:
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            last_id = -1
            for i in range(expected):
                # Schedule against t0 so slow writes don't stretch the burst.
                delay = t0 + i / fps - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                frame_id, frame = self.camera.get_raw_frame()
                if frame is None:
                    raise RuntimeError("camera disconnected during burst")
                if frame_id == last_id:
                    # Camera slower than the burst rate: wait briefly for a new frame.
                    frame_id, frame = self.camera.wait_frame(last_id, timeout=0.5)
                    if frame is None:
                        raise RuntimeError("camera stopped delivering frames")
                    frame_id, frame = self.camera.get_raw_frame()
                last_id = frame_id
                path = out / f"{burst_id}_{i:03d}.png"
                _imwrite(path, frame_to_raw(frame))
                files.append(path)
                with self._lock:
                    self._state["captured"] = len(files)
        except Exception as e:  # report to the page instead of dying silently
            log.exception("Burst %s failed", burst_id)
            error = str(e)
        with self._lock:
            if files:
                self._history.append({"class_id": class_id, "session_id": session_id,
                                      "burst_id": burst_id, "files": files})
            self._state["active"] = False
            self._state["error"] = error
        log.info("Burst %s: saved %d/%d frames to %s", burst_id, len(files), expected, out)


def _imwrite(path, img):
    # imencode + write_bytes works with any path characters on Windows (imwrite doesn't).
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"PNG encode failed for {path}")
    Path(path).write_bytes(buf.tobytes())


# ---- overlay for the collect stream -----------------------------------------

def draw_collect_overlay(frame):
    """Copy of `frame` with everything outside the model's crop dimmed and outlined.
    (Finger count doesn't depend on direction, so no orientation arrows.)"""
    img = frame.copy()
    h, w = img.shape[:2]
    x0, y0, cw, ch = crop_box(w, h)
    mask = np.ones((h, w), bool)
    mask[y0:y0 + ch, x0:x0 + cw] = False
    img[mask] = (img[mask] * 0.35).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (74, 162, 201), 3)
    fs = max(0.5, min(cw, ch) / 1100)
    _label(img, "model sees this area (spread your fingers)", (x0 + 14, y0 + 30), fs)
    return img


def _label(img, text, org, fs):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, fs, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, font, fs, (80, 230, 255), 2, cv2.LINE_AA)
