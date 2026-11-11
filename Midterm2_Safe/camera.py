"""Threaded camera reader shared by every browser client.

One background thread owns the cv2.VideoCapture and keeps only the newest frame,
so slow clients never cause the stream to fall behind real time. If the camera is
missing or unplugged, the thread logs it and keeps retrying instead of crashing.
"""
import logging
import platform
import threading
import time

import cv2

from camera_names import find_camera_index, list_camera_names

log = logging.getLogger(__name__)

# Treat the camera as gone if no good frame arrives for this long.
STALE_AFTER_S = 2.0


def _backend():
    # DirectShow opens faster and more reliably than MSMF on Windows.
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


class Camera:
    def __init__(self, index=None, width=1280, height=720, retry_interval=3.0, process=None,
                 name=None):
        """Pick the camera by `name` (looked up on every open, because Windows can
        reorder indexes) or by a fixed `index` if no name is given.

        `process(frame) -> (frame, info)` runs once per captured frame, so the
        (future) neural net costs the same no matter how many tabs are open."""
        if name is None and index is None:
            raise ValueError("give a camera name or index")
        self.name = name
        self.index = index
        self.width = width
        self.height = height
        self.retry_interval = retry_interval
        self._process = process

        self._lock = threading.Lock()
        # Wakes stream generators as soon as a new frame lands (no busy polling).
        self._new_frame = threading.Condition(self._lock)
        self._frame = None
        self._raw = None          # untouched camera frame, for data collection
        self._frame_id = 0
        self._info = {}
        self._last_ok = 0.0
        self._fps = 0.0
        self._resolution = None

        self._cap = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    # ---- public API -------------------------------------------------------

    def get_frame(self):
        """Return (frame_id, frame). frame_id lets callers skip frames they already sent."""
        with self._lock:
            return self._frame_id, self._frame

    def get_raw_frame(self):
        """Return (frame_id, frame) before process_frame touched it (no overlays)."""
        with self._lock:
            return self._frame_id, self._raw

    def wait_frame(self, last_id, timeout=1.0):
        """Block until a frame newer than `last_id` exists. Returns (id, frame),
        or (last_id, None) on timeout or disconnect."""
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._frame_id != last_id or self._frame is None,
                                     timeout=timeout)
            if self._frame is None or self._frame_id == last_id:
                return last_id, None
            return self._frame_id, self._frame

    @property
    def connected(self):
        with self._lock:
            return self._frame is not None and time.monotonic() - self._last_ok < STALE_AFTER_S

    def status(self):
        connected = self.connected
        with self._lock:
            res = self._resolution
            return {
                "connected": connected,
                "camera_index": self.index,
                "camera_name": self.name,
                "resolution": {"width": res[0], "height": res[1]} if res else None,
                "fps": round(self._fps, 1) if connected else 0.0,
                "info": dict(self._info),
            }

    def release(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._close()
        log.info("Camera %s released", self.index)

    # ---- background thread ------------------------------------------------

    def _open(self):
        if self.name is not None:
            index = find_camera_index(self.name)
            if index is None:
                # Never fall back to another camera: that would quietly record the
                # wrong view into the training data.
                log.warning("No camera named %r (found: %s)", self.name,
                            ", ".join(list_camera_names()) or "none")
                return False
            self.index = index
        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Webcams default to uncompressed YUY2, which USB 2.0 can only carry at ~5 fps
        # for 720p. MJPG gets 30 fps. On DirectShow it must be set *after* the size.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Ask the driver not to queue old frames (ignored by some backends).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        log.info("Camera %d (%s) opened (requested %dx%d)", self.index,
                 self.name or "by index", self.width, self.height)
        return True

    def _close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _run(self):
        last_t = None
        while not self._stop.is_set():
            if self._cap is None:
                if not self._open():
                    log.warning("Camera %s not available; retrying in %.0fs",
                                self.name or self.index, self.retry_interval)
                    self._stop.wait(self.retry_interval)
                    continue
                last_t = None
                # Give a freshly opened camera a full grace period to deliver frames.
                self._last_ok = time.monotonic()

            ok, frame = self._cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                # A few dropped reads are normal; a long gap means it was unplugged.
                if now - self._last_ok > STALE_AFTER_S:
                    log.error("Camera %d disconnected; will retry", self.index)
                    self._close()
                    with self._lock:
                        self._frame = None
                        self._raw = None
                        self._fps = 0.0
                        self._new_frame.notify_all()
                    self._stop.wait(self.retry_interval)
                else:
                    time.sleep(0.01)
                continue

            info = {}
            raw = frame
            if self._process is not None:
                try:
                    # A copy, so overlays drawn by process_frame never reach saved data.
                    frame, info = self._process(frame.copy())
                except Exception:
                    log.exception("process_frame failed; showing raw frame")

            # Exponential moving average keeps the FPS readout steady.
            if last_t is not None:
                inst = 1.0 / max(now - last_t, 1e-6)
                fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
            else:
                fps = self._fps
            last_t = now

            with self._lock:
                if self._frame is None:
                    log.info("Camera %d streaming at %dx%d",
                             self.index, frame.shape[1], frame.shape[0])
                self._frame = frame
                self._raw = raw
                self._frame_id += 1
                self._info = info
                self._last_ok = now
                self._fps = fps
                self._resolution = (frame.shape[1], frame.shape[0])
                self._new_frame.notify_all()
