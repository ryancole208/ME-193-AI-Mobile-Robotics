"""
AprilTag (tag16h5) song selection for the bridge.

TagFilter    pure logic: allowed IDs, minimum size, largest tag wins, N passes in a row -> stable,
             nothing seen for TAG_LOST_S -> lost. update() returns the event to send (id or -1).
TagScanner   a thread that runs cv2.aruco on the newest webcam frame at most TAG_RATE_HZ times a
             second. Frames come from PoseTracker's on_frame hook (one camera, two readers), or
             -- with --no-pose -- from its own cv2.VideoCapture. OpenCV releases the GIL while it
             detects, so the pose thread keeps its frame rate.

Tags are read from the RAW camera frame: a mirrored tag doesn't decode.
"""

import math
import threading
import time

import bridge_config as bc


def marker_geometry(corners):
    """(shortest side px, area px^2) of one marker's 4 corners [(x, y), ...] in order."""
    pts = [(float(x), float(y)) for x, y in corners]
    sides = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    area = 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4)))
    return min(sides), area


class TagFilter:
    """Turns raw detections into one-shot tag events. Not thread-safe: one caller (the scanner)."""

    def __init__(self, allowed=None, stable_frames=None, min_side_px=None, lost_s=None):
        self.allowed = frozenset(bc.ALLOWED_TAG_IDS if allowed is None else allowed)
        self.stable_frames = bc.TAG_STABLE_FRAMES if stable_frames is None else stable_frames
        self.min_side_px = bc.TAG_MIN_SIDE_PX if min_side_px is None else min_side_px
        self.lost_s = bc.TAG_LOST_S if lost_s is None else lost_s
        self.current = None          # stable ID, -1 after it was lost, None before the first one
        self._run_id = None          # ID picked in the latest passes...
        self._run = 0                # ...and how many passes in a row
        self._last_seen = None       # when the current stable ID was last detected
        self.last_side_px = None     # shortest side of the latest pick (for the console)

    @property
    def status_id(self):
        """For the status packet: the stable ID, or -1 (none yet / lost)."""
        return -1 if self.current is None else self.current

    def pick(self, detections):
        """detections: [(id, corners)] -> (id, side px) of the largest allowed, big-enough tag, or None."""
        best = None
        for tag_id, corners in detections:
            if tag_id not in self.allowed:
                continue
            side, area = marker_geometry(corners)
            if side < self.min_side_px:
                continue
            if best is None or area > best[0]:
                best = (area, tag_id, side)
        return None if best is None else (best[1], best[2])

    def update(self, t, detections):
        """One detection pass at time t. Returns the ID to send (a newly stable tag, or -1 when the
        stable tag has been gone for lost_s), else None."""
        chosen = self.pick(detections)
        if chosen is None:
            self._run_id, self._run = None, 0
        else:
            tag_id, self.last_side_px = chosen
            self._run = self._run + 1 if tag_id == self._run_id else 1
            self._run_id = tag_id
            if tag_id == self.current:
                self._last_seen = t
            elif self._run >= self.stable_frames:
                self.current, self._last_seen = tag_id, t
                return tag_id
        if self.current not in (None, -1) and t - self._last_seen >= self.lost_s:
            self.current = -1
            return -1
        return None


class TagScanner:
    """on_event(tag_id, tag_filter) runs on the scanner thread -- keep it quick."""

    def __init__(self, on_event, *, own_camera=False, rate_hz=None, tag_filter=None, frame_listener=None):
        self.on_event = on_event
        self.own_camera = own_camera
        self.frame_listener = frame_listener  # own camera only: gets every frame too (the camera view)
        self.rate_hz = bc.TAG_RATE_HZ if rate_hz is None else rate_hz
        self.filter = tag_filter or TagFilter()
        self.status = "idle"
        self.passes = 0
        self.last_detections = (0.0, [])      # (t, [(id, corners)]) of the latest pass, raw frame coords
        self._latest = None                   # (seq, t, bgr frame) from the camera thread
        self._seq = 0
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    # -- camera thread (PoseTracker.on_frame) -----------------------------------
    def on_frame(self, t, frame):
        """Only keeps a reference: cap.read() returns a fresh array every frame."""
        self._seq += 1
        self._latest = (self._seq, t, frame)
        self._new_frame.set()

    # -- lifecycle --------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="TagScanner", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._new_frame.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.status = "stopped"

    # -- detection --------------------------------------------------------------
    @staticmethod
    def make_detector(cv2):
        params = cv2.aruco.DetectorParameters()
        params.errorCorrectionRate = bc.TAG_ERROR_CORRECTION
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        return cv2.aruco.ArucoDetector(dictionary, params)

    @staticmethod
    def detect(cv2, detector, frame):
        """BGR (or gray) frame -> [(id, corners 4x2)]."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)
        if ids is None:
            return []
        return [(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]

    def process(self, cv2, detector, t, frame):
        detections = self.detect(cv2, detector, frame)
        self.last_detections = (t, detections)
        event = self.filter.update(t, detections)
        self.passes += 1
        if event is not None:
            self.on_event(event, self.filter)

    def _run(self):
        try:
            import cv2
            detector = self.make_detector(cv2)
        except Exception as e:                 # no OpenCV / no aruco: the bridge still runs
            self.status = f"unavailable: {e}"
            print(f"[tag] {self.status}")
            return
        if self.own_camera:
            self._run_own_camera(cv2, detector)
        else:
            self._run_shared(cv2, detector)

    def _run_shared(self, cv2, detector):
        self.status = "waiting for camera frames"
        period = 1.0 / self.rate_hz
        done_seq = 0
        while not self._stop.is_set():
            self._new_frame.wait(0.5)
            self._new_frame.clear()
            latest = self._latest
            if latest is None or latest[0] == done_seq:
                continue
            done_seq, t, frame = latest
            t0 = time.monotonic()
            try:
                self.process(cv2, detector, t, frame)
                self.status = "scanning"
            except Exception as e:
                self.status = f"error: {e}"
            self._stop.wait(max(0.0, period - (time.monotonic() - t0)))    # at most rate_hz passes

    def _run_own_camera(self, cv2, detector):
        import config
        self.status = "opening camera"
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") \
            else cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            self.status = f"no camera at index {config.CAMERA_INDEX}"
            print(f"[tag] {self.status}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        period = 1.0 / self.rate_hz
        next_pass = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()             # read every frame, so the newest one is never stale
                if not ok:
                    self.status = "camera read failed"
                    self._stop.wait(0.05)
                    continue
                t = time.monotonic()
                if self.frame_listener is not None:
                    self.frame_listener(t, frame)
                if t < next_pass:
                    continue
                next_pass = t + period
                try:
                    self.process(cv2, detector, t, frame)
                    self.status = "scanning (own camera)"
                except Exception as e:
                    self.status = f"error: {e}"
        finally:
            cap.release()
