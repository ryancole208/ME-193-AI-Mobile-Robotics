"""
Webcam pose tracking with the MediaPipe Tasks PoseLandmarker (VIDEO running mode).

Tracks the dominant-hand wrist, elbow and both shoulders. Inference runs on the
un-flipped frame so MediaPipe's anatomical LEFT/RIGHT labels stay correct; mirroring
is applied afterwards to x-coordinates (and to the preview image).

Outputs (thread-safe):
  * paddle_x_at(t)   -- lateral paddle position on the table (m) near time t
  * direction_at(t)  -- "left" / "center" / "right" from wrist motion over the last N frames
  * snapshot()       -- latest PoseSnapshot incl. a small preview image with landmarks
"""

import threading
import time
import urllib.request
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field

import config

# MediaPipe pose landmark indices (vision.PoseLandmark)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
TRACKED = {
    "right": {"wrist": R_WRIST, "elbow": R_ELBOW},
    "left": {"wrist": L_WRIST, "elbow": L_ELBOW},
}
# Arm/shoulder segments drawn on the preview
_SKELETON = [(L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
             (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST)]


# =============================================================================
# Pure helpers (unit-tested)
# =============================================================================

def mirror_x(x, mirror=config.MIRROR_CAMERA):
    return 1.0 - x if mirror else x


def map_to_table(x_norm, x_range=config.POSE_X_RANGE, table_width=config.TABLE_WIDTH_M):
    """Map a (mirrored) normalised image x to a lateral table position in metres."""
    lo, hi = x_range
    f = (x_norm - lo) / (hi - lo)
    f = min(1.0, max(0.0, f))
    return (f - 0.5) * table_width


class Smoother:
    """Exponential moving average; alpha = weight of the previous value."""

    def __init__(self, alpha=config.POSE_SMOOTHING):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * self.value + (1 - self.alpha) * x
        return self.value

    def reset(self):
        self.value = None


class WristHistory:
    """Time-stamped (mirrored) wrist x samples and paddle positions."""

    def __init__(self, keep_s=config.POSE_HISTORY_S):
        self.keep_s = keep_s
        self._t = deque()
        self._x = deque()       # mirrored normalised wrist x
        self._paddle = deque()  # smoothed paddle position (m)

    def add(self, t, x, paddle_m):
        self._t.append(t)
        self._x.append(x)
        self._paddle.append(paddle_m)
        while self._t and t - self._t[0] > self.keep_s:
            self._t.popleft(); self._x.popleft(); self._paddle.popleft()

    def __len__(self):
        return len(self._t)

    def _index_at(self, t):
        """Index of the last sample at or before t (or the first sample if t is earlier)."""
        i = bisect_right(self._t, t) - 1
        return max(i, 0)

    def paddle_at(self, t, max_age_s=0.5):
        if not self._t:
            return None
        i = self._index_at(t)
        if abs(self._t[i] - t) > max_age_s:
            return None
        return self._paddle[i]

    def direction_at(self, t, n_frames=config.POSE_DIRECTION_FRAMES,
                     threshold=config.POSE_DIRECTION_THRESHOLD):
        """Wrist motion over the n frames ending at time t -> left/center/right."""
        if len(self._t) < 2:
            return "center"
        end = self._index_at(t)
        start = max(0, end - (n_frames - 1))
        dx = self._x[end] - self._x[start]
        if dx > threshold:
            return "right"
        if dx < -threshold:
            return "left"
        return "center"


@dataclass
class PoseSnapshot:
    t: float = 0.0
    tracking: bool = False
    wrist_x: float | None = None        # mirrored, normalised 0..1
    paddle_x: float | None = None       # metres
    points: dict = field(default_factory=dict)  # landmark idx -> (x, y, visibility), mirrored
    preview_rgb: object = None          # numpy HxWx3 RGB preview (with overlay) or None


def extract_points(landmarks, mirror=config.MIRROR_CAMERA):
    """Pull the tracked landmarks out of a MediaPipe landmark list (mirrored x)."""
    pts = {}
    for idx in (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST):
        if idx < len(landmarks):
            lm = landmarks[idx]
            vis = getattr(lm, "visibility", None)
            pts[idx] = (mirror_x(lm.x, mirror), lm.y, 1.0 if vis is None else vis)
    return pts


# =============================================================================
# Tracker (background thread)
# =============================================================================

def ensure_model(path=config.POSE_MODEL_PATH, url=config.POSE_MODEL_URL):
    """Download the PoseLandmarker .task model on first run."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pose] downloading model -> {path}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


class PoseTracker:
    def __init__(self, hand=config.DOMINANT_HAND):
        if hand not in TRACKED:
            raise ValueError(f"DOMINANT_HAND must be 'left' or 'right', got {hand!r}")
        self.hand = hand
        self.status = "idle"
        self.fps = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._history = WristHistory()
        self._smoother = Smoother()
        self._snapshot = PoseSnapshot()

    @property
    def tracking(self):
        return self._snapshot.tracking

    # -- public, thread-safe ----------------------------------------------------
    def snapshot(self):
        with self._lock:
            return self._snapshot

    def paddle_x_at(self, t):
        with self._lock:
            return self._history.paddle_at(t)

    def direction_at(self, t):
        with self._lock:
            return self._history.direction_at(t)

    # -- processing (also used directly by unit tests) ---------------------------
    def process_landmarks(self, t, landmarks, preview_rgb=None):
        """Update state from one frame's landmark list (None/empty = no person)."""
        pts = extract_points(landmarks) if landmarks else {}
        wrist = pts.get(TRACKED[self.hand]["wrist"])
        tracking = wrist is not None and wrist[2] >= config.POSE_MIN_WRIST_VISIBILITY
        with self._lock:
            if tracking:
                paddle = self._smoother.update(map_to_table(wrist[0]))
                self._history.add(t, wrist[0], paddle)
            else:
                self._smoother.reset()
                paddle = None
            self._snapshot = PoseSnapshot(t=t, tracking=tracking,
                                          wrist_x=wrist[0] if tracking else None,
                                          paddle_x=paddle, points=pts, preview_rgb=preview_rgb)
        return self._snapshot

    # -- lifecycle --------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="PoseTracker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.status = "stopped"

    def _run(self):
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            self.status = f"unavailable: {e}"
            return

        try:
            self.status = "loading model"
            model = ensure_model()
        except Exception as e:
            self.status = f"model download failed: {e}"
            print(f"[pose] {self.status}")
            return

        self.status = "opening camera"
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") \
            else cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            self.status = f"no camera at index {config.CAMERA_INDEX}"
            print(f"[pose] {self.status}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        try:
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                self.status = "no person"
                last_ts_ms = -1
                frames, fps_t0 = 0, time.monotonic()
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self.status = "camera read failed"
                        time.sleep(0.05)
                        continue
                    t = time.monotonic()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ts_ms = max(int(t * 1000), last_ts_ms + 1)  # must strictly increase
                    last_ts_ms = ts_ms
                    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
                    landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

                    preview = None
                    if config.SHOW_CAMERA_PREVIEW:
                        preview = self._make_preview(cv2, rgb, landmarks)
                    snap = self.process_landmarks(t, landmarks, preview)
                    self.status = "tracking" if snap.tracking else "no person"

                    frames += 1
                    if t - fps_t0 >= 1.0:
                        self.fps = frames / (t - fps_t0)
                        frames, fps_t0 = 0, t
        except Exception as e:
            self.status = f"error: {e}"
            print(f"[pose] {self.status}")
        finally:
            cap.release()

    def _make_preview(self, cv2, rgb, landmarks):
        h, w = rgb.shape[:2]
        pw = config.PREVIEW_WIDTH
        ph = int(h * pw / w)
        img = cv2.resize(rgb, (pw, ph))
        if config.MIRROR_CAMERA:
            img = cv2.flip(img, 1)
        if landmarks:
            pts = extract_points(landmarks)
            px = {i: (int(x * pw), int(y * ph)) for i, (x, y, v) in pts.items() if v > 0.3}
            for a, b in _SKELETON:
                if a in px and b in px:
                    cv2.line(img, px[a], px[b], (80, 220, 255), 2)
            for i, p in px.items():
                is_wrist = i == TRACKED[self.hand]["wrist"]
                cv2.circle(img, p, 6 if is_wrist else 4, (255, 80, 80) if is_wrist else (255, 255, 255), -1)
        return img
