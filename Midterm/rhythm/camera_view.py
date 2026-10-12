"""
Camera view for the bridge: an OpenCV window showing what the webcam sees, to check that the
pose and the AprilTags are picked up.

  * the image is mirrored like the other games' camera preview (MIRROR_CAMERA);
  * pose: the same overlay as ../pose_tracker.py's preview -- shoulder / elbow / wrist lines
    (light blue), joints (white), the tracked wrist (red);
  * tags: each detection's outline and ID -- green = allowed (thick + "STABLE" when it is the
    stable tag), red = ignored, with the reason (not allowed / too small);
  * status lines in a dark band at the bottom: pose status, paddle x, the stable tag.

Runs on its own thread (all OpenCV window calls stay on it), so drawing never delays the
bridge's packets. Frames arrive through on_frame (PoseTracker's hook, or the tag scanner's own
camera). Closing the window (or Esc / Q in it) only closes the view; the bridge keeps running.
"""

import threading
import time

import bridge_config as bc
from tags import marker_geometry

# BGR, matching pose_tracker._make_preview (which draws in RGB)
SKELETON_COLOR = (255, 220, 80)
JOINT_COLOR = (255, 255, 255)
WRIST_COLOR = (80, 80, 255)
TAG_OK_COLOR = (90, 200, 90)
TAG_BAD_COLOR = (60, 60, 230)
TEXT_COLOR = (240, 240, 240)
DETECTION_MAX_AGE_S = 0.5            # tag outlines older than this are not drawn


class CameraView:
    def __init__(self, pose=None, tags=None, *, fps=None, width=None, title=None):
        import config
        self.config = config
        self.pose = pose
        self.tags = tags
        self.fps = bc.DISPLAY_FPS if fps is None else fps
        self.width = bc.DISPLAY_WIDTH if width is None else width
        self.title = title or bc.DISPLAY_TITLE
        self.status = "idle"
        self._latest = None              # (seq, t, bgr frame)
        self._seq = 0
        self._stop = threading.Event()
        self._thread = None

    # -- camera thread ----------------------------------------------------------
    def on_frame(self, t, frame):
        self._seq += 1
        self._latest = (self._seq, t, frame)

    # -- drawing (pure: frame in, image out) --------------------------------------
    def compose(self, cv2, frame, now=None):
        """Mirrored copy of the frame with the pose, tags and status line drawn on it."""
        mirror = self.config.MIRROR_CAMERA
        img = cv2.flip(frame, 1) if mirror else frame.copy()
        h, w = img.shape[:2]
        lines = []
        if self.pose is not None:
            snap = self.pose.snapshot()
            self._draw_pose(cv2, img, snap.points, w, h)
            x = "--" if snap.paddle_x is None else f"{snap.paddle_x / self.config.TABLE_WIDTH_M + 0.5:.2f}"
            lines.append(f"pose: {self.pose.status}   x {x}")
        if self.tags is not None:
            self._draw_tags(cv2, img, w, mirror, time.monotonic() if now is None else now)
            stable = self.tags.filter.status_id
            lines.append(f"tag: {'none' if stable == -1 else stable}   ({self.tags.status})")
        if lines:                        # a dark band at the bottom, clear of tag labels near the top
            top = h - 12 - 24 * len(lines)
            img[top:] = img[top:] // 3
            for i, line in enumerate(lines):
                cv2.putText(img, line, (10, top + 20 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            TEXT_COLOR, 1, cv2.LINE_AA)
        if self.width and w != self.width:
            img = cv2.resize(img, (self.width, int(h * self.width / w)))
        return img

    def _draw_pose(self, cv2, img, points, w, h):
        """points: landmark idx -> (x, y, visibility), normalised and already mirrored like the image."""
        from pose_tracker import _SKELETON, TRACKED
        px = {i: (int(x * w), int(y * h)) for i, (x, y, v) in points.items() if v > 0.3}
        for a, b in _SKELETON:
            if a in px and b in px:
                cv2.line(img, px[a], px[b], SKELETON_COLOR, 2)
        wrist = TRACKED[self.pose.hand]["wrist"]
        for i, p in px.items():
            cv2.circle(img, p, 8 if i == wrist else 5, WRIST_COLOR if i == wrist else JOINT_COLOR, -1)

    def _draw_tags(self, cv2, img, w, mirror, now):
        import numpy as np
        t, detections = self.tags.last_detections
        if not detections or now - t > DETECTION_MAX_AGE_S:
            return
        f = self.tags.filter
        for tag_id, corners in detections:
            side, _area = marker_geometry(corners)
            pts = np.array([((w - 1 - x) if mirror else x, y) for x, y in corners], np.int32)
            thick = 2
            if tag_id not in f.allowed:
                color, label = TAG_BAD_COLOR, f"ID {tag_id} (not allowed)"
            elif side < f.min_side_px:
                color, label = TAG_BAD_COLOR, f"ID {tag_id} (too small: {side:.0f} px)"
            elif tag_id == f.status_id:
                color, label, thick = TAG_OK_COLOR, f"ID {tag_id}  STABLE", 4
            else:
                color, label = TAG_OK_COLOR, f"ID {tag_id}"
            cv2.polylines(img, [pts], True, color, thick)
            org = (int(pts[:, 0].min()), max(16, int(pts[:, 1].min()) - 8))
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # -- lifecycle --------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="CameraView", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.status = "stopped"

    def _run(self):
        try:
            import cv2
            cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            self.status = f"unavailable: {e}"
            print(f"[view] {self.status}")
            return
        self.status = "waiting for camera frames"
        period = 1.0 / self.fps
        done_seq = 0
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                latest = self._latest
                if latest is not None and latest[0] != done_seq:
                    done_seq, _t, frame = latest
                    try:
                        cv2.imshow(self.title, self.compose(cv2, frame))
                        self.status = "showing"
                    except Exception as e:
                        self.status = f"error: {e}"
                key = cv2.waitKey(1) & 0xFF            # also pumps the window's events
                if key in (27, ord("q")) or cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    self.status = "closed"
                    print("[view] camera view closed (the bridge keeps running)")
                    break
                self._stop.wait(max(0.0, period - (time.monotonic() - t0)))
        finally:
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
