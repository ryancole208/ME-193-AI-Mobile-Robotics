"""camera_view.py drawing, with fake pose / tags on a blank frame (no camera, no window)."""

import cv2
import numpy as np

import config
from camera_view import TAG_BAD_COLOR, TAG_OK_COLOR, WRIST_COLOR, CameraView
from tags import TagFilter


class FakeSnap:
    def __init__(self, points, paddle_x=None):
        self.points, self.paddle_x = points, paddle_x


class FakePose:
    hand = "right"
    status = "tracking"

    def __init__(self, points, paddle_x=0.0):
        self._snap = FakeSnap(points, paddle_x)

    def snapshot(self):
        return self._snap


class FakeTags:
    status = "scanning"

    def __init__(self, detections, t=10.0):
        self.filter = TagFilter(allowed=(7, 9), stable_frames=1, min_side_px=30, lost_s=1.0)
        self.last_detections = (t, detections)


def sq(x, y, side):
    return np.array([(x, y), (x + side, y), (x + side, y + side), (x, y + side)], float)


def blank(w=640, h=480):
    return np.zeros((h, w, 3), np.uint8)


def has_color(img, color, region):
    x0, y0, x1, y1 = region
    return bool((img[y0:y1, x0:x1] == np.array(color, np.uint8)).all(axis=2).any())


def view(**kw):
    return CameraView(width=640, **kw)


def test_frame_is_mirrored_like_the_other_games(monkeypatch):
    monkeypatch.setattr(config, "MIRROR_CAMERA", True)
    f = blank()
    f[:, :10] = 255                                      # a white strip on the raw left edge
    img = view().compose(cv2, f)
    assert img.shape == (480, 640, 3)
    assert img[240, 635].tolist() == [255, 255, 255] and img[240, 5].tolist() == [0, 0, 0]


def test_tracked_wrist_drawn_in_red_where_the_pose_says():
    # pose points are already mirrored (normalised), like PoseSnapshot.points
    pose = FakePose({16: (0.5, 0.5, 0.9), 14: (0.4, 0.4, 0.9), 12: (0.3, 0.3, 0.9)})
    img = view(pose=pose).compose(cv2, blank())
    assert has_color(img, WRIST_COLOR, (310, 230, 330, 250))


def test_invisible_landmarks_are_not_drawn():
    pose = FakePose({16: (0.5, 0.5, 0.1)})
    img = view(pose=pose).compose(cv2, blank())
    assert not has_color(img, WRIST_COLOR, (300, 220, 340, 260))


def test_tags_are_outlined_at_the_mirrored_spot_with_allowed_and_ignored_colours(monkeypatch):
    monkeypatch.setattr(config, "MIRROR_CAMERA", True)
    tags = FakeTags([(7, sq(40, 200, 100)),               # allowed -> green, raw left = shown right
                     (3, sq(400, 200, 100)),              # not allowed -> red
                     (9, sq(250, 50, 20))])               # allowed but too small -> red
    img = view(tags=tags).compose(cv2, blank(), now=10.1)
    assert has_color(img, TAG_OK_COLOR, (495, 195, 605, 305))       # mirrored: 640-1-140 .. 640-1-40
    assert not has_color(img, TAG_OK_COLOR, (30, 195, 150, 305))
    assert has_color(img, TAG_BAD_COLOR, (135, 195, 245, 305))
    assert has_color(img, TAG_BAD_COLOR, (365, 40, 395, 75))


def test_old_tag_outlines_are_not_drawn():
    tags = FakeTags([(7, sq(40, 200, 100))], t=10.0)
    img = view(tags=tags).compose(cv2, blank(), now=11.0)
    assert not has_color(img, TAG_OK_COLOR, (0, 100, 640, 480))


def test_scaled_to_the_display_width():
    img = CameraView(width=320).compose(cv2, blank())
    assert img.shape == (240, 320, 3)
