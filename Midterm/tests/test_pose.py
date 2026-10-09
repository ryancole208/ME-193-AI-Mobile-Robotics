"""Pose tracking logic with synthetic landmarks (no webcam needed)."""

import time
from types import SimpleNamespace

import pytest

import config
import pose_tracker
from pose_tracker import (L_WRIST, R_SHOULDER, R_WRIST, PoseTracker, Smoother, WristHistory,
                          extract_points, map_to_table, mirror_x)


def landmarks(right_wrist_x=0.5, left_wrist_x=0.5, vis=0.9):
    """33 MediaPipe-like landmarks in RAW (un-mirrored) image coordinates."""
    lms = [SimpleNamespace(x=0.5, y=0.5, visibility=vis) for _ in range(33)]
    lms[R_WRIST] = SimpleNamespace(x=right_wrist_x, y=0.6, visibility=vis)
    lms[L_WRIST] = SimpleNamespace(x=left_wrist_x, y=0.6, visibility=vis)
    return lms


def test_mirror():
    assert mirror_x(0.2, True) == pytest.approx(0.8)
    assert mirror_x(0.2, False) == 0.2


def test_map_to_table_edges_and_clamp():
    w = config.TABLE_WIDTH_M
    assert map_to_table(0.5, (0.15, 0.85), w) == pytest.approx(0.0)
    assert map_to_table(0.15, (0.15, 0.85), w) == pytest.approx(-w / 2)
    assert map_to_table(0.85, (0.15, 0.85), w) == pytest.approx(w / 2)
    assert map_to_table(0.0, (0.15, 0.85), w) == pytest.approx(-w / 2)   # clamped
    assert map_to_table(1.0, (0.15, 0.85), w) == pytest.approx(w / 2)


def test_smoother():
    s = Smoother(alpha=0.5)
    assert s.update(1.0) == 1.0
    assert s.update(0.0) == 0.5
    s.reset()
    assert s.update(2.0) == 2.0


def test_extract_points_mirrors_x():
    pts = extract_points(landmarks(right_wrist_x=0.3), mirror=True)
    assert pts[R_WRIST][0] == pytest.approx(0.7)
    assert R_SHOULDER in pts


def test_history_direction():
    h = WristHistory(keep_s=5)
    for i in range(10):
        h.add(i * 0.033, 0.3 + i * 0.02, 0.0)   # moving right in mirrored image
    assert h.direction_at(9 * 0.033, n_frames=6, threshold=0.04) == "right"
    h2 = WristHistory(keep_s=5)
    for i in range(10):
        h2.add(i * 0.033, 0.7 - i * 0.02, 0.0)
    assert h2.direction_at(9 * 0.033, n_frames=6, threshold=0.04) == "left"
    h3 = WristHistory(keep_s=5)
    for i in range(10):
        h3.add(i * 0.033, 0.5 + (i % 2) * 0.005, 0.0)
    assert h3.direction_at(9 * 0.033, n_frames=6, threshold=0.04) == "center"


def test_history_direction_uses_time_of_swing():
    """Direction is evaluated at the swing onset, not 'now'."""
    h = WristHistory(keep_s=5)
    for i in range(10):   # moves right during t = 0..0.3
        h.add(i * 0.033, 0.3 + i * 0.03, 0.0)
    for i in range(10, 20):  # then moves back left
        h.add(i * 0.033, 0.57 - (i - 10) * 0.03, 0.0)
    assert h.direction_at(0.3) == "right"
    assert h.direction_at(19 * 0.033) == "left"


def test_history_paddle_at_and_expiry():
    h = WristHistory(keep_s=1.0)
    h.add(0.0, 0.5, -0.2)
    h.add(0.1, 0.5, 0.3)
    assert h.paddle_at(0.05) == -0.2
    assert h.paddle_at(0.12) == 0.3
    assert h.paddle_at(5.0) is None   # too old
    h.add(2.0, 0.5, 0.1)
    assert len(h) == 1                # earlier samples expired


def test_tracker_uses_dominant_hand_and_mirrors(monkeypatch):
    monkeypatch.setattr(config, "POSE_SMOOTHING", 0.0)
    trk = PoseTracker(hand="right")
    trk._smoother = Smoother(alpha=0.0)
    # Person's right wrist is on the image's LEFT (raw x=0.15) -> mirrored 0.85 -> right table edge
    snap = trk.process_landmarks(1.0, landmarks(right_wrist_x=0.15, left_wrist_x=0.9))
    assert snap.tracking
    assert snap.wrist_x == pytest.approx(0.85)
    assert snap.paddle_x == pytest.approx(config.TABLE_WIDTH_M / 2)
    assert trk.paddle_x_at(1.0) == pytest.approx(config.TABLE_WIDTH_M / 2)

    left = PoseTracker(hand="left")
    left._smoother = Smoother(alpha=0.0)
    snap = left.process_landmarks(1.0, landmarks(right_wrist_x=0.15, left_wrist_x=0.85))
    assert snap.paddle_x == pytest.approx(-config.TABLE_WIDTH_M / 2)


def test_tracker_no_person_and_low_visibility():
    trk = PoseTracker()
    assert not trk.process_landmarks(0.0, None).tracking
    assert not trk.process_landmarks(0.1, landmarks(vis=0.1)).tracking
    assert trk.snapshot().paddle_x is None


def test_bad_hand_rejected():
    with pytest.raises(ValueError):
        PoseTracker(hand="both")


def test_tasks_api_available():
    """MediaPipe Tasks PoseLandmarker exists in the installed version."""
    from mediapipe.tasks.python import vision
    assert hasattr(vision, "PoseLandmarker")
    assert vision.PoseLandmark.RIGHT_WRIST == R_WRIST
    assert vision.PoseLandmark.LEFT_WRIST == L_WRIST


@pytest.mark.hardware
def test_model_download_and_landmarker_loads():
    path = pose_tracker.ensure_model()
    assert path.exists() and path.stat().st_size > 1_000_000
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(path)),
                                        running_mode=vision.RunningMode.VIDEO)
    with vision.PoseLandmarker.create_from_options(opts):
        pass


@pytest.mark.hardware
def test_webcam_pose_tracking_runs():
    """Opens the real webcam. Stand in view with your dominant hand visible."""
    trk = PoseTracker()
    trk.start()
    try:
        end = time.monotonic() + 20
        while time.monotonic() < end and trk.status != "tracking":
            time.sleep(0.1)
        assert trk.status == "tracking", trk.status
        assert trk.snapshot().preview_rgb is not None
    finally:
        trk.stop()
