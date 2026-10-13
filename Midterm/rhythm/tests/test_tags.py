"""tags.py: the tag16h5 filter (pure logic) and the scanner on synthetic images (no camera)."""

import cv2
import numpy as np
import pytest

from tags import TagFilter, TagScanner, marker_geometry


def sq(x, y, side):
    return [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]


def make_filter(**kw):
    kw = {"allowed": (0, 6, 7, 8, 9, 10, 11), "stable_frames": 3, "min_side_px": 30, "lost_s": 1.0, **kw}
    return TagFilter(**kw)


def feed(f, passes, t0=0.0, dt=0.1):
    """Run detection passes 0.1 s apart; returns [(t, event)] for the passes that sent something."""
    out = []
    for i, dets in enumerate(passes):
        t = t0 + i * dt
        ev = f.update(t, dets)
        if ev is not None:
            out.append((round(t, 3), ev))
    return out


def test_geometry_shortest_side_and_area():
    side, area = marker_geometry([(0, 0), (40, 0), (40, 30), (0, 30)])
    assert side == 30 and area == 1200


def test_stable_after_n_passes_in_a_row_sent_once():
    f = make_filter()
    tag7 = [(7, sq(100, 100, 60))]
    assert feed(f, [tag7] * 6) == [(0.2, 7)]               # 3rd pass, then nothing more
    assert f.status_id == 7


def test_a_gap_restarts_the_count():
    f = make_filter()
    tag7 = [(7, sq(100, 100, 60))]
    assert feed(f, [tag7, tag7, [], tag7, tag7]) == []
    assert f.status_id == -1                               # nothing sent before the first stable tag


def test_ids_outside_the_allowed_list_are_ignored():
    f = make_filter()
    assert feed(f, [[(3, sq(0, 0, 80))]] * 5) == []


def test_small_tags_are_ignored():
    f = make_filter()
    assert feed(f, [[(7, sq(0, 0, 29))]] * 5) == []
    assert feed(make_filter(), [[(7, sq(0, 0, 30))]] * 3) == [(0.2, 7)]


def test_the_largest_allowed_tag_wins():
    f = make_filter()
    both = [(6, sq(0, 0, 40)), (9, sq(200, 0, 90)), (2, sq(400, 0, 200))]   # 2 is not allowed
    assert feed(f, [both] * 3) == [(0.2, 9)]


def test_switching_tags_sends_only_the_new_id():
    f = make_filter()
    tag7, tag9 = [(7, sq(0, 0, 60))], [(9, sq(0, 0, 60))]
    assert feed(f, [tag7] * 3 + [tag9] * 3) == [(0.2, 7), (0.5, 9)]


def test_lost_after_lost_s_once_then_found_again():
    f = make_filter()
    tag7 = [(7, sq(0, 0, 60))]
    events = feed(f, [tag7] * 3 + [[]] * 15 + [tag7] * 3)
    # last seen at 0.2 s -> lost at 1.2 s (once) -> stable again after 3 more passes
    assert events == [(0.2, 7), (1.2, -1), (2.0, 7)]


def test_flicker_shorter_than_lost_s_keeps_the_tag():
    f = make_filter()
    tag7 = [(7, sq(0, 0, 60))]
    assert feed(f, [tag7] * 3 + ([[]] * 5 + [tag7]) * 4) == [(0.2, 7)]


def test_a_false_positive_burst_never_beats_the_stable_count():
    f = make_filter()
    tag7, fp = [(7, sq(0, 0, 60))], [(11, sq(300, 300, 35))]
    assert feed(f, [tag7] * 3 + [fp, fp, tag7] * 4) == [(0.2, 7)]


# -- the real detector on synthetic images -----------------------------------------

def image_with(tags, size=(480, 640)):
    img = np.full(size, 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    for tag_id, x, y, side in tags:
        img[y:y + side, x:x + side] = cv2.aruco.generateImageMarker(d, tag_id, side)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_opencv_has_tag16h5():
    assert hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_APRILTAG_16h5")


def test_scanner_detects_and_reports_the_largest_tag():
    events = []
    sc = TagScanner(lambda tag_id, f: events.append(tag_id), tag_filter=make_filter())
    det = sc.make_detector(cv2)
    frame = image_with([(7, 40, 40, 100), (10, 300, 120, 180)])
    found = sorted(i for i, _ in sc.detect(cv2, det, frame))
    assert found == [7, 10]
    for i in range(3):
        sc.process(cv2, det, i * 0.1, frame)
    assert events == [10] and sc.passes == 3


def test_scanner_ignores_a_mirrored_tag():
    sc = TagScanner(lambda *a: None, tag_filter=make_filter())
    det = sc.make_detector(cv2)
    frame = cv2.flip(image_with([(7, 200, 150, 120)]), 1)
    assert [i for i, _ in sc.detect(cv2, det, frame) if i == 7] == []


def test_shared_frames_thread_runs_and_stops():
    import threading
    got = threading.Event()
    sc = TagScanner(lambda tag_id, f: got.set(), rate_hz=50, tag_filter=make_filter(stable_frames=2))
    sc.start()
    frame = image_with([(8, 200, 150, 120)])
    try:
        for i in range(40):
            sc.on_frame(i * 0.02, frame)
            if got.wait(0.02):
                break
        assert got.is_set() and sc.filter.status_id == 8
    finally:
        sc.stop()
    assert sc.status == "stopped"
