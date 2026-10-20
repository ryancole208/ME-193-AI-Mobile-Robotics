"""KNN classifier, persistence, debouncer and speed-aware debounce."""

import random

import pytest

import config
from color_classifier import (BLUE, FLOOR, UNKNOWN, Debouncer, KnnColorClassifier, LegoColorClassifier,
                              features_from_sensor, samples_needed)
from simulator import BLUE_READING, FLOOR_READING, SimColor


def training(n=30, seed=0):
    rng = random.Random(seed)
    return [(lab, tuple(v + rng.gauss(0, 3) for v in base))
            for lab, base in ((FLOOR, FLOOR_READING), (BLUE, BLUE_READING)) for _ in range(n)]


def test_knn_separates_floor_and_blue():
    clf = KnnColorClassifier(training())
    rng = random.Random(5)
    for lab, base in ((FLOOR, FLOOR_READING), (BLUE, BLUE_READING)):
        for _ in range(50):
            assert clf.classify(tuple(v + rng.gauss(0, 3) for v in base)) == lab


def test_far_reading_is_unknown():
    clf = KnnColorClassifier(training())
    assert clf.classify((10, 10, 10, 2)) == UNKNOWN       # e.g. sensor lifted / black robot
    assert clf.classify(None) == UNKNOWN


def test_needs_both_labels():
    with pytest.raises(ValueError):
        KnnColorClassifier([(FLOOR, (1, 2, 3, 4))])


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    KnnColorClassifier.save_samples(p, training(5))
    loaded = KnnColorClassifier.load_samples(p)
    assert len(loaded) == 10 and loaded[0][0] == FLOOR
    assert KnnColorClassifier(loaded).classify(BLUE_READING) == BLUE


def test_features_from_sensor_and_nan():
    assert features_from_sensor(SimColor(1, 2, 3, 4, color=0)) == (1.0, 2.0, 3.0, 4.0)
    assert features_from_sensor(SimColor(float("nan"), 2, 3, 4, color=0)) is None


def test_lego_classifier():
    fb = LegoColorClassifier()
    assert fb.classify_sensor(SimColor(0, 0, 0, 0, color=3)) == BLUE
    assert fb.classify_sensor(SimColor(0, 0, 0, 0, color=0)) == FLOOR


def test_debouncer_needs_n_consecutive():
    d = Debouncer(n=2, log=lambda *_: None)
    assert d.update(BLUE) is None
    assert d.update(FLOOR) is None          # single blip ignored
    assert d.update(BLUE) is None
    assert d.update(BLUE) == BLUE
    assert d.update(FLOOR) is None
    assert d.update(FLOOR) == FLOOR


def test_debouncer_unknown_neither_counts_nor_resets():
    d = Debouncer(n=2, log=lambda *_: None)
    d.update(BLUE)
    d.update(UNKNOWN)
    assert d.update(BLUE) == BLUE


def test_debouncer_required_override():
    d = Debouncer(n=3, log=lambda *_: None)
    assert d.update(BLUE, required=1) == BLUE


def test_samples_needed_drops_to_one_when_too_fast():
    slow = 0.1        # 3 cm tape at 0.1 m/s = 10 readings
    fast = config.TAPE_WIDTH_M / config.COLOR_SAMPLE_PERIOD_S / config.DEBOUNCE_SAMPLES  # ~0.5 m/s
    assert samples_needed(slow) == config.DEBOUNCE_SAMPLES
    assert samples_needed(0.0) == config.DEBOUNCE_SAMPLES
    assert samples_needed(fast) == 1
    assert samples_needed(-fast) == 1


def test_lego_classifier_accepts_blue_and_azure():
    """In "lego" mode both LEGO blue (3) and azure (10) count as tape."""
    from types import SimpleNamespace
    from color_classifier import LegoColorClassifier
    fb = LegoColorClassifier()
    assert fb.classify_sensor(SimpleNamespace(color=3)) == BLUE
    assert fb.classify_sensor(SimpleNamespace(color=10)) == BLUE
    assert fb.classify_sensor(SimpleNamespace(color=7)) == FLOOR      # white
    assert fb.classify_sensor(SimpleNamespace(color=float("nan"))) == "unknown"


def test_color_method_setting(tmp_path, monkeypatch):
    """config.COLOR_METHOD picks the classifier, and never switches silently."""
    from color_classifier import load_classifier
    quiet = dict(log=lambda *_: None)
    assert isinstance(load_classifier("lego", **quiet), LegoColorClassifier)
    missing = tmp_path / "none.json"
    with pytest.raises(ValueError, match="calibrate_color.py"):
        load_classifier("knn", path=missing, **quiet)
    p = tmp_path / "s.json"
    KnnColorClassifier.save_samples(p, training(5))
    assert isinstance(load_classifier("knn", path=p, **quiet), KnnColorClassifier)
    with pytest.raises(ValueError, match="COLOR_METHOD"):
        load_classifier("auto", **quiet)
    monkeypatch.setattr(config, "COLOR_METHOD", "lego")
    assert isinstance(load_classifier(**quiet), LegoColorClassifier)
