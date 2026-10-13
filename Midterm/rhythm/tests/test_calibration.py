"""Latency calibration maths and persistence."""

import json
import random

import pytest

import config
import calibration
from calibration import InputCalibration, analyze


def beats(n=16, spb=0.6, start=2.4):
    return [start + k * spb for k in range(n)]


def test_constant_offset_is_measured():
    b = beats()
    r = analyze([t + 0.087 for t in b], b)
    assert r.ok and r.offset_ms == pytest.approx(87) and r.used == 16 and r.dropped == 0


def test_noise_and_outliers():
    rng = random.Random(1)
    b = beats()
    swings = [t + 0.05 + rng.gauss(0, 0.012) for t in b]
    swings[3] += 0.2                                           # one sloppy swing
    r = analyze(swings, b)
    assert r.dropped == 1 and r.offset_ms == pytest.approx(50, abs=10) and r.std_ms < 20


def test_needs_enough_swings_and_ignores_far_ones():
    b = beats()
    r = analyze([b[0] + 0.03, b[1] + 0.03, b[5] + 0.9], b)
    assert not r.ok and r.used == 2


def test_one_swing_per_click():
    b = beats(n=10)
    swings = [t + 0.02 for t in b] + [b[0] + 0.03]           # a double swing on the first click
    r = analyze(swings, b)
    assert r.used == 10


def test_calibration_run_uses_measured_beats_and_majority_source():
    cal = InputCalibration(bpm=100, count_in=4, beats=12)
    for t in cal.click_times:
        cal.add_swing(t + 0.06, "imu")                        # count-in swings are ignored
    cal.add_swing(cal.measured_times[2] + 0.01, "keyboard")
    r = cal.finish()
    assert cal.source == "imu" and r.ok and r.offset_ms == pytest.approx(60) and r.used == 12


def test_save_merges_and_updates_live_config(tmp_path):
    p = tmp_path / "cal.json"
    calibration.save(input_offsets={"imu": 91.234}, path=p)
    calibration.save(visual_offset_ms=-12.0, path=p)
    calibration.save(input_offsets={"keyboard": 15.0}, path=p)
    data = json.loads(p.read_text())
    assert data == {"input_offset_ms": {"imu": 91.2, "keyboard": 15.0}, "visual_offset_ms": -12.0}
    assert config.INPUT_OFFSET_MS["imu"] == pytest.approx(91.234) and config.VISUAL_OFFSET_MS == -12.0


def test_config_loads_saved_calibration(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"input_offset_ms": {"imu": 120}, "visual_offset_ms": 33}))
    config.load_calibration(p)
    assert config.INPUT_OFFSET_MS["imu"] == 120.0 and config.VISUAL_OFFSET_MS == 33.0
    assert config.load_calibration(tmp_path / "missing.json") == {}
