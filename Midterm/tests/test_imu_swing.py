"""IMU data handling: sample parsing, gravity calibration, swing detection."""

import math
from types import SimpleNamespace

import pytest

from motor_imu import GravityCalibrator, ImuSample, SwingDetector

G = 1000.0  # raw units per g used in these tests


def rest(t):
    return ImuSample(t, 0, 0, G, 0, 0, 0)


def run(det, samples):
    return [e for s in samples if (e := det.update(s)) is not None]


def make_detector(**kw):
    defaults = dict(accel_threshold_g=1.0, gyro_threshold=2000, mode="either",
                    cooldown_s=0.4, peak_window_s=0.1, gravity_alpha=0.95)
    defaults.update(kw)
    return SwingDetector(G, **defaults)


def test_sample_from_notification():
    imu = SimpleNamespace(accelerometerX=1, accelerometerY=-2, accelerometerZ=1000,
                          gyroscopeX=10, gyroscopeY=0, gyroscopeZ=-5)
    s = ImuSample.from_notification(1.5, imu)
    assert (s.t, s.ax, s.ay, s.az, s.gx, s.gz) == (1.5, 1, -2, 1000, 10, -5)
    assert s.accel_mag == pytest.approx(math.sqrt(1 + 4 + 1000 ** 2))


def test_sample_not_ready_returns_none():
    nan = float("nan")
    imu = SimpleNamespace(accelerometerX=nan, accelerometerY=nan, accelerometerZ=nan,
                          gyroscopeX=nan, gyroscopeY=nan, gyroscopeZ=nan)
    assert ImuSample.from_notification(0, imu) is None


def test_calibration_measures_gravity_when_still():
    cal = GravityCalibrator(n=10, max_rel_std=0.05, fallback=123)
    for i in range(9):
        assert cal.add(ImuSample(i, 0, 0, 980 + (i % 2) * 10, 0, 0, 0)) is None
    assert cal.add(ImuSample(9, 0, 0, 985, 0, 0, 0)) == pytest.approx(985, abs=6)
    assert not cal.used_fallback


def test_calibration_falls_back_when_moving():
    cal = GravityCalibrator(n=4, max_rel_std=0.05, fallback=123)
    for i, z in enumerate([500, 1500, 300, 2000]):
        cal.add(ImuSample(i, 0, 0, z, 0, 0, 0))
    assert cal.raw_per_g == 123 and cal.used_fallback


def test_no_swing_at_rest():
    det = make_detector()
    assert run(det, [rest(i * 0.02) for i in range(100)]) == []


def test_accel_swing_detected_with_onset_time_and_peak():
    det = make_detector()
    samples = [rest(i * 0.02) for i in range(20)]           # t = 0 .. 0.38
    samples += [ImuSample(0.40, 2.0 * G, 0, G, 0, 0, 0),     # 2 g dynamic -> onset
                ImuSample(0.42, 3.0 * G, 0, G, 0, 0, 0),     # 3 g peak
                ImuSample(0.44, 1.0 * G, 0, G, 0, 0, 0)]
    samples += [rest(0.46 + i * 0.02) for i in range(10)]
    events = run(det, samples)
    assert len(events) == 1
    ev = events[0]
    assert ev.t == pytest.approx(0.40)
    assert ev.accel_peak_g == pytest.approx(3.0, rel=0.02)
    assert ev.strength == pytest.approx(3.0, rel=0.02)  # 3 g / 1 g threshold
    assert ev.source == "imu"


def test_gravity_is_removed():
    """A paddle resting on its side (gravity along x) is not a swing."""
    det = make_detector()
    samples = [rest(i * 0.02) for i in range(5)]
    # orientation changes slowly from z-down to x-down: low-pass tracks it
    for i in range(200):
        a = math.radians(90 * i / 200)
        samples.append(ImuSample(0.1 + i * 0.02, G * math.sin(a), 0, G * math.cos(a), 0, 0, 0))
    assert run(det, samples) == []


def test_gyro_swing_detected():
    det = make_detector()
    samples = [rest(i * 0.02) for i in range(10)]
    samples += [ImuSample(0.2 + i * 0.02, 0, 0, G, 0, 0, 5000) for i in range(8)]
    events = run(det, samples)
    assert len(events) == 1
    assert events[0].gyro_peak_raw == 5000
    assert events[0].strength == pytest.approx(2.5)


def test_cooldown_counts_one_swing_once():
    det = make_detector(cooldown_s=0.4)
    samples = [rest(i * 0.02) for i in range(10)]
    # 0.6 s of continuous violent motion = one physical swing... but longer than cooldown
    samples += [ImuSample(0.2 + i * 0.02, 0, 0, G, 0, 0, 6000) for i in range(15)]  # 0.2..0.48
    samples += [rest(0.5 + i * 0.02) for i in range(20)]
    assert len(run(det, samples)) == 1


def test_two_separate_swings_both_count():
    det = make_detector(cooldown_s=0.4)
    swing = lambda t0: [ImuSample(t0 + i * 0.02, 0, 0, G, 0, 0, 6000) for i in range(4)]
    samples = [rest(i * 0.02) for i in range(10)] + swing(0.2) + [rest(0.3 + i * 0.02) for i in range(25)]
    samples += swing(0.9) + [rest(1.0 + i * 0.02) for i in range(10)]
    events = run(det, samples)
    assert [round(e.t, 2) for e in events] == [0.2, 0.9]


@pytest.mark.parametrize("mode,accel,gyro,expected", [
    ("accel", 2.0, 0, 1), ("accel", 0, 5000, 0),
    ("gyro", 2.0, 0, 0), ("gyro", 0, 5000, 1),
    ("either", 2.0, 0, 1), ("either", 0, 5000, 1),
    ("both", 2.0, 0, 0), ("both", 2.0, 5000, 1),
])
def test_swing_modes(mode, accel, gyro, expected):
    det = make_detector(mode=mode)
    samples = [rest(i * 0.02) for i in range(10)]
    samples += [ImuSample(0.2 + i * 0.02, accel * G, 0, G, 0, 0, gyro) for i in range(8)]
    assert len(run(det, samples)) == expected


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        make_detector(mode="sometimes")
