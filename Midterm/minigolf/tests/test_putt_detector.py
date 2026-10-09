"""IMU putts: a quick swing fires the putt and sets its power; twisting turns the putter."""

import math
from types import SimpleNamespace

import pytest

import config
import putt_detector
from motor_imu import GravityCalibrator, ImuSample
from putt_detector import GolfMotor, PuttDetector, PuttEvent, split_rotation, strength_to_speed

G = 1000.0
DT = 0.02


def make(**kw):
    defaults = dict(accel_threshold_g=0.5, gyro_threshold=1000, mode="gyro", confirm_samples=2,
                    peak_window_s=0.12, cooldown_s=1.0, twist_axis="vertical", twist_sign=1.0,
                    twist_deg_per_raw_s=0.1, twist_deadzone=80, twist_freeze_fraction=0.5, gravity_alpha=0.95)
    defaults.update(kw)
    return PuttDetector(G, **defaults)


def sample(t, gx=0.0, gy=0.0, gz=0.0):
    """Motor hanging like a putter: gravity along +z. Swing = rotation about x, twist = about z."""
    return ImuSample(t, 0, 0, G, gx, gy, gz)


def stroke(t0, back=600, fwd=3000):
    """Rest, slow backswing (0.5 s), quick forward swing (0.3 s), follow-through, rest."""
    samples, t = [], t0
    def add(gx):
        nonlocal t
        samples.append(sample(t, gx=gx))
        t += DT
    for _ in range(10):
        add(0)
    for i in range(25):
        add(-back * math.sin(math.pi * i / 25))
    for i in range(15):
        add(fwd * math.sin(math.pi * i / 15))
    for i in range(8):
        add(400 * math.sin(math.pi * i / 8))
    for _ in range(20):
        add(0)
    return samples


def run(det, samples):
    return [(s.t, e) for s in samples if (e := det.update(s)) is not None]


def test_quick_swing_fires_once_right_away_with_peak_power():
    det = make()
    samples = stroke(0.0)
    fired = run(det, samples)
    assert len(fired) == 1
    t_emit, ev = fired[0]
    first_over = next(s.t for s in samples if s.gx >= 1000)
    assert ev.t_onset == pytest.approx(first_over)                  # fires on the quick swing...
    assert t_emit == pytest.approx(first_over + 0.12, abs=DT + 1e-9)  # ...after the short power window
    assert t_emit < samples[-25].t                                  # not waiting for the motor to settle
    assert ev.strength > 2.0 and ev.t_impact >= ev.t_onset
    assert det.last_event is ev and not det.active


def test_harder_swing_gives_more_power():
    soft = run(make(), stroke(0.0, fwd=1500))[0][1]
    hard = run(make(), stroke(0.0, fwd=3500))[0][1]
    assert hard.strength > soft.strength


def test_slow_backswing_alone_does_not_putt():
    det = make()
    samples = [sample(i * DT, gx=-700 * math.sin(math.pi * i / 40)) for i in range(40)]
    assert run(det, samples) == []


def test_single_sample_bump_does_not_putt():
    det = make()
    samples = [sample(i * DT) for i in range(10)] + [sample(0.2, gx=4000)] + [sample(0.22 + i * DT) for i in range(20)]
    assert run(det, samples) == []


def test_live_strength_while_measuring():
    det = make()
    seen = []
    for s in stroke(0.0):
        det.update(s)
        seen.append(det.live_strength)
    assert max(seen) > 1.0 and seen[-1] == 0.0


def test_cooldown_skips_follow_through_then_second_putt():
    det = make(cooldown_s=1.0)
    first = stroke(0.0)
    rebound = [sample(first[-1].t + DT * (1 + i), gx=-2500) for i in range(5)]     # inside cooldown
    later = stroke(first[-1].t + 1.5)
    assert len(run(det, first + rebound + later)) == 2


def test_twist_integrates_into_putter_angle_without_putting():
    det = make()
    # 1 s twisting at 450 raw/s about vertical = 45 deg with 0.1 deg per raw unit
    samples = [sample(i * DT) for i in range(5)] + [sample(0.1 + i * DT, gz=450) for i in range(51)]
    assert run(det, samples) == []
    assert det.twist_deg == pytest.approx(45.0, abs=1.5)
    det.recenter()
    assert det.twist_deg == 0.0


def test_twist_sign_flips_direction():
    det = make(twist_sign=-1.0)
    run(det, [sample(i * DT, gz=500) for i in range(26)])
    assert det.twist_deg < -20


def test_slow_drift_is_ignored():
    det = make()
    run(det, [sample(i * DT, gz=40) for i in range(500)])
    assert det.twist_deg == 0.0


def test_twist_is_frozen_while_swinging():
    det = make()
    run(det, [sample(i * DT, gx=900, gz=300) for i in range(30)])    # swinging hard (below fire threshold)
    assert det.twist_deg == 0.0


def test_vertical_twist_follows_gravity_however_the_motor_is_held():
    tw, sw = split_rotation((0, 500, 0), (0, G, 0), "vertical")       # gravity along y now
    assert tw == pytest.approx(500) and sw == pytest.approx(0)
    tw, sw = split_rotation((300, 0, 400), (0, 0, G), "vertical")
    assert tw == pytest.approx(400) and sw == pytest.approx(300)
    tw, sw = split_rotation((300, 200, 400), (0, 0, G), "y")
    assert tw == 200 and sw == pytest.approx(500)


def test_strength_to_speed_is_monotonic_and_clamped():
    speeds = [strength_to_speed(s) for s in (-5.0, 1.0, 2.0, 3.0, 50.0)]
    assert speeds == sorted(speeds)
    assert speeds[0] == config.PUTT_SPEED_MIN and speeds[-1] == config.PUTT_SPEED_MAX
    assert strength_to_speed(1.0) == pytest.approx(config.PUTT_SPEED_AT_THRESHOLD)


def test_putt_thresholds_are_gentler_than_ping_pong():
    assert config.PUTT_ACCEL_THRESHOLD_G < config.SWING_ACCEL_THRESHOLD_G
    assert config.PUTT_GYRO_THRESHOLD_RAW < config.SWING_GYRO_THRESHOLD_RAW
    assert config.TWIST_AXIS in ("vertical", "x", "y", "z")


def test_golf_motor_calibrates_then_reports_putts_and_twist(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(putt_detector.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(putt_detector, "PuttDetector", lambda raw_per_g: make())
    got = []
    m = GolfMotor(on_putt=got.append)
    assert m.twist_deg is None
    m.recenter_twist()                                               # harmless before calibration
    m._dm = SimpleNamespace(imu_device=SimpleNamespace(accelerometerX=0, accelerometerY=0, accelerometerZ=G,
                                                       gyroscopeX=0, gyroscopeY=0, gyroscopeZ=0))
    m.calibrator = GravityCalibrator(n=5, max_rel_std=0.05, fallback=1.0)
    twist = [sample(0.1 + i * DT, gz=500) for i in range(20)]
    for s in [sample(i * DT) for i in range(5)] + twist + stroke(0.6):
        clock["t"] = s.t
        imu = m._dm.imu_device
        imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ = s.ax, s.ay, s.az
        imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ = s.gx, s.gy, s.gz
        m._on_notification(None)
    assert m.detector is not None and m.status.startswith("connected")
    assert len(got) == 1 and isinstance(got[0], PuttEvent)
    assert m.twist_deg > 10
    m.recenter_twist()
    assert m.twist_deg == 0.0
