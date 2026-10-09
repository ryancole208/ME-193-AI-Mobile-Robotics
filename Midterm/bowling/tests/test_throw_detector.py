"""IMU throw detection: pendulum swing -> one throw, release at peak, spin from twist."""

import math
from types import SimpleNamespace

import pytest

import config
import throw_detector
from motor_imu import GravityCalibrator, ImuSample
from throw_detector import BowlingMotor, ThrowDetector, ThrowEvent, spin_from_twist, strength_to_speed

G = 1000.0
DT = 0.02


def make(**kw):
    defaults = dict(accel_threshold_g=1.0, gyro_threshold=2000, mode="gyro", end_fraction=0.5,
                    quiet_s=0.2, min_s=0.3, max_s=2.0, cooldown_s=1.0, spin_window_s=0.1, gravity_alpha=0.95)
    defaults.update(kw)
    return ThrowDetector(G, **defaults)


def pendulum(t0, back=3000, fwd=6000, twist=0.0, axis=1):
    """Rest, backswing (0.5 s), forward swing (0.5 s, faster), follow-through, rest. Swing on gyro x."""
    samples, t = [], t0
    def add(gx, tw=0.0):
        nonlocal t
        g = [gx, 0.0, 0.0]
        g[axis] += tw
        samples.append(ImuSample(t, 0, 0, G, *g))
        t += DT
    for _ in range(10):
        add(0)
    for i in range(25):
        add(-back * math.sin(math.pi * i / 25))
    for i in range(25):
        u = math.sin(math.pi * i / 25)
        add(fwd * u, twist * u)
    for i in range(10):
        add(1500 * math.sin(math.pi * i / 10))
    for _ in range(20):
        add(0)
    return samples


def run(det, samples):
    return [e for s in samples if (e := det.update(s)) is not None]


def test_one_event_per_throw_with_release_at_forward_peak():
    det = make()
    events = run(det, pendulum(0.0))
    assert len(events) == 1
    ev = events[0]
    assert ev.strength == pytest.approx(6000 / 2000, rel=0.02)
    peak_t = 0.0 + (10 + 25 + 12) * DT          # middle of the forward swing
    assert ev.t_release == pytest.approx(peak_t, abs=0.03)
    assert ev.t_onset < ev.t_release
    assert ev.spin == 0.0


def test_twist_at_release_gives_spin(monkeypatch):
    det = make()
    monkeypatch.setattr(config, "SPIN_GYRO_AXIS", "y")
    ev = run(det, pendulum(0.0, twist=3000, axis=1))[0]
    assert ev.twist_raw[1] > 2000
    assert spin_from_twist(ev.twist_raw, axis="y", sign=1.0, full=4000) > 0.5
    assert spin_from_twist(ev.twist_raw, axis="y", sign=-1.0, full=4000) < -0.5


def test_twitch_is_not_a_throw():
    det = make()
    samples = [ImuSample(i * DT, 0, 0, G, 0, 0, 0) for i in range(10)]
    samples += [ImuSample(0.2 + i * DT, 0, 0, G, 2500, 0, 0) for i in range(3)]
    samples += [ImuSample(0.26 + i * DT, 0, 0, G, 0, 0, 0) for i in range(30)]
    assert run(det, samples) == []


def test_cooldown_then_second_throw():
    det = make(cooldown_s=1.0)
    first = pendulum(0.0)
    end = first[-1].t
    assert len(run(det, first + pendulum(end + DT))) == 2


def test_long_motion_is_cut_at_max_duration():
    det = make(max_s=0.5)
    samples = [ImuSample(i * DT, 0, 0, G, 3000, 0, 0) for i in range(60)]
    events = run(det, samples)
    assert len(events) >= 1
    assert events[0].t_onset == 0.0


def test_spin_helpers():
    assert spin_from_twist((0, 9999, 0), axis=None) == 0.0
    assert spin_from_twist((0, 9999, 0), axis="y", sign=1, full=4000) == 1.0
    assert spin_from_twist((0, 100, 0), axis="y", sign=1, full=4000, deadzone=0.1) == 0.0
    assert spin_from_twist((-2000, 0, 0), axis="x", sign=1, full=4000, deadzone=0.1) == -0.5


def test_strength_to_speed_is_monotonic_and_clamped():
    speeds = [strength_to_speed(s) for s in (0.0, 1.0, 2.0, 3.0, 50.0)]
    assert speeds == sorted(speeds)
    assert speeds[0] == config.BALL_SPEED_MIN and speeds[-1] == config.BALL_SPEED_MAX
    assert strength_to_speed(1.0) == pytest.approx(config.BALL_SPEED_AT_THRESHOLD)


def test_bowling_motor_calibrates_then_reports_throws(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(throw_detector.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(throw_detector, "ThrowDetector", lambda raw_per_g: make())
    got = []
    m = BowlingMotor(on_throw=got.append)
    m._dm = SimpleNamespace(imu_device=SimpleNamespace(accelerometerX=0, accelerometerY=0, accelerometerZ=G,
                                                       gyroscopeX=0, gyroscopeY=0, gyroscopeZ=0))
    m.calibrator = GravityCalibrator(n=5, max_rel_std=0.05, fallback=1.0)
    for s in [ImuSample(i * DT, 0, 0, G, 0, 0, 0) for i in range(5)] + pendulum(0.2):
        clock["t"] = s.t
        imu = m._dm.imu_device
        imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ = s.ax, s.ay, s.az
        imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ = s.gx, s.gy, s.gz
        m._on_notification(None)
    assert m.detector is not None and m.status.startswith("connected")
    assert len(got) == 1 and isinstance(got[0], ThrowEvent)
