"""Rhythm swing detection (peak timestamp), per-song cooldown, haptic budget."""

from types import SimpleNamespace

import pytest

import config
import rhythm_motor
from chart import scan_songs
from helpers import chart_with_cues
from motor_imu import GravityCalibrator, ImuSample
from rhythm_motor import (RhythmMotor, RhythmSwing, RhythmSwingDetector, cooldown_for, haptic_plan,
                          haptic_pulse_ms)
from scheduler import build_schedule

G = 1000.0
DT = 0.02


def swing_samples(t0, peak_at=2, n=8, peak_gyro=9000.0):
    """Gyro magnitude ramps up to a peak at sample `peak_at` and falls (swing axis = y, not roll)."""
    out = []
    for i in range(n):
        g = peak_gyro * max(0.0, 1 - abs(i - peak_at) / 3)
        out.append(ImuSample(t0 + i * DT, 0, 0, G, 0, g, 0))
    return out


def detector(**kw):
    return RhythmSwingDetector(G, accel_threshold_g=99, gyro_threshold=3000, mode="gyro", roll_axis="x", **kw)


def test_swing_time_is_the_peak_sample_and_reported_quickly():
    d = detector()
    for i in range(5):
        assert d.update(ImuSample(i * DT, 0, 0, G, 0, 0, 0)) is None
    got = []
    for s in swing_samples(1.0, peak_at=2):
        ev = d.update(s)
        if ev:
            got.append((ev, s.t))
    assert len(got) == 1
    ev, reported_at = got[0]
    assert isinstance(ev, RhythmSwing)
    assert ev.t == pytest.approx(1.0 + 2 * DT)                # the peak sample, not the onset
    assert ev.onset < ev.t
    assert reported_at - ev.t <= config.RHYTHM_SWING_PEAK_WINDOW_S   # sent once the peak is over


def test_cooldown_is_settable():
    d = detector(cooldown_s=0.15)
    a = [d.update(s) for s in swing_samples(1.0)]
    b = [d.update(s) for s in swing_samples(1.0 + 8 * DT)]      # next swing 160 ms later
    assert sum(e is not None for e in a) == 1 and sum(e is not None for e in b) == 1


def test_cooldown_for_chart():
    # 600 ms gaps (200 BPM, every other beat) keep the default cooldown
    assert cooldown_for(0.6) == (config.RHYTHM_SWING_COOLDOWN_S, False)
    cd, short = cooldown_for(0.353)                              # one beat at 170 BPM
    assert short and cd == pytest.approx(0.353 - 2 * config.GOOD_WINDOW_MS / 1000)
    assert cooldown_for(0.1)[0] == config.RHYTHM_SWING_MIN_COOLDOWN_S


def test_haptic_pulse_fits_before_the_next_window():
    full = config.RHYTHM_HAPTIC_PULSE_MS
    assert haptic_pulse_ms(1000) == full
    budget = full - 10 + config.RHYTHM_HAPTIC_SETTLE_MS + config.RHYTHM_HAPTIC_SAFETY_MS
    assert haptic_pulse_ms(budget) == full - 10                  # shortened
    assert haptic_pulse_ms(config.RHYTHM_HAPTIC_SETTLE_MS) == 0  # skipped


def test_haptic_plan_for_200_bpm_every_other_beat_is_full_length():
    ch = chart_with_cues(range(4, 40, 2), bpm=200.0)
    plan = haptic_plan(build_schedule(ch), input_offset_ms=80)
    assert all(p == config.RHYTHM_HAPTIC_PULSE_MS for p in plan)


@pytest.mark.parametrize("hit_beats", [None, [2, 4]])
def test_haptic_plan_real_songs_fit_every_pulse(monkeypatch, hit_beats):
    monkeypatch.setattr(config, "HIT_BEATS", hit_beats)
    for entry in scan_songs()[0]:
        plan = haptic_plan(build_schedule(entry.chart), input_offset_ms=80)
        assert all(p > 0 for p in plan), entry.chart.title   # [2, 4] at 160 BPM still leaves room


def test_haptic_plan_shortens_or_skips_for_close_cues():
    ch = chart_with_cues((4.0, 6.0, 7.5, 8.5, 10.0), bpm=200.0)
    plan = haptic_plan(build_schedule(ch), input_offset_ms=80)
    full = config.RHYTHM_HAPTIC_PULSE_MS
    assert plan[0] == full                   # 600 ms gap
    assert 0 < plan[1] <= full               # 450 ms gap (upbeat)
    assert plan[2] == 0                      # 300 ms gap: the IMU must stay live, so no pulse


def test_motor_notification_path_uses_the_rhythm_detector(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(rhythm_motor, "clock", lambda: clock["t"])
    monkeypatch.setattr(config, "SWING_MODE", "gyro")
    got = []
    m = RhythmMotor(on_swing=got.append)
    m._dm = SimpleNamespace(imu_device=SimpleNamespace(accelerometerX=0, accelerometerY=0, accelerometerZ=G,
                                                       gyroscopeX=0, gyroscopeY=0, gyroscopeZ=0))
    m.calibrator = GravityCalibrator(n=5, max_rel_std=0.05, fallback=1.0)
    m.set_cooldown(0.2)
    samples = [ImuSample(i * DT, 0, 0, G, 0, 0, 0) for i in range(8)] + swing_samples(1.0)
    for s in samples:
        clock["t"] = s.t
        imu = m._dm.imu_device
        imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ = s.ax, s.ay, s.az
        imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ = s.gx, s.gy, s.gz
        m._on_notification(None)
    assert isinstance(m.detector, RhythmSwingDetector) and m.detector.cooldown_s == 0.2
    assert len(got) == 1 and got[0].t == pytest.approx(1.0 + 2 * DT)


def test_roll_display_ignores_small_twists_swings_and_drift():
    from rhythm_motor import RollDisplay
    rd = RollDisplay()
    t = 0.0
    for _ in range(30):                                   # small wobble: shown straight
        t += 1 / 60
        assert rd.update(t, 8.0 * (-1) ** int(t * 60)) == 0.0
    for _ in range(30):                                   # a real twist shows (scaled down)
        t += 1 / 60
        shown = rd.update(t, 40.0)
    assert 5.0 < shown < 40.0
    held = shown
    for _ in range(20):                                   # swinging: the twist holds
        t += 1 / 60
        assert rd.update(t, 90.0, swinging=True) == held
    for _ in range(int(60 * config.ROLL_DISPLAY_RECENTER_S * 4)):   # drift fades back to neutral
        t += 1 / 60
        shown = rd.update(t, 40.0)
    assert abs(shown) < 1.0


def test_roll_display_can_be_switched_off(monkeypatch):
    from rhythm_motor import RollDisplay
    monkeypatch.setattr(config, "ROLL_DISPLAY_ENABLED", False)
    assert RollDisplay().update(1.0, 45.0) == 0.0


def test_haptic_pulse_does_not_twist_the_paddle(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(rhythm_motor, "clock", lambda: clock["t"])
    m = RhythmMotor(on_swing=lambda e: None)
    m._dm = SimpleNamespace(imu_device=SimpleNamespace(accelerometerX=0, accelerometerY=0, accelerometerZ=G,
                                                       gyroscopeX=0, gyroscopeY=0, gyroscopeZ=0))
    m.calibrator = GravityCalibrator(n=5, max_rel_std=0.05, fallback=1.0)
    imu = m._dm.imu_device
    for i in range(8):
        clock["t"] = i * DT
        m._on_notification(None)
    before = m.roll_deg
    m._quiet_until = 1.0                                  # a pulse is running
    imu.gyroscopeX = 4000.0                               # the motor kicks the hub around the roll axis
    for i in range(8, 40):
        clock["t"] = i * DT
        m._on_notification(None)
    assert m.roll_deg == pytest.approx(before)


def test_quiet_window_blocks_swings(monkeypatch):
    d = detector()
    for i in range(5):
        d.update(ImuSample(i * DT, 0, 0, G, 0, 0, 0))
    assert all(d.update(s, ignore=True) is None for s in swing_samples(1.0))
