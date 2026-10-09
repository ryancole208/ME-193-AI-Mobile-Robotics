"""Spin: roll filter, swing -> spin mapping, spin flights and the opponent's response."""

import math
import random

import pytest

import config
from events import SwingEvent
from game import GameLogic, Flight, make_flight
from motor_imu import ImuSample, SwingDetector
from opponent import MISS, IncomingBall, Shot, SimulatedOpponent
from spin import (NO_SPIN, RollFilter, Spin, flight_params, magnus_curve_m, side_from_imu, spin_for_swing,
                  spin_label, wrap_deg)

G = 1000.0          # raw accel units per g
DPR = 0.1           # gyro deg/s per raw unit used in these tests


def tilted(t, roll_deg, gyro_x=0.0, accel_scale=1.0):
    """Sample of a hub whose handle (x axis) is horizontal and rolled by roll_deg about x."""
    a = math.radians(roll_deg)
    # body rolled by +a about x: gravity (0, 0, G) seen in the body turns by -a
    ay, az = G * math.sin(a), G * math.cos(a)
    return ImuSample(t, 0.0, ay * accel_scale, az * accel_scale, gyro_x, 0.0, 0.0)


def make_filter(**kw):
    defaults = dict(axis="x", sign=1.0, accel_sign=1.0, deg_per_raw=DPR, alpha=0.98, still_tol_g=0.15,
                    min_gravity_fraction=0.35, deadzone_dps=2.0)
    defaults.update(kw)
    return RollFilter(G, **defaults)


# -- roll filter --------------------------------------------------------------------------

def test_accel_roll_matches_geometry():
    f = make_filter()
    for deg in (-60, -20, 0, 30, 80):
        assert f.accel_roll(tilted(0, deg)) == pytest.approx(deg, abs=1e-6)


def test_vertical_handle_gives_no_accel_roll():
    f = make_filter()
    upright = ImuSample(0, G, 0, 0, 0, 0, 0)    # gravity along the handle
    assert f.accel_roll(upright) is None


def test_gyro_integration_tracks_a_fast_twist():
    f = make_filter()
    f.update(tilted(0.0, 0))
    f.calibrate_neutral()
    rate_raw = 900 / DPR      # 900 deg/s twist for 0.1 s = 90 deg
    t = 0.0
    for i in range(5):
        t += 0.02
        f.update(tilted(t, 18 * (i + 1), gyro_x=rate_raw, accel_scale=2.5), freeze_accel=False)  # swing: |a| >> 1 g
    assert f.roll_deg == pytest.approx(90, abs=1.0)
    assert not f.still


def test_accel_ignored_mid_swing_but_corrects_drift_when_still():
    f = make_filter()
    f.update(tilted(0.0, 0))
    f.calibrate_neutral()
    # during a violent swing the accelerometer "roll" is garbage and must not pull the estimate
    t = 0.0
    for i in range(20):
        t += 0.02
        f.update(ImuSample(t, 2.5 * G, 1.8 * G, -0.4 * G, 0, 0, 0))
    assert abs(f.roll_deg) < 1.0
    # a gyro bias drifts the estimate while the paddle rests at 0 deg...
    f2 = make_filter()
    t = 0.0
    for i in range(500):
        t += 0.02
        f2.update(tilted(t, 0, gyro_x=50 / DPR if i < 50 else 0))   # 50 deg/s for 1 s = 50 deg of fake twist
    # ...and the accelerometer pulls it back while still
    assert abs(f2.roll_deg) < 2.0


def test_frozen_accel_correction_during_haptics():
    f = make_filter()
    f.update(tilted(0.0, 0))
    f.calibrate_neutral()
    t = 0.0
    for i in range(50):
        t += 0.02
        f.update(tilted(t, 40), freeze_accel=True)   # the hub reports 40 deg, but we're in the haptic window
    assert abs(f.roll_deg) < 0.5
    assert f.frozen


def test_calibrate_neutral_sets_zero_and_wraps():
    f = make_filter()
    f.update(tilted(0.0, 170))
    f.calibrate_neutral()
    assert f.roll_deg == pytest.approx(0)
    f.update(tilted(0.02, -170, gyro_x=20 / 0.02 / DPR))  # +20 deg twist through the +/-180 seam
    assert f.roll_deg == pytest.approx(20, abs=1.0)
    assert wrap_deg(190) == pytest.approx(-170)


def test_roll_history_lookup():
    f = make_filter()
    f.update(tilted(0.0, 0))
    f.calibrate_neutral()
    for i in range(1, 11):
        f.update(tilted(i * 0.02, 0, gyro_x=500 / DPR, accel_scale=2.0))
    assert f.roll_at(0.1) == pytest.approx(50, abs=1.5)
    assert f.roll_at(0.021) == pytest.approx(10, abs=0.5)
    assert f.roll_at(-1.0) == pytest.approx(10, abs=0.5)   # earlier than the history: oldest value


# -- swing detector: twist is not a swing, but becomes spin --------------------------------

def test_twisting_the_handle_is_not_a_swing():
    det = SwingDetector(G, accel_threshold_g=1.0, gyro_threshold=2000, mode="either", cooldown_s=0.4,
                        peak_window_s=0.1, roll_axis="x", exclude_roll=True)
    samples = [ImuSample(i * 0.02, 0, 0, G, 8000, 0, 0) for i in range(30)]   # big roll-axis rate only
    assert [e for s in samples if (e := det.update(s))] == []


def test_swing_records_twist_rate_and_vertical_motion():
    det = SwingDetector(G, accel_threshold_g=1.0, gyro_threshold=2000, mode="either", cooldown_s=0.4,
                        peak_window_s=0.1, roll_axis="x", exclude_roll=True, roll_sign=1.0, deg_per_raw=DPR)
    rest = [ImuSample(i * 0.02, 0, 0, G, 0, 0, 0) for i in range(10)]
    swing = [ImuSample(0.2 + i * 0.02, 0, 0, G + 2.0 * G, -6000 if i == 2 else -1000, 0, 4000) for i in range(8)]
    events = [e for s in rest + swing if (e := det.update(s))]
    assert len(events) == 1
    ev = events[0]
    assert ev.roll_rate_dps == pytest.approx(-600)       # strongest signed twist
    assert ev.vertical_g == pytest.approx(2.0, rel=0.05)  # upward swing


def test_ignored_samples_never_start_a_swing():
    det = SwingDetector(G, accel_threshold_g=1.0, gyro_threshold=2000, mode="either", cooldown_s=0.4,
                        peak_window_s=0.1)
    shaky = [ImuSample(i * 0.02, 3 * G, 0, G, 0, 0, 9000) for i in range(20)]
    assert [e for s in shaky if (e := det.update(s, ignore=True))] == []


# -- swing -> spin -------------------------------------------------------------------------

def test_spin_dead_zones_and_signs(monkeypatch):
    monkeypatch.setattr(config, "SPIN_SIDE_SIGN", 1.0)
    assert side_from_imu(config.SPIN_RATE_DEADZONE_DPS * 0.9, 0, "rate") == 0
    assert side_from_imu(config.SPIN_RATE_FULL_DPS, 0, "rate") == pytest.approx(1.0)
    assert side_from_imu(-config.SPIN_RATE_FULL_DPS, 0, "rate") == pytest.approx(-1.0)
    assert side_from_imu(0, config.SPIN_ANGLE_FULL_DEG, "angle") == pytest.approx(1.0)
    assert side_from_imu(0, config.SPIN_ANGLE_FULL_DEG, "rate") == 0
    both = side_from_imu(config.SPIN_RATE_FULL_DPS, config.SPIN_ANGLE_FULL_DEG, "both")
    assert both == pytest.approx(2.0)
    with pytest.raises(ValueError):
        side_from_imu(0, 0, "vibes")


def test_spin_for_swing_sources_and_clamp():
    big = SwingEvent(0, 1.0, "imu", roll_rate_dps=10 * config.SPIN_RATE_FULL_DPS, vertical_g=10.0)
    s = spin_for_swing(big, "rate")
    assert abs(s.side) == pytest.approx(config.SPIN_MAX) and s.top == pytest.approx(config.SPIN_MAX)
    down = SwingEvent(0, 1.0, "imu", vertical_g=-config.SPIN_TOP_FULL_G)
    assert spin_for_swing(down).top == pytest.approx(-1.0)        # downward chop = backspin
    calm = SwingEvent(0, 1.0, "imu", roll_rate_dps=10, vertical_g=0.1)
    assert spin_for_swing(calm) == NO_SPIN
    kb = SwingEvent(0, 1.0, "keyboard", spin_side=-0.8, spin_top=None)
    assert spin_for_swing(kb) == Spin(-0.8, 0.0)
    assert spin_for_swing(SwingEvent(0, 1.0, "keyboard")) == NO_SPIN


def test_spin_labels():
    assert spin_label(Spin(0.9, 0.1)) == "CURVE RIGHT!"
    assert spin_label(Spin(-0.9, 0.1)) == "CURVE LEFT!"
    assert spin_label(Spin(0.1, 0.9)) == "TOPSPIN!"
    assert spin_label(Spin(0.1, -0.9)) == "BACKSPIN!"
    assert spin_label(Spin(0.1, 0.1)) is None


# -- spin flights --------------------------------------------------------------------------

def test_no_spin_flight_is_unchanged():
    f = Flight(0, 3.0, 0.5, -0.1, t0=1.0, speed=3.1)
    assert f.duration == pytest.approx(1.0)
    assert f.curve_x == 0


def test_sidespin_curves_and_lands_where_reported():
    f = make_flight(0.0, config.PLAYER_HIT_Y_M, 0.0, config.OPPONENT_HIT_Y_M, 0.0, 3.0, Spin(1.0, 0.0))
    assert f.end_x > 0.05                                    # curved to the right of the aim
    assert f.curve_x == pytest.approx(magnus_curve_m(1.0, 3.0, f.duration))
    x_end, y_end, _ = f.position(f.t_end)
    assert x_end == pytest.approx(f.end_x) and y_end == pytest.approx(config.OPPONENT_HIT_Y_M)
    xm, _, _ = f.position(f.t0 + f.duration / 2)
    assert xm < f.end_x / 2                                  # bends late: a curve, not a straight line


def test_sidespin_never_leaves_the_table():
    half = config.TABLE_WIDTH_M / 2
    f = make_flight(0.6, config.PLAYER_HIT_Y_M, 0.65, config.OPPONENT_HIT_Y_M, 0.0, 4.5, Spin(5.0, 0.0))
    assert abs(f.end_x) < half


def test_topspin_dips_and_kicks_backspin_floats_and_checks():
    top, back, flat = flight_params(Spin(0, 1)), flight_params(Spin(0, -1)), flight_params(NO_SPIN)
    assert top.arc_mult < flat.arc_mult < back.arc_mult
    assert top.bounce_fraction < flat.bounce_fraction < back.bounce_fraction
    assert top.post_speed_mult > 1 > back.post_speed_mult
    assert top.second_arc_ratio < flat.second_arc_ratio < back.second_arc_ratio
    ft = make_flight(0, 0, 0, 3, 0, 3.0, Spin(0, 1))
    fb = make_flight(0, 0, 0, 3, 0, 3.0, Spin(0, -1))
    f0 = make_flight(0, 0, 0, 3, 0, 3.0)
    assert ft.duration < f0.duration < fb.duration
    # ball is on the table at the bounce, arrives at the receiver's plane on time
    assert ft.position(ft.t0 + ft.t_bounce)[2] == pytest.approx(0, abs=1e-9)
    assert fb.position(fb.t_end)[1] == pytest.approx(3.0)


def test_spin_goes_to_opponent_and_label_shows():
    class Inp:
        def paddle_x_at(self, t): return 0.0
        def direction_at(self, t): return "center"
        def current_paddle_x(self): return 0.0

    seen = []

    class Opp(SimulatedOpponent):
        def serve(self, now):
            return Shot(0.0, 0.0, self.ball_speed)

        def ball_incoming(self, ball):
            seen.append(ball)
            super().ball_incoming(ball)

    g = GameLogic(Opp(rng=random.Random(0)), Inp(), rng=random.Random(0))
    g.start(0)
    g.update(config.SERVE_DELAY_S)
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(arrive, 1.0, "keyboard", spin_side=0.0, spin_top=0.9))
    g.update(arrive + 0.01)
    assert g.state == "to_opponent"
    assert seen[-1].spin == Spin(0.0, 0.9)
    assert seen[-1].x == pytest.approx(g.flight.end_x)
    assert seen[-1].arrival_time == pytest.approx(g.flight.t_end)
    assert g.spin_text == "TOPSPIN!"
    assert g.event_times["player_hit"] == pytest.approx(g.flight.t0)


# -- opponent reacts to spin -----------------------------------------------------------------

def test_heavy_spin_makes_simulated_opponent_miss_more(monkeypatch):
    def miss_rate(spin, difficulty):
        opp = SimulatedOpponent(rng=random.Random(4), miss_probability=0.0)
        opp.set_difficulty(difficulty, 3.0)
        misses = 0
        for i in range(2000):
            opp.ball_incoming(IncomingBall(0.0, 1.0, 3.0, spin=spin))
            misses += opp.poll(2.0) is MISS
        return misses / 2000

    assert miss_rate(NO_SPIN, "Easy") == 0
    easy = miss_rate(Spin(1.0, 0.0), "Easy")
    hard = miss_rate(Spin(1.0, 0.0), "Hard")
    assert easy == pytest.approx(config.OPPONENT_SPIN_MISS_MAX * config.OPPONENT_SPIN_PENALTY["Easy"], abs=0.04)
    assert easy > hard > 0


def test_spin_scatter_stays_on_table():
    opp = SimulatedOpponent(rng=random.Random(1), miss_probability=0.0)
    opp.set_difficulty("Easy", 3.0)
    monkey_half = config.TABLE_WIDTH_M / 2
    for _ in range(300):
        opp.ball_incoming(IncomingBall(0.0, 1.0, 3.0, spin=Spin(1.0, 1.0)))
        shot = opp.poll(2.0)
        if shot is not MISS:
            assert abs(shot.target_x) < monkey_half
