"""
Tests for the helper tools (calibrate_color.py, hardware_check.py), the
robot<->motor frame mapping in lego_io.py, and the overshoot estimate.
No hardware: the "sensor data" is synthetic, generated from known answers.
"""

import math
import random

import pytest

import config
from calibrate_color import class_summary, leave_one_out_accuracy, separation
from color_classifier import BLUE, FLOOR, KnnColorClassifier
from hardware_check import (direct_drive_analysis, push_analysis, rate_hz, rest_analysis,
                            spin_analysis, tilt_analysis)
from lego_io import from_motor_frame, to_motor_frame


# ---------------------------------------------------------------------------
# calibrate_color.py
# ---------------------------------------------------------------------------

def _samples(floor=(195, 190, 192, 45), blue=(45, 85, 170, 14), noise=3, n=30, seed=1):
    rng = random.Random(seed)
    out = []
    for label, base in ((FLOOR, floor), (BLUE, blue)):
        out += [(label, tuple(v + rng.gauss(0, noise) for v in base)) for _ in range(n)]
    return out


def test_well_separated_surfaces_score_high():
    s = _samples()
    acc, mistakes = leave_one_out_accuracy(s)
    assert acc == 1.0 and not mistakes
    assert separation(s) > 10


def test_similar_surfaces_score_low():
    s = _samples(blue=(190, 186, 195, 43), noise=6)
    acc, _ = leave_one_out_accuracy(s)
    assert separation(s) < 3
    assert acc < 0.95


def test_mislabeled_recording_is_caught():
    s = _samples()
    s += [(BLUE, f) for _, f in _samples(n=5, seed=7) if _ == FLOOR]   # floor readings saved as blue
    acc, mistakes = leave_one_out_accuracy(s)
    assert mistakes and acc < 1.0


def test_class_summary_and_save_roundtrip(tmp_path):
    s = _samples(n=10)
    summ = class_summary(s)
    assert summ[FLOOR][0] == 10 and summ[BLUE][0] == 10
    assert summ[FLOOR][1][3] == pytest.approx(45, abs=3)
    path = tmp_path / "c.json"
    KnnColorClassifier.save_samples(path, s, note="test")
    back = KnnColorClassifier.load_samples(path)
    assert len(back) == 20 and back[0][0] == FLOOR


# ---------------------------------------------------------------------------
# hardware_check.py: synthetic recordings with a known "truth"
# ---------------------------------------------------------------------------

TRUE_WHEEL_TO_MPS = 0.0028


def _push_rows(left_sign=-1, right_sign=+1, fwd_axis=0, fwd_sign=+1, distance=0.30,
               swap=False, dt=0.03, seed=2):
    """Hand push: speed ramps up, cruises, ramps down (a trapezoid), total = distance."""
    rng = random.Random(seed)
    t_acc, t_cruise = 0.25, 0.75
    v_max = distance / (t_acc + t_cruise)          # area of the trapezoid = distance
    rows, t, v_prev = [], 0.0, 0.0
    total = 2 * t_acc + t_cruise + 0.3
    while t <= total:
        if t < t_acc:
            v = v_max * t / t_acc
        elif t < t_acc + t_cruise:
            v = v_max
        elif t < 2 * t_acc + t_cruise:
            v = v_max * (2 * t_acc + t_cruise - t) / t_acc
        else:
            v = 0.0
        a = (v - v_prev) / dt
        v_prev = v
        units = v / TRUE_WHEEL_TO_MPS
        robot_l, robot_r = left_sign * units, right_sign * units
        m0, m1 = (robot_r, robot_l) if swap else (robot_l, robot_r)
        acc = [rng.gauss(0, 3), rng.gauss(0, 3), 1000 + rng.gauss(0, 3)]
        acc[fwd_axis] += fwd_sign * a / 9.81 * 1000
        rows.append((t, m0 + rng.gauss(0, 0.5), m1 + rng.gauss(0, 0.5), *acc, 0.0, 0.0))
        t += dt
    return rows


def _rest_rows(n=60):
    rng = random.Random(3)
    return [(i * 0.03, 0, 0, rng.gauss(0, 3), rng.gauss(0, 3), 1000 + rng.gauss(0, 3), 15.0, -4.0)
            for i in range(n)]


def test_rest_finds_gravity_axis():
    rest = rest_analysis(_rest_rows())
    assert rest["gravity_axis"] == "z"
    assert rest["raw_per_g"] == pytest.approx(1000, abs=5)


@pytest.mark.parametrize("ls,rs,axis,sgn", [(-1, +1, 0, +1), (+1, -1, 1, -1), (+1, +1, 0, -1)])
def test_push_recovers_the_truth(ls, rs, axis, sgn):
    rest = rest_analysis(_rest_rows())
    out = push_analysis(_push_rows(ls, rs, axis, sgn), 0.30, rest)
    assert (out["left_sign"], out["right_sign"]) == (ls, rs)
    assert out["wheel_to_mps"] == pytest.approx(TRUE_WHEEL_TO_MPS, rel=0.05)
    assert out["forward_axis"] == "xy"[axis] and out["forward_sign"] == sgn
    assert abs(out["forward_corr"]) > 0.7


def test_push_with_swapped_sides_reports_robot_frame_signs():
    rest = rest_analysis(_rest_rows())
    out = push_analysis(_push_rows(-1, +1, swap=True), 0.30, rest, swap=True)
    assert (out["left_sign"], out["right_sign"]) == (-1, +1)


def test_push_without_motion_is_an_error():
    rest = rest_analysis(_rest_rows())
    out = push_analysis(_rest_rows(), 0.30, rest)
    assert "error" in out


def test_tilt_gain():
    rest = rest_analysis(_rest_rows())
    # nose lifted 25 deg = 250 decidegrees on "pitch"; forward axis x sees sin(25 deg) g
    leak = 1000 * math.sin(math.radians(25))
    rows = [(0, 0, 0, leak, 0, 906, 15.0 + 250, -4.0)] * 20
    out = tilt_analysis(rest, rows, "x")
    assert out["field"] == "pitch"
    assert out["gain"] == pytest.approx(leak / 250, rel=0.02)
    barely = tilt_analysis(rest, [(0, 0, 0, 5, 0, 1000, 25.0, -4.0)] * 5, "x")
    assert "error" in barely


def test_direct_drive_detection():
    still = [(i, 0, 0, 100, 100) for i in range(30)]
    moving = [(i, 60, -60, 100, 100) for i in range(30)]
    no_push = [(i, 60, -60, 0, 0) for i in range(30)]
    assert direct_drive_analysis(still) == (True, False)
    assert direct_drive_analysis(moving) == (True, True)
    assert direct_drive_analysis(no_push) == (False, False)


@pytest.mark.parametrize("speed0,fwd,side,inverted,swap", [
    (+40, (+1, -1), "l", False, False),   # normal build
    (-40, (+1, -1), "l", True, False),    # + command turns the wheel backward
    (-40, (+1, -1), "r", True, True),     # Double Motor mounted facing backward
])
def test_spin_analysis(speed0, fwd, side, inverted, swap):
    rows = [(i, speed0, 0, 0, 0, 1000, 0, 0) for i in range(20)]
    out = spin_analysis(rows, fwd, side)
    assert out["inverted"] is inverted and out["swap"] is swap


def test_rate():
    assert rate_hz([0, 0.03, 0.06, 0.09]) == pytest.approx(33.3, abs=0.1)
    assert rate_hz([1.0]) == 0.0


# ---------------------------------------------------------------------------
# lego_io robot frame <-> motor frame
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inv,swap,expected", [
    (False, False, (60, 20)), (True, False, (-60, -20)),
    (False, True, (20, 60)), (True, True, (-20, -60)),
])
def test_to_motor_frame(inv, swap, expected):
    assert to_motor_frame(60, 20, inverted=inv, swap=swap) == expected


def test_from_motor_frame():
    assert from_motor_frame("m0", "m1", swap=False) == ("m0", "m1")
    assert from_motor_frame("m0", "m1", swap=True) == ("m1", "m0")


def test_default_config_is_identity():
    assert to_motor_frame(30, -10) == (30, -10)
    assert config.DRIVE_INVERTED is False and config.DRIVE_SWAP_SIDES is False
