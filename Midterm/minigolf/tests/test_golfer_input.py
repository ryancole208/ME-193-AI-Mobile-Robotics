"""Aim from the keyboard and a (fake) webcam wrist joystick, push/pull, keyboard power meter."""

import pytest

import config
from golfer_input import GolferInput, wrap_deg
from pose_tracker import PoseSnapshot


class FakeTwist:
    """Stands in for GolfMotor's twist_deg / recenter_twist()."""

    def __init__(self, deg=0.0):
        self.twist_deg = deg
        self.recentered = 0

    def recenter_twist(self):
        self.recentered += 1
        if self.twist_deg is not None:
            self.twist_deg = 0.0


class FakePose:
    def __init__(self):
        self.snap = PoseSnapshot()

    @property
    def tracking(self):
        return self.snap.tracking

    def snapshot(self):
        return self.snap


def wrist(pose, t, x):
    pose.snap = PoseSnapshot(t=t, tracking=True, wrist_x=x)


def test_wrap():
    assert wrap_deg(190) == -170 and wrap_deg(-190) == 170 and wrap_deg(30) == 30


def test_keyboard_turns_aim_and_fine_is_slower():
    g = GolferInput()
    g.update(0.0, 1.0, right=True)
    assert g.aim_deg == pytest.approx(config.AIM_KEY_RATE_DEG_S)
    g.update(1.0, 1.0, left=True, fine=True)
    assert g.aim_deg == pytest.approx(config.AIM_KEY_RATE_DEG_S * (1 - config.AIM_KEY_FINE_FACTOR))
    assert g.source == "keyboard"


def test_frozen_aim_does_not_turn():
    g = GolferInput()
    g.update(0.0, 1.0, right=True, frozen=True)
    assert g.aim_deg == 0.0


def test_wrist_joystick_is_off_by_default():
    pose = FakePose()
    g = GolferInput(pose)
    g._smoother.alpha = 0.0
    wrist(pose, 0.0, config.POSE_X_RANGE[1])
    g.update(0.0, 1.0)
    assert g.stick == pytest.approx(1.0) and g.aim_deg == 0.0


def test_wrist_joystick_turns_aim_outside_deadzone(monkeypatch):
    monkeypatch.setattr(config, "POSE_SMOOTHING", 0.0)
    monkeypatch.setattr(config, "AIM_POSE_JOYSTICK", True)
    pose = FakePose()
    g = GolferInput(pose)
    g._smoother.alpha = 0.0
    wrist(pose, 0.0, 0.5)                               # centre: holds still
    g.update(0.0, 1.0)
    assert g.stick == 0.0 and g.aim_deg == 0.0 and g.source == "pose"
    wrist(pose, 0.1, config.POSE_X_RANGE[1])            # far right: full rate
    g.update(0.1, 1.0)
    assert g.stick == pytest.approx(1.0)
    assert g.aim_deg == pytest.approx(config.AIM_POSE_RATE_DEG_S)
    wrist(pose, 0.2, config.POSE_X_RANGE[0])            # far left
    g.update(0.2, 0.5)
    assert g.stick == pytest.approx(-1.0)
    assert g.aim_deg == pytest.approx(config.AIM_POSE_RATE_DEG_S / 2)


def test_lost_tracking_stops_the_joystick(monkeypatch):
    monkeypatch.setattr(config, "AIM_POSE_JOYSTICK", True)
    pose = FakePose()
    g = GolferInput(pose)
    g._smoother.alpha = 0.0
    wrist(pose, 0.0, config.POSE_X_RANGE[1])
    g.update(0.0, 0.0)
    pose.snap = PoseSnapshot(t=0.1, tracking=False)
    g.update(0.1, 1.0)
    assert g.stick == 0.0 and g.aim_deg == 0.0


def test_aim_history_reads_the_aim_at_stroke_onset():
    g = GolferInput()
    for k in range(10):
        g.update(k * 0.1, 0.1, right=True)
    early = g.aim_at(0.2)
    assert early == pytest.approx(3 * 0.1 * config.AIM_KEY_RATE_DEG_S)
    assert g.aim_at(0.95) == pytest.approx(g.aim_deg)


def test_set_aim_resets_history():
    g = GolferInput()
    g.update(0.0, 1.0, right=True)
    g.set_aim(-30.0, 5.0)
    assert g.aim_at(0.0) == -30.0 and g.aim_at(5.0) == -30.0


def test_push_from_sideways_wrist_motion_around_impact():
    pose = FakePose()
    g = GolferInput(pose)
    for k in range(20):
        t = k * 0.03
        wrist(pose, t, 0.5 + 0.004 * k)                 # drifting right during the stroke
        g.update(t, 0.03)
    push = g.push_deg(0.4)
    assert 0 < push <= config.PUSH_MAX_DEG
    assert g.putt_heading_deg(0.0, 0.4) == pytest.approx(g.aim_at(0.0) + push)


def test_no_push_without_tracking_or_inside_deadzone():
    assert GolferInput().push_deg(1.0) == 0.0
    pose = FakePose()
    g = GolferInput(pose)
    for k in range(20):
        wrist(pose, k * 0.03, 0.5)
        g.update(k * 0.03, 0.03)
    assert g.push_deg(0.4) == 0.0


def test_keyboard_charge_meter_sweeps_up_and_down():
    g = GolferInput()
    assert g.release_charge(0.0) is None
    g.begin_charge(10.0)
    half = config.KEYBOARD_CHARGE_S / 2
    assert g.charge_fraction(10.0 + half) == pytest.approx(0.5)
    assert g.charge_fraction(10.0 + config.KEYBOARD_CHARGE_S) == pytest.approx(1.0)
    assert g.charge_fraction(10.0 + 1.5 * config.KEYBOARD_CHARGE_S) == pytest.approx(0.5)
    lo, hi = config.KEYBOARD_STRENGTH_RANGE
    assert g.release_charge(10.0 + config.KEYBOARD_CHARGE_S) == pytest.approx(hi)
    assert g.charge_t is None
    g.begin_charge(0.0)
    assert g.release_charge(0.0) == pytest.approx(lo)


def test_motor_twist_turns_the_aim_from_the_suggested_line():
    tw = FakeTwist()
    g = GolferInput(twist=tw)
    g.set_aim(30.0, 0.0)
    assert tw.recentered == 1 and g.aim_deg == 30.0 and g.twisting
    tw.twist_deg = 15.0
    g.update(0.1, 0.1)
    assert g.twist_offset_deg == pytest.approx(15.0 * config.AIM_TWIST_GAIN)
    assert g.aim_deg == pytest.approx(30.0 + 15.0 * config.AIM_TWIST_GAIN)
    g.update(0.2, 1.0, right=True)                       # keys turn the base on top of the twist
    assert g.base_aim_deg == pytest.approx(30.0 + config.AIM_KEY_RATE_DEG_S)
    assert g.aim_deg == pytest.approx(g.base_aim_deg + g.twist_offset_deg)


def test_twist_is_clamped_and_frozen_while_rolling():
    tw = FakeTwist()
    g = GolferInput(twist=tw)
    tw.twist_deg = 500.0
    g.update(0.0, 0.1)
    assert g.twist_offset_deg == config.AIM_TWIST_MAX_DEG
    tw.twist_deg = -10.0
    g.update(0.1, 0.1, frozen=True)
    assert g.aim_deg == config.AIM_TWIST_MAX_DEG


def test_new_putt_and_c_key_recenter_the_twist():
    tw = FakeTwist(20.0)
    g = GolferInput(twist=tw)
    g.update(0.0, 0.1)
    assert g.aim_deg == pytest.approx(20.0 * config.AIM_TWIST_GAIN)
    g.recenter()
    assert tw.twist_deg == 0.0 and g.twist_offset_deg == 0.0 and g.aim_deg == 0.0
    tw.twist_deg = 5.0
    g.set_aim(-90.0, 1.0)
    assert g.aim_deg == -90.0 and tw.twist_deg == 0.0


def test_uncalibrated_motor_adds_no_twist():
    g = GolferInput(twist=FakeTwist(None))
    g.update(0.0, 0.1)
    assert not g.twisting and g.aim_deg == 0.0


def test_aim_history_includes_twist_at_swing_time():
    tw = FakeTwist()
    g = GolferInput(twist=tw)
    for k in range(10):
        tw.twist_deg = 2.0 * k
        g.update(k * 0.1, 0.1)
    assert g.aim_at(0.31) == pytest.approx(6.0 * config.AIM_TWIST_GAIN)
