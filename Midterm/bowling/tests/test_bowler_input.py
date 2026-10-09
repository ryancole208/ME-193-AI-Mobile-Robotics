"""Bowler position and aim from the keyboard and from (fake) pose snapshots."""

import pytest

import config
from bowler_input import STAND_RANGE_M, BowlerInput
from pose_tracker import PoseSnapshot


class FakePose:
    def __init__(self):
        self.snap = PoseSnapshot()

    @property
    def tracking(self):
        return self.snap.tracking

    def snapshot(self):
        return self.snap


def test_keyboard_moves_and_clamps():
    b = BowlerInput()
    b.update(0, 10.0, right=True)
    assert b.keyboard_x == pytest.approx(STAND_RANGE_M / 2)
    assert b.current_position() == b.keyboard_x and b.source == "keyboard"
    b.update(0, 10.0, aim_left=True)
    assert b.preset_aim_deg == -config.PRESET_AIM_MAX_DEG


def test_keyboard_strength_steps_and_clamps():
    b = BowlerInput()
    b.change_strength(+100)
    assert b.kb_strength == config.KEYBOARD_STRENGTH_RANGE[1]
    b.change_strength(-100)
    assert b.kb_strength == config.KEYBOARD_STRENGTH_RANGE[0]


def test_pose_position_follows_wrist(monkeypatch):
    monkeypatch.setattr(config, "POSE_SMOOTHING", 0.0)
    pose = FakePose()
    b = BowlerInput(pose)
    b._smoother.alpha = 0.0
    lo, hi = config.POSE_X_RANGE
    pose.snap = PoseSnapshot(t=1.0, tracking=True, wrist_x=hi)
    b.update(1.0, 0.016)
    assert b.source == "pose"
    assert b.current_position() == pytest.approx(STAND_RANGE_M / 2)
    assert b.position_at(1.0) == pytest.approx(STAND_RANGE_M / 2)
    pose.snap = PoseSnapshot(t=2.0, tracking=False)
    b.update(2.0, 0.016)
    assert b.current_position() == b.keyboard_x


def test_swing_aim_from_wrist_motion():
    pose = FakePose()
    b = BowlerInput(pose)
    for i in range(30):          # wrist drifts right during the forward swing
        t = i * 0.033
        pose.snap = PoseSnapshot(t=t, tracking=True, wrist_x=0.5 + 0.004 * i)
        b.update(t, 0.033)
    aim = b.swing_aim_deg(0.6)
    assert 0 < aim <= config.SWING_AIM_MAX_DEG
    b.preset_aim_deg = -1.0
    assert b.launch_angle_deg(0.6) == pytest.approx(-1.0 + aim)


def test_no_swing_aim_without_tracking_or_motion():
    b = BowlerInput()
    assert b.swing_aim_deg(1.0) == 0.0
    pose = FakePose()
    b = BowlerInput(pose)
    for i in range(30):
        pose.snap = PoseSnapshot(t=i * 0.033, tracking=True, wrist_x=0.5)
        b.update(i * 0.033, 0.033)
    assert b.swing_aim_deg(0.6) == 0.0
