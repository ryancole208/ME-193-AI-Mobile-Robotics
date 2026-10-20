"""Every line-crossing case of the state machine."""

import pytest

from line_tracker import LineTracker, Zone
from motion import Direction

FWD, BACK, UNC = Direction.FORWARD, Direction.BACKWARD, Direction.UNCERTAIN


def tracker(**kw):
    return LineTracker(log=lambda *_: None, **kw)


def test_starts_inside():
    assert tracker().zone is Zone.INSIDE


def test_blue_while_forward_is_edge_still_inside():
    t = tracker()
    t.on_blue(FWD)
    assert t.zone is Zone.ON_EDGE and not t.is_out


def test_blue_while_backward_is_out():
    t = tracker()
    t.on_blue(BACK)
    assert t.is_out and "backward" in t.out_reason


def test_blue_then_floor_forward_is_out_by_default():
    t = tracker()
    t.on_blue(FWD)
    t.on_floor(FWD)
    assert t.is_out and "crossed" in t.out_reason


def test_blue_then_floor_forward_option_off_stays_inside():
    t = tracker(cross_forward_is_out=False)
    t.on_blue(FWD)
    t.on_floor(FWD)
    assert t.zone is Zone.INSIDE


def test_blue_then_floor_backward_is_back_inside():
    t = tracker()
    t.on_blue(FWD)
    t.on_floor(BACK)
    assert t.zone is Zone.INSIDE
    t.on_blue(FWD)                     # can happen again and again
    assert t.zone is Zone.ON_EDGE


@pytest.mark.parametrize("lean,expect_out", [("in", False), ("out", True)])
def test_uncertain_blue_follows_lean(lean, expect_out):
    t = tracker(lean=lean)
    t.on_blue(UNC)
    assert t.is_out is expect_out
    assert t.uncertain_events == 1


@pytest.mark.parametrize("lean,expect_out", [("in", False), ("out", True)])
def test_uncertain_floor_follows_lean(lean, expect_out):
    t = tracker(lean=lean)
    t.on_blue(FWD)
    t.on_floor(UNC)
    assert t.is_out is expect_out


def test_out_is_latched():
    t = tracker()
    t.on_blue(BACK)
    t.on_floor(BACK)
    t.on_blue(FWD)
    assert t.is_out


def test_events_in_wrong_state_are_ignored():
    t = tracker()
    t.on_floor(FWD)                    # already on floor
    assert t.zone is Zone.INSIDE
    t.on_blue(FWD)
    t.on_blue(BACK)                    # already on the tape
    assert t.zone is Zone.ON_EDGE


def test_overshoot_within_grace_is_not_out_and_recovers():
    t = tracker(grace_s=0.5, recover_s=2.0)
    t.on_blue(FWD, now=10.0)
    t.on_floor(FWD, now=10.3)          # momentum carried the sensor past the tape
    assert t.zone is Zone.ON_EDGE and not t.is_out      # keep reversing
    t.update(11.0)
    t.on_blue(BACK, now=11.2)          # reversing back over the tape
    t.on_floor(BACK, now=11.4)         # ... and off it, inside
    assert t.zone is Zone.INSIDE
    t.update(20.0)
    assert not t.is_out


def test_forward_crossing_after_grace_is_out():
    t = tracker(grace_s=0.5)
    t.on_blue(FWD, now=10.0)
    t.on_floor(FWD, now=10.6)
    assert t.is_out and "crossed" in t.out_reason


def test_overshoot_that_never_comes_back_is_out():
    t = tracker(grace_s=0.5, recover_s=2.0)
    t.on_blue(FWD, now=10.0)
    t.on_floor(FWD, now=10.2)
    t.update(12.0)
    assert not t.is_out
    t.update(12.3)
    assert t.is_out and "did not come back" in t.out_reason and "crossed" in t.out_reason


def test_flicker_while_backing_off_is_same_tape_not_out():
    t = tracker(retape_s=1.5)
    t.on_blue(FWD, now=10.0)
    t.on_floor(BACK, now=10.4)         # backing off the tape ...
    t.on_blue(BACK, now=10.5)          # ... reading flickers back to blue at its edge
    assert t.zone is Zone.ON_EDGE and not t.is_out
    t.on_floor(BACK, now=10.6)
    assert t.zone is Zone.INSIDE


def test_blue_while_backward_long_after_backing_off_is_out():
    t = tracker(retape_s=1.5)
    t.on_blue(FWD, now=10.0)
    t.on_floor(BACK, now=10.4)
    t.on_blue(BACK, now=15.0)          # a different crossing, much later: pushed out
    assert t.is_out and "backward" in t.out_reason


def test_bad_lean_rejected():
    with pytest.raises(ValueError):
        tracker(lean="maybe")
