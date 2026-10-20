"""Behavior-tree engine semantics and the sumo tree's priority order."""

import pytest

import config
from behavior_tree import Action, Condition, Selector, Sequence, Status
from line_tracker import LineTracker, Zone
from robot_tree import Blackboard, active_branch, build_tree, lever_to_speed
from vision import Opponent

S, F, R = Status.SUCCESS, Status.FAILURE, Status.RUNNING


# ---------------------------------------------------------------- engine
class Script(Action):
    """Returns statuses from a list; records ticks and halts."""

    def __init__(self, name, statuses):
        super().__init__(name)
        self.statuses = list(statuses)
        self.ticks = 0
        self.halts = 0

    def tick(self, bb):
        self.ticks += 1
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def halt(self):
        self.halts += 1


def test_sequence_fails_fast_and_succeeds_when_all_succeed():
    a, b = Script("a", [F]), Script("b", [S])
    assert Sequence("s", [a, b]).tick(None) is F
    assert b.ticks == 0
    assert Sequence("s", [Script("a", [S]), Script("b", [S])]).tick(None) is S


def test_sequence_memory_resumes_running_child_without_rechecking_condition():
    cond_calls = []
    cond = Condition("c", lambda bb: cond_calls.append(1) or len(cond_calls) == 1)  # True only once
    act = Script("act", [R, R, S])
    seq = Sequence("s", [cond, act])
    assert seq.tick(None) is R
    assert seq.tick(None) is R      # condition would now be False, but isn't re-checked
    assert seq.tick(None) is S
    assert len(cond_calls) == 1


def test_selector_picks_first_non_failure_left_to_right():
    a, b, c = Script("a", [F]), Script("b", [S]), Script("c", [S])
    sel = Selector("sel", [a, b, c])
    assert sel.tick(None) is S
    assert sel.active_index == 1 and c.ticks == 0


def test_selector_is_reactive_and_halts_preempted_branch():
    flag = {"high": False}
    high = Sequence("high", [Condition("hc", lambda bb: flag["high"]), Script("ha", [S])])
    low = Script("low", [R])
    sel = Selector("sel", [high, low])
    assert sel.tick(None) is R and sel.active_index == 1
    flag["high"] = True
    assert sel.tick(None) is S and sel.active_index == 0
    assert low.halts == 1


def test_selector_all_fail():
    assert Selector("s", [Script("a", [F]), Script("b", [F])]).tick(None) is F


def test_action_must_return_status():
    with pytest.raises(TypeError):
        Action("bad", lambda bb: True).tick(None)


# ---------------------------------------------------------------- the sumo tree
class FakeDrive:
    def __init__(self):
        self.commanded = (0, 0)

    def set(self, l, r):
        self.commanded = (round(l), round(r))

    def stop(self):
        self.commanded = (0, 0)


class FakeLink:
    expects_song = True

    def __init__(self):
        self.song_done = False
        self.lost = []

    def announce_lost(self, reason):
        self.lost.append(reason)


def make_bb(**kw):
    bb = Blackboard(tracker=LineTracker(log=lambda *_: None), drive=FakeDrive(), link=FakeLink(),
                    levers=(0.0, 0.0), teleop_unlocked=True)
    for k, v in kw.items():
        setattr(bb, k, v)
    return bb


OPP = Opponent(seq=1, visible=True, x=0.5, distance_cm=30.0, confidence=0.9, frame_age_ms=40, received_at=0.0)


def tick(tree, bb, now=0.0):
    bb.now = now
    tree.tick(bb)
    return active_branch(tree)


def test_fallback_is_idle_stop():
    tree, bb = build_tree(), make_bb()
    bb.drive.commanded = (50, 50)
    assert tick(tree, bb) == "idle"
    assert bb.drive.commanded == (0, 0)


def test_teleop_beats_idle_and_respects_deadband_and_lock():
    tree = build_tree()
    assert tick(tree, make_bb(levers=(config.TELEOP_DEADBAND - 1, 0.0))) == "idle"
    assert tick(tree, make_bb(levers=(80.0, -80.0), teleop_unlocked=False)) == "idle"
    bb = make_bb(levers=(100.0, -100.0))
    assert tick(tree, bb) == "teleop"
    assert bb.drive.commanded == (config.TELEOP_MAX_SPEED, -config.TELEOP_MAX_SPEED)


def test_chase_beats_teleop():
    tree, bb = build_tree(), make_bb(levers=(80.0, 80.0), opponent=OPP)
    assert tick(tree, bb) == "chase"
    l, r = bb.drive.commanded
    assert l > r            # opponent on the right (x>0): turn right


def test_reverse_beats_chase_and_driver_and_runs_full_time():
    tree, bb = build_tree(), make_bb(levers=(80.0, 80.0), opponent=OPP)
    bb.tracker.zone = Zone.ON_EDGE
    assert tick(tree, bb, 0.0) == "reverse"
    assert bb.drive.commanded == (-config.REVERSE_SPEED, -config.REVERSE_SPEED)
    bb.tracker.zone = Zone.INSIDE                # blue disappears ...
    assert tick(tree, bb, 0.5) == "reverse"      # ... still reversing
    assert tick(tree, bb, config.REVERSE_TIME_S + 0.01) == "reverse"   # finishes (SUCCESS) this tick
    assert tick(tree, bb, config.REVERSE_TIME_S + 0.03) == "chase"


def test_color_safety_beats_reverse():
    tree, bb = build_tree(), make_bb(levers=(80.0, 80.0))
    bb.tracker.zone = Zone.ON_EDGE
    assert tick(tree, bb, 0.0) == "reverse"
    bb.color_fresh = False
    assert tick(tree, bb, 0.1) == "color_safety"
    assert bb.drive.commanded == (0, 0)
    # the interrupted reverse was halted: when the sensor returns it starts a fresh 1 s
    bb.color_fresh = True
    assert tick(tree, bb, 0.2) == "reverse"
    assert tick(tree, bb, 0.2 + config.REVERSE_TIME_S - 0.05) == "reverse"


def test_out_of_ring_beats_everything_waits_for_song_then_exits():
    tree, bb = build_tree(), make_bb(levers=(80.0, 80.0), opponent=OPP, color_fresh=False)
    bb.tracker.zone = Zone.OUT
    bb.tracker.out_reason = "test"
    assert tick(tree, bb, 0.0) == "out_of_ring"
    assert bb.drive.commanded == (0, 0) and bb.link.lost == ["test"] and not bb.exit_requested
    tick(tree, bb, 1.0)
    assert not bb.exit_requested
    bb.link.song_done = True
    tick(tree, bb, 1.5)
    assert bb.exit_requested and bb.link.lost == ["test"]     # announced once


def test_out_of_ring_times_out_without_song():
    tree, bb = build_tree(), make_bb()
    bb.tracker.zone = Zone.OUT
    tick(tree, bb, 0.0)
    tick(tree, bb, config.SONG_DONE_TIMEOUT_S - 0.1)
    assert not bb.exit_requested
    tick(tree, bb, config.SONG_DONE_TIMEOUT_S + 0.01)
    assert bb.exit_requested


def test_lever_to_speed_proportional_with_deadband():
    db = config.TELEOP_DEADBAND
    assert lever_to_speed(db) == 0
    assert lever_to_speed(100) == config.TELEOP_MAX_SPEED
    assert lever_to_speed(-100) == -config.TELEOP_MAX_SPEED
    mid = lever_to_speed((100 + db) / 2)
    assert mid == pytest.approx(config.TELEOP_MAX_SPEED / 2)
