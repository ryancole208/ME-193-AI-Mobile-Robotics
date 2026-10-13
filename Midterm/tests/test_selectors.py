"""Manual difficulty selector and the AprilTag stub's tag -> command logic."""

import pytest

from apriltag_selector import AprilTagSelector, tags_to_commands
from difficulty_selector import START, Command, ManualSelector, difficulty


def test_manual_selector_queues_and_clears():
    sel = ManualSelector()
    sel.push(difficulty("Hard"))
    sel.push(START)
    assert sel.poll() == [Command("difficulty", "Hard"), START]
    assert sel.poll() == []


def test_unknown_difficulty_rejected():
    with pytest.raises(ValueError):
        difficulty("Insane")


def test_tags_to_commands_default_map():
    assert tags_to_commands([0, 2, 99]) == [START, Command("difficulty", "Medium")]


def test_apriltag_debounce_and_refire():
    now = [0.0]
    sel = AprilTagSelector(hold_s=0.5, clock=lambda: now[0])
    sel.feed_tag_ids([3]); assert sel.poll() == []
    now[0] = 0.6; sel.feed_tag_ids([3])
    assert sel.poll() == [Command("difficulty", "Hard")]
    now[0] = 1.0; sel.feed_tag_ids([3]); assert sel.poll() == []   # still visible: no repeat
    now[0] = 1.1; sel.feed_tag_ids([])                              # leaves view
    now[0] = 1.2; sel.feed_tag_ids([3])
    now[0] = 1.8; sel.feed_tag_ids([3])
    assert sel.poll() == [Command("difficulty", "Hard")]


def test_apriltag_detection_is_stub():
    with pytest.raises(NotImplementedError):
        AprilTagSelector().feed_frame(None)
