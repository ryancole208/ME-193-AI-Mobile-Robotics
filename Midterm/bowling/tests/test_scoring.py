"""Ten-pin scoring: strikes, spares, the 10th frame, marks and running totals."""

import pytest

from scoring import ScoreSheet


def play(rolls):
    s = ScoreSheet()
    for r in rolls:
        s.add_roll(r)
    return s


def test_perfect_game():
    s = play([10] * 12)
    assert s.game_over
    assert s.frame_scores()[-1] == 300
    assert s.running_total() == 300
    assert s.marks(0) == ["", "X"]
    assert s.marks(9) == ["X", "X", "X"]


def test_gutter_game():
    s = ScoreSheet()
    for _ in range(20):
        s.add_roll(0, gutter=True)
    assert s.game_over and s.running_total() == 0
    assert s.marks(0) == ["G", "G"]


def test_all_spares_with_five():
    s = play([5] * 21)
    assert s.game_over
    assert s.frame_scores()[-1] == 150
    assert s.marks(3) == ["5", "/"]
    assert s.marks(9) == ["5", "/", "5"]


def test_open_frames_and_cumulative_totals():
    s = play([3, 4, 9, 0, 0, 7])
    assert s.frame_scores()[:3] == [7, 16, 23]
    assert s.marks(1) == ["9", "-"]
    assert s.current_frame == 3


def test_strike_and_spare_bonus_pending_until_next_rolls():
    s = play([10])
    assert s.frame_scores()[0] is None
    assert s.running_total() == 10
    s.add_roll(3)
    assert s.frame_scores()[0] is None
    assert s.running_total() == 16      # 10 + 3 (+3 bonus) so far
    s.add_roll(6)
    assert s.frame_scores()[:2] == [19, 28]
    s = play([7, 3])
    assert s.frame_scores()[0] is None
    s.add_roll(4)
    assert s.frame_scores()[0] == 14


def test_later_frame_waits_for_earlier_pending_frame():
    s = play([10, 10, 3])
    assert s.frame_scores()[:3] == [23, None, None]


def test_pins_standing_and_roll_validation():
    s = play([7])
    assert s.pins_standing() == 3
    with pytest.raises(ValueError):
        s.add_roll(4)
    s.add_roll(3)
    assert s.pins_standing() == 10


def test_tenth_frame_variants():
    base = [0, 0] * 9
    open_tenth = play(base + [3, 4])
    assert open_tenth.game_over and open_tenth.running_total() == 7

    spare_tenth = play(base + [6, 4])
    assert not spare_tenth.game_over and spare_tenth.pins_standing() == 10
    spare_tenth.add_roll(10)
    assert spare_tenth.game_over and spare_tenth.running_total() == 20
    assert spare_tenth.marks(9) == ["6", "/", "X"]

    strike_then_some = play(base + [10, 7])
    assert strike_then_some.pins_standing() == 3
    strike_then_some.add_roll(3)
    assert strike_then_some.marks(9) == ["X", "7", "/"]
    assert strike_then_some.running_total() == 20

    s = play(base + [10, 10])
    assert s.pins_standing() == 10


def test_cannot_roll_after_game_over():
    s = play([10] * 12)
    with pytest.raises(ValueError):
        s.add_roll(1)
