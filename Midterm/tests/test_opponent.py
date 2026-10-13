"""Simulated opponent behind the Opponent interface."""

import random

import pytest

import config
from opponent import MISS, IncomingBall, Opponent, Shot, SimulatedOpponent, clamp_to_table


def test_interface_is_abstract():
    with pytest.raises(TypeError):
        Opponent()


def test_serve_uses_difficulty_speed_and_stays_on_table():
    opp = SimulatedOpponent(rng=random.Random(0))
    opp.set_difficulty("Medium", 3.2)
    for _ in range(100):
        s = opp.serve(0)
        assert s.speed == 3.2
        assert abs(s.target_x) <= config.TABLE_WIDTH_M / 2


def test_returns_after_arrival_with_random_placement():
    opp = SimulatedOpponent(rng=random.Random(0), reaction_s=0.05, miss_probability=0.0)
    targets = set()
    for i in range(20):
        opp.ball_incoming(IncomingBall(x=0.3, arrival_time=10.0, speed=3.0))
        assert opp.poll(9.99) is None
        shot = opp.poll(10.06)
        assert isinstance(shot, Shot) and shot.start_x == 0.3
        targets.add(round(shot.target_x, 3))
        assert opp.poll(10.1) is None   # only once per ball
    assert len(targets) > 5            # randomness in placement


def test_miss_probability():
    opp = SimulatedOpponent(rng=random.Random(0), miss_probability=1.0)
    opp.ball_incoming(IncomingBall(0, 1.0, 3.0))
    assert opp.poll(2.0) is MISS


def test_reset_forgets_ball():
    opp = SimulatedOpponent()
    opp.ball_incoming(IncomingBall(0, 1.0, 3.0))
    opp.reset()
    assert opp.poll(5.0) is None


def test_clamp_to_table():
    half = config.TABLE_WIDTH_M / 2
    assert clamp_to_table(5.0) < half and clamp_to_table(-5.0) > -half
    assert clamp_to_table(0.1) == 0.1
