"""Lane geometry and roll physics: pin layout, splits, pocket hits, gutters, bumpers, hook."""

import random

import pytest

import config
from lane import ALL_PINS, LANE_HALF, PIN_POS, Launch, RollSim, is_split


def roll(x=0.0, angle=0.0, speed=7.5, spin=0.0, standing=ALL_PINS, bumpers=False, seed=0):
    return RollSim(Launch(x, angle, speed, spin), standing, bumpers, random.Random(seed)).run()


def test_pin_layout():
    assert PIN_POS[1] == (0.0, config.HEAD_PIN_Y_M)
    assert PIN_POS[7][0] < PIN_POS[8][0] < PIN_POS[9][0] < PIN_POS[10][0]
    assert PIN_POS[2][0] < 0 < PIN_POS[3][0]
    assert all(abs(x) < LANE_HALF for x, _ in PIN_POS.values())


@pytest.mark.parametrize("standing, split", [
    ({7, 10}, True), ({4, 6}, True), ({2, 7}, True), ({3, 10}, True),
    ({2, 3}, False), ({6, 10}, False), ({10}, False), ({1, 7, 10}, False), (set(), False),
])
def test_split_detection(standing, split):
    assert is_split(standing) is split


def test_pocket_hit_is_a_strike():
    r = roll(x=0.08)
    assert r.knocked == set(ALL_PINS)
    assert not r.gutter and r.first_contact_t is not None


def test_thin_hit_leaves_pins():
    r = roll(x=0.36)
    assert 0 < len(r.knocked) < 10
    assert 1 not in r.knocked


def test_gutter_ball_without_bumpers():
    r = roll(x=0.4, angle=2.0)
    assert r.gutter
    assert r.knocked == set()
    assert r.ball.in_gutter


def test_bumpers_keep_the_ball_in_play():
    r = roll(x=0.4, angle=2.0, bumpers=True)
    assert not r.gutter and r.bumper_hits >= 1
    assert len(r.knocked) > 0


def test_only_standing_pins_are_simulated():
    r = roll(x=0.0, standing={7, 10})
    assert r.knocked <= {7, 10}
    assert {p.number for p in r.pins} == {7, 10}


def ball_x_at(y, **kw):
    sim = RollSim(Launch(kw.pop("x", 0.0), 0.0, 7.0, kw.pop("spin", 0.0)), standing=set(),
                  bumpers=True, rng=random.Random(0))
    while sim.ball.y < y:
        sim.step()
    return sim.ball.x


def test_hook_direction_and_back_end():
    straight = ball_x_at(17.0)
    right = ball_x_at(17.0, spin=0.5)
    left = ball_x_at(17.0, spin=-0.5)
    assert left < straight - 0.1 and right > straight + 0.1
    # most of the hook happens on the dry back end, past the oil
    on_oil = ball_x_at(config.OIL_LENGTH_M, spin=0.5)
    assert abs(on_oil) < abs(right - on_oil)


def test_same_seed_same_result():
    a = roll(x=-0.15, spin=0.3, seed=4)
    b = roll(x=-0.15, spin=0.3, seed=4)
    assert a.knocked == b.knocked and a.t == pytest.approx(b.t)


def test_roll_finishes_well_before_safety_limit():
    r = roll(x=0.05)
    assert r.finished and r.t < config.MAX_ROLL_S - 1


def test_launch_position_is_clamped_onto_the_lane():
    r = RollSim(Launch(5.0, 0, 7, 0))
    assert r.ball.x == pytest.approx(LANE_HALF - config.BALL_RADIUS_M)
