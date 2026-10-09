"""Putt physics on hand-built holes: rolling, walls, cup capture and lip-outs, obstacles, hazards."""

import math
import random

import pytest

import config
from course import C, Ghost, Hole, Pit, Pumpkin, Slime, Spinner, Tombstone, _walls, generate_course, heading_deg
from physics import PuttSim

EASY = config.DIFFICULTIES["Easy"]


def straight_hole(n=6, obstacles=(), cup=None):
    cells = [(0, j) for j in range(n)]
    cup = cup or (0.0, (n - 1) * C)
    return Hole(1, cells, ["N"] * n, set(cells), _walls(set(cells)), (0.0, -0.3), cup, 2, list(obstacles))


def putt(hole, heading, speed, start=None, t0=0.0, run_s=30.0, **kw):
    opts = dict(cup_radius=EASY["cup_radius_m"], capture_speed=EASY["capture_speed"])
    opts.update(kw)
    sim = PuttSim(hole, start or hole.tee, heading, speed, t0, **opts)
    sim.advance_to(t0 + run_s)
    return sim


def test_ball_rolls_straight_and_stops():
    h = straight_hole(cup=(0.5, 100.0))   # cup out of the way
    sim = putt(h, 0.0, 2.0)
    assert sim.finished and sim.outcome == "stopped"
    assert sim.ball.x == pytest.approx(0.0, abs=1e-6)
    assert 2.0 < sim.ball.y + 0.3 < 4.0                      # v^2 / 2a with a little drag
    assert sim.wall_hits == 0


def test_faster_putts_roll_farther():
    h = straight_hole(n=20, cup=(0.5, 100.0))
    ys = [putt(h, 0.0, v).ball.y for v in (1.0, 2.0, 3.0)]
    assert ys == sorted(ys)


def test_wall_bounce_reflects_and_keeps_ball_inside():
    h = straight_hole(cup=(0.5, 100.0))
    sim = putt(h, 90.0, 2.0, start=(0.0, 2.0))               # straight at the east wall
    assert sim.wall_hits >= 1
    assert sim.ball.x < 0.6 and h.contains(sim.ball.x, sim.ball.y)


def test_back_wall_stops_overpowered_putt():
    h = straight_hole(n=3, cup=(0.5, 100.0))
    sim = putt(h, 0.0, 4.6)
    assert h.contains(sim.ball.x, sim.ball.y)
    assert sim.wall_hits >= 1


def test_centred_putt_at_good_speed_drops():
    h = straight_hole()
    dist = (h.cup[1] - h.tee[1])
    sim = putt(h, 0.0, math.sqrt(2 * 0.75 * dist) + 0.2)
    assert sim.outcome == "holed" and sim.holed_t is not None
    assert sim.ball.z < 0


def test_too_fast_lips_out_or_flies_over():
    h = straight_hole(n=8, cup=(0.0, 2.0))
    sim = putt(h, 0.0, 4.6, capture_speed=1.0)
    assert sim.outcome != "holed"


def test_edge_putt_is_harder_than_centred_putt():
    h = straight_hole(n=8, cup=(0.0, 3.0))
    R = EASY["cup_radius_m"]
    centre, edge = PuttSim(h, (0, 0), 0, 1, 0, cup_radius=R, capture_speed=1.6), \
        PuttSim(h, (0.9 * R, 0), 0, 1, 0, cup_radius=R, capture_speed=1.6)
    for sim in (centre, edge):
        sim.ball.y = 3.0 - R - 0.001
        sim.ball.vx, sim.ball.vy = 0.0, 1.3
        for _ in range(100):
            if not sim.finished:
                sim._step(config.SIM_DT_S)
    assert centre.holed and not edge.holed and edge.lipped


def test_slow_ball_over_cup_drops():
    h = straight_hole()
    sim = PuttSim(h, h.cup, 0.0, 0.1, 0.0, cup_radius=0.054, capture_speed=1.3)
    sim.advance_to(1.0)
    assert sim.holed


def test_pumpkin_deflects():
    h = straight_hole(obstacles=[Pumpkin(0.0, 2.0)], cup=(0.5, 100.0))
    sim = putt(h, 0.0, 2.5)
    assert sim.ball.y < 2.0                                   # bounced back off it


def test_tombstone_blocks():
    t = Tombstone(C / 2, 2.0, -0.1, 2.0)
    h = straight_hole(obstacles=[t], cup=(0.5, 100.0))
    sim = putt(h, 0.0, 2.5, start=(0.3, 0.0))
    assert sim.ball.y < 2.0
    sim = putt(h, 0.0, 2.5, start=(-0.35, 0.0))               # the open side
    assert sim.ball.y > 2.0


def test_slime_slows_the_ball():
    plain = straight_hole(n=12, cup=(0.5, 100.0))
    slimy = straight_hole(n=12, obstacles=[Slime(0.0, 1.5, r=1.0)], cup=(0.5, 100.0))
    assert putt(slimy, 0.0, 3.0).ball.y < putt(plain, 0.0, 3.0).ball.y - 1.0


def test_pit_is_a_hazard_and_remembers_start():
    h = straight_hole(obstacles=[Pit(0.0, 2.0)], cup=(0.5, 100.0))
    sim = putt(h, 0.0, 2.5, start=(0.0, 0.5))
    assert sim.outcome == "hazard" and sim.finished
    assert sim.start == (0.0, 0.5)


def test_ghost_pushes_the_ball():
    g = Ghost(-0.4, 2.0, 0.4, 2.0, period=3.0, phase=0.0, axis=(0, 1))
    h = straight_hole(obstacles=[g], cup=(0.5, 100.0))
    hit_any = False
    for t0 in [k * 0.1 for k in range(30)]:
        sim = putt(h, 0.0, 2.0, start=(0.0, 1.0), t0=t0)
        hit_any |= sim.ball.y < 2.0 or abs(sim.ball.x) > 0.05
    assert hit_any


def test_resting_ball_is_moved_out_of_a_moving_obstacles_path():
    s = Spinner(0.0, 3.0, omega=1.5, phase=0.0, axis=(0, 1))
    h = straight_hole(obstacles=[s], cup=(0.5, 100.0))
    sim = PuttSim(h, (0.0, 3.0 - 0.3), 0.0, 0.0, 0.0, cup_radius=0.054, capture_speed=1.3)
    sim.ball.vx = sim.ball.vy = 0.0
    sim.ball.x, sim.ball.y = 0.3, 2.9
    sim._clear_moving_obstacles()
    assert math.hypot(sim.ball.x - s.x, sim.ball.y - s.y) >= s.length + s.r
    assert sim.ball.x == pytest.approx(0.3)                   # moved along the lane only


def test_spinner_surface_velocity_is_tangential():
    s = Spinner(0.0, 0.0, omega=2.0, phase=0.0, axis=(0, 1))
    (_, _, _, _, _, surf), _ = s.capsules(0.0)
    vx, vy = surf(0.5, 0.0)
    assert vx == pytest.approx(0.0) and vy == pytest.approx(1.0)


def test_sim_follows_absolute_time():
    h = straight_hole(cup=(0.5, 100.0))
    sim = PuttSim(h, h.tee, 0.0, 2.0, 100.0, cup_radius=0.054, capture_speed=1.3)
    sim.advance_to(100.5)
    assert sim.t == pytest.approx(100.5, abs=config.SIM_DT_S)
    assert not sim.finished


def test_random_putts_never_escape_generated_holes():
    rng = random.Random(1)
    for h in generate_course(rng, "Hard") + generate_course(rng, "Easy"):
        for _ in range(8):
            sim = putt(h, rng.uniform(-180, 180), rng.uniform(0.4, 4.6), t0=rng.uniform(0, 50), run_s=40,
                       cup_radius=0.054, capture_speed=1.3)
            assert sim.finished
            if sim.outcome == "hazard":
                assert any(p.contains(sim.ball.x, sim.ball.y) for p in h.pits)
            else:
                assert h.contains(sim.ball.x, sim.ball.y)


def test_every_generated_hole_is_playable_from_the_tee():
    """A bot that putts at the default aim with distance-matched speed sinks each hole eventually."""
    rng = random.Random(0)
    for seed in range(6):
        for h in generate_course(random.Random(seed), "Easy"):
            ball, t, jitter = h.tee, 0.0, 0.0
            for stroke in range(25):
                aim = h.default_aim_deg(ball) + jitter
                d = math.hypot(h.cup[0] - ball[0], h.cup[1] - ball[1])
                if h.clear_line(ball, h.cup):
                    aim = heading_deg(h.cup[0] - ball[0], h.cup[1] - ball[1]) + jitter
                    speed = min(4.6, math.sqrt(2 * 0.7 * d) + 0.25)
                else:
                    speed = 2.2
                sim = putt(h, aim, speed, start=ball, t0=t, run_s=40)
                t = sim.t + 1.0
                if sim.holed:
                    break
                stuck = sim.outcome == "hazard" or math.hypot(sim.ball.x - ball[0], sim.ball.y - ball[1]) < 0.3
                jitter = rng.uniform(-25, 25) if stuck else 0.0      # like a player trying another line
                ball = sim.start if sim.outcome == "hazard" else (sim.ball.x, sim.ball.y)
            assert sim.holed, f"seed {seed} hole {h.number} not sunk"
