"""Random hole generation: layout, walls, obstacles only on later holes, fair gaps, aiming."""

import math
import random

import pytest

import config
from course import C, DIRS, Hole, _walls, generate_course, generate_hole, heading_deg

SEEDS = range(25)


def all_holes():
    for seed in SEEDS:
        yield from generate_course(random.Random(seed), "Hard")


def test_course_has_five_holes_and_is_reproducible():
    a = generate_course(random.Random(7))
    b = generate_course(random.Random(7))
    c = generate_course(random.Random(8))
    assert len(a) == config.HOLES_PER_GAME == 5
    assert [h.number for h in a] == [1, 2, 3, 4, 5]
    assert [h.path for h in a] == [h.path for h in b]
    assert [h.path for h in a] != [h.path for h in c]


def test_path_is_a_connected_corridor_within_plan():
    for h in all_holes():
        lo, hi, tmin, tmax, _ = config.HOLE_PLANS[h.number - 1]
        assert lo <= len(h.path) <= hi
        assert tmin <= h.turns <= tmax
        assert len(set(h.path)) == len(h.path)
        for (i0, j0), (i1, j1), d in zip(h.path, h.path[1:], h.dirs):
            assert (i1 - i0, j1 - j0) == DIRS[d]
        assert set(h.path) <= h.cells
        assert h.contains(*h.tee) and h.contains(*h.cup)
        assert 2 <= h.par <= 4


def test_corridor_never_touches_itself():
    """Non-consecutive path cells are never side by side (no missing walls / shortcuts)."""
    for h in all_holes():
        index = {c: k for k, c in enumerate(h.path)}
        for k, (i, j) in enumerate(h.path):
            for dx, dy in DIRS.values():
                n = index.get((i + dx, j + dy))
                assert n is None or abs(n - k) == 1


def test_walls_enclose_every_cell_edge_to_the_outside():
    for h in all_holes():
        for i, j in h.cells:
            for dx, dy in DIRS.values():
                if (i + dx, j + dy) in h.cells:
                    continue
                mx, my = (i + dx / 2) * C, (j + dy / 2) * C   # middle of that edge
                assert any(abs((bx - ax) * (my - ay) - (by - ay) * (mx - ax)) < 1e-6 and
                           min(ax, bx) - 1e-6 <= mx <= max(ax, bx) + 1e-6 and
                           min(ay, by) - 1e-6 <= my <= max(ay, by) + 1e-6 for ax, ay, bx, by in h.walls)


def test_walls_are_merged():
    cells = {(0, 0), (0, 1), (0, 2)}
    walls = _walls(cells)
    assert len(walls) == 4
    lengths = sorted(math.hypot(bx - ax, by - ay) for ax, ay, bx, by in walls)
    assert lengths == pytest.approx([C, C, 3 * C, 3 * C])


def test_first_two_holes_are_clear_and_later_holes_have_obstacles():
    counts = {n: 0 for n in range(1, 6)}
    kinds = {n: set() for n in range(1, 6)}
    for h in all_holes():
        counts[h.number] += len(h.obstacles)
        kinds[h.number] |= {o.kind for o in h.obstacles}
        assert {o.kind for o in h.obstacles} <= set(config.HOLE_OBSTACLES[h.number - 1])
    assert counts[1] == counts[2] == 0
    assert all(counts[n] >= len(SEEDS) for n in (3, 4, 5))       # at least one per hole on average
    assert kinds[5] >= {"spinner", "ghost"}
    assert counts[3] / len(SEEDS) < counts[5] / len(SEEDS) + 1    # hole 5 is at least as busy


def test_signature_obstacle_always_appears():
    for seed in SEEDS:
        course = generate_course(random.Random(seed))
        for h in course[2:]:
            assert config.HOLE_OBSTACLES[h.number - 1][0] in {o.kind for o in h.obstacles}


def test_obstacles_leave_a_gap_and_stay_off_tee_and_cup():
    ball = 2 * config.GOLF_BALL_RADIUS_M
    for h in all_holes():
        for ob in h.obstacles:
            if ob.kind in ("pumpkin", "pit", "slime"):
                x, y = ob.x, ob.y
            elif ob.kind == "tombstone":
                x, y = (ob.ax + ob.bx) / 2, (ob.ay + ob.by) / 2
            elif ob.kind == "ghost":
                x, y = (ob.ax + ob.bx) / 2, (ob.ay + ob.by) / 2
            else:
                x, y = ob.x, ob.y
            k = h.path.index(h.cell_of(x, y))
            assert 2 <= k <= len(h.path) - 3
            if ob.kind == "pumpkin" or ob.kind == "pit":
                half = C / 2 - config.WALL_HALF_THICK_M
                cell = (h.path[k][0] * C, h.path[k][1] * C)
                rx, ry = DIRS[{"N": "E", "E": "S", "S": "W", "W": "N"}[h.dirs[k]]]
                lat = (x - cell[0]) * rx + (y - cell[1]) * ry
                assert half - abs(lat) - ob.r > ball or half + abs(lat) - ob.r > 3 * ball
            if ob.kind == "tombstone":
                assert math.hypot(ob.bx - ob.ax, ob.by - ob.ay) + ob.r + config.WALL_HALF_THICK_M < C - 4 * ball
            if ob.kind == "spinner":
                assert ob.length + ob.r < C / 2 - config.WALL_HALF_THICK_M - ball


def test_decorations_stay_outside_the_course():
    for h in all_holes():
        assert h.decorations
        for d in h.decorations:
            assert not h.contains(d.x, d.y)


def test_default_aim_points_at_cup_when_visible():
    h = Hole(1, [(0, 0), (0, 1), (0, 2)], ["N"] * 3, {(0, 0), (0, 1), (0, 2)},
             _walls({(0, 0), (0, 1), (0, 2)}), (0.0, -0.3), (0.2, 2 * C), 2)
    assert h.default_aim_deg(h.tee) == pytest.approx(heading_deg(0.2, 2 * C + 0.3))


def test_default_aim_around_a_corner_stays_inside():
    for h in all_holes():
        aim = h.default_aim_deg(h.tee)
        a = math.radians(aim)
        p = (h.tee[0] + math.sin(a) * 1.0, h.tee[1] + math.cos(a) * 1.0)
        assert h.contains(*p)


def test_generate_hole_obstacle_speed_follows_difficulty():
    easy = generate_hole(random.Random(3), 5, "Easy")
    hard = generate_hole(random.Random(3), 5, "Hard")
    movers_e = [o.speed for o in easy.obstacles if hasattr(o, "speed")]
    movers_h = [o.speed for o in hard.obstacles if hasattr(o, "speed")]
    assert movers_e and all(s == config.DIFFICULTIES["Easy"]["obstacle_speed"] for s in movers_e)
    assert all(s == config.DIFFICULTIES["Hard"]["obstacle_speed"] for s in movers_h)


def test_default_aim_never_points_back_down_the_course():
    """From anywhere on the course the suggested line heads toward the cup's end, never back to the tee."""
    for h in all_holes():
        for k, (i, j) in enumerate(h.path[1:-1], start=1):
            ball = (i * C, j * C)
            a = math.radians(h.default_aim_deg(ball))
            nxt = h.path[k + 1]
            to_next = ((nxt[0] - i), (nxt[1] - j))
            assert math.sin(a) * to_next[0] + math.cos(a) * to_next[1] > -1e-6
