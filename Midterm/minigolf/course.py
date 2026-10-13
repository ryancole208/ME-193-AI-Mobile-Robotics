"""
Random Halloween minigolf holes (no pygame -- unit tested).

Each hole is a corridor of square cells (CELL_M wide) laid out by a random walk from
the tee to the cup: straight runs joined by 90° turns, never touching itself, so
every corridor is closed by walls. Walls run along each cell edge that borders the
outside and are merged into long segments. Later holes may get a second lane on a
long straight, and Halloween obstacles on straight cells:

    pumpkin    static round bumper
    tombstone  static slab jutting in from a wall
    ghost      floats back and forth across the lane (you bounce off it)
    spinner    a bone spinning around a post
    slime      sticky puddle: much more rolling friction
    pit        witch's cauldron: +1 stroke, ball returns to where it was

Every obstacle sits on one side of its cell (or moves), so there is always a gap.
Coordinates: x = east, y = north (metres). Headings are degrees clockwise from north.
"""

import math
import random
from dataclasses import dataclass, field

import config

C = config.CELL_M
DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
RIGHT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}
LEFT_OF = {v: k for k, v in RIGHT_OF.items()}


def heading_deg(dx, dy):
    return math.degrees(math.atan2(dx, dy))


def heading_vec(deg):
    a = math.radians(deg)
    return math.sin(a), math.cos(a)


def closest_on_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return ax, ay
    u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return ax + u * dx, ay + u * dy


def dist_to_segment(px, py, ax, ay, bx, by):
    qx, qy = closest_on_segment(px, py, ax, ay, bx, by)
    return math.hypot(px - qx, py - qy)


# =============================================================================
# Obstacles. Static ones collide at any time; moving ones take the absolute time t.
# =============================================================================

@dataclass
class Pumpkin:
    x: float
    y: float
    r: float = config.PUMPKIN_RADIUS_M
    kind: str = "pumpkin"

    def capsules(self, t):
        """[(ax, ay, bx, by, radius, vx_fn)] -- vx_fn(px, py) -> surface velocity at a contact point."""
        return [(self.x, self.y, self.x, self.y, self.r, None)]


@dataclass
class Tombstone:
    ax: float
    ay: float
    bx: float
    by: float
    r: float = config.TOMBSTONE_HALF_THICK_M
    height: float = config.TOMBSTONE_HEIGHT_M
    kind: str = "tombstone"

    def capsules(self, t):
        return [(self.ax, self.ay, self.bx, self.by, self.r, None)]


@dataclass
class Ghost:
    """Floats along the track (ax,ay)-(bx,by) and back, sinusoidally."""
    ax: float
    ay: float
    bx: float
    by: float
    period: float
    phase: float
    axis: tuple            # unit vector along the lane (to push a resting ball out of the way)
    r: float = config.GHOST_RADIUS_M
    speed: float = 1.0     # difficulty multiplier
    kind: str = "ghost"

    def _u(self, t):
        return 0.5 + 0.5 * math.sin(2 * math.pi * t * self.speed / self.period + self.phase)

    def pos(self, t):
        u = self._u(t)
        return self.ax + (self.bx - self.ax) * u, self.ay + (self.by - self.ay) * u

    def vel(self, t):
        du = 0.5 * math.cos(2 * math.pi * t * self.speed / self.period + self.phase) * \
            2 * math.pi * self.speed / self.period
        return (self.bx - self.ax) * du, (self.by - self.ay) * du

    def capsules(self, t):
        x, y = self.pos(t)
        vx, vy = self.vel(t)
        return [(x, y, x, y, self.r, lambda px, py: (vx, vy))]

    def sweep_exit(self, px, py, ball_r):
        """A resting ball inside the ghost's path is moved along the lane until it's clear."""
        reach = self.r + ball_r + 0.03
        if dist_to_segment(px, py, self.ax, self.ay, self.bx, self.by) >= reach:
            return None
        return _slide_clear(px, py, self.axis, lambda x, y: dist_to_segment(x, y, self.ax, self.ay,
                                                                           self.bx, self.by) >= reach)


@dataclass
class Spinner:
    """A bone of half-length L spinning around a post at (x, y)."""
    x: float
    y: float
    omega: float           # rad/s, + = counter-clockwise
    phase: float
    axis: tuple
    length: float = config.SPINNER_LENGTH_M
    r: float = config.SPINNER_HALF_THICK_M
    speed: float = 1.0
    kind: str = "spinner"

    def angle(self, t):
        return self.phase + self.omega * self.speed * t

    def ends(self, t):
        a = self.angle(t)
        dx, dy = math.cos(a) * self.length, math.sin(a) * self.length
        return (self.x - dx, self.y - dy), (self.x + dx, self.y + dy)

    def capsules(self, t):
        (ax, ay), (bx, by) = self.ends(t)
        w = self.omega * self.speed
        surf = lambda px, py: (-w * (py - self.y), w * (px - self.x))
        return [(ax, ay, bx, by, self.r, surf), (self.x, self.y, self.x, self.y, 0.05, None)]

    def sweep_exit(self, px, py, ball_r):
        reach = self.length + self.r + ball_r + 0.03
        if math.hypot(px - self.x, py - self.y) >= reach:
            return None
        return _slide_clear(px, py, self.axis, lambda x, y: math.hypot(x - self.x, y - self.y) >= reach)


@dataclass
class Slime:
    x: float
    y: float
    r: float = config.SLIME_RADIUS_M
    kind: str = "slime"

    def contains(self, px, py):
        return math.hypot(px - self.x, py - self.y) < self.r


@dataclass
class Pit:
    x: float
    y: float
    r: float = config.PIT_RADIUS_M
    kind: str = "pit"

    def contains(self, px, py):
        return math.hypot(px - self.x, py - self.y) < self.r


def _slide_clear(px, py, axis, clear, step=0.01, max_d=2.0):
    """Nearest point along +/-axis from (px, py) where clear(x, y) holds."""
    ax, ay = axis
    d = step
    while d <= max_d:
        for s in (1, -1):
            x, y = px + s * d * ax, py + s * d * ay
            if clear(x, y):
                return x, y
        d += step
    return None


@dataclass
class Decoration:
    kind: str              # "tree", "grave", "lantern", "fence"
    x: float
    y: float
    size: float = 1.0
    seed: int = 0


# =============================================================================
# Holes
# =============================================================================

@dataclass
class Hole:
    number: int
    path: list                         # ordered cells (i, j) from tee to cup
    dirs: list                         # direction ("N"/"E"/"S"/"W") leaving each path cell (cup: arriving)
    cells: set                         # path cells + any widened lane
    walls: list                        # (ax, ay, bx, by) along cell edges
    tee: tuple
    cup: tuple
    par: int
    obstacles: list = field(default_factory=list)
    decorations: list = field(default_factory=list)

    @property
    def turns(self):
        return sum(1 for a, b in zip(self.dirs, self.dirs[1:]) if a != b)

    @staticmethod
    def cell_of(x, y):
        return math.floor(x / C + 0.5), math.floor(y / C + 0.5)

    def contains(self, x, y):
        return self.cell_of(x, y) in self.cells

    @property
    def colliders(self):
        return [o for o in self.obstacles if hasattr(o, "capsules")]

    @property
    def moving(self):
        return [o for o in self.obstacles if hasattr(o, "sweep_exit")]

    @property
    def slimes(self):
        return [o for o in self.obstacles if o.kind == "slime"]

    @property
    def pits(self):
        return [o for o in self.obstacles if o.kind == "pit"]

    def bounds(self):
        xs = [i * C for i, _ in self.cells]
        ys = [j * C for _, j in self.cells]
        return min(xs) - C / 2, min(ys) - C / 2, max(xs) + C / 2, max(ys) + C / 2

    def length_m(self):
        return len(self.path) * C

    # -- aiming -------------------------------------------------------------------
    def clear_line(self, p, q, margin=config.GOLF_BALL_RADIUS_M + config.WALL_HALF_THICK_M + 0.01,
                   avoid_obstacles=False):
        """True if a ball rolled straight from p to q stays inside the walls (and, optionally, misses
        pumpkins, tombstones and cauldrons; moving obstacles are a matter of timing)."""
        (px, py), (qx, qy) = p, q
        n = max(2, int(math.hypot(qx - px, qy - py) / 0.05))
        for k in range(n + 1):
            x, y = px + (qx - px) * k / n, py + (qy - py) * k / n
            if not self.contains(x, y):
                return False
        for ax, ay, bx, by in self.walls:
            if _seg_seg_dist(px, py, qx, qy, ax, ay, bx, by) < margin:
                return False
        if avoid_obstacles:
            br = config.GOLF_BALL_RADIUS_M + 0.02
            for ob in self.obstacles:
                if ob.kind in ("pumpkin", "pit") and dist_to_segment(ob.x, ob.y, px, py, qx, qy) < ob.r + br:
                    return False
                if ob.kind == "tombstone" and _seg_seg_dist(px, py, qx, qy, ob.ax, ob.ay, ob.bx, ob.by) < ob.r + br:
                    return False
        return True

    def default_aim_deg(self, ball):
        """Heading at the cup if it's in sight, else at the farthest corridor cell in sight --
        preferring lines that miss the static obstacles, like Wii's suggested line."""
        bx, by = ball
        k = self.progress_index(ball)
        targets = [self.cup] + [(i * C, j * C) for i, j in reversed(self.path[k + 1:])]
        for avoid in (True, False):
            for tx, ty in targets:
                if math.hypot(tx - bx, ty - by) > 0.05 and self.clear_line(ball, (tx, ty), avoid_obstacles=avoid):
                    return heading_deg(tx - bx, ty - by)
        dx, dy = DIRS[self.dirs[k]]
        return heading_deg(dx, dy)

    def progress_index(self, p):
        """Index of the path cell the point is in (a widened lane counts as its path neighbour)."""
        cell = self.cell_of(*p)
        if cell in self.path:
            return self.path.index(cell)
        beside = [k for k, c in enumerate(self.path) if c in _neighbours(cell)]
        if beside:
            return max(beside)
        return min(range(len(self.path)), key=lambda k: math.hypot(self.path[k][0] * C - p[0],
                                                                    self.path[k][1] * C - p[1]))


def _seg_seg_dist(ax, ay, bx, by, cx, cy, dx, dy):
    def cross(ox, oy, px, py, qx, qy):
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)
    d1, d2 = cross(cx, cy, dx, dy, ax, ay), cross(cx, cy, dx, dy, bx, by)
    d3, d4 = cross(ax, ay, bx, by, cx, cy), cross(ax, ay, bx, by, dx, dy)
    if (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0):
        return 0.0
    return min(dist_to_segment(ax, ay, cx, cy, dx, dy), dist_to_segment(bx, by, cx, cy, dx, dy),
               dist_to_segment(cx, cy, ax, ay, bx, by), dist_to_segment(dx, dy, ax, ay, bx, by))


# =============================================================================
# Generation
# =============================================================================

def _neighbours(c):
    i, j = c
    return [(i + dx, j + dy) for dx, dy in DIRS.values()]


def _walk(rng, min_cells, max_cells, min_turns, max_turns, tries=400):
    """Random corridor: list of cells and the travel direction at each."""
    lo, hi = config.RUN_LENGTH
    for _ in range(tries):
        turns = rng.randint(min_turns, max_turns)
        runs = [rng.randint(lo, hi) for _ in range(turns + 1)]
        runs[0] += 1  # the tee cell
        total = sum(runs) - turns  # consecutive runs share their corner cell
        if not min_cells <= total <= max_cells:
            continue
        path, dirs, d = [(0, 0)], ["N"], "N"
        ok = True
        for k, n in enumerate(runs):
            if k > 0:
                d = rng.choice((LEFT_OF[d], RIGHT_OF[d]))
                dirs[-1] = d
                n -= 1
            for _ in range(n - (1 if k == 0 else 0)):
                dx, dy = DIRS[d]
                nxt = (path[-1][0] + dx, path[-1][1] + dy)
                if nxt in path or any(nb in path and nb != path[-1] for nb in _neighbours(nxt)):
                    ok = False
                    break
                path.append(nxt)
                dirs.append(d)
            if not ok:
                break
        if ok and len(path) == total:
            return path, dirs
    raise RuntimeError("could not lay out a hole")


def _widen(rng, path, dirs, cells):
    """Maybe give one long straight a second lane (always beside straight, non-end cells)."""
    straight = [k for k in range(2, len(path) - 1) if dirs[k - 1] == dirs[k] == dirs[k + 1]]
    runs = []
    for k in straight:
        if runs and runs[-1][1] == k - 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    runs = [(a, b) for a, b in runs if b - a >= 1]
    rng.shuffle(runs)
    for a, b in runs:
        if rng.random() > config.WIDEN_CHANCE:
            continue
        for side in rng.sample((RIGHT_OF, LEFT_OF), 2):
            sx, sy = DIRS[side[dirs[a]]]
            extra = [(path[k][0] + sx, path[k][1] + sy) for k in range(a, b + 1)]
            allowed = set(extra) | {path[k] for k in range(a, b + 1)}
            if all(c not in cells and all(nb not in cells or nb in allowed for nb in _neighbours(c)) for c in extra):
                cells.update(extra)
                return extra
    return []


def _walls(cells):
    horiz, vert = {}, {}
    for i, j in cells:
        for d, (dx, dy) in DIRS.items():
            if (i + dx, j + dy) in cells:
                continue
            if dy:
                horiz.setdefault((j + dy * 0.5) * C, []).append(((i - 0.5) * C, (i + 0.5) * C))
            else:
                vert.setdefault((i + dx * 0.5) * C, []).append(((j - 0.5) * C, (j + 0.5) * C))
    walls = []
    for fixed, spans, horizontal in [(k, v, True) for k, v in horiz.items()] + [(k, v, False) for k, v in vert.items()]:
        spans.sort()
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if abs(a - merged[-1][1]) < 1e-9:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            walls.append((a, fixed, b, fixed) if horizontal else (fixed, a, fixed, b))
    return walls


def _par(n_cells, turns, n_obstacles):
    return max(2, min(4, 2 + (n_cells >= 8) + (turns >= 3 or n_cells >= 12 or n_obstacles >= 4)))


def _place_obstacles(rng, path, dirs, cells, kinds, budget, speed):
    if not kinds or budget <= 0:
        return []
    straight = [k for k in range(2, len(path) - 2) if dirs[k] == dirs[k - 1]]
    rng.shuffle(straight)
    used, out = [], []
    rest = list(kinds[1:])
    rng.shuffle(rest)
    order = [kinds[0]] + rest
    for k in straight:
        if len(out) >= budget:
            break
        if any(abs(k - u) < 2 for u in used):
            continue
        kind = order[len(out) % len(order)]
        ob = _make_obstacle(rng, kind, path[k], dirs[k], cells, speed)
        if ob is not None:
            out.append(ob)
            used.append(k)
    return out


def _make_obstacle(rng, kind, cell, d, cells, speed):
    cx, cy = cell[0] * C, cell[1] * C
    fx, fy = DIRS[d]
    rx, ry = DIRS[RIGHT_OF[d]]
    walled = [s for s in (1, -1) if (cell[0] + s * rx, cell[1] + s * ry) not in cells]
    side = rng.choice((1, -1))
    along = rng.uniform(-0.25, 0.25)
    bx, by = cx + fx * along, cy + fy * along
    if kind == "pumpkin":
        off = 0.28
        return Pumpkin(bx + side * off * rx, by + side * off * ry)
    if kind == "tombstone":
        if not walled:
            return None
        s = rng.choice(walled)
        a = (bx + s * rx * C / 2, by + s * ry * C / 2)
        b = (bx + s * rx * (C / 2 - config.TOMBSTONE_LENGTH_M), by + s * ry * (C / 2 - config.TOMBSTONE_LENGTH_M))
        return Tombstone(a[0], a[1], b[0], b[1])
    if kind == "ghost":
        amp = C / 2 - config.GHOST_RADIUS_M - 0.02
        return Ghost(cx - rx * amp, cy - ry * amp, cx + rx * amp, cy + ry * amp,
                     period=rng.uniform(*config.GHOST_PERIOD_S), phase=rng.uniform(0, 2 * math.pi),
                     axis=(fx, fy), speed=speed)
    if kind == "spinner":
        w = rng.uniform(*config.SPINNER_OMEGA_RAD_S) * rng.choice((1, -1))
        return Spinner(cx, cy, omega=w, phase=rng.uniform(0, math.pi), axis=(fx, fy), speed=speed)
    if kind == "slime":
        off = 0.2
        return Slime(bx + side * off * rx, by + side * off * ry)
    if kind == "pit":
        off = 0.31
        return Pit(bx + side * off * rx, by + side * off * ry)
    raise ValueError(f"unknown obstacle {kind!r}")


def _decorations(rng, cells):
    ring = set()
    for i, j in cells:
        for di in range(-3, 4):
            for dj in range(-3, 4):
                c = (i + di, j + dj)
                if c not in cells:
                    ring.add(c)
    out = []
    for c in sorted(ring):
        near = any((c[0] + di, c[1] + dj) in cells for di in (-1, 0, 1) for dj in (-1, 0, 1))
        if rng.random() > (0.45 if near else 0.2):
            continue
        kind = rng.choice(("lantern", "grave", "grave", "fence") if near else ("tree", "tree", "grave", "lantern"))
        x = c[0] * C + rng.uniform(-0.3, 0.3)
        y = c[1] * C + rng.uniform(-0.3, 0.3)
        out.append(Decoration(kind, x, y, size=rng.uniform(0.8, 1.25), seed=rng.randrange(1 << 16)))
    return out


def generate_hole(rng, number, difficulty=config.DEFAULT_DIFFICULTY):
    min_c, max_c, min_t, max_t, budget = config.HOLE_PLANS[number - 1]
    kinds = config.HOLE_OBSTACLES[number - 1]
    speed = config.DIFFICULTIES[difficulty]["obstacle_speed"]
    path, dirs = _walk(rng, min_c, max_c, min_t, max_t)
    cells = set(path)
    if kinds:
        _widen(rng, path, dirs, cells)
    obstacles = _place_obstacles(rng, path, dirs, cells, kinds, budget, speed)
    fx, fy = DIRS[dirs[0]]
    tee = (path[0][0] * C - fx * config.TEE_BACK_M, path[0][1] * C - fy * config.TEE_BACK_M)
    lx, ly = DIRS[RIGHT_OF[dirs[-1]]]
    fx, fy = DIRS[dirs[-1]]
    off, along = rng.uniform(-0.22, 0.22), rng.uniform(-0.1, 0.2)
    cup = (path[-1][0] * C + lx * off + fx * along, path[-1][1] * C + ly * off + fy * along)
    hole = Hole(number, path, dirs, cells, _walls(cells), tee, cup, 0, obstacles, _decorations(rng, cells))
    hole.par = _par(len(path), hole.turns, len(obstacles))
    return hole


def generate_course(rng=None, difficulty=config.DEFAULT_DIFFICULTY, holes=config.HOLES_PER_GAME):
    rng = rng or random.Random()
    return [generate_hole(rng, n, difficulty) for n in range(1, holes + 1)]
