"""
Lane geometry and roll physics (pure Python, no pygame -- unit tested).

Coordinates (metres, top-down): x lateral (left -, right +, 0 = lane centre),
y along the lane (0 = foul line, HEAD_PIN_Y = head pin), z height.

A roll is a small 2D rigid-circle simulation: the ball (with a hook from its spin
on the dry back end), the standing pins, and fallen pins that slide and take out
their neighbours (pin action). A pin goes down when a hit gives it at least
PIN_TOPPLE_SPEED; weaker hits above PIN_WOBBLE_SPEED knock it over by chance.
Without bumpers a ball past the lane edge drops into the gutter and can't hit pins;
with bumpers it bounces back. RollSim is stepped in real time by the game
(advance_to) or run to completion at once by tests (run).
"""

import math
import random
from dataclasses import dataclass

import config

LANE_HALF = config.LANE_WIDTH_M / 2


def pin_positions():
    """Standard numbering: 1 = head pin; then 2-3, 4-6, 7-10, left to right as the bowler sees them."""
    pos, n = {}, 1
    for row in range(4):
        for i in range(row + 1):
            pos[n] = ((i - row / 2) * config.PIN_SPACING_M, config.HEAD_PIN_Y_M + row * config.PIN_ROW_SPACING_M)
            n += 1
    return pos


PIN_POS = pin_positions()
ALL_PINS = frozenset(PIN_POS)


def adjacent(a, b):
    return math.dist(PIN_POS[a], PIN_POS[b]) < config.PIN_SPACING_M * 1.05


def is_split(standing):
    """Head pin down and the standing pins form more than one separate group (e.g. 7-10)."""
    standing = set(standing)
    if 1 in standing or len(standing) < 2:
        return False
    seen, groups = set(), 0
    for p in standing:
        if p in seen:
            continue
        groups += 1
        stack = [p]
        seen.add(p)
        while stack:
            q = stack.pop()
            for r in standing - seen:
                if adjacent(q, r):
                    seen.add(r)
                    stack.append(r)
    return groups > 1


@dataclass(frozen=True)
class Launch:
    x: float           # lateral release position at the foul line (m)
    angle_deg: float   # launch direction, + = toward the right
    speed: float       # m/s
    spin: float = 0.0  # -1..1, + hooks toward the right


class Body:
    __slots__ = ("x", "y", "vx", "vy", "r", "m")

    def __init__(self, x, y, r, m, vx=0.0, vy=0.0):
        self.x, self.y, self.r, self.m, self.vx, self.vy = x, y, r, m, vx, vy

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)


class Pin(Body):
    __slots__ = ("number", "fall_t", "heading", "gone")

    def __init__(self, number):
        x, y = PIN_POS[number]
        super().__init__(x, y, config.PIN_RADIUS_M, config.PIN_MASS_KG)
        self.number = number
        self.fall_t = None    # sim time the pin started toppling
        self.heading = 0.0    # direction it topples (radians, 0 = down the lane, + = toward +x)
        self.gone = False     # slid off the deck into the pit or a gutter

    @property
    def down(self):
        return self.fall_t is not None or self.gone

    def tilt(self, t):
        """0 = upright .. 1 = lying flat."""
        if self.fall_t is None:
            return 0.0
        return min(1.0, (t - self.fall_t) / config.PIN_FALL_TIME_S)


class Ball(Body):
    __slots__ = ("spin", "in_gutter", "gutter_x", "done", "distance")

    def __init__(self, launch):
        x = max(-LANE_HALF + config.BALL_RADIUS_M, min(LANE_HALF - config.BALL_RADIUS_M, launch.x))
        a = math.radians(launch.angle_deg)
        super().__init__(x, 0.0, config.BALL_RADIUS_M, config.BALL_MASS_KG,
                         launch.speed * math.sin(a), launch.speed * math.cos(a))
        self.spin = max(-1.0, min(1.0, launch.spin))
        self.in_gutter = False
        self.gutter_x = 0.0
        self.done = False       # reached the pit
        self.distance = 0.0     # rolled distance, for drawing the rotating finger holes


def collide(a, b, restitution):
    """Resolve overlap + impulse between two circles. Returns b's speed change (m/s), 0 if no contact."""
    dx, dy = b.x - a.x, b.y - a.y
    d = math.hypot(dx, dy)
    min_d = a.r + b.r
    if d >= min_d or d < 1e-9:
        return 0.0
    nx, ny = dx / d, dy / d
    inv_a, inv_b = 1 / a.m, 1 / b.m
    push = (min_d - d) / (inv_a + inv_b)
    a.x -= nx * push * inv_a
    a.y -= ny * push * inv_a
    b.x += nx * push * inv_b
    b.y += ny * push * inv_b
    rv = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
    if rv >= 0:
        return 0.0
    j = -(1 + restitution) * rv / (inv_a + inv_b)
    a.vx -= j * nx * inv_a
    a.vy -= j * ny * inv_a
    b.vx += j * nx * inv_b
    b.vy += j * ny * inv_b
    return j * inv_b


class RollSim:
    def __init__(self, launch, standing=ALL_PINS, bumpers=False, rng=None, dt=config.SIM_DT_S):
        self.launch = launch
        self.bumpers = bumpers
        self.rng = rng or random.Random()
        self.dt = dt
        self.t = 0.0
        self.ball = Ball(launch)
        self.pins = [Pin(n) for n in sorted(standing)]
        self.first_contact_t = None
        self.gutter = False        # the ball went into a gutter
        self.bumper_hits = 0
        self.finished = False

    # -- results ----------------------------------------------------------------
    @property
    def knocked(self):
        return {p.number for p in self.pins if p.down}

    @property
    def standing(self):
        return {p.number for p in self.pins if not p.down}

    # -- stepping ---------------------------------------------------------------
    def advance_to(self, t):
        while not self.finished and self.t < t:
            self.step()

    def run(self):
        while not self.finished:
            self.step()
        return self

    def step(self):
        dt, t = self.dt, self.t
        self._move_ball(dt)
        b = self.ball
        if not b.in_gutter and not b.done:
            for p in self.pins:
                if not p.gone:
                    self._hit(p, collide(b, p, config.BALL_PIN_RESTITUTION), b, t)
        for i, p in enumerate(self.pins):
            if p.gone:
                continue
            for q in self.pins[i + 1:]:
                if q.gone or not (p.down or q.down):
                    continue
                if p.down and q.down and p.speed < config.SETTLE_SPEED and q.speed < config.SETTLE_SPEED:
                    continue
                mover, target = (p, q) if p.speed >= q.speed else (q, p)
                self._hit(target, collide(mover, target, config.PIN_PIN_RESTITUTION), mover, t)
        for p in self.pins:
            self._move_pin(p, dt)
        self.t = t + dt
        if self._settled() or self.t >= config.MAX_ROLL_S:
            self.finished = True

    def _move_ball(self, dt):
        b = self.ball
        if b.done:
            return
        if not b.in_gutter and b.y < config.PIT_Y_M:
            hook = config.HOOK_ACCEL_M_S2 * b.spin
            if b.y < config.OIL_LENGTH_M:
                hook *= config.OIL_HOOK_FRACTION
            b.vx += hook * dt
        if b.vy > 0.5:
            b.vy = max(0.5, b.vy - config.LANE_DECEL_M_S2 * dt)
        b.x += b.vx * dt
        b.y += b.vy * dt
        b.distance += b.speed * dt

        if b.in_gutter:
            b.x += (b.gutter_x - b.x) * min(1.0, 12 * dt)
        elif self.bumpers:
            lim = LANE_HALF - b.r
            if abs(b.x) > lim and b.vx * b.x > 0:
                b.x = math.copysign(lim, b.x)
                b.vx = -b.vx * config.BUMPER_RESTITUTION
                self.bumper_hits += 1
        elif abs(b.x) > LANE_HALF:
            b.in_gutter = True
            self.gutter = self.first_contact_t is None  # off the deck after hitting pins isn't a gutter ball
            b.gutter_x = math.copysign(LANE_HALF + config.GUTTER_WIDTH_M / 2, b.x)
            b.vx = 0.0
        if b.y > config.PIT_Y_M + 0.3 or b.y < -1.0:
            b.done = True

    def _hit(self, pin, dv, by, t):
        if dv <= 0:
            return
        if self.first_contact_t is None and isinstance(by, Ball):
            self.first_contact_t = t
        if pin.fall_t is not None:
            return
        lo, hi = config.PIN_WOBBLE_SPEED, config.PIN_TOPPLE_SPEED
        if dv >= hi or (dv > lo and self.rng.random() < (dv - lo) / (hi - lo)):
            pin.fall_t = t
            pin.r = config.PIN_FALLEN_RADIUS_M
            pin.heading = math.atan2(pin.vx, pin.vy)
        else:  # it wobbles but stays up
            pin.vx = pin.vy = 0.0

    def _move_pin(self, p, dt):
        if p.gone or p.fall_t is None:
            return
        s = p.speed
        if s > 0:
            new = max(0.0, s - config.PIN_FRICTION_M_S2 * dt)
            p.vx *= new / s
            p.vy *= new / s
            p.x += p.vx * dt
            p.y += p.vy * dt
        if p.y > config.PIT_Y_M or abs(p.x) > LANE_HALF + config.GUTTER_WIDTH_M * 0.5:
            p.gone = True
            p.vx = p.vy = 0.0

    def _settled(self):
        if not self.ball.done:
            return False
        for p in self.pins:
            if p.gone:
                continue
            if p.fall_t is not None and (p.speed > config.SETTLE_SPEED or p.tilt(self.t) < 1.0):
                return False
        return True
