"""
One putt, simulated in 2D (no pygame -- unit tested).

The ball rolls with constant + speed-proportional rolling resistance (much more on
slime), bounces off walls and obstacles (moving ghosts and spinning bones push it
with their own surface velocity), sinks in a cauldron pit (hazard), and drops into
the cup if it's slow enough for how centred it is -- otherwise it lips out.

Obstacles move with absolute time, so the sim runs on the same clock as the game:
advance_to(now) steps it forward in SIM_DT_S increments.
"""

import math

import config
from course import closest_on_segment, heading_vec

BR = config.GOLF_BALL_RADIUS_M


class Ball:
    def __init__(self, x, y, vx, vy):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.z = 0.0            # drops below 0 into the cup
        self.distance = 0.0     # rolled so far (for the rolling animation)

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)


class PuttSim:
    def __init__(self, hole, start, heading_deg, speed, t0, *, cup_radius, capture_speed):
        self.hole = hole
        self.start = tuple(start)
        hx, hy = heading_vec(heading_deg)
        self.ball = Ball(start[0], start[1], hx * speed, hy * speed)
        self.t0 = self.t = t0
        self.cup_radius = cup_radius
        self.capture_speed = capture_speed
        self.finished = False
        self.outcome = None        # "holed", "stopped" or "hazard"
        self.holed_t = None
        self.hazard_t = None
        self.wall_hits = 0
        self.lipped = False
        self._over_cup = False

    def advance_to(self, t):
        dt = config.SIM_DT_S
        while not self.finished and self.t + dt <= t + 1e-9:
            self.t += dt
            self._step(dt)

    # -- one step -----------------------------------------------------------------
    def _step(self, dt):
        b = self.ball
        if self.outcome == "holed":
            k = min(1.0, (self.t - self.holed_t) / config.CUP_DROP_S)
            cx, cy = self.hole.cup
            b.x += (cx - b.x) * min(1.0, 12 * dt)
            b.y += (cy - b.y) * min(1.0, 12 * dt)
            b.z = -0.1 * k
            self.finished = k >= 1.0
            return
        if self.outcome == "hazard":
            b.z = -0.1 * min(1.0, (self.t - self.hazard_t) / 0.4)
            self.finished = self.t - self.hazard_t >= 0.6
            return

        speed = b.speed
        if speed > 0:
            decel = config.ROLL_DECEL_M_S2 + config.ROLL_DRAG_PER_S * speed
            if any(s.contains(b.x, b.y) for s in self.hole.slimes):
                decel *= config.SLIME_DECEL_FACTOR
            k = max(0.0, speed - decel * dt) / speed
            b.vx *= k
            b.vy *= k
        b.x += b.vx * dt
        b.y += b.vy * dt
        b.distance += b.speed * dt

        for ax, ay, bx, by in self.hole.walls:
            if self._collide(ax, ay, bx, by, config.WALL_HALF_THICK_M, None, config.WALL_RESTITUTION):
                self.wall_hits += 1
        for ob in self.hole.colliders:
            for ax, ay, bx, by, r, surf in ob.capsules(self.t):
                self._collide(ax, ay, bx, by, r, surf, config.OBSTACLE_RESTITUTION)

        if any(p.contains(b.x, b.y) for p in self.hole.pits) or not self.hole.contains(b.x, b.y):
            self.outcome, self.hazard_t = "hazard", self.t
            b.vx = b.vy = 0.0
            return
        if self._cup():
            return
        if b.speed < config.STOP_SPEED or self.t - self.t0 > config.MAX_PUTT_S:
            b.vx = b.vy = 0.0
            self._clear_moving_obstacles()
            self.outcome = "stopped"
            self.finished = True

    def _collide(self, ax, ay, bx, by, radius, surf, restitution):
        b = self.ball
        qx, qy = closest_on_segment(b.x, b.y, ax, ay, bx, by)
        dx, dy = b.x - qx, b.y - qy
        reach = radius + BR
        d2 = dx * dx + dy * dy
        if d2 >= reach * reach:
            return False
        d = math.sqrt(d2)
        if d < 1e-9:   # dead centre: push out against the velocity
            sp = b.speed or 1.0
            nx, ny = -b.vx / sp, -b.vy / sp
        else:
            nx, ny = dx / d, dy / d
        b.x, b.y = qx + nx * reach, qy + ny * reach
        ux, uy = surf(qx, qy) if surf else (0.0, 0.0)
        vn = (b.vx - ux) * nx + (b.vy - uy) * ny
        if vn < 0:
            b.vx -= (1 + restitution) * vn * nx
            b.vy -= (1 + restitution) * vn * ny
            return True
        return False

    def _cup(self):
        b = self.ball
        cx, cy = self.hole.cup
        R = self.cup_radius
        rx, ry = cx - b.x, cy - b.y
        if math.hypot(rx, ry) >= R:
            self._over_cup = False
            return False
        speed = b.speed
        if speed < 0.25 * self.capture_speed:
            return self._hole()
        if self._over_cup:
            return False
        self._over_cup = True
        ux, uy = b.vx / speed, b.vy / speed
        cross = ux * ry - uy * rx                       # > 0: cup centre is left of the ball's path
        off = min(1.0, abs(cross) / R)
        limit = self.capture_speed * (0.35 + 0.65 * math.sqrt(max(0.0, 1 - off * off)))
        if speed <= limit:
            return self._hole()
        # lip out: deflected away from the centre and slowed
        self.lipped = True
        a = math.radians(config.CUP_LIP_DEFLECT_DEG) * max(0.2, off) * (-1 if cross > 0 else 1)
        ca, sa = math.cos(a), math.sin(a)
        k = config.CUP_LIP_SPEED_KEEP
        b.vx, b.vy = (b.vx * ca - b.vy * sa) * k, (b.vx * sa + b.vy * ca) * k
        return False

    def _hole(self):
        self.outcome, self.holed_t = "holed", self.t
        self.ball.vx = self.ball.vy = 0.0
        return True

    def _clear_moving_obstacles(self):
        b = self.ball
        for ob in self.hole.moving:
            p = ob.sweep_exit(b.x, b.y, BR)
            if p is not None and self.hole.contains(*p):
                b.x, b.y = p

    # -- for the game / renderer ------------------------------------------------------
    @property
    def holed(self):
        return self.outcome == "holed"
