"""
Drawing (pygame): a Wii Sports-style putting view from behind the ball, at night on
a Halloween course. Each hole opens with an overhead flyover that swoops down behind
the tee. While aiming the camera sits behind the ball looking along the aim; once the
ball is struck it turns to watch it and follows when it gets far away. Hold V for the
overhead view. A scorecard runs across the top and a minimap shows the whole hole.

The 3D view is a pinhole camera with yaw and pitch. Flat things (turf, slime, pits,
cup, aim guide) are drawn first; everything standing up is depth-sorted.
"""

import math
import random

import config
from course import DIRS, generate_hole, heading_vec
from game import status_color          # ../game.py (ping pong), reused for the status lights

C = config.CELL_M
BR = config.GOLF_BALL_RADIUS_M

SKY_TOP, SKY_HORIZON = (8, 6, 22), (74, 36, 86)
GROUND, FOG = (20, 24, 26), (52, 34, 70)
TURF_A, TURF_B, TEE_MAT = (44, 116, 64), (38, 104, 58), (30, 82, 46)
WALL_SIDE, WALL_TOP = (72, 40, 100), (242, 126, 34)
STONE, STONE_TOP, STONE_DARK = (124, 126, 138), (160, 162, 174), (70, 72, 84)
PUMPKIN, PUMPKIN_DARK, GLOW = (238, 120, 24), (190, 84, 14), (255, 222, 90)
GHOST, BONE = (236, 240, 255), (238, 230, 206)
SLIME, SLIME_DARK = (120, 224, 60), (66, 160, 40)
BREW, BREW_DARK, IRON = (150, 70, 200), (90, 30, 130), (44, 44, 54)
GOLD, ORANGE, PURPLE = (255, 214, 80), (255, 150, 30), (150, 90, 210)
MOON_AZ_DEG, MOON_EL_DEG = 25.0, 16.0


def _mix(a, b, k):
    return tuple(int(x + (y - x) * k) for x, y in zip(a, b))


def _ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def _lerp_angle(a, b, k):
    d = (b - a + 180) % 360 - 180
    return a + d * k


class Camera:
    """Pinhole camera at (x, y, h) looking along yaw (deg clockwise from north), pitched down."""

    def __init__(self, width):
        self.w = width
        self.F = config.VIEW_FOCAL_PX
        self.cy = config.VIEW_CENTER_Y_PX
        self.set(0, -2, 1, 0, 20)

    def set(self, x, y, h, yaw, pitch):
        self.x, self.y, self.h, self.yaw, self.pitch = x, y, h, yaw, pitch
        fx, fy = heading_vec(yaw)
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        self.right = (fy, -fx, 0.0)
        self.fwd = (fx * cp, fy * cp, -sp)
        self.up = (fx * sp, fy * sp, cp)
        self.flat_fwd = (fx, fy)

    @property
    def horizon_y(self):
        return self.cy - self.F * math.tan(math.radians(self.pitch))

    def to_cam(self, X, Y, Z=0.0):
        dx, dy, dz = X - self.x, Y - self.y, Z - self.h
        r, u, f = self.right, self.up, self.fwd
        return (dx * r[0] + dy * r[1], dx * u[0] + dy * u[1] + dz * u[2], dx * f[0] + dy * f[1] + dz * f[2])

    def depth(self, X, Y, Z=0.0):
        return self.to_cam(X, Y, Z)[2]

    def screen(self, c):
        xc, yc, zc = c
        return self.w / 2 + self.F * xc / zc, self.cy - self.F * yc / zc

    def project(self, X, Y, Z=0.0):
        """(screen x, screen y, px per metre), or None if behind the near plane."""
        c = self.to_cam(X, Y, Z)
        if c[2] < config.VIEW_NEAR_M:
            return None
        sx, sy = self.screen(c)
        return sx, sy, self.F / c[2]

    def polygon(self, pts):
        """Screen polygon for 3D points, clipped at the near plane (None if fully behind)."""
        cam = [self.to_cam(*p) for p in pts]
        near = config.VIEW_NEAR_M
        out = []
        for i, a in enumerate(cam):
            b = cam[(i + 1) % len(cam)]
            ina, inb = a[2] >= near, b[2] >= near
            if ina:
                out.append(a)
            if ina != inb:
                k = (near - a[2]) / (b[2] - a[2])
                out.append(tuple(a[j] + (b[j] - a[j]) * k for j in range(3)))
        if len(out) < 3:
            return None
        return [self.screen(c) for c in out]


class Renderer:
    def __init__(self, screen):
        import pygame
        self.pg = pygame
        self.screen = screen
        self.w, self.h = screen.get_size()
        self.cam = Camera(self.w)
        font = "segoeui,arial,helvetica"
        self.f_huge = pygame.font.SysFont(font, 72, bold=True)
        self.f_big = pygame.font.SysFont(font, 44, bold=True)
        self.f_med = pygame.font.SysFont(font, 26, bold=True)
        self.f_small = pygame.font.SysFont(font, 18)
        self.f_card = pygame.font.SysFont(font, 17, bold=True)
        self.f_mono = pygame.font.SysFont("consolas,couriernew,monospace", 16)
        self.buttons = {}
        self._sky = self._make_sky()
        self._demo = generate_hole(random.Random(31), 5)
        self._mini = None          # (hole, surface, to_px)
        self._glow = {}
        self._last_now = None
        self._cam_state = None     # (x, y, h, yaw, pitch) of the chase camera
        rng = random.Random(4)
        self._stars = [(rng.uniform(-180, 180), rng.uniform(4, 60), rng.choice((1, 1, 2))) for _ in range(120)]

    # -- helpers ------------------------------------------------------------------
    def _make_sky(self):
        h = self.h * 2
        surf = self.pg.Surface((self.w, h))
        for row in range(h):
            k = min(1.0, max(0.0, (row - h * 0.35) / (h * 0.65)))
            self.pg.draw.line(surf, _mix(SKY_TOP, SKY_HORIZON, k ** 1.6), (0, row), (self.w, row))
        return surf

    def _text(self, text, font, color, pos, anchor="topleft", shadow=False):
        if shadow:
            s = font.render(text, True, (0, 0, 0))
            self.screen.blit(s, s.get_rect(**{anchor: pos}).move(3, 3))
        surf = font.render(text, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def _fog(self, color, depth):
        k = min(0.75, max(0.0, (depth - 3.0) / 22.0))
        return _mix(color, FOG, k)

    def _poly3(self, pts, color, depth=None):
        scr = self.cam.polygon(pts)
        if scr:
            if depth is None:
                depth = sum(self.cam.depth(*p) for p in pts) / len(pts)
            self.pg.draw.polygon(self.screen, self._fog(color, depth), scr)

    def _circle3(self, x, y, r, color, z=0.0, n=20):
        pts = [(x + r * math.cos(2 * math.pi * k / n), y + r * math.sin(2 * math.pi * k / n), z) for k in range(n)]
        self._poly3(pts, color)

    def _glow_at(self, sx, sy, radius, color=(255, 170, 60), strength=0.35):
        """Soft additive halo (colours are pre-multiplied, so black adds nothing)."""
        r = max(4, min(140, int(radius / 4) * 4))
        key = (r, color, strength)
        surf = self._glow.get(key)
        if surf is None:
            pg = self.pg
            surf = pg.Surface((2 * r, 2 * r))
            surf.fill((0, 0, 0))
            for k in range(r, 0, -2):
                a = strength * (1 - k / r) ** 2
                pg.draw.circle(surf, tuple(int(c * a) for c in color), (r, r), k)
            self._glow[key] = surf
        self.screen.blit(surf, (sx - r, sy - r), special_flags=self.pg.BLEND_RGB_ADD)

    # -- camera -------------------------------------------------------------------
    def _aim_pose(self, ball, aim_deg):
        fx, fy = heading_vec(aim_deg)
        b = config.VIEW_AIM_BACK_M
        return ball[0] - fx * b, ball[1] - fy * b, config.VIEW_AIM_HEIGHT_M, aim_deg, config.VIEW_AIM_PITCH_DEG

    @staticmethod
    def _overview_pose(hole, yaw_offset=0.0):
        x0, y0, x1, y1 = hole.bounds()
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ext = max(x1 - x0, y1 - y0)
        yaw = math.degrees(math.atan2(mx - hole.tee[0], my - hole.tee[1])) + yaw_offset
        pitch = config.VIEW_OVERVIEW_PITCH_DEG
        dist = 1.15 * ext + 2.0
        fx, fy = heading_vec(yaw)
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        return mx - fx * dist * cp, my - fy * dist * cp, dist * sp, yaw, pitch

    def _update_camera(self, now, game, overview):
        dt = 0.0 if self._last_now is None else max(0.0, min(0.1, now - self._last_now))
        self._last_now = now
        hole = game.hole or self._demo
        st = game.state
        if st in ("menu", "game_over"):
            self.cam.set(*self._overview_pose(hole, yaw_offset=15 * math.sin(now * 0.15)))
            self._cam_state = None
            return
        if overview:
            pose = self._overview_pose(hole)
        elif st == "intro":
            u = _ease((now - game.state_since) / (config.INTRO_S * 0.9))
            a, b = self._overview_pose(hole), self._aim_pose(game.ball, game.golfer.aim_deg)
            pose = tuple(x + (y - x) * u for x, y in zip(a[:3], b[:3])) + (
                _lerp_angle(a[3], b[3], u), a[4] + (b[4] - a[4]) * u)
        elif st == "aiming":
            pose = self._aim_pose(game.ball, game.golfer.aim_deg)
        else:
            pose = self._chase_pose(dt, game)
        if st in ("rolling", "result", "holed") and not overview:
            self._cam_state = pose
        else:
            self._cam_state = None
        self.cam.set(*pose)

    def _chase_pose(self, dt, game):
        if self._cam_state is None:
            self._cam_state = (self.cam.x, self.cam.y, self.cam.h, self.cam.yaw, self.cam.pitch)
        x, y, h, yaw, pitch = self._cam_state
        b = game.sim.ball if game.sim else None
        bx, by = (b.x, b.y) if b else game.ball
        k = min(1.0, config.VIEW_SMOOTHING * dt)
        d = math.hypot(bx - x, by - y)
        if d > config.VIEW_FOLLOW_MAX_M:
            pull = (d - config.VIEW_FOLLOW_MAX_M) * k
            x += (bx - x) / d * pull
            y += (by - y) / d * pull
        if d > 0.3:
            yaw = _lerp_angle(yaw, math.degrees(math.atan2(bx - x, by - y)), k)
        h += (config.VIEW_ROLL_HEIGHT_M - h) * k
        pitch += (config.VIEW_ROLL_PITCH_DEG - pitch) * k
        return x, y, h, yaw, pitch

    # -- sky and ground ---------------------------------------------------------------
    def _draw_sky(self, now):
        pg = self.pg
        hy = self.cam.horizon_y
        if hy > 0:
            self.screen.blit(self._sky, (0, hy - self._sky.get_height()))
            F = self.cam.F
            for az, el, size in self._stars:
                da = (az - self.cam.yaw + 180) % 360 - 180
                if abs(da) < 60:
                    sx = self.w / 2 + F * math.tan(math.radians(da))
                    sy = self.cam.cy - F * math.tan(math.radians(el + self.cam.pitch))
                    if 0 < sy < hy - 4:
                        tw = 150 + int(80 * math.sin(now * 2 + az))
                        pg.draw.circle(self.screen, (tw, tw, min(255, tw + 30)), (sx, sy), size)
            da = (MOON_AZ_DEG - self.cam.yaw + 180) % 360 - 180
            if abs(da) < 55:
                mx = self.w / 2 + F * math.tan(math.radians(da))
                my = self.cam.cy - F * math.tan(math.radians(MOON_EL_DEG + self.cam.pitch))
                if my > -60:
                    self._glow_at(mx, my, 140, (150, 140, 190), 0.5)
                    pg.draw.circle(self.screen, (250, 240, 200), (mx, my), 46)
                    for ox, oy, r in ((-14, -8, 9), (12, 10, 7), (6, -18, 5)):
                        pg.draw.circle(self.screen, (226, 214, 176), (mx + ox, my + oy), r)
                    for i in range(3):   # bats crossing the moon
                        t = now * 0.6 + i * 2.1
                        bx = mx - 140 + (t * 60 % 320)
                        by = my - 30 + 20 * math.sin(t * 1.7 + i)
                        flap = 6 * math.sin(now * 14 + i)
                        pg.draw.lines(self.screen, (10, 8, 16), False,
                                      [(bx - 12, by - flap), (bx - 5, by - 2), (bx, by + 2), (bx + 5, by - 2),
                                       (bx + 12, by - flap)], 3)
        top = max(0, int(hy))
        if top < self.h:
            pg.draw.rect(self.screen, GROUND, (0, top, self.w, self.h - top))
            band = pg.Surface((self.w, 60), pg.SRCALPHA)
            for row in range(60):
                pg.draw.line(band, (*FOG, int(200 * (1 - row / 60))), (0, row), (self.w, row))
            self.screen.blit(band, (0, top))

    def _draw_turf(self, hole):
        for (i, j) in hole.cells:
            x, y = i * C, j * C
            col = TURF_A if (i + j) % 2 == 0 else TURF_B
            self._poly3([(x - C / 2, y - C / 2, 0), (x + C / 2, y - C / 2, 0), (x + C / 2, y + C / 2, 0),
                         (x - C / 2, y + C / 2, 0)], col)
        tx, ty = hole.tee
        fx, fy = DIRS[hole.dirs[0]]
        rx, ry = fy, -fx
        pts = [(tx + sa * rx * 0.25 + sb * fx * 0.18, ty + sa * ry * 0.25 + sb * fy * 0.18, 0.001)
               for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        self._poly3(pts, TEE_MAT)

    def _draw_floor_decals(self, now, game, hole):
        pg = self.pg
        for ob in hole.obstacles:
            if ob.kind == "slime":
                self._circle3(ob.x, ob.y, ob.r, SLIME_DARK, 0.002)
                self._circle3(ob.x, ob.y, ob.r * 0.8, SLIME, 0.003)
                for k in range(4):
                    a = k * 1.7 + now * 0.4
                    rr = ob.r * (0.25 + 0.15 * k)
                    bub = 0.03 + 0.015 * math.sin(now * 3 + k)
                    self._circle3(ob.x + rr * math.cos(a), ob.y + rr * math.sin(a), bub, (180, 255, 120), 0.004, 8)
            elif ob.kind == "pit":
                self._circle3(ob.x, ob.y, ob.r + 0.05, IRON, 0.002)
                self._circle3(ob.x, ob.y, ob.r, BREW_DARK, 0.003)
                self._circle3(ob.x, ob.y, ob.r * 0.75, BREW, 0.004)
                for k in range(5):
                    ph = (now * 0.9 + k * 0.37) % 1.0
                    a = k * 2.4
                    self._circle3(ob.x + ob.r * 0.5 * math.cos(a), ob.y + ob.r * 0.5 * math.sin(a),
                                  0.015 + 0.04 * ph, (200, 130, 240), 0.005, 8)
        cx, cy = hole.cup
        R = game.settings["cup_radius_m"]
        self._circle3(cx, cy, R + 0.012, (230, 230, 235), 0.004)
        self._circle3(cx, cy, R, (8, 8, 10), 0.005)
        if game.state == "aiming":
            self._draw_aim_guide(game.ball, game.golfer.aim_deg, game.settings["guide_m"])
        ball = self._ball_pos(game)
        if ball and ball[2] >= 0:
            self._circle3(ball[0] + 0.01, ball[1] - 0.01, BR * 1.1, (16, 40, 24), 0.006, 10)

    def _draw_aim_guide(self, ball, aim_deg, length):
        fx, fy = heading_vec(aim_deg)
        rx, ry = fy, -fx
        w = 0.018
        d = 0.12
        while d < length:
            d1 = min(length, d + 0.12)
            a = (ball[0] + fx * d, ball[1] + fy * d)
            b = (ball[0] + fx * d1, ball[1] + fy * d1)
            fade = 1 - 0.6 * d / length
            col = _mix((40, 60, 40), (255, 240, 120), fade)
            self._poly3([(a[0] - rx * w, a[1] - ry * w, 0.007), (b[0] - rx * w, b[1] - ry * w, 0.007),
                         (b[0] + rx * w, b[1] + ry * w, 0.007), (a[0] + rx * w, a[1] + ry * w, 0.007)], col)
            d += 0.24
        tip = (ball[0] + fx * (length + 0.18), ball[1] + fy * (length + 0.18))
        base = (ball[0] + fx * length, ball[1] + fy * length)
        self._poly3([(tip[0], tip[1], 0.007), (base[0] + rx * 0.07, base[1] + ry * 0.07, 0.007),
                     (base[0] - rx * 0.07, base[1] - ry * 0.07, 0.007)], (255, 220, 100))

    # -- standing things ------------------------------------------------------------------
    def _prism(self, footprint, z0, z1, side, top, light=(0.5, -0.8)):
        """Extruded convex polygon (counter-clockwise footprint)."""
        cx, cy = self.cam.x, self.cam.y
        n = len(footprint)
        faces = []
        for i in range(n):
            (ax, ay), (bx, by) = footprint[i], footprint[(i + 1) % n]
            nx, ny = by - ay, -(bx - ax)
            mx, my = (ax + bx) / 2, (ay + by) / 2
            if nx * (cx - mx) + ny * (cy - my) > 0:
                L = math.hypot(nx, ny) or 1
                shade = 0.65 + 0.35 * max(0.0, (nx * light[0] + ny * light[1]) / L)
                col = tuple(int(c * shade) for c in side)
                faces.append((self.cam.depth(mx, my, (z0 + z1) / 2),
                              [(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)], col))
        for d, pts, col in sorted(faces, key=lambda f: -f[0]):
            self._poly3(pts, col, d)
        self._poly3([(x, y, z1) for x, y in footprint], top)

    @staticmethod
    def _slab(ax, ay, bx, by, half, extend=0.0):
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) or 1e-6
        ux, uy = dx / L, dy / L
        nx, ny = -uy * half, ux * half
        ax, ay, bx, by = ax - ux * extend, ay - uy * extend, bx + ux * extend, by + uy * extend
        return [(ax - nx, ay - ny), (bx - nx, by - ny), (bx + nx, by + ny), (ax + nx, ay + ny)]

    def _wall_items(self, hole):
        items = []
        half = 0.04
        for ax, ay, bx, by in hole.walls:
            L = math.hypot(bx - ax, by - ay)
            n = max(1, math.ceil(L / 0.6))
            for k in range(n):
                pa = (ax + (bx - ax) * k / n, ay + (by - ay) * k / n)
                pb = (ax + (bx - ax) * (k + 1) / n, ay + (by - ay) * (k + 1) / n)
                fp = self._slab(*pa, *pb, half, extend=half)
                mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
                items.append((self.cam.depth(mx, my), lambda fp=fp: self._prism(fp, 0, config.WALL_HEIGHT_M,
                                                                                WALL_SIDE, WALL_TOP)))
        return items

    def _draw_pumpkin(self, x, y, r, glow=True):
        pg = self.pg
        p = self.cam.project(x, y, r * 0.8)
        if p is None:
            return
        sx, sy, s = p
        R = r * s
        if R < 1:
            return
        d = self.cam.depth(x, y)
        if glow:
            self._glow_at(sx, sy, R * 3.0)
        body, dark = self._fog(PUMPKIN, d), self._fog(PUMPKIN_DARK, d)
        pg.draw.ellipse(self.screen, dark, (sx - R * 1.15, sy - R * 0.85, R * 2.3, R * 1.75))
        pg.draw.ellipse(self.screen, body, (sx - R * 0.95, sy - R * 0.85, R * 1.9, R * 1.75))
        pg.draw.ellipse(self.screen, dark, (sx - R * 0.35, sy - R * 0.85, R * 0.7, R * 1.75), max(1, int(R * 0.08)))
        pg.draw.rect(self.screen, (70, 90, 30), (sx - R * 0.1, sy - R * 1.15, R * 0.22, R * 0.38))
        if R > 4:
            for side in (-1, 1):
                ex = sx + side * R * 0.38
                pg.draw.polygon(self.screen, GLOW, [(ex - R * 0.16, sy - R * 0.05), (ex + R * 0.16, sy - R * 0.05),
                                                    (ex, sy - R * 0.35)])
            pg.draw.polygon(self.screen, GLOW, [(sx - R * 0.55, sy + R * 0.2), (sx - R * 0.25, sy + R * 0.35),
                                                (sx, sy + R * 0.22), (sx + R * 0.25, sy + R * 0.35),
                                                (sx + R * 0.55, sy + R * 0.2), (sx, sy + R * 0.58)])

    def _draw_ghost(self, now, ob):
        pg = self.pg
        x, y = ob.pos(now)
        bob = 0.04 * math.sin(now * 3 + ob.phase)
        p = self.cam.project(x, y, 0.3 + bob)
        foot = self.cam.project(x, y, 0.0)
        if p is None or foot is None:
            return
        sx, sy, s = p
        R = ob.r * s
        col = self._fog(GHOST, self.cam.depth(x, y))
        pg.draw.ellipse(self.screen, (10, 20, 14), (foot[0] - R * 0.8, foot[1] - R * 0.2, R * 1.6, R * 0.4))
        self._glow_at(sx, sy, R * 2.4, (120, 150, 200))
        pg.draw.circle(self.screen, col, (sx, sy - R * 0.3), R)
        hem = sy + R * 1.1
        pts = [(sx - R, sy - R * 0.3), (sx + R, sy - R * 0.3)]
        for k in range(7):
            u = 1 - k / 6
            wave = R * 0.18 * math.sin(now * 8 + k * 1.3)
            pts.append((sx - R + 2 * R * u, hem + (wave if k % 2 else -R * 0.2)))
        pg.draw.polygon(self.screen, col, pts)
        if R > 3:
            for side in (-1, 1):
                pg.draw.ellipse(self.screen, (20, 20, 30), (sx + side * R * 0.35 - R * 0.14, sy - R * 0.6,
                                                            R * 0.28, R * 0.42))
            pg.draw.ellipse(self.screen, (20, 20, 30), (sx - R * 0.15, sy - R * 0.05, R * 0.3, R * 0.36))

    def _draw_spinner(self, now, ob):
        (ax, ay), (bx, by) = ob.ends(now)
        self._prism(self._slab(ax, ay, bx, by, ob.r), 0.01, 0.08, BONE, (250, 246, 230))
        for ex, ey in ((ax, ay), (bx, by)):
            for side in (-1, 1):
                nx, ny = -(by - ay), bx - ax
                L = math.hypot(nx, ny) or 1
                p = self.cam.project(ex + side * nx / L * 0.035, ey + side * ny / L * 0.035, 0.05)
                if p:
                    self.pg.draw.circle(self.screen, self._fog(BONE, self.cam.depth(ex, ey)), p[:2], 0.045 * p[2])

    def _draw_post_and_skull(self, ob):
        s = 0.045
        fp = [(ob.x - s, ob.y - s), (ob.x + s, ob.y - s), (ob.x + s, ob.y + s), (ob.x - s, ob.y + s)]
        self._prism(fp, 0, 0.32, (60, 50, 60), (90, 80, 90))
        p = self.cam.project(ob.x, ob.y, 0.4)
        if p:
            pg = self.pg
            sx, sy, sc = p
            R = 0.08 * sc
            col = self._fog(BONE, self.cam.depth(ob.x, ob.y))
            pg.draw.circle(self.screen, col, (sx, sy), R)
            pg.draw.rect(self.screen, col, (sx - R * 0.55, sy + R * 0.5, R * 1.1, R * 0.6))
            if R > 3:
                for side in (-1, 1):
                    pg.draw.circle(self.screen, (20, 10, 20), (sx + side * R * 0.38, sy), R * 0.28)

    def _draw_tombstone(self, ob):
        fp = self._slab(ob.ax, ob.ay, ob.bx, ob.by, ob.r)
        self._prism(fp, 0, ob.height, STONE, STONE_TOP)
        dx, dy = ob.bx - ob.ax, ob.by - ob.ay
        L = math.hypot(dx, dy)
        nx, ny = -dy / L, dx / L
        if nx * (self.cam.x - ob.ax) + ny * (self.cam.y - ob.ay) < 0:
            nx, ny = -nx, -ny
        mx, my = (ob.ax + ob.bx) / 2 + nx * (ob.r + 0.002), (ob.ay + ob.by) / 2 + ny * (ob.r + 0.002)
        ux, uy = dx / L, dy / L
        a = self.cam.project(mx, my, ob.height * 0.25)
        b = self.cam.project(mx, my, ob.height * 0.85)
        c = self.cam.project(mx - ux * 0.12, my - uy * 0.12, ob.height * 0.65)
        e = self.cam.project(mx + ux * 0.12, my + uy * 0.12, ob.height * 0.65)
        if a and b and c and e:
            wdt = max(1, int(0.03 * a[2]))
            col = self._fog(STONE_DARK, self.cam.depth(mx, my))
            self.pg.draw.line(self.screen, col, a[:2], b[:2], wdt)
            self.pg.draw.line(self.screen, col, c[:2], e[:2], wdt)

    def _draw_decoration(self, now, d):
        pg = self.pg
        foot = self.cam.project(d.x, d.y, 0)
        if foot is None:
            return
        sx, sy, s = foot
        depth = self.cam.depth(d.x, d.y)
        if d.kind == "tree":
            col = self._fog((34, 24, 36), depth)
            H = 2.4 * d.size * s
            trunk_w = max(2, int(0.12 * d.size * s))
            pg.draw.line(self.screen, col, (sx, sy), (sx, sy - H), trunk_w)
            rng = random.Random(d.seed)
            for k in range(5):
                hy = sy - H * rng.uniform(0.45, 0.95)
                side = 1 if k % 2 else -1
                L = H * rng.uniform(0.2, 0.38)
                ex, ey = sx + side * L, hy - L * rng.uniform(0.2, 0.7)
                pg.draw.line(self.screen, col, (sx, hy), (ex, ey), max(1, trunk_w // 2))
                pg.draw.line(self.screen, col, (ex, ey), (ex + side * L * 0.4, ey - L * 0.35), max(1, trunk_w // 3))
        elif d.kind == "grave":
            col = self._fog(STONE, depth)
            W, H = 0.42 * d.size * s, 0.5 * d.size * s
            pg.draw.rect(self.screen, col, (sx - W / 2, sy - H, W, H), border_top_left_radius=int(W / 2),
                         border_top_right_radius=int(W / 2))
            pg.draw.rect(self.screen, self._fog(STONE_DARK, depth), (sx - W / 2, sy - H, W, H), max(1, int(W * 0.06)),
                         border_top_left_radius=int(W / 2), border_top_right_radius=int(W / 2))
            if W > 10:
                self._text("RIP", self.f_small if W > 34 else self.f_mono, self._fog(STONE_DARK, depth),
                           (sx, sy - H * 0.45), "center")
        elif d.kind == "lantern":
            self._draw_pumpkin(d.x, d.y, 0.16 * d.size)
        elif d.kind == "fence":
            col = self._fog((26, 22, 30), depth)
            W, H = 0.9 * s, 0.6 * d.size * s
            w = max(1, int(0.025 * s))
            pg.draw.line(self.screen, col, (sx - W / 2, sy - H * 0.75), (sx + W / 2, sy - H * 0.75), w)
            pg.draw.line(self.screen, col, (sx - W / 2, sy - H * 0.25), (sx + W / 2, sy - H * 0.25), w)
            for k in range(6):
                bx = sx - W / 2 + W * k / 5
                pg.draw.line(self.screen, col, (bx, sy), (bx, sy - H), w)
                pg.draw.polygon(self.screen, col, [(bx - w * 2, sy - H), (bx + w * 2, sy - H), (bx, sy - H - w * 4)])

    def _draw_flag(self, now, hole):
        pg = self.pg
        cx, cy = hole.cup
        base, top = self.cam.project(cx, cy, 0), self.cam.project(cx, cy, 1.0)
        if base is None or top is None:
            return
        pg.draw.line(self.screen, (230, 230, 236), base[:2], top[:2], max(1, int(0.012 * top[2])))
        rx, ry = self.cam.right[0], self.cam.right[1]
        wave = 0.03 * math.sin(now * 3)
        pts = [self.cam.project(cx, cy, 1.0), self.cam.project(cx, cy, 0.78),
               self.cam.project(cx + rx * 0.34, cy + ry * 0.34, 0.89 + wave)]
        if all(pts):
            pg.draw.polygon(self.screen, self._fog(ORANGE, self.cam.depth(cx, cy)), [p[:2] for p in pts])
            mid = self.cam.project(cx + rx * 0.12, cy + ry * 0.12, 0.89)
            pg.draw.circle(self.screen, (20, 10, 20), mid[:2], max(1, 0.035 * mid[2]))

    def _draw_ball(self, x, y, z):
        p = self.cam.project(x, y, BR + z)
        if p is None:
            return
        r = max(1.5, BR * p[2])
        col = (250, 250, 252) if z >= 0 else (120, 120, 126)
        self.pg.draw.circle(self.screen, self._fog(col, self.cam.depth(x, y)), p[:2], r)
        if r > 3:
            self.pg.draw.circle(self.screen, (205, 205, 215), (p[0] + r * 0.25, p[1] + r * 0.25), r * 0.55, 1)

    def _draw_golfer(self, game, pos, aim_deg, charge):
        """A chunky Mii in a witch hat standing left of the ball, putter behind it."""
        pg = self.pg
        fx, fy = heading_vec(aim_deg)
        rx, ry = fy, -fx
        hand = 1 if config.DOMINANT_HAND == "right" else -1
        gx, gy = pos[0] - hand * rx * 0.42, pos[1] - hand * ry * 0.42
        k = 0.75   # Mii scale
        parts = [  # (along, lateral, z, radius, colour)
            (-0.09, 0, 0.10, 0.065, (40, 36, 60)), (0.09, 0, 0.10, 0.065, (40, 36, 60)),
            (-0.09, 0, 0.30, 0.07, (40, 36, 60)), (0.09, 0, 0.30, 0.07, (40, 36, 60)),
            (0, 0, 0.58, 0.2, (110, 50, 150)), (0, 0, 0.74, 0.18, (110, 50, 150)),
            (0, 0.2, 0.55, 0.055, (240, 200, 165)),
            (0, 0, 1.0, 0.16, (240, 200, 165)),
        ]
        parts = [(a * k, l * k, z * k, r * k, c) for a, l, z, r, c in parts]
        drawn = []
        for along, lat, z, r, col in parts:
            x, y = gx + fx * along + rx * lat * hand, gy + fy * along + ry * lat * hand
            drawn.append((self.cam.depth(x, y, z), x, y, z, r, col))
        for d, x, y, z, r, col in sorted(drawn, key=lambda it: -it[0]):
            p = self.cam.project(x, y, z)
            if p:
                pg.draw.circle(self.screen, self._fog(col, d), p[:2], r * p[2])
        brim, tip = self.cam.project(gx, gy, 1.12 * k), self.cam.project(gx, gy, 1.55 * k)
        if brim and tip:
            w = 0.24 * k * brim[2]
            pg.draw.ellipse(self.screen, (16, 12, 22), (brim[0] - w, brim[1] - w * 0.18, 2 * w, w * 0.36))
            pg.draw.polygon(self.screen, (16, 12, 22), [(brim[0] - w * 0.55, brim[1]), (brim[0] + w * 0.55, brim[1]),
                                                        (tip[0] + w * 0.3, tip[1])])
            pg.draw.line(self.screen, ORANGE, (brim[0] - w * 0.5, brim[1] - w * 0.12),
                         (brim[0] + w * 0.5, brim[1] - w * 0.12), max(1, int(w * 0.12)))
        back = 0.05 + 0.3 * charge
        hx, hy = gx + rx * 0.2 * k * hand, gy + ry * 0.2 * k * hand
        cx, cy = pos[0] - fx * back, pos[1] - fy * back
        a, b = self.cam.project(hx, hy, 0.55 * k), self.cam.project(cx, cy, 0.03)
        if a and b:
            pg.draw.line(self.screen, (190, 190, 200), a[:2], b[:2], max(1, int(0.015 * b[2])))
            fp = self._slab(cx - rx * 0.05, cy - ry * 0.05, cx + rx * 0.05, cy + ry * 0.05, 0.015)
            self._prism(fp, 0, 0.03, (150, 150, 160), (200, 200, 210))

    def _ball_pos(self, game):
        if game.state in ("rolling", "result", "holed") and game.sim is not None:
            b = game.sim.ball
            if game.sim.outcome == "hazard" and game.state != "rolling":
                return None
            return b.x, b.y, b.z
        if game.state in ("menu", "game_over") or game.hole is None:
            return None
        return game.ball[0], game.ball[1], 0.0

    def _draw_scene(self, now, game, hole):
        items = self._wall_items(hole)
        for ob in hole.obstacles:
            if ob.kind == "pumpkin":
                items.append((self.cam.depth(ob.x, ob.y), lambda ob=ob: self._draw_pumpkin(ob.x, ob.y, ob.r)))
            elif ob.kind == "tombstone":
                items.append((self.cam.depth((ob.ax + ob.bx) / 2, (ob.ay + ob.by) / 2),
                              lambda ob=ob: self._draw_tombstone(ob)))
            elif ob.kind == "ghost":
                gx, gy = ob.pos(now)
                items.append((self.cam.depth(gx, gy), lambda ob=ob: self._draw_ghost(now, ob)))
            elif ob.kind == "spinner":
                items.append((self.cam.depth(ob.x, ob.y) + 0.01, lambda ob=ob: self._draw_spinner(now, ob)))
                items.append((self.cam.depth(ob.x, ob.y), lambda ob=ob: self._draw_post_and_skull(ob)))
        for d in hole.decorations:
            dd = self.cam.depth(d.x, d.y)
            if dd > 2.6:   # nothing right in front of the lens
                items.append((dd, lambda d=d: self._draw_decoration(now, d)))
        cx, cy = hole.cup
        items.append((self.cam.depth(cx, cy), lambda: self._draw_flag(now, hole)))
        ball = self._ball_pos(game)
        if ball is not None and ball[2] > -0.05:
            items.append((self.cam.depth(ball[0], ball[1]) - 0.001, lambda: self._draw_ball(*ball)))
        if game.state in ("aiming", "intro") or (game.state == "rolling" and game.sim is not None):
            pos = game.ball if game.state != "rolling" else game.sim.start
            aim = game.golfer.aim_deg if game.state != "rolling" else (game.last_putt or {}).get("heading_deg", 0)
            charge = game.golfer.charge_fraction(now) if game.state == "aiming" else 0.0
            fx, fy = heading_vec(aim)
            hand = 1 if config.DOMINANT_HAND == "right" else -1
            gx, gy = pos[0] - hand * fy * 0.42, pos[1] + hand * fx * 0.42
            items.append((self.cam.depth(gx, gy), lambda: self._draw_golfer(game, pos, aim, charge)))
        for _, fn in sorted(items, key=lambda it: -it[0]):
            fn()

    # -- HUD --------------------------------------------------------------------------
    def _draw_scorecard(self, game):
        pg = self.pg
        n = config.HOLES_PER_GAME
        cw, lw, tw, rh = 58, 96, 78, 26
        width = lw + n * cw + tw
        x0, y0 = (self.w - width) // 2, 10
        pg.draw.rect(self.screen, (250, 248, 240), (x0, y0, width, rh * 3), border_radius=6)
        rows = [("HOLE", [str(i + 1) for i in range(n)], "TOT")]
        pars = [str(h.par) for h in game.course] or ["-"] * n
        rows.append(("PAR", pars + [""] * (n - len(pars)), str(sum(h.par for h in game.course)) if game.course else "-"))
        scores = [str(s) for s in game.strokes]
        rows.append((config.PLAYER_NAME[:9], scores + [""] * (n - len(scores)), str(game.total_strokes)))
        for r, (label, cells, total) in enumerate(rows):
            y = y0 + r * rh
            hdr = r == 0
            if hdr:
                pg.draw.rect(self.screen, (60, 30, 80), (x0, y, width, rh), border_top_left_radius=6,
                             border_top_right_radius=6)
            col = (255, 255, 255) if hdr else (30, 30, 40)
            self._text(label, self.f_card, ORANGE if hdr else col, (x0 + 8, y + rh // 2), "midleft")
            for i, txt in enumerate(cells):
                cx = x0 + lw + i * cw + cw // 2
                c = col
                if r == 2 and txt and i < len(game.course):
                    d = int(txt) - game.course[i].par
                    c = (200, 40, 40) if d < 0 else ((40, 80, 200) if d > 0 else col)
                self._text(txt, self.f_card, c, (cx, y + rh // 2), "center")
            self._text(total, self.f_card, ORANGE if hdr else col, (x0 + lw + n * cw + tw // 2, y + rh // 2), "center")
        for i in range(n + 1):
            x = x0 + lw + i * cw
            pg.draw.line(self.screen, (180, 176, 170), (x, y0 + rh), (x, y0 + rh * 3))
        if game.course and game.state not in ("menu", "game_over"):
            cur = pg.Rect(x0 + lw + game.hole_index * cw, y0, cw, rh * 3)
            pg.draw.rect(self.screen, GOLD, cur.inflate(2, 2), 3, border_radius=3)

    def _draw_info(self, game, now):
        pg = self.pg
        hole = game.hole
        lines = [f"{game.difficulty}   input: {game.golfer.source}"]
        if hole:
            lines.append(f"Hole {hole.number}   Par {hole.par}")
            nxt = game.hole_strokes + (1 if game.state in ("aiming", "intro") else 0)
            lines.append(f"Stroke {max(1, nxt)}   To cup {game.distance_to_cup():.1f} m")
            lines.append(f"Aim {game.golfer.aim_deg:+.0f}°")
        t = game.last_putt
        if t:
            push = "" if abs(t["push_deg"]) < 0.05 else f"  {'pull' if t['push_deg'] < 0 else 'push'} {abs(t['push_deg']):.1f}°"
            lines.append(f"Last: {t['speed']:.1f} m/s{push}")
        panel = pg.Surface((260, 22 * len(lines) + 46), pg.SRCALPHA)
        panel.fill((0, 0, 0, 130))
        self.screen.blit(panel, (16, 100))
        for i, line in enumerate(lines):
            self._text(line, self.f_small, (230, 230, 240), (26, 106 + 22 * i))
        y = 106 + 22 * len(lines) + 12
        x0, wdt = 26, 240
        pg.draw.rect(self.screen, (70, 70, 84), (x0, y, wdt, 10), border_radius=5)
        if config.AIM_POSE_JOYSTICK:
            # wrist joystick: deadzone in the middle, marker where the wrist is
            dz = config.AIM_POSE_DEADZONE
            pg.draw.rect(self.screen, (110, 110, 130), (x0 + wdt * (0.5 - dz / 2), y, wdt * dz, 10))
            if game.golfer.tracking:
                st = game.golfer.stick
                u = 0.5 + (dz / 2 + (1 - dz) / 2 * abs(st)) * (1 if st > 0 else -1) if st else 0.5
                pg.draw.circle(self.screen, GOLD if st else (230, 230, 240), (x0 + wdt * u, y + 5), 7)
            self._text("wrist aim", self.f_mono, (150, 150, 165), (x0 + wdt, y + 14), "topright")
        else:
            # motor twist: centre = suggested line, ends = AIM_TWIST_MAX_DEG
            pg.draw.line(self.screen, (130, 130, 150), (x0 + wdt / 2, y - 2), (x0 + wdt / 2, y + 12), 2)
            if game.golfer.twisting:
                u = 0.5 + 0.5 * game.golfer.twist_offset_deg / config.AIM_TWIST_MAX_DEG
                pg.draw.circle(self.screen, GOLD, (x0 + wdt * u, y + 5), 7)
                label = f"twist {game.golfer.twist_offset_deg:+.0f}°   (C recenter)"
            else:
                label = "twist: motor not ready"
            self._text(label, self.f_mono, (150, 150, 165), (x0 + wdt, y + 14), "topright")

    def _draw_power(self, game, now, live_strength):
        pg = self.pg
        f = game.golfer.charge_fraction(now)
        if game.golfer.charge_t is None:
            lo, hi = config.KEYBOARD_STRENGTH_RANGE
            if live_strength <= 0:
                return
            f = max(0.0, min(1.0, (live_strength - lo) / (hi - lo)))
        x, y, w, h = 30, self.h - 260, 26, 190
        pg.draw.rect(self.screen, (0, 0, 0), (x - 4, y - 4, w + 8, h + 8), border_radius=6)
        fill = int(h * f)
        col = _mix((90, 220, 90), (255, 80, 40), f)
        pg.draw.rect(self.screen, col, (x, y + h - fill, w, fill), border_radius=4)
        pg.draw.rect(self.screen, (230, 230, 235), (x, y, w, h), 2, border_radius=4)
        self._text("POWER", self.f_mono, (230, 230, 235), (x + w // 2, y + h + 10), "midtop")

    def _draw_statuses(self, statuses):
        x, y = self.w - 16, 100
        for label, kind, status in statuses:
            surf = self.f_small.render(f"{label}: {status}", True, (230, 230, 235))
            r = surf.get_rect(topright=(x, y))
            self.screen.blit(surf, r)
            self.pg.draw.circle(self.screen, status_color(kind, status), (r.left - 12, r.centery), 7)
            y += 24
        return y

    def _minimap(self, hole, size):
        if self._mini is not None and self._mini[0] is hole:
            return self._mini
        pg = self.pg
        x0, y0, x1, y1 = hole.bounds()
        pad = 8
        scale = (size - 2 * pad) / max(x1 - x0, y1 - y0)
        ox = pad + ((size - 2 * pad) - (x1 - x0) * scale) / 2
        oy = pad + ((size - 2 * pad) - (y1 - y0) * scale) / 2

        def to_px(x, y):
            return ox + (x - x0) * scale, size - (oy + (y - y0) * scale)

        surf = pg.Surface((size, size), pg.SRCALPHA)
        surf.fill((0, 0, 0, 140))
        for i, j in hole.cells:
            a = to_px((i - 0.5) * C, (j + 0.5) * C)
            pg.draw.rect(surf, TURF_A, (a[0], a[1], C * scale + 1, C * scale + 1))
        for ax, ay, bx, by in hole.walls:
            pg.draw.line(surf, WALL_TOP, to_px(ax, ay), to_px(bx, by), 2)
        for ob in hole.obstacles:
            if ob.kind in ("slime", "pit"):
                pg.draw.circle(surf, SLIME if ob.kind == "slime" else BREW, to_px(ob.x, ob.y), max(2, ob.r * scale))
            elif ob.kind == "pumpkin":
                pg.draw.circle(surf, PUMPKIN, to_px(ob.x, ob.y), max(2, ob.r * scale))
            elif ob.kind == "tombstone":
                pg.draw.line(surf, STONE_TOP, to_px(ob.ax, ob.ay), to_px(ob.bx, ob.by), max(2, int(ob.r * 2 * scale)))
            elif ob.kind == "spinner":
                pg.draw.circle(surf, (120, 110, 100), to_px(ob.x, ob.y), ob.length * scale, 1)
            elif ob.kind == "ghost":
                pg.draw.line(surf, (120, 130, 160), to_px(ob.ax, ob.ay), to_px(ob.bx, ob.by), 1)
        pg.draw.circle(surf, (8, 8, 10), to_px(*hole.cup), 4)
        pg.draw.circle(surf, ORANGE, to_px(*hole.cup), 4, 1)
        self._mini = (hole, surf, to_px)
        return self._mini

    def _draw_minimap(self, game, now, x, y, size=200):
        hole = game.hole
        if hole is None or game.state in ("menu", "game_over"):
            return
        pg = self.pg
        _, surf, to_px = self._minimap(hole, size)
        self.screen.blit(surf, (x, y))
        off = lambda p: (p[0] + x, p[1] + y)
        for ob in hole.obstacles:
            if ob.kind == "ghost":
                pg.draw.circle(self.screen, GHOST, off(to_px(*ob.pos(now))), 4)
            elif ob.kind == "spinner":
                a, b = ob.ends(now)
                pg.draw.line(self.screen, BONE, off(to_px(*a)), off(to_px(*b)), 3)
        ball = self._ball_pos(game)
        if ball:
            bp = off(to_px(ball[0], ball[1]))
            if game.state == "aiming":
                fx, fy = heading_vec(game.golfer.aim_deg)
                L = game.settings["guide_m"] + 0.5
                pg.draw.line(self.screen, GOLD, bp, off(to_px(ball[0] + fx * L, ball[1] + fy * L)), 2)
            pg.draw.circle(self.screen, (255, 255, 255), bp, 4)
        pg.draw.rect(self.screen, (230, 230, 235), (x, y, size, size), 1)

    def _draw_preview(self, snap):
        if snap is None or snap.preview_rgb is None:
            return
        img = snap.preview_rgb
        h, w = img.shape[:2]
        surf = self.pg.image.frombuffer(img.tobytes(), (w, h), "RGB")
        rect = surf.get_rect(bottomright=(self.w - 16, self.h - 40))
        self.screen.blit(surf, rect)
        self.pg.draw.rect(self.screen, (230, 230, 235), rect, 2)

    def _shade(self, alpha=150):
        shade = self.pg.Surface((self.w, self.h), self.pg.SRCALPHA)
        shade.fill((0, 0, 0, alpha))
        self.screen.blit(shade, (0, 0))

    def _button(self, key, rect, label, color):
        pg = self.pg
        pg.draw.rect(self.screen, color, rect, border_radius=12)
        pg.draw.rect(self.screen, (230, 230, 235), rect, 2, border_radius=12)
        self._text(label, self.f_med, (255, 255, 255), rect.center, "center")
        self.buttons[key] = rect

    def _draw_menu(self, game):
        pg = self.pg
        self._shade(110)
        cx = self.w // 2
        self._text("Spooky Minigolf", self.f_huge, ORANGE, (cx, 210), "center", shadow=True)
        self._text(f"{config.HOLES_PER_GAME} haunted holes, new every game.  Choose 1 / 2, then Start (Enter)",
                   self.f_small, (220, 210, 230), (cx, 262), "center")
        names = list(config.DIFFICULTIES)
        bw, gap = 200, 24
        x0 = cx - (len(names) * bw + (len(names) - 1) * gap) // 2
        for i, name in enumerate(names):
            r = pg.Rect(x0 + i * (bw + gap), 300, bw, 58)
            col = (150, 70, 200) if name == game.difficulty else (60, 54, 76)
            self._button(f"difficulty:{name}", r, f"{i + 1}  {name}", col)
        self._button("start", pg.Rect(cx - 110, 386, 220, 60), "START", (220, 110, 20))

    def _draw_game_over(self, game):
        pg = self.pg
        self._shade(140)
        cx = self.w // 2
        total, par = game.total_strokes, sum(h.par for h in game.course)
        d = total - par
        rel = "Even par" if d == 0 else f"{d:+d}"
        self._text("Final Score", self.f_med, (220, 220, 230), (cx, 200), "center")
        self._text(f"{total} strokes", self.f_huge, GOLD, (cx, 262), "center", shadow=True)
        self._text(f"{rel}  (par {par})      Best: {game.best}", self.f_small, (220, 220, 230), (cx, 318), "center")
        self._button("start", pg.Rect(cx - 230, 360, 220, 60), "Play again", (220, 110, 20))
        self._button("menu", pg.Rect(cx + 10, 360, 220, 60), "Menu", (60, 54, 76))

    def _draw_debug(self, lines):
        pg = self.pg
        h = 20 * len(lines) + 12
        panel = pg.Surface((500, h), pg.SRCALPHA)
        panel.fill((0, 0, 0, 170))
        self.screen.blit(panel, (16, 300))
        for i, line in enumerate(lines):
            self._text(line, self.f_mono, (170, 255, 170), (24, 306 + 20 * i))

    # -- frame --------------------------------------------------------------------------
    def draw(self, now, game, statuses, snapshot=None, debug_lines=None, hint="", overview=False, live_strength=0.0):
        self._update_camera(now, game, overview)
        hole = game.hole if game.hole is not None and game.state not in ("menu",) else self._demo
        self._draw_sky(now)
        self._draw_turf(hole)
        if hole is game.hole:
            self._draw_floor_decals(now, game, hole)
        self._draw_scene(now, game if hole is game.hole else _NoGame(game), hole)

        self._draw_scorecard(game)
        if game.state not in ("menu", "game_over"):
            self._draw_info(game, now)
        y = self._draw_statuses(statuses)
        self._draw_minimap(game, now, self.w - 216, y + 8)
        self._draw_power(game, now, live_strength)
        if config.SHOW_CAMERA_PREVIEW:
            self._draw_preview(snapshot)
        if game.message and now < game.message_until and game.state not in ("menu", "game_over"):
            big = game.message.startswith(("HOLE IN ONE", "Birdie", "Eagle", "Albatross", "Condor"))
            col = GOLD if big else (255, 255, 255)
            self._text(game.message, self.f_huge if big else self.f_big, col, (self.w // 2, 200), "center",
                       shadow=True)
            if game.state == "intro" and game.hole and game.hole.obstacles:
                kinds = sorted({o.kind for o in game.hole.obstacles})
                self._text("Watch out: " + ", ".join(kinds), self.f_med, ORANGE, (self.w // 2, 250), "center",
                           shadow=True)
        elif game.state == "aiming":
            self._text("Swing to putt!", self.f_med, (255, 255, 255), (self.w // 2, 160), "center", shadow=True)
        if debug_lines:
            self._draw_debug(debug_lines)
        self.buttons = {}
        if game.state == "menu":
            self._draw_menu(game)
        elif game.state == "game_over":
            self._draw_game_over(game)
        self._text(hint, self.f_small, (170, 170, 185), (16, self.h - 30))


class _NoGame:
    """Stand-in so the menu's demo hole draws without a ball or golfer."""

    def __init__(self, game):
        self.state = "menu"
        self.golfer = game.golfer
        self.sim = None
        self.hole = None
        self.ball = (0.0, 0.0)
        self.last_putt = None
