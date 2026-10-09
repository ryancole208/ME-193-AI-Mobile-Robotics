"""
Drawing (pygame): a Wii Sports-style view from behind the bowler. While aiming the
camera sits behind the foul line; once the ball is released it follows the ball
down the lane and stops in front of the pins. A scorecard runs across the top and a
pin diagram shows which pins are standing.
"""

import math

import config
from game import status_color          # ../game.py (ping pong), reused for the status lights
from lane import LANE_HALF, PIN_POS

# Pin silhouette: (fraction of height, radius in m) from base to top
PIN_PROFILE = [(0.02, 0.030), (0.12, 0.048), (0.24, 0.058), (0.34, 0.0605), (0.44, 0.055), (0.53, 0.040),
               (0.61, 0.026), (0.68, 0.024), (0.76, 0.030), (0.84, 0.032), (0.91, 0.028), (0.96, 0.018)]
STRIPE_FRACTIONS = (0.61, 0.68)
BOARD_W = config.LANE_WIDTH_M / 39

WOOD, WOOD_DARK, APPROACH, DECK = (214, 170, 112), (196, 150, 94), (226, 188, 136), (232, 204, 158)
GUTTER, CAPPING, PIT = (92, 96, 108), (58, 52, 60), (14, 14, 20)
BALL_COLOR, BALL_HI = (44, 92, 210), (150, 190, 255)
GOLD = (255, 214, 80)


class Camera:
    def __init__(self, width):
        self.w = width
        self.y = config.VIEW_CAMERA_AIM_Y_M
        self.h = config.VIEW_CAMERA_HEIGHT_M

    def project(self, x, y, z=0.0):
        """(screen x, screen y, px per metre), or None if behind the near plane."""
        d = y - self.y
        if d < config.VIEW_NEAR_M - 1e-6:
            return None
        s = config.VIEW_FOCAL_PX / d
        return self.w / 2 + x * s, config.VIEW_HORIZON_Y_PX + (self.h - z) * s, s


class Renderer:
    BG_TOP, BG_BOTTOM = (16, 18, 34), (40, 34, 52)

    def __init__(self, screen):
        import pygame
        self.pg = pygame
        self.screen = screen
        self.w, self.h = screen.get_size()
        self.cam = Camera(self.w)
        font = "segoeui,arial,helvetica"
        self.f_huge = pygame.font.SysFont(font, 76, bold=True)
        self.f_big = pygame.font.SysFont(font, 44, bold=True)
        self.f_med = pygame.font.SysFont(font, 26, bold=True)
        self.f_small = pygame.font.SysFont(font, 18)
        self.f_card = pygame.font.SysFont(font, 17, bold=True)
        self.f_mono = pygame.font.SysFont("consolas,couriernew,monospace", 16)
        self.buttons = {}
        self._bg = self._make_background()
        self._last_now = None
        self._last_state = None

    def _make_background(self):
        surf = self.pg.Surface((self.w, self.h))
        for row in range(self.h):
            k = row / self.h
            c = [int(a + (b - a) * k) for a, b in zip(self.BG_TOP, self.BG_BOTTOM)]
            self.pg.draw.line(surf, c, (0, row), (self.w, row))
        return surf

    def _text(self, text, font, color, pos, anchor="topleft", shadow=False):
        if shadow:
            s = font.render(text, True, (0, 0, 0))
            r = s.get_rect(**{anchor: pos})
            self.screen.blit(s, r.move(3, 3))
        surf = font.render(text, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    # -- camera -------------------------------------------------------------------
    def _update_camera(self, now, game):
        dt = 0.0 if self._last_now is None else max(0.0, min(0.1, now - self._last_now))
        self._last_now = now
        if game.state in ("rolling", "result") and game.roll is not None:
            target = min(game.roll.ball.y - config.VIEW_FOLLOW_BACK_M,
                         config.HEAD_PIN_Y_M - config.VIEW_PIN_STOP_BACK_M)
            target = max(config.VIEW_CAMERA_AIM_Y_M, target)
            k = min(1.0, config.VIEW_FOLLOW_SMOOTHING * dt)
            self.cam.y += (target - self.cam.y) * k
        else:
            self.cam.y = config.VIEW_CAMERA_AIM_Y_M  # cut back to the bowler, like the Wii
        self._last_state = game.state

    # -- ground ---------------------------------------------------------------------
    def _quad(self, x0, x1, y0, y1, color, z=0.0):
        y0 = max(y0, self.cam.y + config.VIEW_NEAR_M)
        if y1 <= y0:
            return
        pts = [self.cam.project(x, y, z) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        self.pg.draw.polygon(self.screen, color, [(p[0], p[1]) for p in pts])

    def _line(self, a, b, color, width=1):
        (x0, y0, z0), (x1, y1, z1) = a, b
        near = self.cam.y + config.VIEW_NEAR_M
        if max(y0, y1) <= near:
            return
        if y0 < near:
            f = (near - y0) / (y1 - y0)
            x0, y0, z0 = x0 + (x1 - x0) * f, near, z0 + (z1 - z0) * f
        p, q = self.cam.project(x0, y0, z0), self.cam.project(x1, y1, z1)
        self.pg.draw.line(self.screen, color, p[:2], q[:2], width)

    def _draw_alley(self, bumpers):
        pg = self.pg
        far = config.PIT_Y_M + 0.7
        # dark pit opening, with the masking unit hanging above it
        tl, mid, br = self.cam.project(-8, far, 2.6), self.cam.project(8, far, 0.62), self.cam.project(8, far, 0)
        if tl and br:
            pg.draw.rect(self.screen, PIT, (tl[0], mid[1], br[0] - tl[0], br[1] - mid[1]))
            pg.draw.rect(self.screen, (24, 30, 64), (tl[0], tl[1], br[0] - tl[0], mid[1] - tl[1]))
            band = self.cam.project(-8, far, 0.95)
            pg.draw.rect(self.screen, (60, 120, 220), (tl[0], band[1], br[0] - tl[0], max(2, band[2] * 0.08)))
            pg.draw.line(self.screen, (120, 170, 255), (tl[0], mid[1]), (br[0], mid[1]), 2)
        pitch = config.LANE_WIDTH_M + 2 * config.GUTTER_WIDTH_M + 0.5
        for k in (-2, -1, 1, 2):  # neighbouring lanes
            cx = k * pitch
            self._quad(cx - 3, cx + 3, -config.APPROACH_LENGTH_M, far, CAPPING)
            self._quad(cx - LANE_HALF, cx + LANE_HALF, 0, config.PIT_Y_M, WOOD_DARK)
            self._quad(cx - LANE_HALF, cx + LANE_HALF, -config.APPROACH_LENGTH_M, 0, (190, 160, 118))
            self._quad(cx - LANE_HALF, cx + LANE_HALF, config.PIT_Y_M, far, PIT)
        # this lane
        g = LANE_HALF + config.GUTTER_WIDTH_M
        self._quad(-g - 0.25, g + 0.25, -config.APPROACH_LENGTH_M, far, CAPPING)
        self._quad(-g, g, config.PIT_Y_M, far, PIT)
        self._quad(-g, -LANE_HALF, 0, config.PIT_Y_M, GUTTER)
        self._quad(LANE_HALF, g, 0, config.PIT_Y_M, GUTTER)
        self._quad(-LANE_HALF, LANE_HALF, -config.APPROACH_LENGTH_M, 0, APPROACH)
        self._quad(-LANE_HALF, LANE_HALF, 0, config.PIT_Y_M, WOOD)
        self._quad(-LANE_HALF, LANE_HALF, config.HEAD_PIN_Y_M - 0.45, config.PIT_Y_M, DECK)
        for b in range(5, 39, 5):
            x = -LANE_HALF + b * BOARD_W
            self._line((x, 0, 0), (x, config.HEAD_PIN_Y_M - 0.45, 0), (200, 154, 98))
        self._quad(-LANE_HALF, LANE_HALF, 0, 0.03, (150, 30, 30))       # foul line
        for k in range(-3, 4):                                           # target arrows
            x, y = k * 5 * BOARD_W, 4.88 - abs(k) * 0.3
            tip, l, r = (x, y + 0.3, 0), (x - 0.035, y, 0), (x + 0.035, y, 0)
            pts = [self.cam.project(*p) for p in (tip, l, r)]
            if all(pts):
                pg.draw.polygon(self.screen, (120, 70, 40), [p[:2] for p in pts])
        for b in (3, 5, 8, 11, 14):                                      # dots
            for side in (-1, 1):
                p = self.cam.project(side * (LANE_HALF - (b - 0.5) * BOARD_W), 2.13, 0)
                if p:
                    pg.draw.circle(self.screen, (120, 70, 40), p[:2], max(1, 0.012 * p[2]))
        if bumpers:
            for side in (-1, 1):
                x = side * (LANE_HALF + 0.03)
                y0 = max(0.0, self.cam.y + config.VIEW_NEAR_M)
                y1 = config.HEAD_PIN_Y_M - 0.3
                if y1 > y0:
                    pts = [self.cam.project(x, y0, 0), self.cam.project(x, y1, 0),
                           self.cam.project(x, y1, 0.13), self.cam.project(x, y0, 0.13)]
                    pg.draw.polygon(self.screen, (70, 150, 240), [p[:2] for p in pts])
                    pg.draw.line(self.screen, (200, 230, 255), pts[3][:2], pts[2][:2], 2)

    # -- objects ----------------------------------------------------------------------
    def _draw_pin(self, x, y, tilt=0.0, heading=0.0):
        pg = self.pg
        phi = tilt * math.pi / 2
        ax, ay, az = math.sin(phi) * math.sin(heading), math.sin(phi) * math.cos(heading), math.cos(phi)
        H = config.PIN_HEIGHT_M
        circles = []
        for f, r in PIN_PROFILE:
            p = self.cam.project(x + ax * f * H, y + ay * f * H, az * f * H + r * math.sin(phi))
            if p is None:
                return
            circles.append((f, p, r * p[2]))
        for _, p, r in circles:
            pg.draw.circle(self.screen, (150, 150, 158), p[:2], r + 1.5)
        for f, p, r in circles:
            col = (196, 30, 40) if any(abs(f - s) < 0.02 for s in STRIPE_FRACTIONS) else (250, 250, 252)
            pg.draw.circle(self.screen, col, p[:2], r)

    def _draw_ball(self, x, y, z, distance=0.0):
        pg = self.pg
        r_m = config.BALL_RADIUS_M
        sh = self.cam.project(x, y, max(0.0, z - r_m))
        p = self.cam.project(x, y, z)
        if p is None or sh is None:
            return
        r = max(2, r_m * p[2])
        pg.draw.ellipse(self.screen, (40, 30, 30), (sh[0] - r, sh[1] - r * 0.3, 2 * r, 0.6 * r))
        pg.draw.circle(self.screen, BALL_COLOR, p[:2], r)
        theta = distance / r_m
        for k, dx in enumerate((-0.22, 0.22, 0.0)):    # finger holes rolling over the top
            a = theta + (0.0 if k < 2 else 0.55)
            if math.sin(a) > 0.2:
                hy = -math.cos(a) * 0.62
                pg.draw.circle(self.screen, (14, 24, 60), (p[0] + dx * r, p[1] + hy * r), max(1, 0.12 * r))
        pg.draw.circle(self.screen, BALL_HI, (p[0] - 0.35 * r, p[1] - 0.4 * r), max(1, 0.22 * r))

    def _draw_bowler(self, x, holding):
        """A chunky Mii-like figure seen from behind, standing behind the foul line."""
        pg = self.pg
        y = -0.45
        hand = 1 if config.DOMINANT_HAND == "right" else -1
        parts = [  # (dx, z, radius m, colour)
            (-0.09, 0.12, 0.07, (40, 46, 70)), (0.09, 0.12, 0.07, (40, 46, 70)),
            (-0.09, 0.32, 0.075, (40, 46, 70)), (0.09, 0.32, 0.075, (40, 46, 70)),
            (0.0, 0.62, 0.21, (210, 60, 60)), (0.0, 0.76, 0.19, (210, 60, 60)),
            (-hand * 0.24, 0.66, 0.06, (210, 60, 60)), (-hand * 0.27, 0.5, 0.055, (240, 200, 165)),
            (hand * 0.25, 0.66, 0.06, (210, 60, 60)), (hand * 0.28, 0.5, 0.055, (240, 200, 165)),
            (0.0, 1.03, 0.17, (60, 40, 30)),
        ]
        for dx, z, r, col in parts:
            p = self.cam.project(x + dx, y, z)
            if p is None:
                return
            pg.draw.circle(self.screen, col, p[:2], r * p[2])
        if holding:
            self._draw_ball(x + hand * 0.3, y + 0.05, 0.38)

    def _draw_aim_guide(self, x, angle_deg):
        a = math.radians(angle_deg)
        for i in range(1, 9):
            d0, d1 = i * 0.9, i * 0.9 + 0.45
            p0 = (x + math.sin(a) * d0, math.cos(a) * d0, 0.005)
            p1 = (x + math.sin(a) * d1, math.cos(a) * d1, 0.005)
            self._line(p0, p1, (255, 240, 120), 3)

    def _draw_scene(self, game):
        roll = game.roll if game.state in ("rolling", "result") else None
        items = []  # (y, draw fn)
        if roll is not None:
            t = roll.t
            for p in roll.pins:
                if not p.gone:
                    items.append((p.y, lambda p=p: self._draw_pin(p.x, p.y, p.tilt(t), p.heading)))
            b = roll.ball
            if not b.done:
                z = config.BALL_RADIUS_M - (0.06 if b.in_gutter else 0.0)
                items.append((b.y, lambda b=b, z=z: self._draw_ball(b.x, b.y, z, b.distance)))
        else:
            for n in game.standing:
                px, py = PIN_POS[n]
                items.append((py, lambda px=px, py=py: self._draw_pin(px, py)))
        bx = game.bowler.current_position() if game.state == "aiming" else (
            game.last_throw["x"] if game.last_throw else 0.0)
        items.append((-0.45, lambda: self._draw_bowler(bx, game.state in ("aiming", "menu", "game_over"))))
        for _, fn in sorted(items, key=lambda it: -it[0]):
            fn()

    # -- HUD --------------------------------------------------------------------------
    def _draw_scorecard(self, game):
        pg = self.pg
        sheet = game.sheet
        fw, tw, totw, h = 74, 104, 92, 66
        x0 = (self.w - (9 * fw + tw + totw)) // 2
        y0 = 10
        cur = sheet.current_frame
        frame_scores = sheet.frame_scores()
        pg.draw.rect(self.screen, (250, 250, 252), (x0, y0, 9 * fw + tw + totw, h), border_radius=6)
        for i in range(10):
            w = tw if i == 9 else fw
            x = x0 + i * fw
            rect = pg.Rect(x, y0, w, h)
            pg.draw.rect(self.screen, (40, 44, 60), rect, 2)
            self._text(str(i + 1), self.f_small, (110, 110, 125), (x + 5, y0 + 2))
            marks = sheet.marks(i)
            nbox = 3 if i == 9 else 2
            bw = 24
            for j in range(nbox):
                br = pg.Rect(x + w - (nbox - j) * bw, y0, bw, 26)
                pg.draw.rect(self.screen, (40, 44, 60), br, 1)
                if j < len(marks) and marks[j]:
                    col = (200, 40, 40) if marks[j] in ("X", "/") else (30, 30, 40)
                    self._text(marks[j], self.f_card, col, br.center, "center")
            if frame_scores[i] is not None:
                self._text(str(frame_scores[i]), self.f_med, (20, 20, 30), (x + w // 2, y0 + 46), "center")
            if i == cur and game.state not in ("menu", "game_over"):
                pg.draw.rect(self.screen, GOLD, rect.inflate(2, 2), 4, border_radius=3)
        tx = x0 + 9 * fw + tw
        pg.draw.rect(self.screen, (40, 44, 60), (tx, y0, totw, h), 2)
        self._text("TOTAL", self.f_small, (110, 110, 125), (tx + totw // 2, y0 + 14), "center")
        self._text(str(sheet.running_total()), self.f_med, (20, 20, 30), (tx + totw // 2, y0 + 44), "center")
        self._text(config.PLAYER_NAME, self.f_card, (230, 230, 240), (x0 - 10, y0 + h // 2), "midright")

    def _draw_pin_diagram(self, game, x, y):
        pg = self.pg
        standing = game.roll.standing if game.state in ("rolling", "result") and game.roll else game.standing
        panel = pg.Surface((120, 104), pg.SRCALPHA)
        panel.fill((0, 0, 0, 120))
        self.screen.blit(panel, (x, y))
        for n, (px, py) in PIN_POS.items():
            cx = x + 60 + px / config.PIN_SPACING_M * 26
            cy = y + 88 - (py - config.HEAD_PIN_Y_M) / config.PIN_ROW_SPACING_M * 24
            up = n in standing
            pg.draw.circle(self.screen, (250, 250, 252) if up else (70, 70, 84), (cx, cy), 9)
            if up:
                pg.draw.circle(self.screen, (196, 30, 40), (cx, cy), 3)

    def _draw_info(self, game, bowler_source):
        pg = self.pg
        lines = [f"{game.difficulty}", f"Frame {game.frame_number or 10}   input: {bowler_source}",
                 f"Aim {game.bowler.preset_aim_deg:+.1f}°"]
        if bowler_source == "keyboard":
            lines.append(f"Power {game.bowler.kb_strength:.2f}  (↑/↓)")
        t = game.last_throw
        if t:
            hook = "straight" if abs(t["spin"]) < 0.05 else ("hook ←" if t["spin"] < 0 else "hook →")
            lines.append(f"Last: {t['speed'] * 3.6:.0f} km/h  {hook}")
        panel = pg.Surface((250, 22 * len(lines) + 12), pg.SRCALPHA)
        panel.fill((0, 0, 0, 120))
        self.screen.blit(panel, (16, 88))
        for i, line in enumerate(lines):
            self._text(line, self.f_small, (230, 230, 240), (26, 94 + 22 * i))

    def _draw_statuses(self, statuses):
        x, y = self.w - 16, 88
        for label, kind, status in statuses:
            surf = self.f_small.render(f"{label}: {status}", True, (230, 230, 235))
            r = surf.get_rect(topright=(x, y))
            self.screen.blit(surf, r)
            self.pg.draw.circle(self.screen, status_color(kind, status), (r.left - 12, r.centery), 7)
            y += 24
        return y

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
        self._shade()
        cx = self.w // 2
        self._text("Bowling", self.f_huge, (255, 255, 255), (cx, 200), "center", shadow=True)
        self._text("Choose 1 / 2, then Start (Enter)", self.f_small, (210, 210, 220), (cx, 252), "center")
        names = list(config.DIFFICULTIES)
        bw, gap = 220, 24
        x0 = cx - (len(names) * bw + (len(names) - 1) * gap) // 2
        for i, name in enumerate(names):
            r = pg.Rect(x0 + i * (bw + gap), 290, bw, 58)
            col = (70, 140, 230) if name == game.difficulty else (60, 64, 80)
            self._button(f"difficulty:{name}", r, f"{i + 1}  {name}", col)
        self._button("start", pg.Rect(cx - 110, 376, 220, 60), "START", (60, 170, 100))

    def _draw_game_over(self, game):
        pg = self.pg
        self._shade(140)
        cx = self.w // 2
        total = game.sheet.running_total()
        self._text("Final Score", self.f_med, (220, 220, 230), (cx, 200), "center")
        self._text(str(total), self.f_huge, GOLD, (cx, 260), "center", shadow=True)
        self._text(f"Best: {game.best}" + ("   PERFECT GAME!" if total == 300 else ""),
                   self.f_small, (220, 220, 230), (cx, 314), "center")
        self._button("start", pg.Rect(cx - 230, 360, 220, 60), "Play again", (60, 170, 100))
        self._button("menu", pg.Rect(cx + 10, 360, 220, 60), "Menu", (60, 64, 80))

    def _draw_debug(self, lines):
        pg = self.pg
        h = 20 * len(lines) + 12
        panel = pg.Surface((470, h), pg.SRCALPHA)
        panel.fill((0, 0, 0, 170))
        self.screen.blit(panel, (16, 240))
        for i, line in enumerate(lines):
            self._text(line, self.f_mono, (170, 255, 170), (24, 246 + 20 * i))

    # -- frame --------------------------------------------------------------------------
    def draw(self, now, game, statuses, snapshot=None, debug_lines=None, hint=""):
        self._update_camera(now, game)
        self.screen.blit(self._bg, (0, 0))
        self._draw_alley(game.bumpers)
        if game.state == "aiming":
            self._draw_aim_guide(game.bowler.current_position(), game.bowler.preset_aim_deg)
        self._draw_scene(game)

        self._draw_scorecard(game)
        self._draw_info(game, game.bowler.source)
        y = self._draw_statuses(statuses)
        self._draw_pin_diagram(game, self.w - 136, y + 8)
        if config.SHOW_CAMERA_PREVIEW:
            self._draw_preview(snapshot)
        if game.message and now < game.message_until and game.state not in ("menu", "game_over"):
            big = game.message.startswith(("STRIKE", "SPARE"))
            self._text(game.message, self.f_huge if big else self.f_big, GOLD if big else (255, 255, 255),
                       (self.w // 2, 190), "center", shadow=True)
        elif game.state == "aiming":
            self._text("Swing to bowl!", self.f_med, (255, 255, 255), (self.w // 2, 160), "center", shadow=True)
        if debug_lines:
            self._draw_debug(debug_lines)
        self.buttons = {}
        if game.state == "menu":
            self._draw_menu(game)
        elif game.state == "game_over":
            self._draw_game_over(game)
        self._text(hint, self.f_small, (170, 170, 185), (16, self.h - 30))
