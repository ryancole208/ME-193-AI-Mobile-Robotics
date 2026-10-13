"""
Pseudo-3D renderer (pygame-ce): perspective camera, pre-rendered scene layers, shaded ball
with a shadow, spin marking and trail, 3D paddles, the skeleton opponent and the themed HUD.

Draw order each frame (back to front):
    backdrop (sky, scenery, floor)  ->  ambient animation  ->  ball shadow on the floor
    ->  skeleton + its paddle  ->  table  ->  ball shadow on the table  ->  far-side ball
    ->  net  ->  near-side ball  ->  my paddle  ->  HUD / Dynamic panel / debug / menu
    ->  controls bar (wraps to the window width; H toggles it)
"""

import math
from dataclasses import dataclass

import config
from camera3d import Camera
from game import status_color
from paddle import player_paddle, swing_pose
from scene import Scene
from skeleton import Skeleton
from spin import NO_SPIN
from theme import get_theme, mix


@dataclass
class PlayerView:
    """What the renderer needs to draw my paddle."""
    x: float = 0.0              # lateral position (m)
    roll_deg: float = 0.0       # handle twist (0 = neutral grip, + = screen-side face turned right)
    swing_t: float | None = None  # time the last swing was registered (animation)
    direction: str = "center"


class Renderer:
    def __init__(self, screen, theme=None):
        import pygame
        self.pg = pygame
        self.screen = screen
        self.w, self.h = screen.get_size()
        self.theme = theme or get_theme()
        self.c = self.theme.c
        self.cam = Camera(self.w, self.h)
        self.scene = Scene(pygame, self.cam, self.theme)
        self.ambient = self.theme.make_ambient(pygame, self.cam)
        self.skeleton = Skeleton()
        self.paddle = player_paddle()
        t = self.theme
        self.f_huge = t.font(64, title=True, bold=True)
        self.f_big = t.font(44, title=True, bold=True)
        self.f_med = t.font(26, bold=True)
        self.f_small = t.font(17)
        self.f_mono = pygame.font.SysFont("consolas,couriernew,monospace", 16)
        self.buttons = {}
        self._text_cache = {}
        self._sprite_cache = {}
        self._icon = t.icon(pygame, 48)
        self._last_label = None

    # -- small helpers ---------------------------------------------------------------
    def _text_surf(self, text, font, color, outline=None):
        key = (text, id(font), color, outline)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 400:
                self._text_cache.clear()
            pg = self.pg
            main = font.render(text, True, color)
            if outline:
                w, h = main.get_size()
                surf = pg.Surface((w + 4, h + 4), pg.SRCALPHA)
                edge = font.render(text, True, outline)
                for dx in (-2, 0, 2):
                    for dy in (-2, 0, 2):
                        if dx or dy:
                            surf.blit(edge, (2 + dx, 2 + dy))
                surf.blit(main, (2, 2))
            else:
                surf = main
            self._text_cache[key] = surf
        return surf

    def _text(self, text, font, color, pos, anchor="topleft", outline=None):
        surf = self._text_surf(text, font, color, outline)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def _panel(self, rect, alpha=200, border=None, radius=14):
        pg = self.pg
        key = ("panel", rect.size, alpha, border, radius)
        surf = self._sprite_cache.get(key)
        if surf is None:
            surf = pg.Surface(rect.size, pg.SRCALPHA)
            pg.draw.rect(surf, (*self.c["panel"], alpha), surf.get_rect(), border_radius=radius)
            if border:
                pg.draw.rect(surf, (*border, 255), surf.get_rect(), 2, border_radius=radius)
            self._sprite_cache[key] = surf
        self.screen.blit(surf, rect)

    def _ball_sprite(self, r):
        r = max(2, int(round(r)))
        key = ("ball", r)
        surf = self._sprite_cache.get(key)
        if surf is None:
            pg = self.pg
            surf = pg.Surface((2 * r + 2, 2 * r + 2), pg.SRCALPHA)
            base = config.BALL_COLOR
            dark = mix(base, config.BALL_SHADE_COLOR, 0.55)
            light = mix(base, (255, 255, 255), 0.75)
            steps = max(3, r)
            for i in range(steps, 0, -1):
                k = i / steps
                col = mix(light, base, min(1.0, k * 1.3)) if k < 0.75 else mix(base, dark, (k - 0.75) * 3)
                off = (1 - k) * r * 0.35
                pg.draw.circle(surf, col, (r + 1 - off, r + 1 - off), k * r)
            pg.draw.circle(surf, config.BALL_OUTLINE_COLOR, (r + 1, r + 1), r, 1 if r < 7 else 2)
            self._sprite_cache[key] = surf
        return surf

    def _blob(self, w, h, color, alpha):
        key = ("blob", int(w), int(h), color, int(alpha) // 8)
        surf = self._sprite_cache.get(key)
        if surf is None:
            if len(self._sprite_cache) > 600:
                self._sprite_cache.clear()
            pg = self.pg
            surf = pg.Surface((max(2, int(w)), max(2, int(h))), pg.SRCALPHA)
            pg.draw.ellipse(surf, (*color, int(alpha)), surf.get_rect())
            self._sprite_cache[key] = surf
        return surf

    # -- ball --------------------------------------------------------------------------
    def _over_table(self, x, y):
        return abs(x) <= config.TABLE_WIDTH_M / 2 and 0 <= y <= config.TABLE_LENGTH_M

    def _draw_shadow(self, pos, on_table):
        x, y, z = pos
        surface_z = 0.0 if on_table else -config.TABLE_HEIGHT_M
        p = self.cam.project(x, y, surface_z)
        if p is None:
            return
        sx, sy, s = p
        height = max(0.0, z - surface_z)
        r = config.BALL_RADIUS_M * config.BALL_VISUAL_SCALE * (1 + 1.2 * height)
        q = self.cam.point(x, y + r, surface_z)
        depth_px = abs(sy - q[1]) if q else r * s * 0.4
        alpha = max(45, min(170, 170 - 220 * height))
        blob = self._blob(2 * r * s, max(2, 2 * depth_px), config.SHADOW_COLOR, alpha)
        self.screen.blit(blob, blob.get_rect(center=(sx, sy)))

    def _draw_ball(self, now, logic, pos, spin):
        x, y, z = pos
        rad = config.BALL_RADIUS_M * config.BALL_VISUAL_SCALE
        if config.TRAIL_ENABLED:
            col = config.TRAIL_COLORS.get(spin.kind, config.TRAIL_COLORS["none"])
            n = config.TRAIL_DOTS
            for i in range(n, 0, -1):
                tp = logic.ball_position(now - config.TRAIL_LENGTH_S * i / n)
                if tp is None:
                    continue
                p = self.cam.project(tp[0], tp[1], tp[2] + rad)
                if p is None:
                    continue
                k = 1 - i / (n + 1)
                r = rad * p[2] * (0.35 + 0.6 * k)
                if config.TRAIL_HALO_ALPHA:   # dark halo so the trail reads on the orange table
                    halo = self._blob(2 * r + 4, 2 * r + 4, config.BALL_OUTLINE_COLOR, config.TRAIL_HALO_ALPHA * k)
                    self.screen.blit(halo, halo.get_rect(center=(p[0], p[1])))
                dot = self._blob(2 * r, 2 * r, col, 150 * k)
                self.screen.blit(dot, dot.get_rect(center=(p[0], p[1])))
        p = self.cam.project(x, y, z + rad)
        if p is None:
            return
        sx, sy, s = p
        r = max(2.5, rad * s)
        spr = self._ball_sprite(r)
        self.screen.blit(spr, spr.get_rect(center=(sx, sy)))
        if r >= 4:
            self._draw_spin_marking(now, logic, sx, sy, r, spin)

    def _draw_spin_marking(self, now, logic, sx, sy, r, spin):
        """A seam (a meridian circle) rotating about the spin axis."""
        mag = spin.magnitude
        if mag > 1e-3:
            ax, ay = spin.top / mag, spin.side / mag     # screen-plane axis: topspin = horizontal
            t0 = logic.flight.t0 if logic.flight else 0.0
            theta = 2 * math.pi * config.BALL_SPIN_VIS_RPS * mag * (now - t0)
        else:
            ax, ay, theta = 0.0, 1.0, 0.6
        axis = (ax, ay, 0.0)
        view = (0.0, 0.0, 1.0)
        uvec = (axis[1] * view[2] - axis[2] * view[1], axis[2] * view[0] - axis[0] * view[2], 0.0)
        ct, st = math.cos(theta), math.sin(theta)
        m = (uvec[0] * ct + view[0] * st, uvec[1] * ct + view[1] * st, view[2] * st)
        pts = []
        for i in range(17):
            phi = math.pi * i / 16
            cp, sp = math.cos(phi), math.sin(phi)
            px, py, pz = (cp * axis[0] + sp * m[0], cp * axis[1] + sp * m[1], cp * axis[2] + sp * m[2])
            if pz >= -0.05:
                pts.append((sx + px * r * 0.92, sy - py * r * 0.92))
            elif len(pts) >= 2:
                break
        if len(pts) >= 2:
            self.pg.draw.lines(self.screen, config.BALL_SEAM_COLOR, False, pts, max(1, int(r / 5)))

    # -- my paddle ----------------------------------------------------------------------
    def _draw_player_paddle(self, now, view):
        dx = dy = dz = tilt = 0.0
        if view.swing_t is not None:
            phase = (now - view.swing_t) / config.SWING_ANIM_S
            if 0 <= phase <= 1:
                for ghost in (0.3, 0.15):
                    gp = phase - ghost
                    if gp > 0:
                        gx, gy, gz, gt = swing_pose(gp, view.direction)
                        self.paddle.draw(self.screen, self.cam,
                                         (view.x + gx, config.PLAYER_HIT_Y_M + gy, config.PADDLE_HEIGHT_M + gz),
                                         view.roll_deg, tilt_deg=gt, lean_deg=self._lean(),
                                         alpha=int(70 * (1 - ghost * 2)))
                dx, dy, dz, tilt = swing_pose(phase, view.direction)
        center = (view.x + dx, config.PLAYER_HIT_Y_M + dy, config.PADDLE_HEIGHT_M + dz)
        self.paddle.draw(self.screen, self.cam, center, view.roll_deg, tilt_deg=tilt, lean_deg=self._lean(),
                         alpha=config.PLAYER_PADDLE_ALPHA)

    @staticmethod
    def _lean():
        return -12.0 if config.DOMINANT_HAND == "right" else 12.0

    # -- HUD ------------------------------------------------------------------------------
    def _draw_hud(self, logic, statuses):
        pg, c = self.pg, self.c
        rect = pg.Rect(16, 16, 230, 152)
        self._panel(rect, 205, c["panel_border"])
        self._text("STREAK", self.f_small, c["dim"], (32, 22))
        self._text(str(logic.streak), self.f_huge, c["accent"], (30, 34), outline=(20, 8, 30))
        self._text(f"Best: {logic.best}", self.f_small, c["text"], (32, 112))
        self._text(f"Difficulty: {logic.difficulty}", self.f_small, c["accent2"], (32, 136))

        rows = len(statuses)
        srect = pg.Rect(0, 0, 300, 14 + 26 * rows)
        srect.topright = (self.w - 16, 16)
        self._panel(srect, 190, c["panel_border"])
        y = srect.top + 8
        for label, kind, status in statuses:
            col = self._status_col(kind, status)
            txt = f"{label}: {status}"
            while len(txt) > 4 and self.f_small.size(txt)[0] > srect.w - 46:
                txt = txt[:-2] + "…"
            r = self._text(txt, self.f_small, c["text"], (srect.left + 34, y))
            pg.draw.circle(self.screen, (20, 8, 30), (srect.left + 20, r.centery), 8)
            pg.draw.circle(self.screen, col, (srect.left + 20, r.centery), 6)
            y += 26

    def _status_col(self, kind, status):
        mapping = {(70, 200, 110): "good", (235, 190, 60): "warn", (225, 80, 70): "bad", (130, 130, 140): "off"}
        return self.c.get(mapping.get(status_color(kind, status), "off"), (130, 130, 140))

    def _draw_preview(self, snap, bottom):
        if snap is None or snap.preview_rgb is None:
            return
        img = snap.preview_rgb
        h, w = img.shape[:2]
        surf = self.pg.image.frombuffer(img.tobytes(), (w, h), "RGB")
        rect = surf.get_rect(bottomright=(self.w - 16, bottom - 12))
        frame = rect.inflate(10, 10)
        self._panel(frame, 220, self.c["panel_border"], radius=8)
        self.screen.blit(surf, rect)

    def _draw_messages(self, now, logic):
        c = self.c
        if logic.message and now < logic.message_until and logic.state != "menu":
            col = c["bad"] if logic.message.startswith("MISS") else c["text"]
            if logic.message.startswith("Opponent"):
                col = c["accent2"]
            self._text(logic.message, self.f_med, col, (self.w // 2, 96), "center", outline=(20, 8, 30))
        if logic.spin_text and now < logic.spin_text_until and logic.state != "menu":
            age = config.SPIN_LABEL_S - (logic.spin_text_until - now)
            pop = 1.0 + 0.35 * max(0.0, 1 - age / 0.18)
            surf = self._text_surf(logic.spin_text, self.f_big, c["accent2"], (40, 10, 60))
            if pop > 1.01:
                surf = self.pg.transform.smoothscale(surf, (int(surf.get_width() * pop), int(surf.get_height() * pop)))
            self.screen.blit(surf, surf.get_rect(center=(self.w // 2, 150)))

    def _draw_menu(self, logic):
        pg, c = self.pg, self.c
        key = ("shade", self.w, self.h)
        shade = self._sprite_cache.get(key)
        if shade is None:
            shade = pg.Surface((self.w, self.h), pg.SRCALPHA)
            shade.fill((*c["panel"], 150))
            self._sprite_cache[key] = shade
        self.screen.blit(shade, (0, 0))
        cx = self.w // 2
        names = list(config.DIFFICULTIES)
        dynamic = logic.difficulty == config.DYNAMIC_DIFFICULTY
        box = pg.Rect(0, 0, min(self.w - 24, 120 + 150 * len(names)), 330 + (62 if dynamic else 0))
        box.center = (cx, min(300 + box.h // 2 - 165, self.h // 2 + 20))
        self._panel(box, 225, c["panel_border"], radius=22)
        title = self._text("Ping Pong: Human vs Robot", self.f_big, c["accent"], (cx, box.top + 46), "center",
                           outline=(20, 8, 30))
        if self._icon is not None:
            self.screen.blit(self._icon, self._icon.get_rect(midright=(title.left - 10, title.centery)))
            self.screen.blit(self._icon, self._icon.get_rect(midleft=(title.right + 10, title.centery)))
        if self.theme.name == "halloween":
            self._text("~ Spooky Halloween Edition ~", self.f_small, c["accent2"], (cx, box.top + 88), "center")
        keys = "/".join(str(i + 1) for i in range(len(names)))
        self._text(f"Choose difficulty ({keys}), then Start (Enter)", self.f_small, c["dim"],
                   (cx, box.top + 116), "center")
        self.buttons = {}
        gap = 14
        bw = min(160, (box.w - 40 - (len(names) - 1) * gap) // len(names))
        x0 = cx - (len(names) * bw + (len(names) - 1) * gap) // 2
        for i, name in enumerate(names):
            r = pg.Rect(x0 + i * (bw + gap), box.top + 150, bw, 56)
            sel = name == logic.difficulty
            pg.draw.rect(self.screen, c["button_selected"] if sel else c["button"], r, border_radius=14)
            pg.draw.rect(self.screen, c["text"] if sel else c["panel_border"], r, 2, border_radius=14)
            label = f"{i + 1}  {name}"
            font = self.f_med if self.f_med.size(label)[0] <= bw - 10 else self.f_small
            self._text(label, font, (30, 10, 40) if sel else c["text"], r.center, "center")
            self.buttons[f"difficulty:{name}"] = r
        if dynamic:
            info = logic.opponent.panel_info() if hasattr(logic.opponent, "panel_info") else None
            note = (info or {}).get("notice")
            lines = [("Dynamic: an AI learns the hardest shots you can still return", c["accent2"]),
                     ("Press R here to reset what it has learned", c["dim"])]
            if note:
                lines.append((note, c["warn"]))
            y = box.top + 312
            for text, col in lines:
                self._text(self._fit(text, self.f_small, box.w - 30), self.f_small, col, (cx, y), "center")
                y += 21
        r = pg.Rect(cx - 120, box.top + 232, 240, 62)
        pg.draw.rect(self.screen, c["start"], r, border_radius=16)
        pg.draw.rect(self.screen, c["text"], r, 2, border_radius=16)
        self._text("START", self.f_big, (30, 10, 40), r.center, "center")
        self.buttons["start"] = r

    def _fit(self, text, font, width):
        while len(text) > 4 and font.size(text)[0] > width:
            text = text[:-2] + "…"
        return text

    def _draw_debug(self, lines, top=180):
        pg = self.pg
        h = 20 * len(lines) + 12
        rect = pg.Rect(16, top, min(520, self.w - 32), h)
        panel = pg.Surface(rect.size, pg.SRCALPHA)
        panel.fill((0, 0, 0, 185))
        self.screen.blit(panel, rect)
        for i, line in enumerate(lines):
            self.screen.blit(self._text_surf(self._fit(line, self.f_mono, rect.w - 16), self.f_mono,
                                             (170, 255, 170)), (24, top + 6 + 20 * i))

    @staticmethod
    def dynamic_lines(info):
        """Text of the Dynamic RL panel (pure -- unit tested)."""
        def num(v, fmt):
            return "--" if v is None else format(v, fmt)
        lo, hi = info["band"]
        rate = info["return_rate"]
        pos = {"in": "in band", "above": "too easy", "below": "too hard"}[info["band_pos"]]
        mode = {True: "EXPLORE", False: "EXPLOIT", None: "--"}[info["explored"]]
        beta = info["beta"]
        source = "--" if beta is None else (f"mostly {'joint' if beta >= 0.5 else 'tables'}"
                                            f" (joint {beta:.0%})")
        lines = [f"eps {info['epsilon']:.2f}   gamma {info['gamma']:.2f}   hits {info['hits']}",
                 f"last: {mode}  {info['action'] or '--'}",
                 f"difficulty {num(info['difficulty'], '.2f')}   reward {num(info['reward'], '+.2f')}",
                 f"estimate: {source}",
                 f"return rate {num(rate, '.0%')}  target {lo:.0%}-{hi:.0%}  ({pos})",
                 f"weights: bonus {info['bonus']:.2f}  penalty {info['penalty']:.2f}"]
        if info.get("notice"):
            lines.append(info["notice"])
        return lines

    def _draw_dynamic_panel(self, info, top):
        pg, c = self.pg, self.c
        lines = self.dynamic_lines(info)
        rect = pg.Rect(16, top, min(400, self.w - 32), 30 + 19 * len(lines))
        self._panel(rect, 205, c["accent2"], radius=12)
        self._text("DYNAMIC AI", self.f_small, c["accent2"], (rect.left + 12, rect.top + 4))
        for i, line in enumerate(lines):
            col = c["warn"] if (info.get("notice") and i == len(lines) - 1) else c["text"]
            self.screen.blit(self._text_surf(self._fit(line, self.f_mono, rect.w - 22), self.f_mono, col),
                             (rect.left + 12, rect.top + 26 + 19 * i))
        return rect.bottom

    def wrap_hint(self, segments, width, font=None):
        """Greedy word-wrap of control segments into lines that fit `width` px (pure-ish, tested)."""
        font = font or self.f_small
        sep = "   "
        lines, cur = [], ""
        for seg in segments:
            cand = seg if not cur else cur + sep + seg
            if font.size(cand)[0] <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = seg
        if cur:
            lines.append(cur)
        return [self._fit(line, font, width) for line in lines]

    def _draw_hint(self, hint):
        """Controls bar at the bottom. Returns its top y (the camera preview sits above it)."""
        if not hint:
            return self.h
        segments = [hint] if isinstance(hint, str) else list(hint)
        lines = self.wrap_hint(segments, self.w - 40)
        lh = self.f_small.get_linesize()
        top = self.h - 10 - lh * len(lines) - 6
        self._panel(self.pg.Rect(8, top, self.w - 16, self.h - 8 - top), 150, None, radius=10)
        for i, line in enumerate(lines):
            self._text(line, self.f_small, self.c["dim"], (20, top + 4 + lh * i), outline=(15, 6, 25))
        return top

    # -- frame ----------------------------------------------------------------------------
    def draw(self, now, logic, view, statuses, snapshot=None, debug_lines=None, hint=""):
        if not isinstance(view, PlayerView):   # plain paddle x (older callers)
            view = PlayerView(x=float(view))
        scr = self.screen
        sc = self.scene
        scr.blit(sc.backdrop, (0, 0))
        if self.ambient is not None:
            self.ambient.draw(scr, now)

        ball = logic.ball_position(now)
        spin = logic.ball_spin(now) if ball is not None else NO_SPIN
        on_table = ball is not None and self._over_table(ball[0], ball[1])
        if ball is not None and not on_table:
            self._draw_shadow(ball, on_table=False)

        self.skeleton.update(now, logic)
        self.skeleton.draw(scr, self.cam, now, logic)
        scr.blit(sc.table, sc.table_pos)

        far = ball is not None and ball[1] > config.TABLE_LENGTH_M / 2
        if on_table:
            self._draw_shadow(ball, on_table=True)
        if far:
            self._draw_ball(now, logic, ball, spin)
        scr.blit(sc.net, sc.net_pos)
        if ball is not None and not far:
            self._draw_ball(now, logic, ball, spin)
        self._draw_player_paddle(now, view)

        self._draw_hud(logic, statuses)
        hint_top = self._draw_hint(hint)
        if config.SHOW_CAMERA_PREVIEW:
            self._draw_preview(snapshot, hint_top)
        self._draw_messages(now, logic)
        top = 180
        info = logic.opponent.panel_info() if hasattr(logic, "opponent") else None
        if info and logic.state != "menu" and (debug_lines or not config.RL_PANEL_DEBUG_ONLY):
            top = self._draw_dynamic_panel(info, top) + 8
        if debug_lines:
            self._draw_debug(debug_lines, top)
        if logic.state == "menu":
            self._draw_menu(logic)
        else:
            self.buttons = {}
