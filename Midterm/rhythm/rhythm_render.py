"""
Draws Concert Rally: a plain backboard and table, the ball, both paddles, the HUD and every screen.

Layer order (back to front): backdrop -> opponent paddle -> table -> arrival hint ->
far ball -> net -> near ball -> my paddle -> HUD / tempo LED / screens.

All motion is computed from the game's song time (shifted by the visual offset), never from
frame counts. Re-create the renderer when the window size changes.
"""

import math
from dataclasses import dataclass

import config
from camera3d import Camera
from paddle import player_paddle, swing_pose
from plain import OpponentPaddle, PlainTheme
from scene import Scene
from judge import contact_distance
from scheduler import BallView, ball_at, opponent_swing_phase, opponent_x
from tempo import led_state
from theme import mix

SHADOW = (0, 0, 0)


@dataclass
class PlayerView:
    x: float = 0.0
    roll_deg: float = 0.0
    swing_t: float | None = None


def draw_tempo_led(pg, screen, center, radius, state, label_font=None, label_color=(236, 238, 242)):
    """The tempo-change LED: a round lamp in a dark metal bezel (fits the plain stage look);
    amber before a speed-up, blue before a slowdown.
    state: tempo.LedState or None (nothing drawn). Also used by make_chart.py preview."""
    if state is None or state.visible <= 0:
        return
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    a = state.visible
    col = config.TEMPO_LED_COLOR if state.faster else config.TEMPO_LED_SLOW_COLOR
    size = 4 * r
    surf = pg.Surface((size * 2, size * 2), pg.SRCALPHA)
    c = (size, size)
    if state.lit > 0:                                   # glow around the lit lens
        for k in range(r * 2, 0, -2):
            alpha = int(110 * state.lit * a * (1 - k / (r * 2)) ** 2)
            pg.draw.circle(surf, (*col, alpha), c, r + k)
    pg.draw.circle(surf, (58, 60, 66, int(255 * a)), c, r + 5)          # bezel
    pg.draw.circle(surf, (24, 25, 29, int(255 * a)), c, r + 2)
    lens = mix(tuple(v // 5 for v in col), col, state.lit)
    pg.draw.circle(surf, (*lens, int(255 * a)), c, r)
    hl = mix(lens, (255, 255, 255), 0.55 if state.lit else 0.2)        # glass highlight
    pg.draw.circle(surf, (*hl, int(200 * a)), (c[0] - r // 3, c[1] - r // 3), max(2, r // 3))
    screen.blit(surf, (cx - size, cy - size))
    if label_font is not None and config.TEMPO_LED_LABEL:
        txt = label_font.render(f"{state.new_bpm:.0f} BPM", True, label_color)
        txt.set_alpha(int(255 * a))
        screen.blit(txt, txt.get_rect(midtop=(cx, cy + r + 9)))


def status_kind(status):
    s = (status or "").lower()
    if s.startswith(("connected", "tracking", "wasapi", "mme", "windows", "directsound", "pygame")):
        return "good"
    if s.startswith(("disabled", "stopped", "idle", "no audio")):
        return "off"
    if s.startswith(("error", "unavailable", "firmware")):
        return "bad"
    return "warn"


class RhythmRenderer:
    def __init__(self, screen):
        import pygame
        self.pg = pygame
        self.screen = screen
        self.w, self.h = screen.get_size()
        self.theme = PlainTheme()
        self.c = self.theme.c
        # the optical-axis row scales with the window height, so the table fits at any size
        self.cam = Camera(self.w, self.h, center_y=config.VIEW_CENTER_Y_PX * self.h / config.WINDOW_HEIGHT)
        self.scene = Scene(pygame, self.cam, self.theme)
        self.opponent = OpponentPaddle()
        self.paddle = player_paddle()
        t = self.theme
        s = max(0.75, min(1.25, min(self.w / 1100, self.h / 720)))
        self.f_huge = t.font(int(64 * s), title=True, bold=True)
        self.f_big = t.font(int(40 * s), title=True, bold=True)
        self.f_med = t.font(int(24 * s), bold=True)
        self.f_small = t.font(max(14, int(18 * s)))
        self.f_mono = pygame.font.SysFont("consolas,couriernew,monospace", 15)
        self._text_cache = {}
        self._sprite_cache = {}
        self._shown_x = 0.0
        self._last_now = None

    # =============================================================================
    # helpers
    # =============================================================================
    def _text_surf(self, text, font, color, shadow=False):
        key = (text, id(font), color, shadow)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 500:
                self._text_cache.clear()
            pg = self.pg
            main = font.render(text, True, color)
            if shadow:                                   # a soft drop shadow for legibility
                w, h = main.get_size()
                surf = pg.Surface((w + 2, h + 2), pg.SRCALPHA)
                sh = font.render(text, True, SHADOW)
                sh.set_alpha(150)
                surf.blit(sh, (2, 2))
                surf.blit(main, (0, 0))
            else:
                surf = main
            self._text_cache[key] = surf
        return surf

    def _text(self, text, font, color, pos, anchor="topleft", shadow=False, alpha=None, scale=1.0):
        surf = self._text_surf(text, font, color, shadow)
        if scale != 1.0:
            surf = self.pg.transform.smoothscale(surf, (max(1, int(surf.get_width() * scale)),
                                                        max(1, int(surf.get_height() * scale))))
        if alpha is not None:
            surf = surf.copy()
            surf.set_alpha(alpha)
        rect = surf.get_rect(**{anchor: pos})
        rect.clamp_ip(self.screen.get_rect())           # never off-screen, at any window size
        self.screen.blit(surf, rect)
        return rect

    def _fit(self, text, font, width):
        while len(text) > 4 and font.size(text)[0] > width:
            text = text[:-2] + "…"
        return text

    def _panel(self, rect, alpha=215, border=None, radius=6):
        pg = self.pg
        key = ("panel", rect.size, alpha, border, radius)
        surf = self._sprite_cache.get(key)
        if surf is None:
            if len(self._sprite_cache) > 300:
                self._sprite_cache.clear()
            surf = pg.Surface(rect.size, pg.SRCALPHA)
            pg.draw.rect(surf, (*self.c["panel"], alpha), surf.get_rect(), border_radius=radius)
            if border:
                pg.draw.rect(surf, (*border, 255), surf.get_rect(), 1, border_radius=radius)
            self._sprite_cache[key] = surf
        self.screen.blit(surf, rect)

    def _blob(self, w, h, color, alpha):
        key = ("blob", int(w), int(h), color, int(alpha) // 8)
        surf = self._sprite_cache.get(key)
        if surf is None:
            if len(self._sprite_cache) > 300:
                self._sprite_cache.clear()
            pg = self.pg
            surf = pg.Surface((max(2, int(w)), max(2, int(h))), pg.SRCALPHA)
            pg.draw.ellipse(surf, (*color, int(alpha)), surf.get_rect())
            self._sprite_cache[key] = surf
        return surf

    def _ball_sprite(self, r, halo=None):
        r = max(2, int(round(r)))
        key = ("ball", r, halo)
        surf = self._sprite_cache.get(key)
        if surf is None:
            pg = self.pg
            pad = max(6, int(r * 0.9)) if halo else 1
            surf = pg.Surface((2 * (r + pad), 2 * (r + pad)), pg.SRCALPHA)
            c0 = r + pad
            if halo:
                for k in range(pad, 0, -1):
                    pg.draw.circle(surf, (*halo, int(150 * (1 - k / pad) ** 1.5)), (c0, c0), r + k)
            base = config.BALL_COLOR
            dark = mix(base, config.BALL_SHADE_COLOR, 0.6)
            light = mix(base, (255, 255, 255), 0.8)
            steps = max(3, r)
            for i in range(steps, 0, -1):
                k = i / steps
                col = mix(light, base, min(1.0, k * 1.3)) if k < 0.75 else mix(base, dark, (k - 0.75) * 3)
                off = (1 - k) * r * 0.3
                pg.draw.circle(surf, col, (c0 - off, c0 - off), k * r)
            pg.draw.circle(surf, config.BALL_OUTLINE_COLOR, (c0, c0), r, 1)
            self._sprite_cache[key] = surf
        return surf

    # =============================================================================
    # frame
    # =============================================================================
    def draw(self, now, game, view, statuses, snapshot=None, debug_lines=None, hint=None):
        scr, sc = self.screen, self.scene
        draw_t = game.draw_time(now)
        scr.blit(sc.backdrop, (0, 0))

        playing = game.state == "play" and draw_t is not None
        flights = game.flights if playing else []
        ox = opponent_x(flights, draw_t) if playing else 0.0
        swing = opponent_swing_phase(flights, draw_t) if playing else None
        self.opponent.draw(scr, self.cam, ox, swing)
        scr.blit(sc.table, sc.table_pos)

        ball = None
        if playing:
            ball = ball_at(flights, draw_t, game.outcomes, game.ball_x_at())
            self._draw_arrival_hint(game, draw_t)
        elif game.state == "calib_av" and draw_t is not None:
            ph = (draw_t / game.av_spb) % 1.0
            ball = BallView((0.0, 0.9, 0.32 * math.sin(math.pi * ph)), 0, "av")   # touches down on each click

        far = ball is not None and ball.pos[1] > config.TABLE_LENGTH_M / 2
        if ball is not None:
            self._draw_shadow(ball)
            if far:
                self._draw_ball(game, ball, draw_t)
        scr.blit(sc.net, sc.net_pos)
        if ball is not None and not far:
            self._draw_ball(game, ball, draw_t)
        self._draw_player_paddle(now, view)

        hint_top = self._draw_hint(hint)
        if config.SHOW_CAMERA_PREVIEW:
            self._draw_preview(snapshot, hint_top)
        if game.state == "play":
            self._draw_hud(game, draw_t, statuses)
            self._draw_tempo_led(game, draw_t)
            self._draw_popup(game, now)
            self._draw_countdown(game, draw_t)
        elif game.state == "title":
            self._draw_title(game, statuses)
        elif game.state == "select":
            self._draw_select(game)
        elif game.state == "calib_input":
            self._draw_calib_input(game, draw_t)
        elif game.state == "calib_av":
            self._draw_calib_av(game, draw_t)
        elif game.state == "results":
            self._draw_results(game)
        if game.message and now < game.message_until:
            self._text(self._fit(game.message, self.f_med, self.w - 40), self.f_med, self.c["text"],
                       (self.w // 2, self.h - 120), "center", shadow=True)
        if debug_lines:
            self._draw_debug(debug_lines)

    # =============================================================================
    # ball and paddles
    # =============================================================================
    def _over_table(self, x, y):
        return abs(x) <= config.TABLE_WIDTH_M / 2 and 0 <= y <= config.TABLE_LENGTH_M

    def _draw_shadow(self, ball):
        x, y, z = ball.pos
        surface_z = 0.0 if self._over_table(x, y) else -config.TABLE_HEIGHT_M
        p = self.cam.project(x, y, surface_z)
        if p is None:
            return
        sx, sy, s = p
        height = max(0.0, z - surface_z)
        r = config.BALL_RADIUS_M * config.BALL_VISUAL_SCALE * (1 + 1.2 * height)
        alpha = max(40, min(150, 150 - 200 * height)) * ball.alpha
        blob = self._blob(2 * r * s, max(2, 0.7 * r * s), config.SHADOW_COLOR, alpha)
        self.screen.blit(blob, blob.get_rect(center=(sx, sy)))

    def _draw_ball(self, game, ball, draw_t):
        rad = config.BALL_RADIUS_M * config.BALL_VISUAL_SCALE
        if config.TRAIL_ENABLED and draw_t is not None and ball.phase in ("in", "out") and game.state == "play":
            n = config.TRAIL_DOTS
            for i in range(n, 0, -1):
                tb = ball_at(game.flights, draw_t - config.TRAIL_LENGTH_S * i / n, game.outcomes,
                             game.ball_x_at())
                if tb is None or tb.index != ball.index:
                    continue
                p = self.cam.project(tb.pos[0], tb.pos[1], tb.pos[2] + rad)
                if p is None:
                    continue
                k = 1 - i / (n + 1)
                r = rad * p[2] * (0.3 + 0.5 * k)
                dot = self._blob(2 * r, 2 * r, config.TRAIL_COLOR, 70 * k * ball.alpha)
                self.screen.blit(dot, dot.get_rect(center=(p[0], p[1])))
        p = self.cam.project(ball.pos[0], ball.pos[1], ball.pos[2] + rad)
        if p is None:
            return
        sx, sy, s = p
        r = max(2.5, rad * s)
        halo = config.UPBEAT_GLOW_COLOR if (ball.upbeat and ball.phase in ("in", "hold")) else None
        spr = self._ball_sprite(r, halo)
        if ball.alpha < 0.999:
            spr = spr.copy()
            spr.set_alpha(int(255 * max(0.0, ball.alpha)))
        self.screen.blit(spr, spr.get_rect(center=(sx, sy)))

    def _draw_arrival_hint(self, game, draw_t):
        """Soft spot where the next ball reaches your end: in aim mode as wide as the zone your
        paddle centre must be in (contact distance each side), in timing mode just a cue, in
        follow mode none (the ball comes to you). For an upbeat cue (future charts), a pulsing
        amber ring as well."""
        fl = game.next_flight(draw_t)
        if fl is None or game.mode == "follow":
            return
        p = self.cam.project(fl.x_arrive, 0.22, 0.0)
        if p is None:
            return
        sx, sy, s = p
        lead = config.ARRIVAL_HINT_BEATS * fl.spb
        k = (draw_t - (fl.t_arrive - lead)) / lead
        if config.ARRIVAL_HINT and 0 <= k <= 1.05:
            w = 2 * contact_distance() if game.mode == "aim" else 0.34
            blob = self._blob(w * s, 0.11 * s, (255, 255, 255), 15 + 35 * min(1.0, k))
            self.screen.blit(blob, blob.get_rect(center=(sx, sy)))
        start = fl.t_warn if fl.t_warn is not None else fl.t_serve
        if fl.upbeat and start <= draw_t <= fl.t_arrive:
            pulse = 0.5 + 0.5 * abs(math.cos(math.pi * game.chart.time_beat(draw_t)))
            rw, rh = 0.40 * s * (0.9 + 0.15 * pulse), 0.13 * s * (0.9 + 0.15 * pulse)
            self.pg.draw.ellipse(self.screen, config.UPBEAT_GLOW_COLOR,
                                 (sx - rw / 2, sy - rh / 2, rw, rh), 3)

    def _draw_player_paddle(self, now, view):
        dt = 0.0 if self._last_now is None else min(0.1, now - self._last_now)
        self._last_now = now
        self._shown_x += (view.x - self._shown_x) * min(1.0, dt * 30)      # visual easing only
        dx = dy = dz = tilt = 0.0
        if view.swing_t is not None:
            phase = (now - view.swing_t) / config.SWING_ANIM_S
            if 0 <= phase <= 1:
                dx, dy, dz, tilt = swing_pose(phase)
        center = (self._shown_x + dx, config.PLAYER_HIT_Y_M + dy, config.PADDLE_HEIGHT_M + dz)
        lean = -12.0 if config.DOMINANT_HAND == "right" else 12.0
        self.paddle.draw(self.screen, self.cam, center, view.roll_deg, tilt_deg=tilt, lean_deg=lean,
                         alpha=config.PLAYER_PADDLE_ALPHA)

    # =============================================================================
    # HUD (play)
    # =============================================================================
    def _draw_statuses(self, statuses, right, top):
        pg = self.pg
        rect = pg.Rect(0, 0, min(260, self.w // 4 + 30), 12 + 24 * len(statuses))
        rect.topright = (right, top)
        self._panel(rect, 200, self.c["panel_border"])
        y = rect.top + 7
        for label, status in statuses:
            txt = self._fit(f"{label}: {status}", self.f_small, rect.w - 36)
            r = self._text(txt, self.f_small, self.c["text"], (rect.left + 26, y))
            pg.draw.circle(self.screen, self.c.get(status_kind(status), self.c["off"]), (rect.left + 14, r.centery), 5)
            y += 24
        return rect

    def _draw_hud(self, game, draw_t, statuses):
        pg, c = self.pg, self.c
        sc = game.score
        left = pg.Rect(14, 14, 240, 138)
        self._panel(left, 215, c["panel_border"])
        self._text("ACCURACY", self.f_small, c["dim"], (left.left + 14, left.top + 6))
        acc = f"{sc.accuracy:.1f}%" if sc.judged else "--.-%"
        self._text(acc, self.f_big, c["text"], (left.left + 12, left.top + 26))
        self._text(f"Streak {sc.streak}   (max {sc.max_streak})", self.f_small, c["text"],
                   (left.left + 14, left.top + 78))
        x = left.left + 14
        for g, key in (("Perfect", "perfect"), ("Good", "goodgrade"), ("Miss", "miss")):
            r = self._text(f"{g} {sc.counts[g]}", self.f_small, c[key], (x, left.top + 104))
            x = r.right + 12
        status_rect = self._draw_statuses(statuses, self.w - 14, 14)
        x0, x1 = left.right + 12, status_rect.left - 12
        top = 14
        if x1 - x0 < 150:
            x0, x1, top = 14, self.w - 14, max(left.bottom, status_rect.bottom) + 8
        mid = pg.Rect(x0, top, x1 - x0, 58)
        self._panel(mid, 200, c["panel_border"])
        self._text(self._fit(game.chart.title, self.f_med, mid.w - 24), self.f_med, c["text"],
                   (mid.centerx, mid.top + 5), "midtop")
        self._text(config.GAME_MODE_NAMES[game.mode], self.f_small, c["dim"], (mid.left + 10, mid.top + 6))
        bar = pg.Rect(mid.left + 16, mid.bottom - 15, mid.w - 32, 5)
        pg.draw.rect(self.screen, c["button"], bar, border_radius=2)
        pg.draw.rect(self.screen, c["accent"], (bar.left, bar.top, int(bar.w * game.progress(draw_t)), bar.h),
                     border_radius=2)

    def _draw_tempo_led(self, game, draw_t):
        state = led_state(game.tempo_changes, draw_t)
        if state is None:
            return
        fx, fy = config.TEMPO_LED_POS
        draw_tempo_led(self.pg, self.screen, (self.w * fx, self.h * fy), config.TEMPO_LED_RADIUS_PX, state,
                       self.f_small, self.c["text"])

    def _draw_popup(self, game, now):
        """Latest grade, beside the table on the side the ball came to (never over its path)."""
        c = self.c
        if not game.popups:
            return
        j, t = game.popups[-1]
        age = now - t
        if not 0 <= age <= config.GRADE_POPUP_S:
            return
        fl = game.flights[j.index]
        x = game.hit_x.get(j.index, fl.x_arrive) if game.mode == "follow" else fl.x_arrive
        side = 1 if x > 0 else -1
        edge = self.cam.point(side * (config.TABLE_WIDTH_M / 2 + 0.08), 0.45, 0.25)
        if edge is None:
            return
        k = age / config.GRADE_POPUP_S
        col = {"Perfect": c["perfect"], "Good": c["goodgrade"]}.get(j.grade, c["miss"])
        label = {"Perfect": "Perfect", "Good": "Good", "Miss": "Miss"}[j.grade]
        alpha = int(255 * (1 - max(0.0, k - 0.6) / 0.4))
        anchor = "midleft" if side > 0 else "midright"
        r = self._text(label, self.f_big, col, (edge[0] + 10 * side, edge[1] - 12 * k), anchor, True, alpha)
        sub = j.reason
        if not sub and j.grade == "Good" and j.error_ms is not None:
            sub = "early" if j.error_ms < 0 else "late"
        if sub:
            self._text(sub, self.f_small, c["dim"], (r.centerx, r.bottom), "midtop", True, alpha)

    def _draw_countdown(self, game, draw_t):
        label = game.countdown_label(draw_t)
        if label is not None:
            self._text(label, self.f_huge, self.c["text"], (self.w // 2, int(self.h * 0.36)), "center", True)

    # =============================================================================
    # screens
    # =============================================================================
    def _shade(self, alpha=150):
        key = ("shade", self.w, self.h, alpha)
        s = self._sprite_cache.get(key)
        if s is None:
            s = self.pg.Surface((self.w, self.h), self.pg.SRCALPHA)
            s.fill((*self.c["panel"], alpha))
            self._sprite_cache[key] = s
        self.screen.blit(s, (0, 0))

    def _box(self, w, h):
        rect = self.pg.Rect(0, 0, min(w, self.w - 24), min(h, self.h - 24))
        rect.center = (self.w // 2, self.h // 2 - 20)
        self._panel(rect, 235, self.c["panel_border"], radius=8)
        return rect

    def _button(self, rect, label, selected):
        c = self.c
        self.pg.draw.rect(self.screen, c["button_selected"] if selected else c["button"], rect, border_radius=6)
        self._text(label, self.f_med, c["panel"] if selected else c["text"], rect.center, "center")

    def _draw_title(self, game, statuses):
        from rhythm_game import TITLE_ITEMS
        c = self.c
        self._shade(120)
        box = self._box(520, 400)
        t = self._text("Concert Rally", self.f_huge, c["text"], (box.centerx, box.top + 22), "midtop")
        self._text("Rhythm ping pong", self.f_small, c["dim"], (box.centerx, t.bottom), "midtop")
        y = t.bottom + 44
        for i, item in enumerate(TITLE_ITEMS):
            r = self.pg.Rect(0, 0, 300, 46)
            r.midtop = (box.centerx, y)
            self._button(r, item, i == game.menu_index)
            y += 56
        self._draw_statuses(statuses, self.w - 14, 14)

    def _draw_select(self, game):
        c, pg = self.c, self.pg
        self._shade(140)
        box = self._box(980, 520)
        from rhythm_game import mode_label
        t = self._text("Choose a song", self.f_big, c["text"], (box.centerx, box.top + 6), "midtop")
        self._text(f"Mode: {mode_label(game.mode)}   ·   M to change", self.f_small, c["accent2"],
                   (box.centerx, t.bottom - 2), "midtop")
        if not game.songs:
            self._text(f"No playable songs in {config.SONGS_DIR.name}/ -- see README (Adding a song)",
                       self.f_small, c["warn"], (box.centerx, box.centery), "center")
            return
        row_h = 52
        top = box.top + 88
        info_h = 70
        rows = max(1, (box.bottom - info_h - top) // row_h)
        first = max(0, min(game.sel_index - rows // 2, len(game.songs) - rows))
        for k, e in enumerate(game.songs[first:first + rows]):
            i = first + k
            r = pg.Rect(box.left + 20, top + k * row_h, box.w - 40, row_h - 6)
            sel = i == game.sel_index
            pg.draw.rect(self.screen, c["button_selected"] if sel else c["button"], r, border_radius=6)
            fg = c["panel"] if sel else c["text"]
            dim = mix(fg, c["button_selected"] if sel else c["button"], 0.35)
            ch = e.chart
            info = game.song_info(e)
            mins, secs = divmod(int(info.length_s), 60)
            stats_x = r.left + int(r.w * 0.52)
            # left: the title, then the author smaller and dimmer in the same row
            t = self._text(self._fit(ch.title, self.f_med, r.w * 0.3), self.f_med, fg,
                           (r.left + 12, r.centery), "midleft")
            if ch.author:
                room = stats_x - 16 - (t.right + 8)
                if room > 40:
                    self._text(self._fit(f"— {ch.author}", self.f_small, room), self.f_small, dim,
                               (t.right + 8, r.centery + 2), "midleft")
            # right: BPM (a range when the tempo changes), length, cues, best score
            self._text(f"{ch.bpm_label()} BPM  ·  {mins}:{secs:02d}  ·  {len(ch.cues)} cues", self.f_small, fg,
                       (stats_x, r.centery), "midleft")
            best = game.best_for(ch)
            txt = f"best {best['best_accuracy']:.1f}%  ·  streak {best['max_streak']}" if best else "not played yet"
            self._text(txt, self.f_small, fg, (r.right - 12, r.centery), "midright")
        info = game.song_info(game.selected_entry())
        y = box.bottom - info_h + 6
        notes = list(info.notes if info else [])
        if game.skipped:
            notes.append(f"{len(game.skipped)} chart(s) skipped: " + ", ".join(n for n, _ in game.skipped)
                         + " (reasons in the console)")
        for line in (notes or ["Chart checks OK"])[:3]:
            self._text(self._fit(line, self.f_small, box.w - 40), self.f_small, c["dim"], (box.left + 20, y))
            y += 21

    def _draw_calib_input(self, game, draw_t):
        c = self.c
        self._shade(160)
        box = self._box(700, 440)
        cal = game.calib
        self._text("Timing calibration", self.f_big, c["text"], (box.centerx, box.top + 12), "midtop")
        if cal is None:
            return
        old = config.INPUT_OFFSET_MS
        y = box.top + 72
        if cal.result is None:
            lines = ["Swing your paddle (or press Space) on every click.",
                     "Listen rather than watch -- the screen gives no beat here.",
                     f"Current offsets: motor {old.get('imu', 0):+.0f} ms, keyboard {old.get('keyboard', 0):+.0f} ms"]
            for line in lines:
                self._text(line, self.f_small, c["text"], (box.centerx, y), "midtop")
                y += 24
            b = cal.beat_index(draw_t) if draw_t is not None else -1
            if b < 0:
                label = "Get ready..."
            elif b < cal.count_in:
                label = f"Count-in {cal.count_in - b}"
            else:
                label = f"Beat {min(cal.beats, b - cal.count_in + 1)} / {cal.beats}"
            self._text(label, self.f_big, c["accent"], (box.centerx, y + 20), "midtop")
            errs = [(t - min(cal.measured_times, key=lambda m: abs(m - t))) * 1000 for t, _ in cal.swings]
            self._draw_error_strip(box, box.bottom - 90, errs)
        else:
            r = cal.result
            if r.ok:
                lines = [(f"Measured {cal.source} offset: {r.offset_ms:+.0f} ms", self.f_med, c["text"]),
                         (f"{r.used} swings used, {r.dropped} dropped, spread ±{r.std_ms:.0f} ms", self.f_small, c["dim"]),
                         (f"Before: your swings were judged {r.mean_ms - old.get(cal.source, 0):+.0f} ms off on average"
                          f" (old offset {old.get(cal.source, 0):+.0f} ms)", self.f_small, c["dim"]),
                         ("After saving: about 0 ms", self.f_small, c["good"]),
                         ("Enter = save    R = try again    Esc = cancel", self.f_small, c["text"])]
            else:
                lines = [(f"Not enough steady swings ({r.used} of {config.CALIB_MIN_SWINGS} needed)", self.f_small, c["warn"]),
                         ("R = try again    Esc = cancel", self.f_small, c["text"])]
            for text, font, col in lines:
                self._text(self._fit(text, font, box.w - 30), font, col, (box.centerx, y), "midtop")
                y += 34 if font is self.f_med else 26
            self._draw_error_strip(box, box.bottom - 90, list(r.errors_ms))

    def _draw_error_strip(self, box, y, errors_ms):
        """Timeline -200..+200 ms with a dot per swing (early left, late right)."""
        pg, c = self.pg, self.c
        line = pg.Rect(box.left + 40, y, box.w - 80, 2)
        pg.draw.rect(self.screen, c["dim"], line)
        mid = line.centerx
        pg.draw.line(self.screen, c["accent"], (mid, y - 12), (mid, y + 14), 2)
        self._text("early", self.f_small, c["dim"], (line.left, y + 10))
        self._text("late", self.f_small, c["dim"], (line.right, y + 10), "topright")
        for e in errors_ms:
            x = mid + max(-1.0, min(1.0, e / 200)) * line.w / 2
            pg.draw.circle(self.screen, c["accent2"], (int(x), y + 1), 5)

    def _draw_calib_av(self, game, draw_t):
        c = self.c
        rect = self.pg.Rect(0, 0, min(760, self.w - 24), 120)
        rect.midtop = (self.w // 2, 14)
        self._panel(rect, 225, c["panel_border"])
        self._text("Visual calibration", self.f_med, c["dim"], (rect.centerx, rect.top + 8), "midtop")
        self._text(f"Visual offset: {game.av_offset_ms:+.0f} ms", self.f_big, c["text"],
                   (rect.centerx, rect.top + 36), "midtop")
        self._text("Left/Right until the ball touches the table exactly on each click (Shift = 1 ms)",
                   self.f_small, c["dim"], (rect.centerx, rect.bottom - 26), "midtop")

    def _draw_results(self, game):
        c = self.c
        res = game.results
        self._shade(160)
        box = self._box(580, 440)
        if res is None:
            return
        self._text(self._fit(res["title"], self.f_big, box.w - 30), self.f_big, c["text"],
                   (box.centerx, box.top + 12), "midtop")
        self._text(res["rank"], self.f_huge, c["text"], (box.left + 90, box.top + 150), "center", scale=1.5)
        x = box.left + 190
        y = box.top + 80
        self._text(f"{res['accuracy']:.1f}%", self.f_big, c["text"], (x, y))
        if res["new_best"]:
            self._text("New best", self.f_small, c["good"], (x + 4, y + 50))
        y += 84
        cnt = res["counts"]
        for g, key in (("Perfect", "perfect"), ("Good", "goodgrade"), ("Miss", "miss")):
            self._text(f"{g}: {cnt[g]}", self.f_med, c[key], (x, y))
            y += 32
        self._text(f"Max streak: {res['max_streak']}" + ("  (new best)" if res["new_streak"] else ""),
                   self.f_med, c["text"], (x, y))
        y += 38
        m = res["mean_error_ms"]
        if m is not None:
            self._text(f"On average {abs(m):.0f} ms {'early' if m < 0 else 'late'}", self.f_small, c["dim"], (x, y))

    # =============================================================================
    # bottom bar, preview, debug
    # =============================================================================
    def wrap_hint(self, segments, width, font=None):
        font = font or self.f_small
        lines, cur = [], ""
        for seg in segments:
            cand = seg if not cur else cur + "   " + seg
            if font.size(cand)[0] <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = seg
        if cur:
            lines.append(cur)
        return [self._fit(line, font, width) for line in lines]

    def _draw_hint(self, hint):
        if not hint:
            return self.h
        lines = self.wrap_hint(list(hint), self.w - 40)
        lh = self.f_small.get_linesize()
        top = self.h - 10 - lh * len(lines) - 6
        self._panel(self.pg.Rect(8, top, self.w - 16, self.h - 8 - top), 170, None)
        for i, line in enumerate(lines):
            self._text(line, self.f_small, self.c["dim"], (20, top + 4 + lh * i))
        return top

    def _draw_preview(self, snap, bottom):
        if snap is None or snap.preview_rgb is None:
            return
        img = snap.preview_rgb
        h, w = img.shape[:2]
        surf = self.pg.image.frombuffer(img.tobytes(), (w, h), "RGB")
        rect = surf.get_rect(bottomright=(self.w - 16, bottom - 12))
        self._panel(rect.inflate(8, 8), 220, self.c["panel_border"], radius=4)
        self.screen.blit(surf, rect)

    def _draw_debug(self, lines):
        pg = self.pg
        rect = pg.Rect(14, 166, min(560, self.w - 28), 20 * len(lines) + 12)
        panel = pg.Surface(rect.size, pg.SRCALPHA)
        panel.fill((0, 0, 0, 190))
        self.screen.blit(panel, rect)
        for i, line in enumerate(lines):
            self.screen.blit(self._text_surf(self._fit(line, self.f_mono, rect.w - 16), self.f_mono, (170, 255, 170)),
                             (rect.left + 8, rect.top + 6 + 20 * i))
