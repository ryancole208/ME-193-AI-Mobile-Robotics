"""
Themes: colours, fonts, the pre-rendered backdrop (sky, scenery, floor) and ambient animation.

All artwork is drawn procedurally here -- no image files. A theme provides:
    colors                     -- dict of named RGB colours (from config.py)
    font(size, title, bold)    -- playful system font (config.FONT_TITLE / FONT_BODY)
    make_backdrop(pg, cam)     -- opaque Surface: sky, scenery and floor (drawn once)
    make_ambient(pg, cam)      -- object with draw(screen, now) for animated scenery
                                  (always drawn behind the table, so it never hides the ball)
    icon(pg, size)             -- small decoration for the menu (or None)

Add a theme by subclassing Theme and registering it in THEMES.
"""

import math
import random

import config


def mix(a, b, k):
    k = max(0.0, min(1.0, k))
    return tuple(int(round(x + (y - x) * k)) for x, y in zip(a, b))


def _soft(pg, surf, factor=3):
    """Cheap blur: shrink and grow back (once, at startup)."""
    w, h = surf.get_size()
    small = pg.transform.smoothscale(surf, (max(1, w // factor), max(1, h // factor)))
    return pg.transform.smoothscale(small, (w, h))


def _alpha_surface(pg, w, h):
    return pg.Surface((max(1, int(w)), max(1, int(h))), pg.SRCALPHA)


class Theme:
    name = "base"
    colors = {}

    def __init__(self):
        self._fonts = {}
        self.c = dict(self.colors)

    def color(self, key, default=(255, 0, 255)):
        return self.c.get(key, default)

    def font(self, size, title=False, bold=False):
        import pygame
        key = (size, title, bold)
        if key not in self._fonts:
            names = config.FONT_TITLE if title else config.FONT_BODY
            self._fonts[key] = pygame.font.SysFont(names, size, bold=bold)
        return self._fonts[key]

    # -- backdrop -----------------------------------------------------------------
    def make_backdrop(self, pg, cam):
        w, h = cam.w, cam.h
        surf = pg.Surface((w, h))
        self.draw_sky(pg, surf, cam)
        self.draw_scenery(pg, surf, cam)
        self.draw_floor(pg, surf, cam)
        return surf

    def draw_sky(self, pg, surf, cam):
        w, h = surf.get_size()
        hz = max(1, int(cam.horizon_y))
        top, bottom, glow = self.color("sky_top"), self.color("sky_bottom"), self.color("horizon_glow")
        for row in range(min(h, hz + 2)):
            k = row / hz
            c = mix(top, bottom, k ** 1.2)
            if k > 0.75:
                c = mix(c, glow, (k - 0.75) / 0.25 * 0.6)
            pg.draw.line(surf, c, (0, row), (w, row))

    def draw_scenery(self, pg, surf, cam):
        pass

    def draw_floor(self, pg, surf, cam):
        """Stone-tile patio under the table, fading into fog at the horizon."""
        w, h = surf.get_size()
        zf = -config.TABLE_HEIGHT_M
        hz = cam.horizon_y
        pg.draw.rect(surf, self.color("fog"), (0, int(hz), w, h - int(hz) + 1))
        tile = 0.6
        a, b, line = self.color("floor_a"), self.color("floor_b"), self.color("floor_line")
        fog = self.color("fog")
        x_range = range(-14, 14)
        y_range = range(int(cam.y / tile) - 1, 40)
        for j in reversed(y_range):          # far rows first
            y0, y1 = j * tile, (j + 1) * tile
            far = cam.depth(0, y1, zf)
            k = min(1.0, max(0.0, (far - 3.0) / 16.0)) ** 0.8   # fog by distance
            for i in x_range:
                x0, x1 = i * tile, (i + 1) * tile
                poly = cam.polygon([(x0, y0, zf), (x1, y0, zf), (x1, y1, zf), (x0, y1, zf)])
                if poly is None:
                    continue
                base = a if (i + j) % 2 == 0 else b
                pg.draw.polygon(surf, mix(base, fog, k), poly)
                if k < 0.9:
                    pg.draw.polygon(surf, mix(line, fog, k), poly, 1)
        # haze band at the horizon
        band = _alpha_surface(pg, w, 70)
        for r in range(70):
            pg.draw.line(band, (*fog, int(230 * (1 - r / 70) ** 1.5)), (0, r), (w, r))
        surf.blit(band, (0, int(hz)))

    # -- animation ------------------------------------------------------------------
    def make_ambient(self, pg, cam):
        return None

    def icon(self, pg, size):
        return None


# =============================================================================
# Classic (plain) theme -- proves the swap works and is the fast fallback
# =============================================================================

class ClassicTheme(Theme):
    name = "classic"

    def __init__(self):
        self.colors = config.CLASSIC_COLORS
        super().__init__()


# =============================================================================
# Halloween (classroom-friendly: cute, not scary)
# =============================================================================

def draw_ghost(pg, size, color, face=0):
    """Cute ghost sprite (size = width in px). face: 0 smile, 1 'oh', 2 wink."""
    w = max(8, int(size))
    h = int(w * 1.25)
    s = _alpha_surface(pg, w, h)
    r = w // 2
    body = (*color, 225)
    pg.draw.circle(s, body, (r, r), r)
    pg.draw.rect(s, body, (0, r, w, h - r - w // 8))
    # wavy hem: alternating bumps
    n = 4
    bw = w / n
    for i in range(n):
        cx = int(bw * i + bw / 2)
        pg.draw.circle(s, body, (cx, h - w // 8 - 1), int(bw / 2) + 1)
    # little arms
    pg.draw.ellipse(s, body, (-w // 10, int(h * 0.45), w // 4, w // 6))
    pg.draw.ellipse(s, body, (w - w // 7, int(h * 0.40), w // 4, w // 6))
    eye = (40, 30, 60, 255)
    ey = int(r * 0.85)
    ew, eh = max(2, w // 9), max(3, w // 6)
    if face == 2:
        pg.draw.ellipse(s, eye, (int(r - w * 0.2) - ew // 2, ey, ew, eh))
        pg.draw.arc(s, eye, (int(r + w * 0.1), ey + eh // 3, ew * 2, eh), 0, math.pi, max(1, w // 30))
    else:
        for dx in (-0.2, 0.2):
            pg.draw.ellipse(s, eye, (int(r + w * dx) - ew // 2, ey, ew, eh))
    cheek = (255, 160, 190, 200)
    for dx in (-0.32, 0.32):
        pg.draw.ellipse(s, cheek, (int(r + w * dx) - w // 14, ey + eh, w // 7, w // 14))
    my = ey + int(eh * 1.6)
    if face == 1:
        pg.draw.ellipse(s, eye, (r - w // 16, my, w // 8, w // 7))
    else:
        pg.draw.arc(s, eye, (r - w // 8, my - w // 16, w // 4, w // 7), math.pi * 1.1, math.pi * 1.9,
                    max(1, w // 28))
    return s


def draw_pumpkin(pg, size, c, face=0, lit=True):
    """Jack-o'-lantern (size = width px). face: 0 happy, 1 silly (tongue), 2 surprised.

    lit=False draws the face holes dark; lit=True draws only the glowing face
    (to be blended on top of the unlit pumpkin for flicker).
    """
    w = max(10, int(size))
    h = int(w * 0.85)
    s = _alpha_surface(pg, w, h + w // 5)
    top = w // 5
    if not lit:
        body, dark = c["pumpkin"], c["pumpkin_dark"]
        for frac, col in ((0.0, dark), (0.18, body), (0.36, dark), (0.5, body)):
            rw = int(w * (1 - frac * 0.9))
            pg.draw.ellipse(s, col, ((w - rw) // 2, top, rw, h))
        pg.draw.ellipse(s, mix(body, (255, 255, 255), 0.25),
                        (int(w * 0.22), top + h // 8, w // 6, h // 3))
        # stem + curly vine
        pg.draw.rect(s, c["stem"], (w // 2 - w // 16, top - w // 7, w // 8, w // 6), border_radius=3)
        pg.draw.arc(s, c["stem"], (w // 2, top - w // 6, w // 4, w // 6), 0, math.pi * 1.2, max(1, w // 30))
    face_col = c["pumpkin_glow"] if lit else (90, 40, 20)
    fy = top + h // 3
    # eyes
    if face == 2:
        for dx in (-0.18, 0.18):
            pg.draw.circle(s, face_col, (int(w / 2 + w * dx), fy), max(2, w // 12))
    else:
        for dx in (-0.18, 0.18):
            cx = int(w / 2 + w * dx)
            pg.draw.polygon(s, face_col, [(cx - w // 12, fy + w // 14), (cx + w // 12, fy + w // 14),
                                          (cx, fy - w // 14)])
    pg.draw.polygon(s, face_col, [(w // 2 - w // 28, fy + w // 7), (w // 2 + w // 28, fy + w // 7),
                                  (w // 2, fy + w // 10)])
    # mouth
    my = fy + w // 5
    if face == 2:
        pg.draw.ellipse(s, face_col, (w // 2 - w // 14, my, w // 7, w // 6))
    else:
        mouth = [(w * 0.25, my), (w * 0.75, my), (w * 0.68, my + w * 0.12), (w * 0.5, my + w * 0.17),
                 (w * 0.32, my + w * 0.12)]
        pg.draw.polygon(s, face_col, mouth)
        if not lit:  # a tooth
            pg.draw.rect(s, c["pumpkin"], (int(w * 0.44), int(my), max(2, w // 12), max(2, w // 14)))
        if face == 1 and not lit:
            pg.draw.ellipse(s, (240, 90, 120), (int(w * 0.52), int(my + w * 0.09), w // 8, w // 9))
    return s


def draw_bat(pg, size, color, frame):
    w = max(10, int(size))
    h = w // 2 + 4
    s = _alpha_surface(pg, w, h)
    cx, cy = w // 2, h // 2
    up = frame == 0
    for side in (-1, 1):
        tip_y = 2 if up else h - 3
        pts = [(cx, cy), (cx + side * w * 0.48, tip_y), (cx + side * w * 0.38, cy + 2),
               (cx + side * w * 0.28, cy + (0 if up else 4)), (cx + side * w * 0.16, cy + 3)]
        pg.draw.polygon(s, color, pts)
    pg.draw.circle(s, color, (cx, cy), max(3, w // 7))
    pg.draw.polygon(s, color, [(cx - w // 10, cy - w // 12), (cx - w // 18, cy - w // 5), (cx - w // 40, cy - w // 10)])
    pg.draw.polygon(s, color, [(cx + w // 10, cy - w // 12), (cx + w // 18, cy - w // 5), (cx + w // 40, cy - w // 10)])
    for dx in (-1, 1):
        pg.draw.circle(s, (255, 250, 230), (cx + dx * max(1, w // 22), cy - 1), max(1, w // 30))
    return s


def draw_leaf(pg, size, color):
    w = max(6, int(size))
    s = _alpha_surface(pg, w, w)
    pts = [(w * 0.5, 0), (w * 0.85, w * 0.35), (w * 0.7, w * 0.8), (w * 0.5, w), (w * 0.3, w * 0.8),
           (w * 0.15, w * 0.35)]
    pg.draw.polygon(s, color, pts)
    pg.draw.line(s, mix(color, (60, 30, 10), 0.5), (w * 0.5, w * 0.1), (w * 0.5, w * 0.95), 1)
    return s


def draw_moon(pg, r, c):
    size = int(r * 3.2)
    s = _alpha_surface(pg, size, size)
    cx = cy = size // 2
    glow = c["moon_glow"]
    for i in range(12, 0, -1):
        rr = int(r * (1 + i * 0.06))
        pg.draw.circle(s, (*glow, int(10 + 3 * (12 - i))), (cx, cy), rr)
    pg.draw.circle(s, c["moon"], (cx, cy), r)
    crater = mix(c["moon"], (200, 180, 140), 0.35)
    for dx, dy, rr in ((-0.4, -0.35, 0.16), (0.42, 0.3, 0.12), (0.15, -0.5, 0.08), (-0.3, 0.45, 0.1)):
        pg.draw.circle(s, crater, (int(cx + dx * r), int(cy + dy * r)), max(2, int(rr * r)))
    # friendly sleepy face
    face = (150, 120, 90)
    lw = max(2, r // 18)
    for dx in (-0.3, 0.3):
        pg.draw.arc(s, face, (int(cx + dx * r - r * 0.13), int(cy - r * 0.12), int(r * 0.26), int(r * 0.2)),
                    math.pi, 2 * math.pi, lw)
    pg.draw.arc(s, face, (int(cx - r * 0.22), int(cy + r * 0.05), int(r * 0.44), int(r * 0.32)),
                math.pi * 1.15, math.pi * 1.85, lw)
    for dx in (-0.48, 0.48):
        pg.draw.ellipse(s, (255, 190, 170), (int(cx + dx * r - r * 0.1), int(cy + r * 0.12), int(r * 0.2),
                                             int(r * 0.11)))
    return s


# =============================================================================
# Decoration layout: seeded (or hand-placed) spots that never overlap each other or gameplay
# =============================================================================

def _hull(points):
    """Convex hull (monotone chain), counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def _in_hull(q, hull):
    n = len(hull)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        c = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
        if c != 0:
            if sign == 0:
                sign = 1 if c > 0 else -1
            elif (c > 0) != (sign > 0):
                return False
    return True


def rect_hits_hull(rect, hull, steps=6):
    """Does a screen rect (x, y, w, h) overlap a convex screen polygon?"""
    x, y, w, h = rect
    if any(x <= hx <= x + w and y <= hy <= y + h for hx, hy in hull):
        return True
    return any(_in_hull((x + w * i / steps, y + h * j / steps), hull)
               for i in range(steps + 1) for j in range(steps + 1))


def rects_overlap(a, b, gap=0.0):
    return not (a[0] + a[2] + gap <= b[0] or b[0] + b[2] + gap <= a[0] or
                a[1] + a[3] + gap <= b[1] or b[1] + b[3] + gap <= a[1])


def gameplay_hulls(cam):
    """Screen polygons decorations must stay clear of: the table with the ball's flight space and
    both paddles, and everywhere the skeleton (and its paddle) can reach."""
    hw, zf = config.TABLE_WIDTH_M / 2, -config.TABLE_HEIGHT_M
    sk_y = config.OPPONENT_HIT_Y_M + config.SKELETON_BEHIND_HIT_PLANE_M
    reach = hw + 0.5 + 0.8
    blade = 0.6 * config.PADDLE_BLADE_W_M * config.PADDLE_DRAW_SCALE
    py = config.PLAYER_HIT_Y_M
    boxes = [(-hw - 0.05, hw + 0.05, 0.0, config.TABLE_LENGTH_M, zf, 0.5),          # table + ball flight
             (-hw - blade, hw + blade, py - 0.15, py + 0.45, -0.15, 0.45),            # my paddle + swing
             (-reach, reach, sk_y - 0.35, sk_y + 0.35, zf, zf + 2.0)]               # skeleton reach
    hulls = []
    for x0, x1, y0, y1, z0, z1 in boxes:
        pts = [cam.point(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        hulls.append(_hull([q for q in pts if q is not None]))
    return hulls


def pumpkin_rect(cam, x, y):
    p = cam.project(x, y, -config.TABLE_HEIGHT_M)
    if p is None:
        return None
    sx, sy, s = p
    w = 0.42 * s
    return (sx - 0.6 * w, sy - 1.08 * w, 1.2 * w, 1.23 * w)   # sprite + its floor shadow


def ghost_rect(cam, x, y, z):
    p = cam.project(x, y, z)
    if p is None:
        return None
    sx, sy, s = p
    w = 0.55 * s
    h = 1.25 * w
    return (sx - w / 2 - 0.03 * s, sy - h / 2 - 0.06 * s, w + 0.06 * s, h + 0.12 * s)   # incl. bobbing


def hud_rects(w, h):
    """Screen areas the HUD covers (score, status lights, Dynamic panel, controls bar)."""
    return [(0, 0, 262, 186), (w - 332, 0, 332, 116), (0, 176, 428, 178), (0, h - 92, w, 92)]


def layout_decorations(cam, *, seed=None, pumpkin_count=None, ghost_count=None, pumpkin_spots=None,
                       ghost_spots=None, gap=None):
    """(ghosts [(x, y, z)], pumpkins [(x, y, z)]) with no overlaps between them or with gameplay."""
    seed = config.THEME_SEED if seed is None else seed
    pumpkin_count = config.PUMPKIN_COUNT if pumpkin_count is None else pumpkin_count
    ghost_count = config.GHOST_COUNT if ghost_count is None else ghost_count
    pumpkin_spots = config.PUMPKIN_SPOTS if pumpkin_spots is None else pumpkin_spots
    ghost_spots = config.GHOST_SPOTS if ghost_spots is None else ghost_spots
    gap = config.DECOR_MIN_GAP_PX if gap is None else gap
    rng = random.Random(seed)
    hulls = gameplay_hulls(cam)
    taken = []
    hud = hud_rects(cam.w, cam.h) if config.DECOR_AVOID_HUD else []
    zf = -config.TABLE_HEIGHT_M

    def fits(rect, on_floor, avoid_hud=True):
        if rect is None:
            return False
        x, y, w, h = rect
        if x < 0 or x + w > cam.w or y < 0 or (on_floor and y + h > cam.h):
            return False
        if any(rect_hits_hull(rect, hull) for hull in hulls):
            return False
        if avoid_hud and any(rects_overlap(rect, r) for r in hud):
            return False
        return not any(rects_overlap(rect, t, gap) for t in taken)

    def place(fixed, count, make, rect_of, kind, on_floor):
        out = []
        if fixed is not None:
            for spot in fixed:
                spot3 = tuple(spot) if len(spot) == 3 else (spot[0], spot[1], zf)
                rect = rect_of(*spot3)
                if fits(rect, on_floor, avoid_hud=False):   # your choice: only gameplay/overlaps block it
                    taken.append(rect)
                    out.append(spot3)
                else:
                    print(f"[theme] {kind} at {tuple(spot)} skipped: it overlaps gameplay or another decoration")
            return out
        for _ in range(count):
            # try to stay clear of the HUD; if there is no room, a spot under a panel is allowed
            for attempt in range(800):
                spot = make()
                rect = rect_of(*spot)
                if fits(rect, on_floor, avoid_hud=attempt < 400):
                    taken.append(rect)
                    out.append(spot)
                    break
        return out

    def side():
        return rng.choice((-1, 1))

    ghosts = place(ghost_spots, ghost_count,
                   lambda: (side() * rng.uniform(2.5, 9.0), rng.uniform(3.0, 14.0), rng.uniform(0.5, 3.2)),
                   lambda x, y, z: ghost_rect(cam, x, y, z), "ghost", False)
    pumpkins = place(pumpkin_spots, pumpkin_count,
                     lambda: (side() * rng.uniform(1.2, 4.5), rng.uniform(0.5, 9.0), zf),
                     lambda x, y, z: pumpkin_rect(cam, x, y), "pumpkin", True)
    return ghosts, sorted(pumpkins, key=lambda q: -q[1])


class _Sprite3D:
    def __init__(self, surf, sx, sy, s):
        self.surf, self.sx, self.sy, self.s = surf, sx, sy, s


class HalloweenAmbient:
    """Animated scenery: bobbing ghosts, flickering pumpkin faces, bats, falling leaves, twinkles."""

    def __init__(self, theme, pg, cam):
        self.pg = pg
        c = theme.c
        rng = random.Random(config.THEME_SEED + 1)
        soften = config.BACKGROUND_SOFTEN
        hz = cam.horizon_y
        theme.layout(cam)
        self.ghosts = []
        for i, (x, y, z) in enumerate(theme.ghost_spots):
            p = cam.project(x, y, z)
            if p is None:
                continue
            sx, sy, s = p
            spr = draw_ghost(pg, 0.55 * s, mix(c["ghost"], c["sky_bottom"], soften * 0.5), i % 3)
            if config.BACKGROUND_BLUR:
                spr = _soft(pg, spr, 2)
            self.ghosts.append((spr, sx, sy, s, rng.uniform(0.5, 0.9), rng.uniform(0, 6.28)))
        self.pumpkins = []
        for i, spr_dim, sx, sy, s in theme.pumpkin_sprites:
            glow = dict(c, pumpkin_glow=mix(c["pumpkin_glow"], c["fog"], soften * 0.4))
            lit = draw_pumpkin(pg, spr_dim.get_width(), glow, i % 3, lit=True)
            if config.BACKGROUND_BLUR:
                lit = _soft(pg, lit, 2)
            self.pumpkins.append((lit, sx, sy, rng.uniform(5, 9), rng.uniform(0, 6.28)))
        bat_col = mix(c["bat"], c["sky_top"], 0.2)
        self.bat_frames = [draw_bat(pg, 40, bat_col, f) for f in (0, 1)]
        self.bats = [(rng.uniform(0, cam.w), rng.uniform(25, max(40, hz - 60)), rng.uniform(30, 60),
                      rng.uniform(0, 6.28)) for _ in range(config.BAT_COUNT if config.AMBIENT_BATS else 0)]
        leaf_cols = c["leaf"]
        self.leaf_imgs = []
        for col in leaf_cols:
            base = draw_leaf(pg, 14, mix(col, c["fog"], soften * 0.4))
            self.leaf_imgs.append([pg.transform.rotate(base, a) for a in range(0, 360, 45)])
        self.leaf_floor = min(cam.h, hz + 140)
        self.leaves = [(rng.uniform(0, cam.w), rng.uniform(0, self.leaf_floor), rng.uniform(18, 35),
                        rng.uniform(0, 6.28), rng.randrange(len(leaf_cols)))
                       for _ in range(config.LEAF_COUNT if config.AMBIENT_LEAVES else 0)]
        self.twinkles = [(x, y, rng.uniform(1.5, 3.5), rng.uniform(0, 6.28)) for x, y in theme.star_spots[:14]]
        self.w = cam.w

    def draw(self, screen, now):
        pg = self.pg
        t = now if config.AMBIENT_ANIMATION else 0.0
        for x, y, f, ph in self.twinkles:
            a = 0.5 + 0.5 * math.sin(t * f + ph)
            if a > 0.6:
                l = int(2 + 3 * a)
                col = (255, 250, 220)
                pg.draw.line(screen, col, (x - l, y), (x + l, y))
                pg.draw.line(screen, col, (x, y - l), (x, y + l))
        for x0, y0, v, ph in self.bats:
            x = (x0 + v * t) % (self.w + 80) - 40
            y = y0 + 10 * math.sin(t * 1.7 + ph)
            img = self.bat_frames[int(t * 7 + ph) % 2]
            screen.blit(img, img.get_rect(center=(x, y)))
        for spr, sx, sy, s, f, ph in self.ghosts:
            dy = math.sin(t * f * 2 * math.pi * 0.5 + ph) * 0.06 * s
            dx = math.sin(t * f * 0.7 + ph) * 0.03 * s
            screen.blit(spr, spr.get_rect(center=(sx + dx, sy + dy)))
        for lit, sx, sy, f, ph in self.pumpkins:
            flick = 0.65 + 0.2 * math.sin(t * f + ph) + 0.15 * math.sin(t * f * 2.3 + ph * 1.7)
            lit.set_alpha(int(255 * max(0.3, min(1.0, flick))))
            screen.blit(lit, lit.get_rect(midbottom=(sx, sy)))
        for x0, y0, v, ph, k in self.leaves:
            y = (y0 + v * t) % self.leaf_floor
            x = x0 + 25 * math.sin(t * 0.9 + ph) + 8 * t % self.w
            x %= self.w
            imgs = self.leaf_imgs[k]
            img = imgs[int(t * 2 + ph * 3) % len(imgs)]
            screen.blit(img, img.get_rect(center=(x, y)))


class HalloweenTheme(Theme):
    name = "halloween"

    def __init__(self):
        self.colors = config.HALLOWEEN_COLORS
        super().__init__()
        self.ghost_spots = self.pumpkin_spots = None   # placed on first use (needs the camera)
        self.pumpkin_sprites = []
        self.star_spots = []

    def layout(self, cam):
        if self.pumpkin_spots is None:
            self.ghost_spots, self.pumpkin_spots = layout_decorations(cam)
        return self.ghost_spots, self.pumpkin_spots

    def draw_sky(self, pg, surf, cam):
        super().draw_sky(pg, surf, cam)
        w = surf.get_width()
        hz = cam.horizon_y
        rng = random.Random(config.THEME_SEED + 2)
        star = self.color("star")
        for _ in range(140):
            x, y = rng.uniform(0, w), rng.uniform(0, hz - 30)
            b = rng.uniform(0.3, 1.0)
            col = mix(self.color("sky_top"), star, b)
            if rng.random() < 0.15:
                pg.draw.circle(surf, col, (int(x), int(y)), 2)
            else:
                surf.set_at((int(x), int(y)), col)
            if b > 0.85:
                self.star_spots.append((int(x), int(y)))

    def draw_scenery(self, pg, surf, cam):
        c = self.c
        w, h = surf.get_size()
        hz = cam.horizon_y
        soften = config.BACKGROUND_SOFTEN
        layer = _alpha_surface(pg, w, h)
        # moon
        r = int(min(58, max(30, hz * 0.3)))
        moon = draw_moon(pg, r, c)
        layer.blit(moon, moon.get_rect(center=(int(w * 0.31), int(max(r + 12, hz * 0.42)))))
        # rolling hills with a couple of curly trees
        for base, amp, col, phase in ((34, 16, c["hill_far"], 1.0), (12, 10, c["hill_near"], 4.0)):
            pts = [(x, hz - (base + amp * math.sin(x * 0.009 + phase) + amp * 0.5 * math.sin(x * 0.023 + phase * 2)))
                   for x in range(0, w + 9, 8)]
            pg.draw.polygon(layer, col, pts + [(w, hz + 3), (0, hz + 3)])
        tree = c["hill_near"]
        for tx in (w * 0.12, w * 0.9):
            gy = hz - 12
            pg.draw.line(layer, tree, (tx, gy), (tx, gy - 60), 6)
            for sign, hgt in ((-1, 40), (1, 50), (-1, 25)):
                ex, ey = tx + sign * 22, gy - hgt - 12
                pg.draw.line(layer, tree, (tx, gy - hgt), (ex, ey), 4)
                pg.draw.arc(layer, tree, (ex - 8 if sign < 0 else ex - 8, ey - 14, 16, 16), 0, math.pi * 1.5, 3)
        if config.BACKGROUND_BLUR:
            layer = _soft(pg, layer, 3)
        surf.blit(layer, (0, 0))
        veil = _alpha_surface(pg, w, int(hz) + 4)
        veil.fill((*c["sky_bottom"], int(80 * soften)))
        surf.blit(veil, (0, 0))

    def draw_floor(self, pg, surf, cam):
        super().draw_floor(pg, surf, cam)
        # pumpkins sit on the floor; their unlit bodies are part of the static backdrop
        muted = dict(self.c)
        for k in ("pumpkin", "pumpkin_dark", "stem"):
            muted[k] = mix(self.c[k], self.c["fog"], config.BACKGROUND_SOFTEN * 0.6)
        self.layout(cam)
        for i, (x, y, z) in enumerate(self.pumpkin_spots):
            p = cam.project(x, y, z)
            if p is None:
                continue
            sx, sy, s = p
            body = draw_pumpkin(pg, 0.42 * s, muted, i % 3, lit=False)
            if config.BACKGROUND_BLUR:
                body = _soft(pg, body, 2)
            shadow = _alpha_surface(pg, body.get_width() * 1.2, body.get_width() * 0.3)
            pg.draw.ellipse(shadow, (10, 5, 20, 110), shadow.get_rect())
            surf.blit(shadow, shadow.get_rect(center=(sx, sy)))
            surf.blit(body, body.get_rect(midbottom=(sx, sy)))
            self.pumpkin_sprites.append((i, body, sx, sy, s))

    def make_ambient(self, pg, cam):
        return HalloweenAmbient(self, pg, cam)

    def icon(self, pg, size):
        body = draw_pumpkin(pg, size, self.c, 0, lit=False)
        body.blit(draw_pumpkin(pg, size, self.c, 0, lit=True), (0, 0))
        return body


THEMES = {"halloween": HalloweenTheme, "classic": ClassicTheme}


def get_theme(name=None):
    name = name or config.THEME
    if name not in THEMES:
        raise ValueError(f"unknown THEME {name!r}; options: {list(THEMES)}")
    return THEMES[name]()
