"""
Static 3D scene pieces, pre-rendered once (the camera never moves):

    backdrop   -- theme sky/scenery/floor + the table's shadow on the floor (opaque)
    table      -- legs, frame, thickness, playing surface and lines (alpha, cropped)
    net        -- mesh, top tape and posts (alpha, cropped)

Per-frame objects (ball, paddles, skeleton) are drawn between these layers by renderer.py.
"""

import config
from theme import _alpha_surface, _soft, mix

LIGHT = (-0.35, -0.45, 0.82)   # moonlight direction (towards the light), for flat shading


def shade(color, normal, ambient=0.55):
    n = sum(a * b for a, b in zip(normal, LIGHT))
    k = ambient + (1 - ambient) * max(0.0, n)
    return tuple(min(255, int(c * k)) for c in color)


def _cropped(pg, surf):
    rect = surf.get_bounding_rect()
    if rect.w == 0 or rect.h == 0:
        return surf.subsurface((0, 0, 1, 1)).copy(), (0, 0)
    return surf.subsurface(rect).copy(), rect.topleft


def box_faces(x0, x1, y0, y1, z0, z1):
    """(normal, 4 corners) for each face of an axis-aligned box."""
    return [
        ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ((0, 1, 0), [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]),
        ((-1, 0, 0), [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)]),
        ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
        ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
        ((0, 0, -1), [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]),
    ]


def draw_box(pg, surf, cam, color, x0, x1, y0, y1, z0, z1, outline=None):
    for normal, pts in box_faces(x0, x1, y0, y1, z0, z1):
        center = tuple(sum(p[i] for p in pts) / 4 for i in range(3))
        if not cam.facing(center, normal):
            continue
        poly = cam.polygon(pts)
        if poly:
            pg.draw.polygon(surf, shade(color, normal), poly)
            if outline:
                pg.draw.polygon(surf, outline, poly, 1)


class Scene:
    def __init__(self, pg, cam, theme):
        self.pg, self.cam, self.theme = pg, cam, theme
        self.hw = config.TABLE_WIDTH_M / 2
        self.L = config.TABLE_LENGTH_M
        self.floor_z = -config.TABLE_HEIGHT_M
        self.backdrop = self._make_backdrop()
        self.table, self.table_pos = _cropped(pg, self._make_table())
        self.net, self.net_pos = _cropped(pg, self._make_net())
        if pg.display.get_surface() is not None:
            self.backdrop = self.backdrop.convert()
            self.table = self.table.convert_alpha()
            self.net = self.net.convert_alpha()

    def leg_positions(self):
        inset_x, inset_y = 0.14, 0.30
        xs = (-self.hw + inset_x, self.hw - inset_x)
        ys = (inset_y, self.L - inset_y)
        return [(x, y) for y in ys for x in xs]

    # -- backdrop + floor shadows ------------------------------------------------------
    def _make_backdrop(self):
        pg, cam = self.pg, self.cam
        surf = self.theme.make_backdrop(pg, cam)
        shadow = _alpha_surface(pg, cam.w, cam.h)
        zf, hw, L = self.floor_z, self.hw, self.L
        off = (0.10, -0.12)  # moonlight pushes the shadow slightly
        poly = cam.polygon([(-hw + off[0], off[1], zf), (hw + off[0], off[1], zf),
                            (hw + off[0], L + off[1], zf), (-hw + off[0], L + off[1], zf)])
        if poly:
            pg.draw.polygon(shadow, (5, 0, 15, 120), poly)
        for x, y in self.leg_positions():
            p = cam.project(x, y, zf)
            if p:
                sx, sy, s = p
                r = 0.09 * s
                pg.draw.ellipse(shadow, (5, 0, 15, 150), (sx - r, sy - r * 0.35, 2 * r, r * 0.7))
        surf.blit(_soft(pg, shadow, 4), (0, 0))
        return surf

    # -- table -------------------------------------------------------------------------
    def _make_table(self):
        pg, cam = self.pg, self.cam
        surf = _alpha_surface(pg, cam.w, cam.h)
        hw, L, zf = self.hw, self.L, self.floor_z
        T = config.TABLE_THICKNESS_M
        leg_col = config.TABLE_LEG_COLOR
        legs = sorted(self.leg_positions(), key=lambda p: -cam.depth(p[0], p[1], zf / 2))
        lw = 0.025
        # side stretchers + end crossbars of the frame
        bar_z = zf + 0.18
        for x in (-hw + 0.14, hw - 0.14):
            draw_box(pg, surf, cam, leg_col, x - 0.012, x + 0.012, 0.30, L - 0.30, bar_z - 0.012, bar_z + 0.012)
        for x, y in legs:
            draw_box(pg, surf, cam, leg_col, x - lw, x + lw, y - lw, y + lw, zf, -T - 0.06)
            # foot
            draw_box(pg, surf, cam, mix(leg_col, (0, 0, 0), 0.4), x - lw * 1.6, x + lw * 1.6,
                     y - lw * 1.6, y + lw * 1.6, zf, zf + 0.02)
        # apron (frame under the top)
        apron = mix(config.TABLE_EDGE_COLOR, (0, 0, 0), 0.4)
        draw_box(pg, surf, cam, apron, -hw + 0.06, hw - 0.06, 0.08, L - 0.08, -T - 0.07, -T)
        # top slab thickness
        draw_box(pg, surf, cam, config.TABLE_EDGE_COLOR, -hw, hw, 0, L, -T, 0)
        # playing surface: shaded strips, lighter toward the far end (moonlit)
        top = config.TABLE_TOP_COLOR
        n = 24
        for i in range(n):
            y0, y1 = L * i / n, L * (i + 1) / n
            k = i / (n - 1)
            col = mix(mix(top, (0, 0, 0), 0.18), mix(top, (255, 255, 255), 0.08), k)
            poly = cam.polygon([(-hw, y0, 0), (hw, y0, 0), (hw, y1, 0), (-hw, y1, 0)])
            if poly:
                pg.draw.polygon(surf, col, poly)
        # soft sheen
        sheen = _alpha_surface(pg, cam.w, cam.h)
        poly = cam.polygon([(-hw * 0.2, L * 0.55, 0), (hw * 0.7, L * 0.55, 0), (hw * 0.5, L * 0.95, 0),
                            (-hw * 0.4, L * 0.95, 0)])
        if poly:
            pg.draw.polygon(sheen, (255, 255, 255, 28), poly)
            surf.blit(_soft(pg, sheen, 6), (0, 0))
        # white lines: 2 cm edges, 3 mm centre line
        line = config.TABLE_LINE_COLOR
        e = 0.02
        for quad in ([(-hw, 0, 0), (-hw + e, 0, 0), (-hw + e, L, 0), (-hw, L, 0)],
                     [(hw - e, 0, 0), (hw, 0, 0), (hw, L, 0), (hw - e, L, 0)],
                     [(-hw, 0, 0), (hw, 0, 0), (hw, e, 0), (-hw, e, 0)],
                     [(-hw, L - e, 0), (hw, L - e, 0), (hw, L, 0), (-hw, L, 0)],
                     [(-0.003, 0, 0), (0.003, 0, 0), (0.003, L, 0), (-0.003, L, 0)]):
            poly = cam.polygon(quad)
            if poly:
                pg.draw.polygon(surf, line, poly)
                pg.draw.aalines(surf, line, True, poly)
        # bright front edge highlight
        a, b = cam.point(-hw, 0, 0), cam.point(hw, 0, 0)
        if a and b:
            pg.draw.line(surf, mix(line, top, 0.3), a, b, 2)
        return surf

    # -- net ---------------------------------------------------------------------------
    def _make_net(self):
        pg, cam = self.pg, self.cam
        surf = _alpha_surface(pg, cam.w, cam.h)
        y = self.L / 2
        hw = self.hw + config.NET_OVERHANG_M
        nh = config.NET_HEIGHT_M
        quad = cam.polygon([(-hw, y, 0), (hw, y, 0), (hw, y, nh), (-hw, y, nh)])
        if quad:
            pg.draw.polygon(surf, (20, 20, 30, 90), quad)
        mesh = (*config.NET_COLOR, 60)
        steps = 40
        for i in range(steps + 1):
            x = -hw + 2 * hw * i / steps
            a, b = cam.point(x, y, 0), cam.point(x, y, nh)
            if a and b:
                pg.draw.line(surf, mesh, a, b)
        for j in range(1, 6):
            z = nh * j / 6
            a, b = cam.point(-hw, y, z), cam.point(hw, y, z)
            if a and b:
                pg.draw.line(surf, mesh, a, b)
        tape = cam.polygon([(-hw, y, nh - 0.015), (hw, y, nh - 0.015), (hw, y, nh), (-hw, y, nh)])
        if tape:
            pg.draw.polygon(surf, (*config.NET_COLOR, 255), tape)
        post = config.NET_POST_COLOR
        for x in (-hw, hw):
            draw_box(pg, surf, cam, post, x - 0.012, x + 0.012, y - 0.012, y + 0.012, -0.03, nh + 0.01)
            # clamp onto the table edge
            cx = -self.hw if x < 0 else self.hw
            draw_box(pg, surf, cam, post, min(x, cx) - 0.01, max(x, cx) + 0.01, y - 0.02, y + 0.02, -0.03, 0.0)
        return surf
