"""
3D table-tennis paddle: rounded blade with visible thickness (wood edge), and a handle.

Paddle frame (before any rotation): the blade is vertical, its FRONT face points toward the
opponent (+y), the handle hangs down (-z). Roll is the twist about the handle axis --
positive roll turns the front face toward +x (the player's right). The face drawn is the one
facing the camera, so twisting past edge-on flips between the front and back colours.

My paddle (PADDLE_FACE_INVERT): in the neutral grip the real paddle has BLACK toward the screen,
so the model's front is black and its back -- the face the camera sees -- is yellow. Only the
colours swap: a twist is the same physical rotation whichever face points where, so the roll
sign is unchanged.

The skeleton builds its own basis from its forearm (draw_basis) and places the blade so the
handle sits in its hand (center_for_grip).
"""

import math

import config
from theme import _alpha_surface

BLADE_POINTS = 28


def _rot(v, axis, deg):
    """Rodrigues rotation of vector v about unit axis by deg."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    kx, ky, kz = axis
    vx, vy, vz = v
    dot = kx * vx + ky * vy + kz * vz
    cx, cy, cz = ky * vz - kz * vy, kz * vx - kx * vz, kx * vy - ky * vx
    return (vx * c + cx * s + kx * dot * (1 - c),
            vy * c + cy * s + ky * dot * (1 - c),
            vz * c + cz * s + kz * dot * (1 - c))


def _add(p, *terms):
    x, y, z = p
    for k, v in terms:
        x, y, z = x + k * v[0], y + k * v[1], z + k * v[2]
    return x, y, z


def _shade(color, k):
    return tuple(max(0, min(255, int(c * k))) for c in color)


def blade_outline(n=BLADE_POINTS):
    """(u, v) unit-ish outline of a blade: a slightly squarer-than-elliptical shape."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        c, s = math.cos(t), math.sin(t)
        u = math.copysign(abs(s) ** 0.85, s) * 0.5
        v = math.copysign(abs(c) ** 0.9, c) * 0.5
        pts.append((u, v))
    return pts


_OUTLINE = blade_outline()


class Paddle:
    def __init__(self, front, back, wood=None, rim=None, scale=None):
        self.front, self.back = front, back
        self.wood = wood or config.PADDLE_WOOD
        self.rim = rim
        self.scale = config.PADDLE_DRAW_SCALE if scale is None else scale

    def basis(self, roll_deg, base_yaw_deg=0.0, tilt_deg=0.0, lean_deg=0.0):
        """(u across the face, v along the handle toward the blade tip, n front normal)."""
        u, v, n = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)
        yaw = base_yaw_deg + roll_deg        # roll about the handle axis (vertical here)
        u, n = _rot(u, v, -yaw), _rot(n, v, -yaw)
        if lean_deg:                          # handle leaning sideways (in the face plane)
            u, v = _rot(u, n, lean_deg), _rot(v, n, lean_deg)
        if tilt_deg:                          # blade tip tipping forward/back (about u)
            v, n = _rot(v, u, tilt_deg), _rot(n, u, tilt_deg)
        return u, v, n

    def front_visible(self, cam, center, roll_deg, **kw):
        _, _, n = self.basis(roll_deg, **kw)
        return cam.facing(center, n)

    def center_for_grip(self, grip, v, along=None):
        """Blade centre that puts the point `along` (0 = blade end, 1 = butt) of the handle at grip."""
        along = config.SKELETON_GRIP_ALONG_HANDLE if along is None else along
        k = self.scale
        d = config.PADDLE_BLADE_H_M * k * 0.42 + config.PADDLE_HANDLE_L_M * k * along
        return _add(grip, (d, v))

    def draw(self, screen, cam, center, roll_deg, *, base_yaw_deg=0.0, tilt_deg=0.0, lean_deg=0.0,
             alpha=255):
        """Draw the paddle with its blade centred at `center` (world metres)."""
        self.draw_basis(screen, cam, center, self.basis(roll_deg, base_yaw_deg, tilt_deg, lean_deg), alpha=alpha)

    def draw_basis(self, screen, cam, center, basis, *, alpha=255):
        """Draw with an explicit (u, v, n) basis: u across the face, v toward the blade tip, n front normal."""
        import pygame as pg
        u, v, n = basis
        k = self.scale
        W, H = config.PADDLE_BLADE_W_M * k, config.PADDLE_BLADE_H_M * k
        T = config.PADDLE_THICKNESS_M * k
        front_vis = cam.facing(center, n)
        near_sign = 1 if front_vis else -1

        def face_pts(sign):
            off = sign * T / 2
            return [_add(center, (pu * W, u), (pv * H, v), (off, n)) for pu, pv in _OUTLINE]

        near3, far3 = face_pts(near_sign), face_pts(-near_sign)
        near = [cam.point(*p) for p in near3]
        far = [cam.point(*p) for p in far3]
        if any(p is None for p in near + far):
            return

        # handle: a flared box hanging off the blade bottom
        hl, hwid, hth = config.PADDLE_HANDLE_L_M * k, 0.028 * k, 0.024 * k
        top = _add(center, (-H * 0.42, v))
        handle = []
        for along, half_w in ((0.0, hwid * 0.55), (hl, hwid * 0.5)):
            for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                handle.append(_add(top, (-along, v), (su * half_w, u), (sn * hth / 2, n)))
        hpts = [cam.point(*p) for p in handle]
        if any(p is None for p in hpts):
            return

        all_pts = near + far + hpts
        xs, ys = [p[0] for p in all_pts], [p[1] for p in all_pts]
        pad = 10
        x0, y0 = int(min(xs)) - pad, int(min(ys)) - pad
        w, h = int(max(xs)) - x0 + pad, int(max(ys)) - y0 + pad
        if w <= 0 or h <= 0 or w > 4000 or h > 4000:
            return
        surf = _alpha_surface(pg, w, h)
        o = lambda pts: [(x - x0, y - y0) for x, y in pts]

        wood = self.wood
        # far face (only seen around the edge) and the edge band
        pg.draw.polygon(surf, _shade(wood, 0.6), o(far))
        m = len(near)
        for i in range(m):
            j = (i + 1) % m
            mid = tuple((near3[i][a] + near3[j][a]) / 2 for a in range(3))
            outward = tuple(mid[a] - center[a] for a in range(3))
            if not cam.facing(mid, outward):
                continue
            lit = 0.8 + 0.3 * outward[2] / (math.sqrt(sum(c * c for c in outward)) or 1)
            pg.draw.polygon(surf, _shade(wood, lit), o([near[i], near[j], far[j], far[i]]))

        face_col = self.front if front_vis else self.back
        if self.rim:
            glow = _alpha_surface(pg, w, h)
            pg.draw.polygon(glow, (*self.rim, 255), o(near), 0)
            pg.draw.lines(glow, (*self.rim, 255), True, o(near), 10)
            glow = pg.transform.smoothscale(pg.transform.smoothscale(glow, (max(1, w // 4), max(1, h // 4))), (w, h))
            glow.set_alpha(150)
            surf.blit(glow, (0, 0))
        pg.draw.polygon(surf, face_col, o(near))
        # rubber sheen: a lighter band on the upper half
        facing_amount = abs(sum((c - p) * nn for c, p, nn in zip(cam.position, center, n)))
        if len(near) > 8 and facing_amount > 0.2:
            sheen = _shade(face_col, 1.35) if sum(face_col) > 120 else (70, 70, 80)
            q = len(near) // 4
            band = [near[i] for i in range(-q + 1, q)]
            cx = sum(p[0] for p in near) / m
            cy = sum(p[1] for p in near) / m
            inner = [((p[0] - cx) * 0.7 + cx, (p[1] - cy) * 0.7 + cy) for p in band]
            if len(inner) >= 3:
                pg.draw.polygon(surf, sheen, o(inner))
        rim_col = self.rim or _shade(wood, 0.8)
        pg.draw.lines(surf, rim_col, True, o(near), 2)

        # handle faces: (indices of the 8 corners) near end 0-3, far end 4-7
        faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
        hcenter = tuple(sum(p[a] for p in handle) / 8 for a in range(3))
        drawn = []
        for idx in faces:
            pts3 = [handle[i] for i in idx]
            fc = tuple(sum(p[a] for p in pts3) / 4 for a in range(3))
            normal = tuple(fc[a] - hcenter[a] for a in range(3))
            if cam.facing(fc, normal):
                drawn.append((cam.depth(*fc), idx, normal))
        for _, idx, normal in sorted(drawn, reverse=True):
            nz = normal[2] / (math.sqrt(sum(c * c for c in normal)) or 1)
            pg.draw.polygon(surf, _shade(wood, 0.8 + 0.25 * nz), o([hpts[i] for i in idx]))
            pg.draw.polygon(surf, _shade(wood, 0.5), o([hpts[i] for i in idx]), 1)

        if alpha < 255:
            surf.set_alpha(alpha)
        screen.blit(surf, (x0, y0))


def player_paddle(invert=None):
    """My paddle. Mirrored (default): the model's front (toward the screen) is black, so the camera
    sees yellow in the neutral grip -- the face that points at me in real life."""
    invert = config.PADDLE_FACE_INVERT if invert is None else invert
    yellow, black = config.PLAYER_PADDLE_FRONT, config.PLAYER_PADDLE_BACK
    front, back = (black, yellow) if invert else (yellow, black)
    return Paddle(front, back, rim=config.PADDLE_RIM_COLOR if config.PADDLE_RIM_GLOW else None)


def opponent_paddle():
    return Paddle(config.OPPONENT_PADDLE_FRONT, config.OPPONENT_PADDLE_BACK, scale=config.PADDLE_DRAW_SCALE * 1.1)


def swing_pose(phase, direction="center"):
    """Offsets for the swing animation at phase 0..1: (dx, dy, dz, tilt_deg)."""
    if phase < 0 or phase > 1:
        return 0.0, 0.0, 0.0, 0.0
    a = math.sin(math.pi * phase)
    side = {"left": -1, "right": 1}.get(direction, 0)
    return 0.08 * side * a, 0.30 * a, 0.10 * a, -40.0 * a
