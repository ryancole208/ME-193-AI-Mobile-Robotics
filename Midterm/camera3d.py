"""
Perspective camera for the pseudo-3D view.

World coordinates (metres) are the game's: x lateral (right +), y along the table toward
the opponent, z height above the table surface (the floor is at -TABLE_HEIGHT_M). The camera
sits behind and above my end, looks down the table (+y) and is pitched down.
"""

import math

import config


class Camera:
    def __init__(self, width, height, *, x=None, back=None, cam_height=None, pitch_deg=None,
                 fov_deg=None, center_y=None, near=None):
        self.w, self.h = width, height
        self.x = config.VIEW_CAMERA_X_M if x is None else x
        self.y = -(config.VIEW_CAMERA_BACK_M if back is None else back)
        self.z = config.VIEW_CAMERA_HEIGHT_M if cam_height is None else cam_height
        self.pitch = config.VIEW_CAMERA_PITCH_DEG if pitch_deg is None else pitch_deg
        fov = config.VIEW_FOV_DEG if fov_deg is None else fov_deg
        self.focal = (width / 2) / math.tan(math.radians(fov) / 2)
        self.cy = config.VIEW_CENTER_Y_PX if center_y is None else center_y
        self.near = config.VIEW_NEAR_M if near is None else near
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        self.right = (1.0, 0.0, 0.0)
        self.fwd = (0.0, cp, -sp)
        self.up = (0.0, sp, cp)

    @property
    def position(self):
        return self.x, self.y, self.z

    @property
    def horizon_y(self):
        """Screen row of the horizon (infinitely far level ground)."""
        return self.cy - self.focal * math.tan(math.radians(self.pitch))

    def to_cam(self, X, Y, Z=0.0):
        dx, dy, dz = X - self.x, Y - self.y, Z - self.z
        r, u, f = self.right, self.up, self.fwd
        return (dx * r[0] + dy * r[1] + dz * r[2],
                dx * u[0] + dy * u[1] + dz * u[2],
                dx * f[0] + dy * f[1] + dz * f[2])

    def depth(self, X, Y, Z=0.0):
        return self.to_cam(X, Y, Z)[2]

    def screen(self, c):
        xc, yc, zc = c
        return self.w / 2 + self.focal * xc / zc, self.cy - self.focal * yc / zc

    def project(self, X, Y, Z=0.0):
        """(screen x, screen y, px per metre) or None if behind the near plane."""
        c = self.to_cam(X, Y, Z)
        if c[2] < self.near:
            return None
        sx, sy = self.screen(c)
        return sx, sy, self.focal / c[2]

    def point(self, X, Y, Z=0.0):
        p = self.project(X, Y, Z)
        return None if p is None else (p[0], p[1])

    def polygon(self, pts):
        """Screen polygon for 3D points, clipped at the near plane (None if fully behind)."""
        cam = [self.to_cam(*p) for p in pts]
        out = []
        for i, a in enumerate(cam):
            b = cam[(i + 1) % len(cam)]
            ina, inb = a[2] >= self.near, b[2] >= self.near
            if ina:
                out.append(a)
            if ina != inb:
                k = (self.near - a[2]) / (b[2] - a[2])
                out.append(tuple(a[j] + (b[j] - a[j]) * k for j in range(3)))
        if len(out) < 3:
            return None
        return [self.screen(c) for c in out]

    def facing(self, point, normal):
        """True if a surface at `point` with `normal` faces the camera."""
        return sum((c - p) * n for c, p, n in zip(self.position, point, normal)) > 0
