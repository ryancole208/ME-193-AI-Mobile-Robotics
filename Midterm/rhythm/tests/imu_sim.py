"""Synthetic Double Motor IMU samples from a known paddle motion (rigid-body physics, no hardware).

World frame: R = player's right, U = up, B = toward the player (x, y, z, right-handed); the screen
is in front of the player (-B). The paddle's pose at time t is

    Q(t) = Rot_world(F, tilt) . Q0 . Rot_hub(H, flip) . Rot_hub(W, pitch)

Q0 maps the hub frame to the world in the neutral grip (black toward the screen), built from the
same two gravity axes the tracker is configured with. Tilt + = top to the player's right
(clockwise as the player sees it); pitch = the forward swing about the axis across the face.
The accelerometer reads +1 g "up" at rest (specific force); the gyro is the right-handed body rate.
"""

import math

from motor_imu import ImuSample
from paddle_tracker import dot, paddle_axes

G = 1000.0               # raw units per g
DEG_PER_RAW = 0.1        # gyro scale used to make raw values
DT = 0.02                # 50 Hz, like the hub

UP = (0.0, 1.0, 0.0)
FORWARD = (0.0, 0.0, -1.0)      # toward the screen
LEFT = (-1.0, 0.0, 0.0)


def mat_from_cols(c0, c1, c2):
    return [[c0[i], c1[i], c2[i]] for i in range(3)]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def apply(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def rot(axis, deg):
    """Right-handed rotation matrix about a unit axis."""
    x, y, z = axis
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    C = 1 - c
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


class Paddle:
    def __init__(self, neutral="+x", tilt_right="+y", mount_deg=0.0):
        self.H, self.W, self.N = paddle_axes(neutral, tilt_right)
        # hub H -> up, W -> player's left (it rises when the top tilts right), N = W x H -> the screen
        self.Q0 = mat_from_cols(*[self._world_of(i) for i in range(3)])
        if mount_deg:      # hub mounted a little crooked in the handle (an in-plane offset)
            self.Q0 = matmul(rot(FORWARD, mount_deg), self.Q0)
        self.t = 0.0
        self.tilt = self.flip = self.pitch = 0.0
        self.extra_accel = (0.0, 0.0, 0.0)      # linear acceleration, world frame, in g
        self.gyro_bias = (0.0, 0.0, 0.0)        # raw units

    def _world_of(self, hub_axis):
        """World direction of hub axis e_i in the neutral grip."""
        e = [0.0, 0.0, 0.0]
        e[hub_axis] = 1.0
        return tuple(dot(e, self.H) * UP[k] + dot(e, self.W) * LEFT[k] + dot(e, self.N) * FORWARD[k]
                     for k in range(3))

    def orientation(self, tilt, flip, pitch):
        q = matmul(rot(FORWARD, tilt), self.Q0)
        q = matmul(q, rot(self.H, flip))
        return matmul(q, rot(self.W, pitch))

    def sample_to(self, tilt=None, flip=None, pitch=None, accel=None):
        """Move to the new pose over one DT; returns the sample at the end of the step."""
        q_a = self.orientation(self.tilt, self.flip, self.pitch)
        self.tilt = self.tilt if tilt is None else tilt
        self.flip = self.flip if flip is None else flip
        self.pitch = self.pitch if pitch is None else pitch
        self.extra_accel = self.extra_accel if accel is None else accel
        q_b = self.orientation(self.tilt, self.flip, self.pitch)
        m = matmul(transpose(q_a), q_b)            # = exp([omega] dt), body frame
        omega = ((m[2][1] - m[1][2]) / (2 * DT), (m[0][2] - m[2][0]) / (2 * DT), (m[1][0] - m[0][1]) / (2 * DT))
        self.t += DT
        sf = tuple(UP[i] + self.extra_accel[i] for i in range(3))
        a = apply(transpose(q_b), sf)
        g = tuple(math.degrees(w) / DEG_PER_RAW + b for w, b in zip(omega, self.gyro_bias))
        return ImuSample(self.t, a[0] * G, a[1] * G, a[2] * G, *g)

    def hold(self, seconds, **kw):
        return [self.sample_to(**kw) for _ in range(round(seconds / DT))]

    def move(self, seconds, **targets):
        """Linear move of tilt / flip / pitch to the targets over `seconds`."""
        n = max(1, round(seconds / DT))
        accel = targets.pop("accel", None)         # held during the move, not interpolated
        start = {k: getattr(self, k) for k in targets}
        return [self.sample_to(accel=accel, **{k: start[k] + (v - start[k]) * (i + 1) / n
                                               for k, v in targets.items()})
                for i in range(n)]


def feed(tracker, samples, swinging=False, quiet=False):
    """Feed samples; returns the output after each one."""
    out = []
    for s in samples:
        tracker.update(s, quiet=quiet, swinging=swinging)
        out.append(tracker.output)
    return out
