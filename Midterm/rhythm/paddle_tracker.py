"""
On-screen TILT and FACE of the paddle for the bridge, from the Double Motor's IMU (the hub IS the
handle). Pure Python, no hardware: fed one ImuSample at a time on the motor's BLE thread.

The paddle's three directions, in hub coordinates, come from two gravity readings
(bridge_config.TRACK_NEUTRAL_GRAVITY / TRACK_TILT_RIGHT_GRAVITY, found with bridge.py --imu-debug):

    H  the accel reading in the neutral grip      (along the handle)
    W  the accel reading after a 90 deg tilt right (across the face)
    N  = W x H                                    (face normal: toward the screen in the neutral grip)

  * FLIP: phi = the gyro about H, integrated. Accel can't see a twist about an upright handle,
    so while the paddle is still phi settles toward the face it is showing (0 or 180 deg).
    face = black / yellow from |phi| with hysteresis (TRACK_FLIP_TO_YELLOW_DEG / _TO_BLACK_DEG)
    and a debounce.
  * TILT: the rotation in the screen plane. Undoing the flip (rotating the readings by phi about
    H) gives what the hub would read in the black-forward grip, so the tilt is the same real
    direction on either face and stays continuous through a flip:
        accel:  theta = atan2(v . W, v . H)     (absolute: gravity is the reference)
        gyro:   d(theta)/dt = (R_H(phi) omega) . N
    blended with a complementary filter (accel only while |a| is near 1 g and gravity lies
    mostly in the screen plane).
  * During a swing (or a haptic pulse) and TRACK_SWING_HOLD_S after it, tilt and flip freeze;
    at swing onset both rewind TRACK_SWING_REWIND_S, because the detector trips a few samples
    into the motion.

output = (tilt_deg, face): tilt relative to the zeroed grip, + = top of the paddle to the
player's right (Godot: clockwise), after deadzone, gain, smoothing and the +/- clamp.
"""

import math
import threading
from collections import deque

import bridge_config as bc

_AXES = {"x": 0, "y": 1, "z": 2}


def axis_vector(spec):
    """'+x' / '-z' / 'y' -> unit vector in hub coordinates."""
    s = str(spec).strip().lower()
    sign = -1.0 if s.startswith("-") else 1.0
    name = s.lstrip("+-")
    if name not in _AXES:
        raise ValueError(f"axis must look like '+x', '-y' or 'z', got {spec!r}")
    v = [0.0, 0.0, 0.0]
    v[_AXES[name]] = sign
    return tuple(v)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def rotate(v, axis, deg):
    """Rodrigues: v rotated by +deg (right hand) about the unit vector axis."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    kxv = cross(axis, v)
    kdv = dot(axis, v)
    return tuple(v[i] * c + kxv[i] * s + axis[i] * kdv * (1 - c) for i in range(3))


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def paddle_axes(neutral=None, tilt_right=None):
    """(H, W, N) unit vectors in hub coordinates from the two gravity readings."""
    h = axis_vector(bc.TRACK_NEUTRAL_GRAVITY if neutral is None else neutral)
    w = axis_vector(bc.TRACK_TILT_RIGHT_GRAVITY if tilt_right is None else tilt_right)
    if abs(dot(h, w)) > 0.5:
        raise ValueError("TRACK_NEUTRAL_GRAVITY and TRACK_TILT_RIGHT_GRAVITY must be different hub axes")
    return h, w, cross(w, h)


class PaddleTracker:
    """update() on the BLE thread; output / request_zero() from any thread."""

    def __init__(self, raw_per_g=None, **overrides):
        p = {k: getattr(bc, k) for k in dir(bc) if k.startswith("TRACK_")}
        for k, v in overrides.items():
            key = "TRACK_" + k.upper()
            if key not in p:
                raise TypeError(f"unknown tracker setting {k!r}")
            p[key] = v
        self.p = p
        self.H, self.W, self.N = paddle_axes(p["TRACK_NEUTRAL_GRAVITY"], p["TRACK_TILT_RIGHT_GRAVITY"])
        self.gyro_scale = p["TRACK_GYRO_DEG_PER_RAW"] * p["TRACK_GYRO_SIGN"]
        self._lock = threading.Lock()
        self._zero_requested = False
        self.reset(raw_per_g)

    # -- lifecycle ----------------------------------------------------------------
    def reset(self, raw_per_g):
        """New calibration (or reconnect): start over; the first sample becomes the zero."""
        self.raw_per_g = raw_per_g
        self.theta = None              # filtered tilt, absolute (deg)
        self.theta0 = 0.0              # zeroed grip
        self.phi = 0.0                 # flip angle about the handle (deg, wrapped)
        self.face = "black"
        self.shown = 0.0               # smoothed, scaled, clamped output tilt
        self.bias = (0.0, 0.0, 0.0)    # gyro bias, deg/s in hub coordinates
        self.accel_theta = None        # latest accel tilt (debug), None if not trustworthy
        self.still = False
        self.holding = False
        self._pending_face = None
        self._pending_since = None
        self._last_t = None
        self._was_swinging = False
        self._hold_until = -math.inf
        self._hist = deque()
        with self._lock:
            self._zero_requested = True
            self._out = (0.0, "black")

    def request_zero(self):
        """The current grip becomes zero tilt, black face (applied with the next sample)."""
        with self._lock:
            self._zero_requested = True

    @property
    def ready(self):
        return self.raw_per_g is not None and self.theta is not None

    @property
    def output(self):
        with self._lock:
            return self._out

    # -- per sample (BLE thread) --------------------------------------------------
    def gyro_dps(self, s):
        """Bias-corrected angular rate in deg/s, hub coordinates."""
        k = self.gyro_scale
        return (s.gx * k - self.bias[0], s.gy * k - self.bias[1], s.gz * k - self.bias[2])

    def accel_tilt(self, f):
        """Tilt from the flip-undone accel vector, or None when the accel can't be trusted."""
        g = self.raw_per_g
        mag = math.sqrt(dot(f, f))
        if abs(mag / g - 1.0) > self.p["TRACK_ACCEL_TOLERANCE_G"]:
            return None
        v = rotate(f, self.H, self.phi)
        a_w, a_h = dot(v, self.W), dot(v, self.H)
        if math.hypot(a_w, a_h) < self.p["TRACK_MIN_PLANE_GRAVITY"] * g:
            return None                            # pointing at / away from the screen: unobservable
        return math.degrees(math.atan2(a_w, a_h))

    def update(self, s, quiet=False, swinging=False):
        if self.raw_per_g is None:
            return
        p = self.p
        dt = 0.0 if self._last_t is None else min(0.1, max(0.0, s.t - self._last_t))
        self._last_t = s.t
        f = (s.ax, s.ay, s.az)
        omega = self.gyro_dps(s)

        # -- hold: swing / haptic pulse, and a moment after ------------------------
        if swinging and not self._was_swinging:
            self._rewind(s.t - p["TRACK_SWING_REWIND_S"])
        self._was_swinging = swinging
        if swinging or quiet:
            self._hold_until = s.t + p["TRACK_SWING_HOLD_S"]
        self.holding = s.t < self._hold_until
        with self._lock:
            zero = self._zero_requested
            self._zero_requested = False

        if self.theta is None or zero:
            self._zero(f)
        elif not self.holding:
            self._track(s, f, omega, dt)
        self._remember(s.t)
        self._publish(dt)

    def _track(self, s, f, omega, dt):
        p = self.p
        gyro_mag = math.sqrt(dot(omega, omega))
        self.still = (abs(math.sqrt(dot(f, f)) / self.raw_per_g - 1.0) <= p["TRACK_ACCEL_TOLERANCE_G"]
                      and gyro_mag < p["TRACK_STILL_GYRO_DPS"])

        # gyro bias: learnt only while really still
        if gyro_mag < p["TRACK_BIAS_STILL_DPS"] and self.still:
            k = min(1.0, dt / p["TRACK_BIAS_TAU_S"])
            raw = (s.gx * self.gyro_scale, s.gy * self.gyro_scale, s.gz * self.gyro_scale)
            self.bias = tuple(b + (r - b) * k for b, r in zip(self.bias, raw))

        # FLIP: twist about the handle
        rate_h = dot(omega, self.H)
        if abs(rate_h) < p["TRACK_GYRO_DEADZONE_DPS"]:
            rate_h = 0.0
        self.phi = wrap_deg(self.phi + rate_h * dt)
        if self.still:                             # settle to the face being shown (kills drift)
            target = 0.0 if self.face == "black" else math.copysign(180.0, self.phi)
            self.phi = wrap_deg(self.phi + wrap_deg(target - self.phi) * min(1.0, dt / p["TRACK_FLIP_SETTLE_S"]))
        self._update_face(s.t)

        # TILT: gyro in the screen plane, pulled toward the accelerometer
        rate_n = dot(rotate(omega, self.H, self.phi), self.N)
        self.theta = wrap_deg(self.theta + rate_n * dt)
        self.accel_theta = self.accel_tilt(f)
        if self.accel_theta is not None:
            k = min(1.0, dt / p["TRACK_ACCEL_TAU_S"])
            self.theta = wrap_deg(self.theta + k * wrap_deg(self.accel_theta - self.theta))

    def _update_face(self, t):
        p = self.p
        a = abs(self.phi)
        want = self.face
        if self.face == "black" and a > p["TRACK_FLIP_TO_YELLOW_DEG"]:
            want = "yellow"
        elif self.face == "yellow" and a < p["TRACK_FLIP_TO_BLACK_DEG"]:
            want = "black"
        if want == self.face:
            self._pending_face = None
            return
        if self._pending_face != want:
            self._pending_face, self._pending_since = want, t
        if t - self._pending_since >= p["TRACK_FLIP_DEBOUNCE_S"]:
            self.face = want
            self._pending_face = None

    def _zero(self, f):
        self.phi = 0.0
        self.face = "black"
        self._pending_face = None
        a = self.accel_tilt(f)
        self.theta = a if a is not None else (self.theta if self.theta is not None else 0.0)
        self.accel_theta = a
        self.theta0 = self.theta
        self.shown = 0.0
        self._hist.clear()

    # -- swing rewind ----------------------------------------------------------------
    def _remember(self, t):
        self._hist.append((t, self.theta, self.phi, self.face, self.shown))
        horizon = max(0.3, 2 * self.p["TRACK_SWING_REWIND_S"])
        while self._hist and t - self._hist[0][0] > horizon:
            self._hist.popleft()

    def _rewind(self, t):
        for entry in self._hist:                   # oldest first: the last one at or before t
            if entry[0] > t:
                break
            _, self.theta, self.phi, self.face, self.shown = entry
        self._pending_face = None

    # -- output ----------------------------------------------------------------------
    def _publish(self, dt):
        p = self.p
        d = wrap_deg(self.theta - self.theta0)
        mag = max(0.0, abs(d) - p["TRACK_DEADZONE_DEG"]) * p["TRACK_GAIN"]
        target = math.copysign(min(p["TRACK_TILT_MAX_DEG"], mag), d)
        if not self.holding:
            k = 1.0 if p["TRACK_SMOOTH_S"] <= 0 else min(1.0, dt / p["TRACK_SMOOTH_S"])
            self.shown += (target - self.shown) * k
        with self._lock:
            self._out = (self.shown, self.face)
