"""
Spin: handle-roll estimation, swing -> spin mapping and the spin's effect on a flight.

Pure Python (no pygame, no BLE) so it is fully unit-tested.

  * RollFilter      -- complementary filter for the twist of the handle (the hub IS the
                       handle). Integrates the gyro roll rate and, only while the paddle is
                       roughly still (|a| near 1 g), pulls slowly toward the accelerometer's
                       gravity-based roll. Swing accelerations swamp gravity, so mid-swing
                       the estimate is gyro-only.
  * spin_for_swing  -- SwingEvent -> Spin(side, top), using SPIN_INPUT ("rate"/"angle"/"both")
  * flight_params   -- Spin -> bounce point, arc heights, post-bounce speed
  * magnus_curve_m  -- sideways displacement a sidespin adds over a flight
"""

import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

import config

AXES = {"x": 0, "y": 1, "z": 2}
# The two axes across the handle, in cyclic order (right-handed rotation about the roll axis).
_CROSS = {0: (1, 2), 1: (2, 0), 2: (0, 1)}

BOUNCE_FRACTION = 0.70   # where along a no-spin flight the ball bounces (receiver's side)
SECOND_ARC_RATIO = 0.6   # height of the post-bounce arc relative to the first


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =============================================================================
# Spin value
# =============================================================================

@dataclass(frozen=True)
class Spin:
    side: float = 0.0   # > 0 curves right (+x)
    top: float = 0.0    # > 0 topspin, < 0 backspin

    @property
    def magnitude(self):
        return math.hypot(self.side, self.top)

    @property
    def kind(self):
        """'none', 'side', 'top' or 'back' -- whichever component dominates."""
        if self.magnitude < 1e-6:
            return "none"
        if abs(self.side) >= abs(self.top):
            return "side"
        return "top" if self.top > 0 else "back"

    def clamped(self, m=None):
        m = config.SPIN_MAX if m is None else m
        return Spin(clamp(self.side, -m, m), clamp(self.top, -m, m))


NO_SPIN = Spin()


def spin_label(spin, threshold=None):
    """Text shown for a strong spin shot, or None."""
    threshold = config.SPIN_LABEL_THRESHOLD if threshold is None else threshold
    if spin.magnitude < threshold:
        return None
    if spin.kind == "side":
        return "CURVE RIGHT!" if spin.side > 0 else "CURVE LEFT!"
    return "TOPSPIN!" if spin.kind == "top" else "BACKSPIN!"


# =============================================================================
# Swing -> spin
# =============================================================================

def deadzone_scale(value, deadzone, full):
    """0 inside the dead zone, then linear to +/-1 at `full`, signed, not clamped."""
    mag = abs(value)
    if mag <= deadzone:
        return 0.0
    return math.copysign((mag - deadzone) / max(1e-9, full - deadzone), value)


def side_from_imu(roll_rate_dps, roll_deg, mode=None):
    mode = mode or config.SPIN_INPUT
    if mode not in ("rate", "angle", "both"):
        raise ValueError(f"bad SPIN_INPUT {mode!r}")
    rate = deadzone_scale(roll_rate_dps, config.SPIN_RATE_DEADZONE_DPS, config.SPIN_RATE_FULL_DPS)
    angle = deadzone_scale(roll_deg, config.SPIN_ANGLE_DEADZONE_DEG, config.SPIN_ANGLE_FULL_DEG)
    side = {"rate": rate, "angle": angle, "both": rate + angle}[mode]
    return config.SPIN_SIDE_SIGN * side


def top_from_imu(vertical_g):
    if config.SPIN_TOP_SOURCE == "off":
        return 0.0
    return deadzone_scale(vertical_g, config.SPIN_TOP_DEADZONE_G, config.SPIN_TOP_FULL_G)


def spin_for_swing(swing, mode=None):
    """Spin carried by a SwingEvent.

    Keyboard swings set spin_side/spin_top directly; IMU swings are mapped from the
    twist rate / roll angle (sidespin) and the vertical swing motion (top/backspin).
    """
    if getattr(swing, "spin_side", None) is not None or getattr(swing, "spin_top", None) is not None:
        return Spin(swing.spin_side or 0.0, swing.spin_top or 0.0).clamped()
    if getattr(swing, "source", "") != "imu":
        return NO_SPIN
    k = config.SPIN_SENSITIVITY
    side = side_from_imu(swing.roll_rate_dps, swing.roll_deg, mode)
    top = top_from_imu(swing.vertical_g)
    return Spin(k * side, k * top).clamped()


# =============================================================================
# Spin -> flight shape
# =============================================================================

@dataclass(frozen=True)
class FlightParams:
    bounce_fraction: float = BOUNCE_FRACTION   # where (fraction of the distance) the ball bounces
    arc_mult: float = 1.0                      # first-arc height multiplier
    second_arc_ratio: float = SECOND_ARC_RATIO  # bounce-arc height relative to the base arc
    post_speed_mult: float = 1.0               # speed along the table after the bounce


def flight_params(spin):
    t = spin.top
    if t >= 0:   # topspin: dips earlier, low skidding bounce that kicks forward
        return FlightParams(
            bounce_fraction=clamp(BOUNCE_FRACTION - config.TOPSPIN_BOUNCE_SHIFT * t, 0.5, 0.85),
            arc_mult=max(0.3, 1 - config.TOPSPIN_ARC_MULT * t),
            second_arc_ratio=SECOND_ARC_RATIO * max(0.25, 1 - config.TOPSPIN_SECOND_ARC_MULT * t),
            post_speed_mult=1 + config.TOPSPIN_KICK * t)
    b = -t       # backspin: floats, bounces later, sits up and checks
    return FlightParams(
        bounce_fraction=clamp(BOUNCE_FRACTION + config.BACKSPIN_BOUNCE_SHIFT * b, 0.5, 0.85),
        arc_mult=1 + config.BACKSPIN_ARC_MULT * b,
        second_arc_ratio=SECOND_ARC_RATIO * (1 + config.BACKSPIN_SECOND_ARC_MULT * b),
        post_speed_mult=max(0.4, 1 - config.BACKSPIN_CHECK * b))


def flight_duration(distance, speed, params):
    d1 = distance * params.bounce_fraction
    return d1 / speed + (distance - d1) / (speed * params.post_speed_mult)


def magnus_curve_m(side, speed, duration):
    """Sideways displacement at the end of a flight from a constant Magnus push."""
    accel = config.MAGNUS_STRENGTH * side * speed
    return 0.5 * accel * duration ** 2


# =============================================================================
# Handle roll: complementary filter
# =============================================================================

class RollFilter:
    """Roll (twist) of the handle in degrees, relative to the neutral grip.

    update() takes one ImuSample. Positive roll = the screen-side face (black, in the neutral grip) turning to the player's right.
    """

    def __init__(self, raw_per_g, *, axis=None, sign=None, accel_sign=None, deg_per_raw=None,
                 alpha=None, still_tol_g=None, min_gravity_fraction=None, deadzone_dps=None,
                 history_s=None):
        axis = config.IMU_ROLL_AXIS if axis is None else axis
        if axis not in AXES:
            raise ValueError(f"IMU_ROLL_AXIS must be x, y or z, got {axis!r}")
        self.raw_per_g = raw_per_g
        self.axis = AXES[axis]
        self.cross = _CROSS[self.axis]
        self.sign = config.IMU_ROLL_SIGN if sign is None else sign
        self.accel_sign = config.IMU_ACCEL_ROLL_SIGN if accel_sign is None else accel_sign
        self.deg_per_raw = config.IMU_GYRO_DEG_PER_RAW if deg_per_raw is None else deg_per_raw
        self.alpha = config.ROLL_FILTER_ALPHA if alpha is None else alpha
        self.still_tol_g = config.ROLL_STILL_TOLERANCE_G if still_tol_g is None else still_tol_g
        self.min_gravity_fraction = (config.ROLL_MIN_GRAVITY_FRACTION if min_gravity_fraction is None
                                     else min_gravity_fraction)
        self.deadzone_dps = config.ROLL_GYRO_DEADZONE_DPS if deadzone_dps is None else deadzone_dps
        self.history_s = config.ROLL_HISTORY_S if history_s is None else history_s

        self.angle = None          # filtered roll, absolute (accelerometer frame), degrees
        self.zero = 0.0            # neutral-grip angle
        self.rate_dps = 0.0        # latest signed roll rate
        self.accel_angle = None    # latest accelerometer roll (absolute) or None if unobservable
        self.still = False
        self.frozen = False
        self._last_t = None
        self._hist_t = deque()
        self._hist_v = deque()

    # -- pure helpers ------------------------------------------------------------
    def rate_from_sample(self, s):
        raw = (s.gx, s.gy, s.gz)[self.axis]
        rate = self.sign * raw * self.deg_per_raw
        return 0.0 if abs(rate) < self.deadzone_dps else rate

    def accel_roll(self, s):
        """Gravity-based roll (deg, absolute) or None when the handle is too close to vertical."""
        a = (s.ax, s.ay, s.az)
        j, k = self.cross
        across = math.hypot(a[j], a[k])
        if across < self.min_gravity_fraction * self.raw_per_g:
            return None
        # Rotating the body by +theta about the axis turns gravity (seen in the body) by -theta,
        # so atan2(a_j, a_k) grows with the roll (0 when gravity lies along axis k).
        return self.accel_sign * self.sign * math.degrees(math.atan2(a[j], a[k]))

    def is_still(self, s):
        return abs(s.accel_mag / self.raw_per_g - 1.0) <= self.still_tol_g

    # -- streaming ---------------------------------------------------------------
    def update(self, s, freeze_accel=False):
        dt = 0.0 if self._last_t is None else clamp(s.t - self._last_t, 0.0, 0.1)
        self._last_t = s.t
        self.rate_dps = self.rate_from_sample(s)
        self.accel_angle = self.accel_roll(s)
        self.still = self.is_still(s)
        self.frozen = freeze_accel

        if self.angle is None:
            self.angle = self.accel_angle if self.accel_angle is not None else 0.0
            self.zero = self.angle
        self.angle += self.rate_dps * dt
        if self.still and not freeze_accel and self.accel_angle is not None:
            k = (1.0 - self.alpha) * dt / 0.02
            self.angle += clamp(k, 0.0, 1.0) * wrap_deg(self.accel_angle - self.angle)

        self._hist_t.append(s.t)
        self._hist_v.append(self.roll_deg)
        while self._hist_t and s.t - self._hist_t[0] > self.history_s:
            self._hist_t.popleft()
            self._hist_v.popleft()
        return self.roll_deg

    def calibrate_neutral(self):
        """Current grip becomes zero roll (snapping to the accelerometer if it is trustworthy)."""
        if self.angle is None:
            return
        if self.still and self.accel_angle is not None:
            self.angle = self.accel_angle
        self.zero = self.angle
        self._hist_t.clear()
        self._hist_v.clear()

    @property
    def roll_deg(self):
        return 0.0 if self.angle is None else wrap_deg(self.angle - self.zero)

    @property
    def accel_roll_deg(self):
        """Raw accelerometer roll relative to neutral (debug), or None."""
        return None if self.accel_angle is None else wrap_deg(self.accel_angle - self.zero)

    def roll_at(self, t):
        if not self._hist_t:
            return self.roll_deg
        i = max(0, bisect_right(self._hist_t, t) - 1)
        return self._hist_v[i]
