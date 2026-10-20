"""
Which way is the robot REALLY moving?  (forward / backward / uncertain)

The commanded direction is not enough: in a pushing match the wheels can turn
one way while the robot slides the other. We combine two measurements that fail
in opposite situations:

  * Wheel speed, MEASURED by the motor encoders (not the command). Correct whenever
    the tyres grip, including when the opponent back-drives our wheels. Wrong when
    the tyres slip (wheels spinning forward while we slide backward).
  * IMU forward acceleration, integrated into a velocity. Reacts instantly to a
    shove, but integration drifts, so it can't be trusted for long on its own.

Slip-aware complementary filter (one step per new IMU notification, dt between them):

    v_pred = v + accel * dt                      # dead-reckon with the IMU
    e      = v_pred - v_wheels                   # do the two sources agree?

    NORMAL mode (|e| <= SLIP_MPS):  v = a*v_pred + (1-a)*v_wheels,  a = tau/(tau+dt)
        Fast changes come from the IMU, the long-term value from the wheels.
        Braking: accel and v_wheels drop together, so v goes to 0, not negative.

    SLIP mode   (|e| >  SLIP_MPS):  v = v_pred   (wheels ignored)
        The wheels disagree with the IMU, so they are slipping or being overpowered.
        Pure IMU integration keeps the shove's velocity through a constant-speed
        slide (when the accelerometer reads ~0), instead of drifting back to the
        spinning wheels. Ends when |e| < SLIP_MPS / 2 again, or after SLIP_MAX_S
        (bounds integration drift), when v is reset to the wheels.

Direction = FORWARD if v > DIRECTION_MIN_MPS, BACKWARD if v < -DIRECTION_MIN_MPS,
else UNCERTAIN (the line tracker then leans per config.LEAN_WHEN_UNCERTAIN and logs it).

Accelerometer corrections:
  * bias: while sitting still, the forward-axis reading slowly re-zeroes (mounting tilt);
  * tilt (optional, IMU_TILT_FIELD/IMU_TILT_GAIN from hardware_check.py): when the
    nose is lifted (e.g. by a wedge), gravity leaks into the forward axis; we
    subtract gain * (tilt - tilt_at_rest).
"""

import math
from dataclasses import dataclass
from enum import Enum

import config

_AXIS = {"x": 0, "y": 1, "z": 2}
G = 9.81


class Direction(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    UNCERTAIN = "uncertain"


@dataclass
class MotionSample:
    """Latest Double Motor readings (library units). None = not available/stale."""
    left_speed: float | None      # motor[0].speed
    right_speed: float | None     # motor[1].speed
    accel: tuple | None           # (ax, ay, az) raw counts (milli-g)
    t: float | None = None        # laptop time the notification arrived
    tilt: float | None = None     # IMU pitch or roll (decidegrees), per IMU_TILT_FIELD


class MotionEstimator:
    def __init__(self, *, tau_s=config.FUSION_TAU_S, slip_mps=config.SLIP_MPS,
                 slip_max_s=config.SLIP_MAX_S, min_mps=config.DIRECTION_MIN_MPS,
                 left_sign=config.LEFT_WHEEL_SIGN, right_sign=config.RIGHT_WHEEL_SIGN,
                 wheel_to_mps=config.WHEEL_SPEED_TO_MPS, axis=config.IMU_FORWARD_AXIS,
                 axis_sign=config.IMU_FORWARD_SIGN, raw_per_g=config.IMU_RAW_PER_G,
                 bias_alpha=config.IMU_REST_BIAS_ALPHA, tilt_gain=config.IMU_TILT_GAIN, log=print):
        self.tau_s, self.slip_mps, self.slip_max_s = tau_s, slip_mps, slip_max_s
        self.min_mps = min_mps
        self.left_sign, self.right_sign = left_sign, right_sign
        self.wheel_to_mps = wheel_to_mps
        self.axis = _AXIS[axis]
        self.axis_sign = axis_sign
        self.raw_per_g = raw_per_g
        self.bias_alpha = bias_alpha
        self.tilt_gain = tilt_gain
        self.log = log
        self.bias_raw = None          # forward-axis accel reading at rest
        self.tilt_rest = None
        self.v = 0.0                  # fused forward speed, m/s
        self.v_wheels = 0.0
        self.accel_mps2 = 0.0
        self.slipping = False
        self._slip_since = None
        self._last_t = None

    def update(self, sample: MotionSample, now: float):
        """Feed the latest sample (call every tick; repeats of the same sample are ignored)."""
        if sample.left_speed is None or sample.accel is None:
            # No Double Motor data: we can't know. Decay toward "uncertain".
            self.v *= 0.9
            self._last_t = None
            return self.direction
        t = sample.t if sample.t is not None else now
        if self._last_t is not None and t <= self._last_t:
            return self.direction            # same notification as last tick
        dt = 0.03 if self._last_t is None else min(0.1, t - self._last_t)
        self._last_t = t

        wl = self.left_sign * sample.left_speed
        wr = self.right_sign * sample.right_speed
        self.v_wheels = 0.5 * (wl + wr) * self.wheel_to_mps

        raw = float(sample.accel[self.axis])
        if self.tilt_gain and sample.tilt is not None:
            if self.tilt_rest is None:
                self.tilt_rest = float(sample.tilt)
            raw -= self.tilt_gain * (float(sample.tilt) - self.tilt_rest)
        if self.bias_raw is None:
            self.bias_raw = raw
        accel = self.axis_sign * (raw - self.bias_raw) / self.raw_per_g * G
        at_rest = abs(self.v_wheels) < 0.01 and abs(self.v) < 0.05 and abs(accel) < 0.5
        if at_rest and not self.slipping:
            # Re-zero at rest. (tilt_rest stays fixed: after tilt compensation the
            # reading no longer depends on tilt, and the bias absorbs the rest.)
            self.bias_raw += self.bias_alpha * (raw - self.bias_raw)
        self.accel_mps2 = accel

        v_pred = self.v + accel * dt
        e = v_pred - self.v_wheels
        if not self.slipping and abs(e) > self.slip_mps:
            self.slipping, self._slip_since = True, t
            self.log(f"[motion] wheels and IMU disagree by {e:+.2f} m/s -> wheels SLIPPING, trusting IMU")
        elif self.slipping and abs(e) < self.slip_mps / 2:
            self.slipping = False
            self.log("[motion] wheels and IMU agree again")
        elif self.slipping and t - self._slip_since > self.slip_max_s:
            self.slipping = False
            v_pred = self.v_wheels
            self.log(f"[motion] slip lasted > {self.slip_max_s:g} s -> resetting to wheel speed")

        if self.slipping:
            self.v = v_pred
        else:
            a = self.tau_s / (self.tau_s + dt)
            self.v = a * v_pred + (1 - a) * self.v_wheels
        if not math.isfinite(self.v):
            self.v = self.v_wheels
        return self.direction

    @property
    def direction(self):
        if self.v > self.min_mps:
            return Direction.FORWARD
        if self.v < -self.min_mps:
            return Direction.BACKWARD
        return Direction.UNCERTAIN

    def describe(self):
        return (f"v={self.v:+.2f} m/s (wheels {self.v_wheels:+.2f}, accel {self.accel_mps2:+.1f} m/s^2"
                f"{', SLIPPING' if self.slipping else ''})")
