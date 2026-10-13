"""
Bowling throws from the Double Motor's IMU.

Reuses ../motor_imu.py unchanged (BLE scan/connect, nearest-device choice, gravity
calibration, haptics). BowlingMotor only swaps its per-sample swing detector for a
ThrowDetector, because a bowling throw is a whole pendulum motion rather than one
quick swing:

    backswing -> forward swing -> RELEASE (fastest point, bottom of the arc) -> follow-through

A throw starts when the thresholds trip (THROW_MODE) and ends once the motor has
stayed below THROW_END_FRACTION of them for THROW_END_QUIET_S. The release is the
sample with peak strength. Spin is the mean rate around SPIN_GYRO_AXIS (twisting the
motor about your forearm) within SPIN_WINDOW_S of the release.
"""

import math
import time
from dataclasses import dataclass

import config
from motor_imu import ImuSample, MotorIMU, SwingDetector

_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class ThrowEvent:
    """One throw.

    t_onset    -- time.monotonic() when the motion started (where the bowler stood)
    t_release  -- time of peak strength (aim is measured around it)
    strength   -- peak strength normalised to the start thresholds (1.0 = just tripped)
    spin       -- -1..1, + hooks right
    """
    t_onset: float
    t_release: float
    strength: float
    spin: float = 0.0
    source: str = "imu"
    accel_peak_g: float = 0.0
    gyro_peak_raw: float = 0.0
    twist_raw: tuple = (0.0, 0.0, 0.0)   # mean gx, gy, gz around release (for picking SPIN_GYRO_AXIS)


def spin_from_twist(twist_raw, axis=config.SPIN_GYRO_AXIS, sign=config.SPIN_SIGN,
                    full=config.SPIN_FULL_RAW, deadzone=config.SPIN_DEADZONE):
    if axis is None:
        return 0.0
    s = sign * twist_raw[_AXES[axis]] / full
    s = max(-1.0, min(1.0, s))
    return 0.0 if abs(s) < deadzone else s


def strength_to_speed(strength):
    v = config.BALL_SPEED_AT_THRESHOLD + config.BALL_SPEED_PER_STRENGTH * (strength - 1.0)
    return max(config.BALL_SPEED_MIN, min(config.BALL_SPEED_MAX, v))


class ThrowDetector(SwingDetector):
    """Turns ImuSamples into ThrowEvents. Shares SwingDetector's gravity removal and thresholds."""

    def __init__(self, raw_per_g, *, accel_threshold_g=config.THROW_ACCEL_THRESHOLD_G,
                 gyro_threshold=config.THROW_GYRO_THRESHOLD_RAW, mode=config.THROW_MODE,
                 end_fraction=config.THROW_END_FRACTION, quiet_s=config.THROW_END_QUIET_S,
                 min_s=config.THROW_MIN_S, max_s=config.THROW_MAX_S, cooldown_s=config.THROW_COOLDOWN_S,
                 spin_window_s=config.SPIN_WINDOW_S, gravity_alpha=config.IMU_GRAVITY_ALPHA):
        super().__init__(raw_per_g, accel_threshold_g=accel_threshold_g, gyro_threshold=gyro_threshold,
                         mode=mode, cooldown_s=cooldown_s, gravity_alpha=gravity_alpha)
        self.end_fraction = end_fraction
        self.quiet_s = quiet_s
        self.min_s = min_s
        self.max_s = max_s
        self.spin_window_s = spin_window_s
        self._last_end = -math.inf
        self._samples = None     # samples of the throw in progress
        self._onset = self._last_motion = 0.0
        self._peak = None        # (strength, t, accel_g, gyro)
        self.last_event = None

    @property
    def active(self):
        return self._samples is not None

    def _moving(self, accel_g, gyro):
        f = self.end_fraction
        return accel_g >= f * self.accel_threshold_g or gyro >= f * self.gyro_threshold

    def update(self, s):
        """Process one sample; returns a ThrowEvent when a throw ends, else None."""
        vec = (s.ax, s.ay, s.az)
        if self._gravity is None:
            self._gravity = vec
        gx, gy, gz = self._gravity
        accel_g = math.sqrt((s.ax - gx) ** 2 + (s.ay - gy) ** 2 + (s.az - gz) ** 2) / self.raw_per_g
        gyro = s.gyro_mag
        self.last_dyn_accel_g, self.last_gyro = accel_g, gyro

        if self._samples is None:
            if self._tripped(accel_g, gyro) and s.t - self._last_end >= self.cooldown_s:
                self._samples = [s]
                self._onset = self._last_motion = s.t
                self._peak = (self._strength(accel_g, gyro), s.t, accel_g, gyro)
            else:
                al = self.gravity_alpha
                self._gravity = tuple(al * g0 + (1 - al) * v for g0, v in zip(self._gravity, vec))
            return None

        self._samples.append(s)
        strength = self._strength(accel_g, gyro)
        if strength > self._peak[0]:
            self._peak = (strength, s.t, accel_g, gyro)
        if self._moving(accel_g, gyro):
            self._last_motion = s.t
        quiet = s.t - self._last_motion >= self.quiet_s
        if not quiet and s.t - self._onset < self.max_s:
            return None

        samples, (peak, t_rel, peak_a, peak_g) = self._samples, self._peak
        self._samples = None
        self._last_end = s.t
        if self._last_motion - self._onset < self.min_s:
            return None  # a twitch, not a throw
        w = self.spin_window_s
        near = [x for x in samples if abs(x.t - t_rel) <= w] or samples
        twist = tuple(sum(getattr(x, a) for x in near) / len(near) for a in ("gx", "gy", "gz"))
        self.last_event = ThrowEvent(t_onset=self._onset, t_release=t_rel, strength=peak,
                                     spin=spin_from_twist(twist), source="imu", accel_peak_g=peak_a,
                                     gyro_peak_raw=peak_g, twist_raw=twist)
        return self.last_event


class BowlingMotor(MotorIMU):
    """MotorIMU with a ThrowDetector. on_throw(ThrowEvent) runs on the BLE thread -- keep it quick."""

    def __init__(self, on_throw, **kw):
        super().__init__(on_swing=on_throw, **kw)

    def _on_notification(self, _notification):
        sample = ImuSample.from_notification(time.monotonic(), self._dm.imu_device)
        if sample is None:
            return
        self.latest = sample
        self.samples_received += 1
        self.recent_samples.append(sample)
        event = None
        with self._lock:
            if self.detector is None:
                if self.calibrator is None:
                    return
                raw_per_g = self.calibrator.add(sample)
                if raw_per_g is not None:
                    self.detector = ThrowDetector(raw_per_g)
                    note = " (FALLBACK - motor moved during calibration)" if self.calibrator.used_fallback else ""
                    print(f"[motor] IMU calibrated: 1 g = {raw_per_g:.1f} raw units{note}")
                    self.status = f"connected: {self.device_name or 'Double Motor'}"
                return
            event = self.detector.update(sample)
        if event is not None:
            self._on_swing(event)
