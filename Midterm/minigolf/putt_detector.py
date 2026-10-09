"""
Putts and putter angle from the Double Motor's IMU.

Reuses ../motor_imu.py unchanged (BLE scan/connect, nearest-device choice, gravity
calibration, haptics). GolfMotor only swaps its per-sample swing detector for a
PuttDetector, which splits each gyro reading into two parts:

    twist rate  -- rotation around TWIST_AXIS ("vertical" = around gravity, i.e. turning
                   a putter face). Integrated into twist_deg = the putter angle.
    swing rate  -- the rest of the rotation (the pendulum swing). When it trips the
                   threshold for PUTT_CONFIRM_SAMPLES samples the putt FIRES; its power
                   is the peak strength over the next PUTT_PEAK_WINDOW_S.

Twisting therefore never fires a putt, and the twist angle is frozen while you swing.
"""

import math
import time
from dataclasses import dataclass

import config
from motor_imu import ImuSample, MotorIMU, SwingDetector

_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PuttEvent:
    """One putt.

    t_onset    -- time.monotonic() when the swing tripped the threshold (the aim is read here)
    t_impact   -- time of peak strength (push/pull is measured around it)
    strength   -- peak strength normalised to the thresholds (1.0 = just tripped)
    """
    t_onset: float
    t_impact: float
    strength: float
    source: str = "imu"
    accel_peak_g: float = 0.0
    gyro_peak_raw: float = 0.0


def strength_to_speed(strength):
    v = config.PUTT_SPEED_AT_THRESHOLD + config.PUTT_SPEED_PER_STRENGTH * (strength - 1.0)
    return max(config.PUTT_SPEED_MIN, min(config.PUTT_SPEED_MAX, v))


def split_rotation(gyro, gravity, axis=config.TWIST_AXIS):
    """(twist rate, swing rate) in raw units. gravity: accel vector at rest (any scale)."""
    gx, gy, gz = gyro
    if axis == "vertical":
        n = math.sqrt(sum(v * v for v in gravity)) or 1.0
        ux, uy, uz = (v / n for v in gravity)
        twist = gx * ux + gy * uy + gz * uz
        sx, sy, sz = gx - twist * ux, gy - twist * uy, gz - twist * uz
        return twist, math.sqrt(sx * sx + sy * sy + sz * sz)
    i = _AXES[axis]
    twist = gyro[i]
    return twist, math.sqrt(sum(v * v for k, v in enumerate(gyro) if k != i))


class PuttDetector(SwingDetector):
    """Turns ImuSamples into PuttEvents and tracks the putter twist angle."""

    def __init__(self, raw_per_g, *, accel_threshold_g=config.PUTT_ACCEL_THRESHOLD_G,
                 gyro_threshold=config.PUTT_GYRO_THRESHOLD_RAW, mode=config.PUTT_MODE,
                 confirm_samples=config.PUTT_CONFIRM_SAMPLES, peak_window_s=config.PUTT_PEAK_WINDOW_S,
                 cooldown_s=config.PUTT_COOLDOWN_S, twist_axis=config.TWIST_AXIS, twist_sign=config.TWIST_SIGN,
                 twist_deg_per_raw_s=config.TWIST_DEG_PER_RAW_S, twist_deadzone=config.TWIST_DEADZONE_RAW,
                 twist_freeze_fraction=config.TWIST_FREEZE_FRACTION, gravity_alpha=config.IMU_GRAVITY_ALPHA):
        super().__init__(raw_per_g, accel_threshold_g=accel_threshold_g, gyro_threshold=gyro_threshold,
                         mode=mode, cooldown_s=cooldown_s, peak_window_s=peak_window_s,
                         gravity_alpha=gravity_alpha)
        self.confirm_samples = confirm_samples
        self.twist_axis = twist_axis
        self.twist_sign = twist_sign
        self.twist_deg_per_raw_s = twist_deg_per_raw_s
        self.twist_deadzone = twist_deadzone
        self.twist_freeze_fraction = twist_freeze_fraction
        self.twist_deg = 0.0
        self.last_twist_rate = 0.0
        self._last_t = None
        self._last_end = -math.inf
        self._over = 0                # consecutive samples over the threshold
        self._first_over_t = None
        self._peak = None             # (strength, t, accel_g, swing) while firing, else None
        self._onset = 0.0
        self.last_event = None

    @property
    def active(self):
        return self._peak is not None

    @property
    def live_strength(self):
        """Peak strength of the putt being measured (0 when idle), for a live power meter."""
        return self._peak[0] if self._peak is not None else 0.0

    def recenter(self):
        self.twist_deg = 0.0

    def update(self, s):
        """Process one sample; returns a PuttEvent when a putt's power window closes, else None."""
        vec = (s.ax, s.ay, s.az)
        if self._gravity is None:
            self._gravity = vec
        gx, gy, gz = self._gravity
        accel_g = math.sqrt((s.ax - gx) ** 2 + (s.ay - gy) ** 2 + (s.az - gz) ** 2) / self.raw_per_g
        twist, swing = split_rotation((s.gx, s.gy, s.gz), self._gravity, self.twist_axis)
        self.last_dyn_accel_g, self.last_gyro, self.last_twist_rate = accel_g, swing, twist
        dt = 0.0 if self._last_t is None else max(0.0, min(0.1, s.t - self._last_t))
        self._last_t = s.t

        # -- measuring a putt's power -------------------------------------------------
        if self._peak is not None:
            strength = self._strength(accel_g, swing)
            if strength > self._peak[0]:
                self._peak = (strength, s.t, accel_g, swing)
            if s.t - self._onset < self.peak_window_s:
                return None
            peak, t_imp, peak_a, peak_g = self._peak
            self._peak = None
            self._last_end = s.t
            self.last_event = PuttEvent(t_onset=self._onset, t_impact=t_imp, strength=peak, source="imu",
                                        accel_peak_g=peak_a, gyro_peak_raw=peak_g)
            return self.last_event

        # -- twist = putter angle (only while not swinging) --------------------------
        if swing < self.twist_freeze_fraction * self.gyro_threshold and abs(twist) > self.twist_deadzone:
            self.twist_deg += self.twist_sign * twist * self.twist_deg_per_raw_s * dt

        # -- a quick swing fires the putt ----------------------------------------------
        if self._tripped(accel_g, swing) and s.t - self._last_end >= self.cooldown_s:
            if self._over == 0:
                self._first_over_t = s.t
            self._over += 1
            if self._over >= self.confirm_samples:
                self._over = 0
                self._onset = self._first_over_t
                self._peak = (self._strength(accel_g, swing), s.t, accel_g, swing)
        else:
            self._over = 0
            if accel_g < self.accel_threshold_g * 0.5:
                # Only track gravity while (nearly) still so the swing doesn't leak into it.
                al = self.gravity_alpha
                self._gravity = tuple(al * g0 + (1 - al) * v for g0, v in zip(self._gravity, vec))
        return None


class GolfMotor(MotorIMU):
    """MotorIMU with a PuttDetector. on_putt(PuttEvent) runs on the BLE thread -- keep it quick."""

    def __init__(self, on_putt, **kw):
        super().__init__(on_swing=on_putt, **kw)

    @property
    def twist_deg(self):
        """Putter twist angle since the last recenter, or None before the IMU is calibrated."""
        det = self.detector
        return det.twist_deg if det is not None else None

    def recenter_twist(self):
        det = self.detector
        if det is not None:
            det.recenter()

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
                    self.detector = PuttDetector(raw_per_g)
                    note = " (FALLBACK - motor moved during calibration)" if self.calibrator.used_fallback else ""
                    print(f"[motor] IMU calibrated: 1 g = {raw_per_g:.1f} raw units{note}")
                    self.status = f"connected: {self.device_name or 'Double Motor'}"
                return
            event = self.detector.update(sample)
        if event is not None:
            self._on_swing(event)
