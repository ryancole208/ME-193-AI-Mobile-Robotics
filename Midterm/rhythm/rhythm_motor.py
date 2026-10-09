"""
The Double Motor paddle for the rhythm game: ../motor_imu.MotorIMU with three changes,
made in subclasses so the ping-pong game is untouched:

  * RhythmSwingDetector timestamps a swing at its PEAK sample (the "contact" moment), not
    its onset, and reports it as soon as the peak is over (strength < SWING_PEAK_DROP x
    peak) -- or after RHYTHM_SWING_PEAK_WINDOW_S at the latest -- to keep the delay short.
  * The swing cooldown is set per song (set_cooldown), short enough for the chart.
  * Haptic pulses have a per-hit length (pulse(ms)), so the IMU-ignore time around a pulse
    can be fitted before the next cue's timing window.

Sample timestamps are taken with time.perf_counter(), the same clock as the audio engine.
"""

from dataclasses import dataclass, replace

import config
from audio_engine import clock
from motor_imu import _HAPTIC_MOTOR_INDEX, ImuSample, MotorIMU, SwingDetector
from spin import RollFilter


@dataclass(frozen=True)
class RhythmSwing:
    t: float                 # wall time (clock()) of the peak sample
    onset: float
    strength: float
    source: str = "imu"
    roll_deg: float = 0.0


class RhythmSwingDetector(SwingDetector):
    def __init__(self, raw_per_g, *, cooldown_s=None, peak_window_s=None, drop=None, **kw):
        k = config.RHYTHM_SWING_THRESHOLD_SCALE
        kw.setdefault("accel_threshold_g", config.SWING_ACCEL_THRESHOLD_G * k)
        kw.setdefault("gyro_threshold", config.SWING_GYRO_THRESHOLD_RAW * k)
        super().__init__(raw_per_g,
                         cooldown_s=config.RHYTHM_SWING_COOLDOWN_S if cooldown_s is None else cooldown_s,
                         peak_window_s=config.RHYTHM_SWING_PEAK_WINDOW_S if peak_window_s is None else peak_window_s,
                         **kw)
        self.drop = config.SWING_PEAK_DROP if drop is None else drop

    def _measure(self, s, accel_g, gyro, dyn):
        before = self._active["peak"]
        super()._measure(s, accel_g, gyro, dyn)
        if self._active["peak"] > before or "peak_t" not in self._active:
            self._active["peak_t"] = s.t

    def update(self, s, ignore=False):
        a = self._active
        ev = super().update(s, ignore)
        if ev is not None:
            return RhythmSwing(a.get("peak_t", ev.t), ev.t, ev.strength)
        if a is not None and self._active is a and not ignore and a["n"] >= 2:
            if self._strength(self.last_dyn_accel_g, self.last_gyro) < self.drop * a["peak"]:
                self._active = None
                return RhythmSwing(a["peak_t"], a["onset"], a["peak"])
        return None


class RhythmMotor(MotorIMU):
    """on_swing(RhythmSwing) runs on the BLE thread -- keep it quick (put it on a queue)."""

    def __init__(self, on_swing, **kw):
        super().__init__(on_swing=on_swing, **kw)
        self.cooldown_s = config.RHYTHM_SWING_COOLDOWN_S
        self._pulse_ms = 0

    def set_cooldown(self, seconds):
        self.cooldown_s = seconds
        with self._lock:
            if self.detector is not None:
                self.detector.cooldown_s = seconds

    def pulse(self, ms):
        """Haptic pulse of `ms` milliseconds (sent from the motor thread)."""
        if ms > 0:
            self._pulse_ms = int(ms)
            self._haptic.set()

    @property
    def quiet(self):
        return clock() < self._quiet_until

    def _pulse(self):
        ms = self._pulse_ms
        if ms <= 0:
            return
        self._quiet_until = clock() + (ms + config.RHYTHM_HAPTIC_SETTLE_MS) / 1000
        try:
            self._dm.motor_run_for_time(ms, motor=_HAPTIC_MOTOR_INDEX[config.HAPTIC_MOTOR],
                                        speed=config.HAPTIC_SPEED, blocking=False)
        except Exception as e:
            print(f"[motor] haptic pulse failed: {e}")

    def _on_notification(self, _notification):
        sample = ImuSample.from_notification(clock(), self._dm.imu_device)
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
                    self.detector = RhythmSwingDetector(raw_per_g, cooldown_s=self.cooldown_s)
                    self.roll = RollFilter(raw_per_g)
                    note = " (FALLBACK - paddle moved during calibration)" if self.calibrator.used_fallback else ""
                    print(f"[motor] IMU calibrated: 1 g = {raw_per_g:.1f} raw units{note}")
                    self.status = f"connected: {self.device_name or 'Double Motor'}"
                return
            quiet = sample.t < self._quiet_until
            if quiet:
                # The haptic pulse twists the hub (motor reaction torque): don't integrate it as roll.
                self.roll._last_t = sample.t
            else:
                self.roll.update(sample)
            event = self.detector.update(sample, ignore=quiet)
            if event is not None:
                event = replace(event, roll_deg=self.roll.roll_at(event.t))
        if event is not None:
            self._on_swing(event)


# =============================================================================
# On-screen paddle twist (visual only -- spin isn't scored in this game)
# =============================================================================

class RollDisplay:
    """Calms the IMU roll for drawing.

    In a rhythm game you swing every second or so, so the roll filter rarely gets the still
    moments it needs to correct gyro drift, and swinging leaks into the roll axis. So:
      * while a swing is in progress (and ROLL_DISPLAY_SWING_HOLD_S after it) the shown twist holds;
      * slow drift fades back to neutral over ROLL_DISPLAY_RECENTER_S (a high-pass filter);
      * twists smaller than ROLL_DISPLAY_DEADZONE_DEG show as 0, the rest x ROLL_DISPLAY_GAIN.
    """

    def __init__(self):
        self.bias = None
        self.shown = 0.0
        self._last = None

    def update(self, now, raw_deg, swinging=False):
        if not config.ROLL_DISPLAY_ENABLED or raw_deg is None:
            self.shown = 0.0
            return 0.0
        dt = 0.0 if self._last is None else min(0.2, max(0.0, now - self._last))
        self._last = now
        if self.bias is None:
            self.bias = 0.0                         # roll_deg is already relative to the neutral grip
        if swinging:
            return self.shown                       # hold: swinging is not twisting
        self.bias += (raw_deg - self.bias) * min(1.0, dt / max(1e-3, config.ROLL_DISPLAY_RECENTER_S))
        d = raw_deg - self.bias
        dz = config.ROLL_DISPLAY_DEADZONE_DEG
        mag = max(0.0, abs(d) - dz) * config.ROLL_DISPLAY_GAIN
        target = min(config.ROLL_DISPLAY_MAX_DEG, mag) * (1 if d >= 0 else -1)
        self.shown += (target - self.shown) * min(1.0, dt * 12)     # light smoothing
        return self.shown

    def reset(self):
        """After re-zeroing the motor's neutral grip."""
        self.bias = 0.0
        self.shown = 0.0


# =============================================================================
# Chart checks: cooldown and haptic budget
# =============================================================================

def cooldown_for(min_gap_s, good_ms=None):
    """Cooldown that can't swallow the next cue's swing: two swings at adjacent cues can be as
    close as (gap - 2 x good window). Returns (cooldown, was shortened?)."""
    good = (config.GOOD_WINDOW_MS if good_ms is None else good_ms) / 1000
    limit = min_gap_s - 2 * good
    cd = config.RHYTHM_SWING_COOLDOWN_S
    if limit >= cd:
        return cd, False
    return max(config.RHYTHM_SWING_MIN_COOLDOWN_S, limit), True


def haptic_pulse_ms(available_ms):
    """Longest pulse whose pulse + settle + safety fits in available_ms (0 = skip)."""
    ms = min(config.RHYTHM_HAPTIC_PULSE_MS,
             available_ms - config.RHYTHM_HAPTIC_SETTLE_MS - config.RHYTHM_HAPTIC_SAFETY_MS)
    return int(ms) if ms >= config.RHYTHM_HAPTIC_MIN_PULSE_MS else 0


def haptic_plan(flights, input_offset_ms, good_ms=None):
    """Expected pulse length after each cue, assuming the hit registers at
    cue + input offset + peak detection. Returns a list of ms (0 = skipped)."""
    good = config.GOOD_WINDOW_MS if good_ms is None else good_ms
    delay = max(0.0, input_offset_ms) + config.RHYTHM_SWING_PEAK_WINDOW_S * 1000
    plan = []
    for a, b in zip(flights, flights[1:]):
        # the next window opens at (b.t_arrive - good) in song time; swings are measured
        # input_offset late, so the IMU must be listening from (b - good + offset)
        avail = (b.t_arrive - a.t_arrive) * 1000 - good + input_offset_ms - delay
        plan.append(haptic_pulse_ms(avail))
    if flights:
        plan.append(config.RHYTHM_HAPTIC_PULSE_MS)
    return plan
