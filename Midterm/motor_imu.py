"""
LEGO Education Double Motor (paddle): BLE scan/connect, IMU streaming, swing detection.

Uses the official `legoeducation` package:
  DoubleMotor.search() / .connect(device, device_notification_delay=...) / .connected
  DoubleMotor.imu_device  (ImuDeviceNotification: accelerometerX/Y/Z, gyroscopeX/Y/Z, ...)
  DoubleMotor.set_notification_callback(cb) / .motor_run_for_time(...) / .disconnect()

`search()` returns bleak BLEDevice objects, which carry no RSSI. To pick the nearest
motor we register a subclass of the library's own BLE transport (via the documented
`legoeducation.ble_transport.register_transport` hook) that records each matching
advertisement's RSSI during the library's normal scan.

The library runs its own asyncio loop in a background thread. Notification
callbacks run on that loop thread, where the synchronous API may NOT be called --
so haptic pulses are sent from our own MotorIMU thread instead.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace

import config
from events import SwingEvent
from spin import AXES, RollFilter

# address -> latest RSSI (dBm) seen for a matching LEGO device during scanning
RSSI_BY_ADDRESS: dict[str, int] = {}

try:
    from legoeducation.basic_ble import BasicBLE
    from legoeducation.ble_transport import register_transport

    class RssiRecordingBLE(BasicBLE):
        """The library's bleak transport, plus RSSI bookkeeping for matching devices."""

        def on_scan(self, device, adv):
            super().on_scan(device, adv)
            if any(d.address == device.address for d in self.device_list):
                rssi = getattr(adv, "rssi", None)
                if rssi is not None:
                    RSSI_BY_ADDRESS[device.address] = rssi

    # Must happen before the library creates its transport (lazily, on the first scan).
    register_transport(RssiRecordingBLE)
    LEGO_AVAILABLE = True
except ImportError:  # package or bleak missing -> keyboard-only still works
    RssiRecordingBLE = None
    LEGO_AVAILABLE = False


def choose_nearest(devices, rssi_by_address=None):
    """Return the device with the strongest RSSI (unknown RSSI ranks last)."""
    if not devices:
        return None
    rssi_by_address = RSSI_BY_ADDRESS if rssi_by_address is None else rssi_by_address
    return max(devices, key=lambda d: rssi_by_address.get(d.address, -999))


# =============================================================================
# IMU math (pure, unit-tested)
# =============================================================================

@dataclass(frozen=True)
class ImuSample:
    """Raw IMU reading straight from the device (library units)."""
    t: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    @property
    def accel_mag(self):
        return math.sqrt(self.ax ** 2 + self.ay ** 2 + self.az ** 2)

    @property
    def gyro_mag(self):
        return math.sqrt(self.gx ** 2 + self.gy ** 2 + self.gz ** 2)

    @classmethod
    def from_notification(cls, t, imu):
        """Build from a legoeducation ImuDeviceNotification; None if not populated yet."""
        vals = (imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ,
                imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ)
        if any(isinstance(v, float) and math.isnan(v) for v in vals):
            return None
        return cls(t, *(float(v) for v in vals))


class GravityCalibrator:
    """Measures |gravity| in raw units while the paddle is held still.

    Feed samples with add(); once `n` samples are in, `raw_per_g` is set either to
    the measured mean (if the motor was still enough) or to the config fallback.
    """

    def __init__(self, n=config.IMU_CALIBRATION_SAMPLES, max_rel_std=config.IMU_CALIBRATION_MAX_STD,
                 fallback=config.IMU_ACCEL_RAW_PER_G_FALLBACK):
        self.n = n
        self.max_rel_std = max_rel_std
        self.fallback = fallback
        self._mags = []
        self.raw_per_g = None
        self.used_fallback = False

    @property
    def done(self):
        return self.raw_per_g is not None

    def add(self, sample):
        if self.done:
            return self.raw_per_g
        self._mags.append(sample.accel_mag)
        if len(self._mags) >= self.n:
            mean = sum(self._mags) / len(self._mags)
            std = math.sqrt(sum((m - mean) ** 2 for m in self._mags) / len(self._mags))
            if mean > 0 and std / mean <= self.max_rel_std:
                self.raw_per_g = mean
            else:
                self.raw_per_g = self.fallback
                self.used_fallback = True
        return self.raw_per_g


class SwingDetector:
    """Turns a stream of ImuSamples into SwingEvents.

    Dynamic acceleration = |a - gravity_estimate| / raw_per_g, where the gravity
    estimate is a low-pass (EMA) of the accel vector, i.e. a simple high-pass filter.
    A swing starts when the configured thresholds trip (SWING_MODE) and no swing
    started within the cooldown. The event is emitted SWING_PEAK_WINDOW_S after onset
    carrying the onset time and the peak strength over that window.

    The hub is the paddle handle, so twisting it (rotation about the roll axis) is spin
    input, not a swing: with exclude_roll the roll-axis gyro is left out of |gyro|. During
    the window the detector also records the strongest signed roll rate and the mean
    vertical (against gravity) dynamic acceleration, which become spin.
    """

    def __init__(self, raw_per_g, *, accel_threshold_g=config.SWING_ACCEL_THRESHOLD_G,
                 gyro_threshold=config.SWING_GYRO_THRESHOLD_RAW, mode=config.SWING_MODE,
                 cooldown_s=config.SWING_COOLDOWN_S, peak_window_s=config.SWING_PEAK_WINDOW_S,
                 gravity_alpha=config.IMU_GRAVITY_ALPHA, roll_axis=None, exclude_roll=None,
                 roll_sign=None, deg_per_raw=None):
        if mode not in ("accel", "gyro", "either", "both"):
            raise ValueError(f"bad SWING_MODE {mode!r}")
        self.raw_per_g = raw_per_g
        self.accel_threshold_g = accel_threshold_g
        self.gyro_threshold = gyro_threshold
        self.mode = mode
        self.cooldown_s = cooldown_s
        self.peak_window_s = peak_window_s
        self.gravity_alpha = gravity_alpha
        self.roll_axis = AXES[config.IMU_ROLL_AXIS if roll_axis is None else roll_axis]
        self.exclude_roll = config.SWING_EXCLUDE_ROLL_GYRO if exclude_roll is None else exclude_roll
        self.roll_sign = config.IMU_ROLL_SIGN if roll_sign is None else roll_sign
        self.deg_per_raw = config.IMU_GYRO_DEG_PER_RAW if deg_per_raw is None else deg_per_raw
        self._gravity = None
        self._last_onset = -math.inf
        self._active = None  # dict while measuring a swing
        self.last_dyn_accel_g = 0.0
        self.last_gyro = 0.0
        self.ignored = False

    def _tripped(self, accel_g, gyro):
        a = accel_g >= self.accel_threshold_g
        g = gyro >= self.gyro_threshold
        return {"accel": a, "gyro": g, "either": a or g, "both": a and g}[self.mode]

    def _strength(self, accel_g, gyro):
        a = accel_g / self.accel_threshold_g
        g = gyro / self.gyro_threshold
        return {"accel": a, "gyro": g}.get(self.mode, max(a, g))

    def _swing_gyro(self, s):
        g = (s.gx, s.gy, s.gz)
        if not self.exclude_roll:
            return s.gyro_mag
        return math.sqrt(sum(v * v for i, v in enumerate(g) if i != self.roll_axis))

    def update(self, s, ignore=False):
        """Process one sample; returns a SwingEvent when a swing completes, else None.

        ignore=True (haptic pulse running): the sample can't start a swing, doesn't feed
        the gravity estimate and doesn't change the peaks of a swing being measured.
        """
        vec = (s.ax, s.ay, s.az)
        if self._gravity is None:
            self._gravity = vec
        gx, gy, gz = self._gravity
        dyn = (s.ax - gx, s.ay - gy, s.az - gz)
        accel_g = math.sqrt(sum(v * v for v in dyn)) / self.raw_per_g
        gyro = self._swing_gyro(s)
        self.last_dyn_accel_g, self.last_gyro = accel_g, gyro
        self.ignored = ignore

        if self._active is None:
            if ignore:
                return None
            if self._tripped(accel_g, gyro) and s.t - self._last_onset >= self.cooldown_s:
                self._last_onset = s.t
                self._active = {"onset": s.t, "peak": 0.0, "peak_a": 0.0, "peak_g": 0.0,
                                "roll_rate": 0.0, "vert_sum": 0.0, "n": 0}
                self._measure(s, accel_g, gyro, dyn)
            else:
                # Only track gravity while not swinging so the swing doesn't leak into it.
                al = self.gravity_alpha
                self._gravity = tuple(al * g0 + (1 - al) * v for g0, v in zip(self._gravity, vec))
            return None

        if not ignore:
            self._measure(s, accel_g, gyro, dyn)
        a = self._active
        if s.t - a["onset"] >= self.peak_window_s:
            self._active = None
            return SwingEvent(t=a["onset"], strength=a["peak"], source="imu",
                              accel_peak_g=a["peak_a"], gyro_peak_raw=a["peak_g"],
                              roll_rate_dps=a["roll_rate"],
                              vertical_g=a["vert_sum"] / max(1, a["n"]))
        return None

    def _measure(self, s, accel_g, gyro, dyn):
        a = self._active
        a["peak"] = max(a["peak"], self._strength(accel_g, gyro))
        a["peak_a"] = max(a["peak_a"], accel_g)
        a["peak_g"] = max(a["peak_g"], gyro)
        rate = self.roll_sign * (s.gx, s.gy, s.gz)[self.roll_axis] * self.deg_per_raw
        if abs(rate) > abs(a["roll_rate"]):
            a["roll_rate"] = rate
        # Accelerometers read +1 g "up" at rest, so the gravity estimate points up:
        # dynamic acceleration along it is upward motion.
        gn = math.sqrt(sum(v * v for v in self._gravity)) or 1.0
        a["vert_sum"] += sum(d * g for d, g in zip(dyn, self._gravity)) / gn / self.raw_per_g
        a["n"] += 1


# =============================================================================
# Device manager (background thread)
# =============================================================================

_HAPTIC_MOTOR_INDEX = {"left": 0, "right": 1, "both": 2}  # == le.MOTOR_LEFT / RIGHT / BOTH


class MotorIMU:
    """Owns the Double Motor connection in a background thread.

    on_swing(SwingEvent) is called from the library's BLE thread -- keep it quick
    (e.g. put the event on a queue).
    """

    def __init__(self, on_swing, *, device_factory=None, sleep=time.sleep):
        self._on_swing = on_swing
        self._device_factory = device_factory or self._default_factory
        self._sleep = sleep
        self._dm = None
        self._thread = None
        self._stop = threading.Event()
        self._haptic = threading.Event()
        self._lock = threading.Lock()
        self.status = "idle"
        self.device_name = None
        self.device_address = None
        self.latest = None            # last ImuSample
        self.calibrator = None
        self.detector = None
        self.roll = None              # RollFilter once calibrated
        self._quiet_until = -math.inf  # IMU ignored until then (haptic pulse + settle)
        self.samples_received = 0
        self.fatal_error = None
        self.recent_samples = deque(maxlen=200)

    @staticmethod
    def _default_factory():
        if not LEGO_AVAILABLE:
            raise RuntimeError("legoeducation (and bleak) are not installed in this venv")
        import legoeducation as le
        return le.DoubleMotor()

    @property
    def connected(self):
        return bool(self._dm is not None and self._dm.connected and self.status.startswith(("calibrating", "connected")))

    # -- lifecycle --------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="MotorIMU", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._dm is not None and self._dm.connected:
            try:
                self._dm.disconnect()
            except Exception as e:  # shutting down anyway
                print(f"[motor] disconnect error: {e}")
        self.status = "stopped"

    def request_haptic(self):
        if config.HAPTIC_ENABLED:
            self._haptic.set()

    # -- handle roll (read from the game thread) ----------------------------------
    @property
    def roll_deg(self):
        """Filtered handle roll relative to neutral, or None before calibration."""
        r = self.roll
        return None if r is None else r.roll_deg

    def calibrate_neutral(self):
        """The current grip becomes zero roll (game start / C key)."""
        with self._lock:
            if self.roll is not None:
                self.roll.calibrate_neutral()

    @property
    def quiet(self):
        return time.monotonic() < self._quiet_until

    # -- thread body ------------------------------------------------------------
    def _run(self):
        try:
            self._dm = self._device_factory()
        except Exception as e:
            self.status = f"unavailable: {e}"
            self.fatal_error = e
            return
        self._dm.set_notification_callback(self._on_notification)

        while not self._stop.is_set():
            if not self._dm.connected:
                if self.status.startswith(("connected", "calibrating")):
                    print("[motor] connection lost -- reconnecting")
                try:
                    ok = self.connect_once()
                except Exception as e:
                    if type(e).__name__ == "VersionMismatchError":
                        self.status = "firmware/package mismatch (see console)"
                        self.fatal_error = e
                        print(f"[motor] {e}")
                        return
                    self.status = f"error: {e}"
                    ok = False
                if not ok:
                    self._stop.wait(config.BLE_RECONNECT_INTERVAL_S)
                continue

            if self._haptic.wait(timeout=0.05):
                self._haptic.clear()
                self._pulse()

    def connect_once(self):
        """One scan + connect attempt. Returns True on success."""
        self.status = "reconnecting: scanning" if self.device_address else "scanning"
        RSSI_BY_ADDRESS.clear()
        devices = self._dm.search(timeout=config.BLE_SCAN_TIMEOUT_S,
                                  device_name=config.BLE_DEVICE_NAME_FILTER,
                                  card_color=config.BLE_CARD_COLOR,
                                  card_serial=config.BLE_CARD_SERIAL)
        if self._stop.is_set():
            return False
        if not devices:
            self.status = "no motor found (retrying)"
            return False

        chosen = choose_nearest(devices)
        rssi = RSSI_BY_ADDRESS.get(chosen.address)
        print(f"[motor] found {len(devices)} Double Motor(s); connecting to nearest: "
              f"{chosen.name} [{chosen.address}] RSSI={rssi} dBm")
        self.status = "connecting"
        # Notifications (and so calibration) can start before connect() returns.
        self.device_name, self.device_address = chosen.name, chosen.address
        with self._lock:
            self.calibrator = GravityCalibrator()
            self.detector = None
        self._dm.connect(chosen, device_notification_delay=config.BLE_NOTIFICATION_MS)
        if not self._dm.connected:
            self.status = "connect failed (retrying)"
            return False
        if self.detector is None:
            self.status = "calibrating - hold paddle still"
        print(f"[motor] connected to {chosen.name} [{chosen.address}]")
        return True

    def _on_notification(self, _notification):
        """Runs on the library's BLE loop thread: parse IMU, feed calibrator/detector."""
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
                    self.detector = SwingDetector(raw_per_g)
                    self.roll = RollFilter(raw_per_g)
                    note = " (FALLBACK - paddle moved during calibration)" if self.calibrator.used_fallback else ""
                    print(f"[motor] IMU calibrated: 1 g = {raw_per_g:.1f} raw units{note}")
                    self.status = f"connected: {self.device_name or 'Double Motor'}"
                return
            quiet = sample.t < self._quiet_until
            if self.roll is not None:
                self.roll.update(sample, freeze_accel=quiet)
            event = self.detector.update(sample, ignore=quiet)
            if event is not None and self.roll is not None:
                event = replace(event, roll_deg=self.roll.roll_at(event.t))
        if event is not None:
            self._on_swing(event)

    def _pulse(self):
        # The pulse shakes the handle: ignore the IMU for its duration plus a settle time.
        self._quiet_until = time.monotonic() + config.HAPTIC_PULSE_MS / 1000 + config.HAPTIC_SETTLE_S
        try:
            self._dm.motor_run_for_time(config.HAPTIC_PULSE_MS,
                                        motor=_HAPTIC_MOTOR_INDEX[config.HAPTIC_MOTOR],
                                        speed=config.HAPTIC_SPEED, blocking=False)
        except Exception as e:
            print(f"[motor] haptic pulse failed: {e}")
