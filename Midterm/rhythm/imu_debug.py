"""
bridge.py --imu-debug: a live IMU readout for finding the paddle's axes (see bridge.py's docstring
for the motions to perform). Fed on the BLE thread, printed from the main thread.

One line per print (IMU_DEBUG_RATE_HZ):
    acc   raw accel per hub axis, in g
    gyro  gyro per hub axis, deg/s (TRACK_GYRO_DEG_PER_RAW, minus the tracker's bias estimate)
    grav  gravity direction in hub coordinates (low-passed unit vector) and its nearest axis, e.g. "-z"
    int   gyro integrated per axis since the last reset (deg)          -> which axis a motion turns
    peak  largest |gyro| per axis since the last reset (deg/s)         -> fast flips / swings
    turn  how far gravity has turned since the reset (deg); with "int" -> the "scale?" check
    check tilt change since the reset from the accel vs the gyro alone (same sign and size = the
          axis mapping, TRACK_GYRO_SIGN and TRACK_GYRO_DEG_PER_RAW agree)
    out   the tracker's tilt (accel / filtered, deg), flip angle phi, the face, and HOLD in swings
Press Enter in the console to reset int / peak / turn.
"""

import math
import threading

from paddle_tracker import dot, rotate, wrap_deg

NAMES = "xyz"
GRAVITY_ALPHA = 0.85          # low-pass per sample for the gravity direction


def nearest_axis(v):
    i = max(range(3), key=lambda k: abs(v[k]))
    return ("+" if v[i] >= 0 else "-") + NAMES[i]


class ImuDebug:
    def __init__(self, tracker):
        self.tracker = tracker
        self._lock = threading.Lock()
        self.latest = None
        self.grav = None
        self.swings = 0
        self.reset()

    def reset(self):
        with self._lock:
            self.integral = [0.0, 0.0, 0.0]
            self.peak = [0.0, 0.0, 0.0]
            self.grav_ref = self.grav
            self.gyro_tilt = 0.0
            self.acc_tilt_ref = None
            self._last_t = None

    def update(self, s):
        """BLE thread, after the tracker has seen the sample."""
        tr = self.tracker
        with self._lock:
            dt = 0.0 if self._last_t is None else min(0.1, max(0.0, s.t - self._last_t))
            self._last_t = s.t
            gyro = tr.gyro_dps(s)
            for i in range(3):
                self.integral[i] += gyro[i] * dt
                self.peak[i] = max(self.peak[i], abs(gyro[i]))
            a = (s.ax, s.ay, s.az)
            mag = math.sqrt(dot(a, a)) or 1.0
            u = tuple(v / mag for v in a)
            if self.grav is None:
                self.grav = u
            else:
                g = tuple(GRAVITY_ALPHA * old + (1 - GRAVITY_ALPHA) * new for old, new in zip(self.grav, u))
                n = math.sqrt(dot(g, g)) or 1.0
                self.grav = tuple(v / n for v in g)
            if self.grav_ref is None:
                self.grav_ref = self.grav
            self.gyro_tilt += dot(rotate(gyro, tr.H, tr.phi), tr.N) * dt
            if self.acc_tilt_ref is None and tr.accel_theta is not None:
                self.acc_tilt_ref = tr.accel_theta
            self.latest = s

    def line(self):
        tr = self.tracker
        with self._lock:
            s, grav = self.latest, self.grav
            if s is None or tr.raw_per_g is None:
                return "[imu] waiting for the motor (connect, then hold still to calibrate)..."
            integral, peak = list(self.integral), list(self.peak)
            ref = self.grav_ref
            gyro_tilt, acc_ref = self.gyro_tilt, self.acc_tilt_ref
        g = tr.raw_per_g
        gyro = tr.gyro_dps(s)
        turned = math.degrees(math.acos(max(-1.0, min(1.0, dot(ref, grav))))) if ref else 0.0
        parts = [
            "acc g " + " ".join(f"{n}{v / g:+.2f}" for n, v in zip(NAMES, (s.ax, s.ay, s.az))),
            "gyro dps " + " ".join(f"{n}{v:+5.0f}" for n, v in zip(NAMES, gyro)),
            "grav " + " ".join(f"{n}{v:+.2f}" for n, v in zip(NAMES, grav)) + f" ({nearest_axis(grav)})",
            "int " + " ".join(f"{n}{v:+5.0f}" for n, v in zip(NAMES, integral)),
            "peak " + " ".join(f"{n}{v:4.0f}" for n, v in zip(NAMES, peak)),
            f"turn {turned:3.0f}",
        ]
        big = max(range(3), key=lambda i: abs(integral[i]))
        if turned >= 30 and abs(integral[big]) >= 10:
            parts.append(f"scale? {tr.p['TRACK_GYRO_DEG_PER_RAW'] * turned / abs(integral[big]):.3f}")
        if acc_ref is not None and tr.accel_theta is not None:
            parts.append(f"check accel{wrap_deg(tr.accel_theta - acc_ref):+4.0f} gyro{gyro_tilt:+4.0f}")
        if tr.ready:
            acc = "  --" if tr.accel_theta is None else f"{wrap_deg(tr.accel_theta - tr.theta0):+4.0f}"
            tilt, face = tr.output
            parts.append(f"out tilt acc{acc} filt{wrap_deg(tr.theta - tr.theta0):+4.0f} shown{tilt:+4.0f} "
                         f"phi{tr.phi:+4.0f} {face}{' HOLD' if tr.holding else ''}")
        return "[imu] " + " | ".join(parts)
