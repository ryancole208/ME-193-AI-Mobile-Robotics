"""
Find which IMU axis runs along the paddle handle (the roll / twist axis), its sign and
the gyro scale, then print the config.py lines to use.

    python imu_axis_test.py           # guided test (about 30 s)
    python imu_axis_test.py --live    # just stream accel + gyro for all axes (Ctrl+C to quit)

The LEGO library doesn't document how the IMU axes sit in the Double Motor, nor the gyro
units, so we measure them:
  1. Hold still in your neutral grip     -> gravity direction
  2. Twist the handle back and forth     -> the axis with the most gyro activity = roll axis
  3. Slowly turn the screen-side (BLACK) face 90 deg to your RIGHT
                                         -> sign (+ = right) and deg/s per raw gyro unit,
                                            by comparing the gyro integral with gravity's turn
"""

import argparse
import math
import time

from motor_imu import MotorIMU
from spin import _CROSS, AXES, wrap_deg

NAMES = ("x", "y", "z")


class Reader:
    """Pulls new ImuSamples from the MotorIMU in order."""

    def __init__(self, motor):
        self.motor = motor
        self.last_t = -math.inf

    def new(self):
        out = [s for s in list(self.motor.recent_samples) if s.t > self.last_t]
        if out:
            self.last_t = out[-1].t
        return out


def live_line(s):
    return (f"accel x{s.ax:7.0f} y{s.ay:7.0f} z{s.az:7.0f}   "
            f"gyro x{s.gx:7.0f} y{s.gy:7.0f} z{s.gz:7.0f}")


def collect(reader, seconds, label, every=0.25):
    print(f"   ... {label} ({seconds:.0f} s)")
    samples, end, last_print = [], time.monotonic() + seconds, 0.0
    reader.new()   # drop anything older
    while time.monotonic() < end:
        time.sleep(0.02)
        got = reader.new()
        samples += got
        if got and time.monotonic() - last_print > every:
            print("     " + live_line(got[-1]))
            last_print = time.monotonic()
    return samples


def mean_vec(samples):
    n = max(1, len(samples))
    return (sum(s.ax for s in samples) / n, sum(s.ay for s in samples) / n, sum(s.az for s in samples) / n)


def wait_enter(msg):
    input(f"\n>> {msg}\n   Press Enter when ready...")


def connect():
    motor = MotorIMU(on_swing=lambda e: None)
    motor.start()
    last = None
    while motor.detector is None:
        if motor.status != last:
            print(f"[motor] {motor.status}")
            last = motor.status
        if motor.fatal_error:
            raise SystemExit(f"Motor unavailable: {motor.fatal_error}")
        time.sleep(0.1)
    return motor


def guided(motor):
    reader = Reader(motor)
    raw_per_g = motor.detector.raw_per_g
    print(f"\nConnected. 1 g = {raw_per_g:.0f} raw accel units.")

    # 1. neutral grip
    wait_enter("STEP 1: hold the paddle in your NEUTRAL grip -- face vertical, BLACK side toward the "
               "screen (yellow toward you) -- and keep it still.")
    still = collect(reader, 2.0, "hold still")
    g0 = mean_vec(still)

    # 2. twist back and forth
    wait_enter("STEP 2: TWIST the handle back and forth like a doorknob (about +/-45 deg), "
               "keeping the hub in place. Keep going until it says done.")
    twist = collect(reader, 4.0, "twist back and forth")
    activity = [sum(abs((s.gx, s.gy, s.gz)[i]) for s in twist) for i in range(3)]
    total = sum(activity) or 1.0
    axis = max(range(3), key=lambda i: activity[i])
    print("\n   gyro activity: " + "   ".join(f"{NAMES[i]} {100 * activity[i] / total:4.0f}%" for i in range(3)))
    print(f"   -> roll axis = {NAMES[axis]!r}")
    if activity[axis] / total < 0.5:
        print("   WARNING: no single axis dominates -- try again twisting more cleanly (only the handle).")

    # 3. slow 90 deg turn to the right
    wait_enter("STEP 3: start in the neutral grip, then SLOWLY (2-3 s) turn the handle so the BLACK face "
               "(the one toward the screen) turns to your RIGHT by about 90 deg (edge-on to the screen), "
               "and hold it there.")
    turn = collect(reader, 5.0, "turn right ~90 deg, then hold")
    integral = 0.0
    for a, b in zip(turn, turn[1:]):
        integral += (a.gx, a.gy, a.gz)[axis] * (b.t - a.t)
    peak_raw = max((abs((s.gx, s.gy, s.gz)[axis]) for s in twist), default=0.0)
    g1 = mean_vec(turn[-20:])
    sign = 1.0 if integral > 0 else -1.0

    j, k = _CROSS[AXES[NAMES[axis]]]
    across0, across1 = math.hypot(g0[j], g0[k]), math.hypot(g1[j], g1[k])
    observable = min(across0, across1) >= 0.35 * raw_per_g
    if observable:
        delta = wrap_deg(math.degrees(math.atan2(g1[j], g1[k]) - math.atan2(g0[j], g0[k])))
        turned = abs(delta)
        accel_sign = 1.0 if sign * delta > 0 else -1.0
        print(f"\n   gravity says you turned {turned:.0f} deg")
    else:
        turned = 90.0
        accel_sign = 1.0
        print("\n   NOTE: the handle is close to vertical, so gravity can't measure the twist.")
        print("   Assuming you turned 90 deg. The game will track roll with the gyro only while")
        print("   the handle points straight down (press C in-game to re-zero if it drifts).")
    if abs(integral) < 1e-6 or turned < 20:
        print("   ERROR: barely any rotation measured -- rerun and turn further.")
        return
    deg_per_raw = turned / abs(integral)
    print(f"   gyro integral {integral:+.0f} raw*s  ->  {deg_per_raw:.4f} deg/s per raw unit")
    print(f"   fastest twist in step 2: {peak_raw * deg_per_raw:.0f} deg/s")
    print(f"   gravity across the handle in neutral grip: {100 * across0 / raw_per_g:.0f}% of 1 g")

    print("\n=== Put these in config.py (and tell Claude the numbers) ===")
    print(f'IMU_ROLL_AXIS = "{NAMES[axis]}"')
    print(f"IMU_ROLL_SIGN = {sign:+.1f}")
    print(f"IMU_ACCEL_ROLL_SIGN = {accel_sign:+.1f}")
    print(f"IMU_GYRO_DEG_PER_RAW = {deg_per_raw:.4f}")
    full = max(300.0, 0.6 * peak_raw * deg_per_raw)
    print(f"SPIN_RATE_FULL_DPS = {full:.0f}          # ~60% of your fastest twist")
    print(f"SPIN_RATE_DEADZONE_DPS = {0.18 * full:.0f}")


def live(motor):
    reader = Reader(motor)
    print("Streaming (Ctrl+C to stop). Twist the handle: the roll axis shows the biggest gyro values.")
    while True:
        time.sleep(0.1)
        got = reader.new()
        if got:
            print(live_line(got[-1]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="only stream raw values")
    args = p.parse_args(argv)
    motor = connect()
    try:
        live(motor) if args.live else guided(motor)
    except KeyboardInterrupt:
        pass
    finally:
        motor.stop()


if __name__ == "__main__":
    main()
