"""
Hardware check: measures the values marked MEASURE in config.py and tests the
LEGO hardware, step by step, with you moving the robot by hand.

    python hardware_check.py            never commands the motors
    python hardware_check.py --spin     also runs a short motor test (WHEELS LIFTED!)

Steps
  1. Connect to the Double Motor, Color Sensor and Controller (card in config.py)
     and measure how often each sends data.
  2. REST     robot still on the floor   -> gravity axis, IMU counts per g
  3. PUSH     you push the robot FORWARD (toward the color sensor) a known
              distance, by hand          -> wheel signs, wheel speed -> m/s,
                                            IMU forward axis and sign
  4. TILT     you lift the nose          -> IMU_TILT_FIELD / IMU_TILT_GAIN
  5. CONTROLLER  levers shown live; then "does the Controller drive the wheels
              DIRECTLY?" (we send no motor commands while you push the levers)
  6. SPIN (--spin only, wheels off the ground): the program turns the motor's
              LEFT side forward for 1 s     -> DRIVE_INVERTED / DRIVE_SWAP_SIDES
At the end it prints a block of lines to paste into config.py.

Why push by hand? A hand push back-drives the wheels, so the encoders report
the robot's real motion without the program ever powering the motors.
"""

import argparse
import math
import sys
import threading
import time

import config
from calibrate_color import get_key, key_pressed
from lego_io import from_motor_frame

G = 9.81
AXES = "xyz"


# ---------------------------------------------------------------------------
# Pure analysis (no hardware; tested in tests/test_hardware_check.py)
# Row formats:
#   motor row : (t, speed0, speed1, ax, ay, az, pitch, roll)   motor[0]/[1] = Double Motor frame
#   lever row : (t, speed0, speed1, lever_left, lever_right)
# ---------------------------------------------------------------------------

def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def rate_hz(times):
    """Average notifications per second from a list of arrival times."""
    if len(times) < 2:
        return 0.0
    return (len(times) - 1) / (times[-1] - times[0])


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def rest_analysis(rows):
    """Robot sitting still. Gravity shows up on one axis at ~1000 counts (1 g)."""
    acc = [mean(r[3 + i] for r in rows) for i in range(3)]
    gravity_axis = max(range(3), key=lambda i: abs(acc[i]))
    return {"accel_mean": acc, "gravity_axis": AXES[gravity_axis],
            "raw_per_g": math.sqrt(sum(a * a for a in acc)),
            "pitch": mean(r[6] for r in rows), "roll": mean(r[7] for r in rows)}


def _integrate(ts, vs):
    return sum(0.5 * (vs[i] + vs[i + 1]) * (ts[i + 1] - ts[i]) for i in range(len(ts) - 1))


def _pearson(a, b):
    ma, mb = mean(a), mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return 0.0 if va <= 0 or vb <= 0 else cov / math.sqrt(va * vb)


def push_analysis(rows, distance_m, rest, swap=False):
    """Hand push FORWARD by distance_m.

    * Wheel signs: whichever sign each robot-frame wheel's speed had (integrated
      over the push) is "forward" for that wheel.
    * WHEEL_SPEED_TO_MPS: distance / integral of the (sign-corrected) average speed.
    * IMU forward axis: the wheel speed tells us the robot's velocity, so its
      derivative is the true forward acceleration. The accelerometer axis whose
      reading is most correlated with it is the forward axis; the sign of the
      correlation is IMU_FORWARD_SIGN. (Gravity's axis is skipped.)
    """
    ts = [r[0] for r in rows]
    lr = [from_motor_frame(r[1], r[2], swap=swap) for r in rows]
    left = [a for a, _ in lr]
    right = [b for _, b in lr]
    out = {"left_sign": sign(_integrate(ts, left)), "right_sign": sign(_integrate(ts, right))}
    if 0 in (out["left_sign"], out["right_sign"]):
        out["error"] = "a wheel did not turn during the push (is the robot on the floor?)"
        return out
    avg = [0.5 * (out["left_sign"] * a + out["right_sign"] * b) for a, b in lr]
    travel_units = _integrate(ts, avg)
    out["wheel_to_mps"] = distance_m / travel_units
    v = [x * out["wheel_to_mps"] for x in avg]
    # Central-difference acceleration, lightly smoothed (3-point average).
    a_wheel = [0.0] * len(v)
    for i in range(1, len(v) - 1):
        dt = ts[i + 1] - ts[i - 1]
        a_wheel[i] = (v[i + 1] - v[i - 1]) / dt if dt > 0 else 0.0
    a_wheel = [mean(a_wheel[max(0, i - 1):i + 2]) for i in range(len(a_wheel))]
    best = None
    for i, axis in enumerate(AXES):
        if axis == rest["gravity_axis"]:
            continue
        acc = [(r[3 + i] - rest["accel_mean"][i]) / rest["raw_per_g"] * G for r in rows]
        a_s = [mean(acc[max(0, j - 1):j + 2]) for j in range(len(acc))]
        c = _pearson(a_wheel, a_s)
        out[f"corr_{axis}"] = c
        if best is None or abs(c) > abs(best[1]):
            best = (axis, c)
    out["forward_axis"], out["forward_corr"] = best[0], best[1]
    out["forward_sign"] = sign(best[1]) or 1
    out["peak_speed_mps"] = max(abs(x) for x in v)
    return out


def tilt_analysis(rest, rows, forward_axis):
    """Nose lifted and held. Which tilt angle changed, and how much does gravity
    leak into the forward axis per decidegree of it?"""
    pitch, roll = mean(r[6] for r in rows), mean(r[7] for r in rows)
    d = {"pitch": pitch - rest["pitch"], "roll": roll - rest["roll"]}
    field = max(d, key=lambda k: abs(d[k]))
    i = AXES.index(forward_axis)
    d_raw = mean(r[3 + i] for r in rows) - rest["accel_mean"][i]
    out = {"field": field, "delta_tilt": d[field], "delta_raw": d_raw}
    if abs(d[field]) < 50:            # 50 decidegrees = 5 degrees
        out["error"] = f"the nose was barely tilted ({d[field] / 10:+.1f} deg)"
    else:
        out["gain"] = d_raw / d[field]
    return out


def direct_drive_analysis(rows, deadband=config.TELEOP_DEADBAND, speed=config.DIRECT_DRIVE_SPEED):
    """While WE send no motor commands, did the wheels turn while a lever was pushed?
    Returns (levers_were_pushed, wheels_moved)."""
    pushed = [r for r in rows if max(abs(r[3]), abs(r[4])) > deadband]
    if not pushed:
        return False, False
    moving = sum(1 for r in pushed if max(abs(r[1]), abs(r[2])) > speed)
    return True, moving >= max(3, len(pushed) // 4)


def spin_analysis(rows, motor_forward_signs, human_side):
    """We ran movement_move_tank(+SPIN, 0): the motor's LEFT side forward.

    motor_forward_signs: (sign of motor[0], sign of motor[1]) when the robot moves
        forward, from the PUSH step (Double Motor frame).
    human_side: "l" or "r", which ROBOT wheel you saw turn.
    """
    s0 = mean(r[1] for r in rows)
    s1 = mean(r[2] for r in rows)
    idx = 0 if abs(s0) >= abs(s1) else 1
    s = s0 if idx == 0 else s1
    out = {"moved_index": idx, "speed": s,
           "swap": human_side == "r"}
    if abs(s) < config.DIRECT_DRIVE_SPEED:
        out["error"] = "the wheel barely moved"
        return out
    fwd = motor_forward_signs[idx] if motor_forward_signs else 0
    if fwd:
        # Commanded +, so if the encoder reads the BACKWARD sign the drive is inverted.
        out["inverted"] = sign(s) != fwd
    return out


# ---------------------------------------------------------------------------
# Hardware recording
# ---------------------------------------------------------------------------

class Recorder:
    """Connects to the three devices and records a row per notification."""

    def __init__(self):
        import legoeducation as le
        self.le = le
        self.dm, self.cs, self.ct = le.DoubleMotor(), le.ColorSensor(), le.Controller()
        self.lock = threading.Lock()
        self.times = {"Double Motor": [], "Color Sensor": [], "Controller": []}
        self.motor_rows = []
        self.recording = False
        self.dm.set_notification_callback(self._dm_cb)
        self.cs.set_notification_callback(lambda _n: self._stamp("Color Sensor"))
        self.ct.set_notification_callback(lambda _n: self._stamp("Controller"))

    def _stamp(self, name):
        with self.lock:
            self.times[name].append(time.monotonic())

    def _dm_cb(self, _n):
        t = time.monotonic()
        m0, m1, imu = self.dm.motor[0], self.dm.motor[1], self.dm.imu_device
        row = (t, m0.speed, m1.speed, imu.accelerometerX, imu.accelerometerY,
               imu.accelerometerZ, imu.pitch, imu.roll)
        if any(isinstance(v, float) and math.isnan(v) for v in row):
            return
        with self.lock:
            self.times["Double Motor"].append(t)
            if self.recording:
                self.motor_rows.append(row)

    def connect(self, required=("Double Motor",)):
        devices = {"Double Motor": self.dm, "Color Sensor": self.cs, "Controller": self.ct}
        for name, dev in devices.items():
            for attempt in range(3):
                print(f"  connecting to {name} (card {config.CARD_SERIAL}), attempt {attempt + 1} ...")
                try:
                    dev.connect(card_color=config.CARD_COLOR, card_serial=config.CARD_SERIAL,
                                device_notification_delay=config.NOTIFICATION_MS)
                except Exception as e:
                    if type(e).__name__ == "VersionMismatchError":
                        raise
                    print(f"    error: {e}")
                if dev.connected:
                    print(f"  {name} connected (firmware {'.'.join(map(str, dev.version_firmware))})")
                    break
            else:
                if name in required:
                    raise SystemExit(f"{name} not found: is it on, charged, and on card {config.CARD_SERIAL}?")
                print(f"  {name} NOT found; its steps will be skipped.")
        return {name: dev.connected for name, dev in devices.items()}

    def record(self, seconds=None, until_key=False):
        """Record Double Motor rows for `seconds`, or until a key is pressed."""
        with self.lock:
            self.motor_rows = []
            self.recording = True
        t_end = None if seconds is None else time.monotonic() + seconds
        while True:
            if t_end is not None and time.monotonic() >= t_end:
                break
            if until_key and key_pressed():
                break
            time.sleep(0.01)
        with self.lock:
            self.recording = False
            return list(self.motor_rows)

    def levers(self):
        s = self.ct.sensor
        return s.leftPercent, s.rightPercent

    def close(self):
        try:
            if self.dm.connected:
                self.dm.movement_stop(blocking=False)
        except Exception:
            pass
        for dev in (self.dm, self.cs, self.ct):
            if dev.connected:
                try:
                    dev.disconnect()
                except Exception as e:
                    print(f"  disconnect error: {e}")


def ask_float(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        print(f"  not a number; using {default}")
        return default


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description="Measure the MEASURE values in config.py")
    p.add_argument("--spin", action="store_true",
                   help="also run a 1 s motor test (lift the robot so the wheels spin freely!)")
    args = p.parse_args(argv)

    rec = Recorder()
    results = {}
    try:
        section("1. Connect and measure update rates")
        present = rec.connect()
        with rec.lock:
            for k in rec.times:
                rec.times[k].clear()
        time.sleep(3.0)
        with rec.lock:
            rates = {k: rate_hz(v) for k, v in rec.times.items()}
        for name, hz in rates.items():
            if present[name]:
                print(f"  {name:13s} {hz:5.1f} updates/s  (every {1000 / hz:.0f} ms)" if hz else
                      f"  {name:13s} NO updates received")
        if rates["Color Sensor"]:
            results["COLOR_SAMPLE_PERIOD_S"] = round(1 / rates["Color Sensor"], 3)

        section("2. REST: put the robot on the floor and don't touch it")
        get_key("  Press any key when it is still ... ")
        rest = rest_analysis(rec.record(seconds=2.0))
        results["IMU_RAW_PER_G"] = round(rest["raw_per_g"], 1)
        print(f"  accel (x, y, z) = {tuple(round(a) for a in rest['accel_mean'])}; gravity is on "
              f"'{rest['gravity_axis']}'; {rest['raw_per_g']:.0f} counts per g")

        spin = None
        if args.spin:
            section("6 (done first). SPIN: LIFT THE ROBOT so the wheels can spin freely")
            if get_key("  Wheels off the ground? Press y to spin the motor's LEFT side "
                       "forward for 1 s, any other key to skip ... ") == "y":
                rec.dm.movement_move_tank(30, 0, blocking=False)
                time.sleep(0.3)                      # let it speed up
                spin_rows = rec.record(seconds=0.7)
                rec.dm.movement_stop(blocking=False)
                rec.dm.movement_move_tank(0, 0, blocking=False)
                side = get_key("  Which wheel turned? Look from BEHIND the robot (color sensor "
                               "pointing away from you): l = left, r = right ... ")
                spin = (spin_rows, side)
            else:
                print("  skipped.")

        section("3. PUSH: measure how the wheels and IMU see FORWARD motion")
        print("  Put the robot on the floor next to a ruler or tape measure, pointing so that")
        print("  FORWARD (toward the color sensor) runs along the ruler. You will push it")
        print("  forward in a STRAIGHT line, briskly: start fast, stop sharply. 20-40 cm is good.")
        get_key("  Press any key, push the robot forward, then press any key again ... ")
        push_rows = rec.record(until_key=True)
        dist_cm = ask_float("  How far did it move, in cm?", 30.0)

        motor_fwd = (sign(_integrate([r[0] for r in push_rows], [r[1] for r in push_rows])),
                     sign(_integrate([r[0] for r in push_rows], [r[2] for r in push_rows])))
        swap, inverted = config.DRIVE_SWAP_SIDES, config.DRIVE_INVERTED
        if spin is not None:
            sp = spin_analysis(spin[0], motor_fwd, spin[1])
            print(f"\n  Spin test: motor[{sp['moved_index']}] turned (speed {sp['speed']:+.0f}).")
            if "error" in sp:
                print(f"  !! {sp['error']}; DRIVE_* values not measured")
            else:
                swap = sp["swap"]
                results["DRIVE_SWAP_SIDES"] = swap
                if "inverted" in sp:
                    inverted = sp["inverted"]
                    results["DRIVE_INVERTED"] = inverted
                if sp["moved_index"] != 0:
                    print("  !! motor[1] turned when the library's LEFT side was commanded; "
                          "that's unexpected, please tell your instructor/TA.")

        push = push_analysis(push_rows, dist_cm / 100, rest, swap=swap)
        if "error" in push:
            print(f"  !! {push['error']}")
        else:
            results.update(LEFT_WHEEL_SIGN=push["left_sign"], RIGHT_WHEEL_SIGN=push["right_sign"],
                           WHEEL_SPEED_TO_MPS=round(push["wheel_to_mps"], 5),
                           IMU_FORWARD_AXIS=push["forward_axis"], IMU_FORWARD_SIGN=push["forward_sign"])
            print(f"  wheel signs: left {push['left_sign']:+d}, right {push['right_sign']:+d}; "
                  f"1 speed unit = {push['wheel_to_mps']:.5f} m/s; peak {push['peak_speed_mps']:.2f} m/s")
            corr = ", ".join(f"{a}: {push.get('corr_' + a, 0):+.2f}" for a in AXES if 'corr_' + a in push)
            print(f"  IMU forward axis '{push['forward_axis']}' sign {push['forward_sign']:+d} "
                  f"(correlation with wheel acceleration: {corr})")
            if abs(push["forward_corr"]) < 0.5:
                print("  !! weak correlation: push again more briskly (sharp start and stop)")

            section("4. TILT: lift the FRONT (sensor end) about 20-30 degrees and hold it still")
            get_key("  Press any key while holding it tilted ... ")
            tilt = tilt_analysis(rest, rec.record(seconds=1.0), push["forward_axis"])
            if "error" in tilt:
                print(f"  !! {tilt['error']}; tilt compensation left off")
            else:
                results["IMU_TILT_FIELD"] = tilt["field"]
                results["IMU_TILT_GAIN"] = round(tilt["gain"], 3)
                print(f"  '{tilt['field']}' changed by {tilt['delta_tilt'] / 10:+.1f} deg; forward "
                      f"axis changed {tilt['delta_raw']:+.0f} counts -> gain {tilt['gain']:+.3f}")

        if present["Controller"]:
            section("5. CONTROLLER")
            print("  Move the levers: values are shown for 5 s.")
            t_end = time.monotonic() + 5
            while time.monotonic() < t_end:
                l, r = rec.levers()
                print(f"\r  left {l:+5.0f} %   right {r:+5.0f} %   ", end="", flush=True)
                time.sleep(0.05)
            print("\n\n  Direct-drive test: does the Controller move the wheels by ITSELF?")
            print("  This program sends NO motor commands now. LIFT THE ROBOT (wheels off the")
            print("  ground), press a key, then push BOTH levers fully for ~2 s.")
            get_key("  Ready? Press any key ... ")
            lever_rows = []
            t_end = time.monotonic() + 3.0
            while time.monotonic() < t_end:
                l, r = rec.levers()
                m0, m1 = rec.dm.motor[0].speed, rec.dm.motor[1].speed
                lever_rows.append((time.monotonic(), m0, m1, l, r))
                time.sleep(0.03)
            pushed, moved = direct_drive_analysis(lever_rows)
            if not pushed:
                print("  !! no lever push seen; test not done")
            elif moved:
                print("  !! YES: the wheels turned although we sent nothing. The Controller drives")
                print("     the Double Motor directly when they share a card. sumo.py re-sends its own")
                print("     command every MOTOR_RESEND_S, but to be safe put the Controller on its own")
                print("     Connection Card (see README > Controller pairing).")
            else:
                print("  OK: the wheels did not move. Lever input only reaches this program.")

        section("Paste into config.py")
        if not args.spin:
            print("# (run with --spin, wheels lifted, to also measure DRIVE_INVERTED / DRIVE_SWAP_SIDES)")
        for k, v in results.items():
            print(f"{k} = {v!r}")
        print(f"TAPE_WIDTH_M = ...   # measure the blue tape with a ruler (currently {config.TAPE_WIDTH_M})")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        rec.close()


if __name__ == "__main__":
    sys.exit(main())
