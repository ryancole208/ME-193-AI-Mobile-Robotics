"""
Live readout of bowling throws for tuning (Ctrl+C to quit).

    python throw_probe.py

Hold the motor still for the first second (gravity calibration), then bowl. Every
detected throw prints its strength, the resulting ball speed, how long the swing
took to reach release, and the mean twist rate on each gyro axis around release.

Picking SPIN_GYRO_AXIS: bowl a few straight throws, then a few twisting the motor
like turning a doorknob as you release. The axis whose twist value changes the most
is your forearm axis. If twisting the way a right-hander hooks (counter-clockwise)
gives a positive value, set SPIN_SIGN = -1.0 so it hooks left.
"""

import time

import config
from throw_detector import BowlingMotor, strength_to_speed


def main():
    def on_throw(ev):
        tw = "  ".join(f"{a}={v:+7.0f}" for a, v in zip("xyz", ev.twist_raw))
        print(f"  >>> THROW  strength={ev.strength:.2f} -> {strength_to_speed(ev.strength) * 3.6:.0f} km/h  "
              f"onset->release={ev.t_release - ev.t_onset:.2f}s  peak |a|={ev.accel_peak_g:.2f} g  "
              f"peak |gyro|={ev.gyro_peak_raw:.0f}\n"
              f"            twist {tw}   spin={ev.spin:+.2f} (axis {config.SPIN_GYRO_AXIS})")

    motor = BowlingMotor(on_throw=on_throw)
    motor.start()
    last_print = 0.0
    try:
        while True:
            time.sleep(0.02)
            det, s = motor.detector, motor.latest
            if det is None or s is None:
                if time.monotonic() - last_print > 1.0:
                    print(f"[{motor.status}]")
                    last_print = time.monotonic()
                continue
            if time.monotonic() - last_print > 0.1:
                print(f"g=({s.gx:6.0f},{s.gy:6.0f},{s.gz:6.0f})  |a|dyn={det.last_dyn_accel_g:5.2f}g "
                      f"(thr {det.accel_threshold_g:.2f})  |gyro|={det.last_gyro:6.0f} (thr {det.gyro_threshold:.0f})"
                      f"{'  THROWING' if det.active else ''}")
                last_print = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        motor.stop()


if __name__ == "__main__":
    main()
