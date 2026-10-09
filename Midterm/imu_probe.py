"""
Live IMU readout for tuning swing thresholds (Ctrl+C to quit).

    python imu_probe.py

Hold the paddle still for the first second (gravity calibration), then swing.
Prints raw accel/gyro, gravity-removed |a| in g, |gyro| in raw units, and every
detected swing. Put the values you like into config.py.
"""

import time

import config
from motor_imu import MotorIMU


def main():
    def on_swing(ev):
        print(f"  >>> SWING  strength={ev.strength:.2f}  peak |a|={ev.accel_peak_g:.2f} g  "
              f"peak |gyro|={ev.gyro_peak_raw:.0f} raw")

    motor = MotorIMU(on_swing=on_swing)
    motor.start()
    peak_a = peak_g = 0.0
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
            peak_a = max(peak_a, det.last_dyn_accel_g)
            peak_g = max(peak_g, det.last_gyro)
            if time.monotonic() - last_print > 0.1:
                print(f"a=({s.ax:6.0f},{s.ay:6.0f},{s.az:6.0f}) g=({s.gx:6.0f},{s.gy:6.0f},{s.gz:6.0f})  "
                      f"|a|dyn={det.last_dyn_accel_g:5.2f}g (thr {config.SWING_ACCEL_THRESHOLD_G})  "
                      f"|gyro|={det.last_gyro:6.0f} (thr {config.SWING_GYRO_THRESHOLD_RAW:.0f})  "
                      f"peaks: {peak_a:.2f}g / {peak_g:.0f}")
                last_print = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        motor.stop()


if __name__ == "__main__":
    main()
