"""
Live readout of putts and putter twist for tuning (Ctrl+C to quit).

    python putt_probe.py

Hold the motor still for the first second (gravity calibration). Then:

  * Twist it like turning a putter face. "twist" shows the integrated angle; turn it
    about 90 degrees and adjust TWIST_DEG_PER_RAW_S until it reads about 90. If it drifts
    while you hold still, raise TWIST_DEADZONE_RAW. If it barely moves, try TWIST_AXIS
    = "x", "y" or "z" (the raw g=(x,y,z) column that changes most while twisting).
    Press Enter to re-zero it.
  * Swing quickly to putt. Every putt prints its strength, the resulting ball speed and
    roughly how far it would roll. If gentle putts don't fire, lower
    PUTT_GYRO_THRESHOLD_RAW; if your backswing fires a putt, raise it (or swing back slower).
"""

import sys
import threading
import time

import config
from putt_detector import GolfMotor, strength_to_speed


def roll_distance(v, dt=0.002):
    d = 0.0
    while v > config.STOP_SPEED:
        v -= (config.ROLL_DECEL_M_S2 + config.ROLL_DRAG_PER_S * v) * dt
        d += max(0.0, v) * dt
    return d


def main():
    def on_putt(ev):
        v = strength_to_speed(ev.strength)
        print(f"  >>> PUTT  strength={ev.strength:.2f} -> {v:.2f} m/s (rolls ~{roll_distance(v):.1f} m)  "
              f"peak |a|={ev.accel_peak_g:.2f} g  peak swing={ev.gyro_peak_raw:.0f}")

    motor = GolfMotor(on_putt=on_putt)
    motor.start()

    def rezero():
        for _ in sys.stdin:
            motor.recenter_twist()
    threading.Thread(target=rezero, daemon=True).start()
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
                print(f"g=({s.gx:6.0f},{s.gy:6.0f},{s.gz:6.0f})  swing={det.last_gyro:6.0f} "
                      f"(thr {det.gyro_threshold:.0f})  twist rate={det.last_twist_rate:+6.0f}  "
                      f"twist={det.twist_deg:+6.1f}°{'  PUTTING' if det.active else ''}")
                last_print = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        motor.stop()


if __name__ == "__main__":
    main()
