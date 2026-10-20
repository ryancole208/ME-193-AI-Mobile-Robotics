"""
Why won't the Double Motor turn?  Connects to the Double Motor ONLY and tries
each way of driving it, printing what the motor reports back after each.

    python motor_test.py            (LIFT THE WHEELS / disconnect the gear train first)

Close sumo.py and code.legoeducation.com first: a device can only have one connection.

For each test it prints the motor's own state (READY, RUNNING, STALLED,
NOT_ALLOWED_TO_RUN, MOTOR_DISCONNECTED, ...), its measured speed, power and
position. If the position changes, that side physically turned.
"""

import sys
import time

import legoeducation as le

import config

STATE_NAMES = {getattr(le, n): n.removeprefix("MOTOR_STATE_") for n in dir(le) if n.startswith("MOTOR_STATE_")}
SPEED = 40
RUN_S = 1.5


def report(dm, label):
    parts = []
    for i, side in enumerate(("left/motor[0]", "right/motor[1]")):
        m = dm.motor[i]
        parts.append(f"{side}: state={STATE_NAMES.get(m.motorState, m.motorState)} speed={m.speed} "
                     f"power={m.power} pos={m.position}")
    print(f"  {label:10s} " + " | ".join(parts))
    return [dm.motor[i].position for i in range(2)]


def trial(dm, name, start, stop):
    print(f"\n=== {name} ===")
    before = report(dm, "before")
    try:
        start()
    except Exception as e:
        print(f"  command FAILED: {type(e).__name__}: {e}")
        return
    t_end = time.monotonic() + RUN_S
    while time.monotonic() < t_end:
        time.sleep(0.5)
        after = report(dm, "running")
    try:
        stop()
    except Exception as e:
        print(f"  stop failed: {e}")
    time.sleep(0.3)
    moved = [a != b for a, b in zip(after, before)]
    print(f"  RESULT: left {'TURNED' if moved[0] else 'did not move'}, "
          f"right {'TURNED' if moved[1] else 'did not move'}")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    dm = le.DoubleMotor()
    print(f"connecting to the Double Motor (card {config.CARD_SERIAL}) ...")
    dm.connect(card_color=config.CARD_COLOR, card_serial=config.CARD_SERIAL,
               device_notification_delay=config.NOTIFICATION_MS)
    if not dm.connected:
        sys.exit("Double Motor not found")
    print(f"connected: {dm.device_name}  firmware {dm.version_firmware}  RPC {dm.version_rpc}")
    time.sleep(0.5)
    info = dm.info_device
    usb = info.UsbPowerState == le.USB_POWER_STATE_USB_CONNECTED
    print(f"battery: {info.batteryLevel}%   USB cable: {'CONNECTED' if usb else 'not connected'}")
    if usb:
        print("  !!! USB is plugged in. LEGO motors usually refuse to run while charging"
              " (state NOT_ALLOWED_TO_RUN). Unplug it and try again.")
    if info.batteryLevel <= 10:
        print("  !!! Battery is very low. The motor may refuse to run; charge it.")
    stop_all = lambda: dm.movement_stop()
    try:
        trial(dm, "1. movement_move_tank blocking=False (what sumo.py uses)",
              lambda: dm.movement_move_tank(SPEED, SPEED, blocking=False), stop_all)
        trial(dm, "2. movement_move_tank blocking=True",
              lambda: dm.movement_move_tank(SPEED, SPEED), stop_all)
        trial(dm, "3. movement_move forward",
              lambda: dm.movement_move(direction=le.MOVEMENT_DIRECTION_FORWARD, speed=SPEED), stop_all)
        trial(dm, "4. motor_run both sides",
              lambda: dm.motor_run(motor=le.MOTOR_BOTH, speed=SPEED), lambda: dm.motor_stop(motor=le.MOTOR_BOTH))
        trial(dm, "5. motor_set_duty_cycle (raw power, no speed regulation)",
              lambda: dm.motor_set_duty_cycle(SPEED, motor=le.MOTOR_BOTH),
              lambda: dm.motor_stop(motor=le.MOTOR_BOTH))
    finally:
        try:
            dm.movement_stop()
            dm.motor_stop(motor=le.MOTOR_BOTH)
        except Exception:
            pass
        dm.disconnect()
        print("\nmotors stopped, disconnected")


if __name__ == "__main__":
    main()
