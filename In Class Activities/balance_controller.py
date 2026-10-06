"""
Self-balancing (inverted-pendulum) controller for a LEGO Education Double
Motor driving base. Uses only the two wheels attached to the Double Motor
- no Single Motor or other actuators are involved.

Hardware: LEGO Education Double Motor, identified by its Azure connection
card, serial 1096.
"""

import time

import legoeducation as le

# --- Hardware identification (Azure card, serial 1096) ---
CARD_COLOR = le.LEGO_COLOR_AZURE
CARD_SERIAL = "1096"

# --- Balance PID gains ---
# Starting points only - these MUST be tuned on the real robot. Raise KP
# until it responds promptly to a lean, add KD to damp oscillation, and
# only add a small KI if it settles with a steady-state lean.
KP = 2.2
KI = 0.0
KD = 0.06

# --- Safety / timing ---
FALL_ANGLE_DEG = 45.0   # if tilt error exceeds this, the robot has fallen - cut power
LOOP_HZ = 50            # control loop rate
MAX_DUTY = 100          # duty cycle cap (%), matches motor_set_duty_cycle's range


def calibrate_upright(doublemotor, samples=100, sample_delay=0.01):
    """Average the pitch reading while the robot is held balanced/upright
    by hand, to find the zero-tilt setpoint (accounts for IMU mounting
    offset)."""
    total = 0.0
    count = 0
    for _ in range(samples):
        pitch = doublemotor.imu_device.pitch
        if pitch == pitch:  # skip NaN (no notification received yet)
            total += pitch
            count += 1
        time.sleep(sample_delay)
    return total / count if count else 0.0


def balance(doublemotor, setpoint_deg):
    """Run the balance PID loop until interrupted (Ctrl+C)."""
    integral = 0.0
    prev_error = 0.0
    prev_time = time.monotonic()
    period = 1.0 / LOOP_HZ

    print("Balancing. Press Ctrl+C to stop.")
    try:
        while True:
            now = time.monotonic()
            dt = now - prev_time
            if dt < period:
                time.sleep(period - dt)
                now = time.monotonic()
                dt = now - prev_time
            prev_time = now

            pitch = doublemotor.imu_device.pitch
            if pitch != pitch:  # NaN guard: no IMU notification yet
                continue

            # NOTE: pitch is the rotation about the wheel axle (nose
            # up/down), which is the correct axis for forward/back
            # balance on a two-wheeled base. If tilting the robot
            # forward/back by hand doesn't move this value, your build
            # is mounted differently - use doublemotor.imu_device.roll
            # instead.
            error = pitch - setpoint_deg

            if abs(error) > FALL_ANGLE_DEG:
                doublemotor.motor_set_duty_cycle(0, motor=le.MOTOR_BOTH, blocking=False)
                integral = 0.0
                prev_error = error
                continue

            integral += error * dt
            derivative = (error - prev_error) / dt if dt > 0 else 0.0
            prev_error = error

            # NOTE: if the robot accelerates away from vertical instead of
            # catching itself, the sign convention is flipped for your
            # mounting - negate KP/KI/KD (or the whole output) and retest.
            output = KP * error + KI * integral + KD * derivative
            duty = max(-MAX_DUTY, min(MAX_DUTY, output))

            doublemotor.motor_set_duty_cycle(duty, motor=le.MOTOR_BOTH, blocking=False)
    except KeyboardInterrupt:
        pass
    finally:
        doublemotor.motor_stop(motor=le.MOTOR_BOTH)


def main():
    doublemotor = le.DoubleMotor()

    print("Connecting to Double Motor (Azure card, serial 1096)...")
    doublemotor.connect(
        card_color=CARD_COLOR,
        card_serial=CARD_SERIAL,
        device_notification_delay=15,  # fastest allowed IMU update rate
    )

    if not doublemotor.connected:
        print("Could not connect. Check the hub is powered on and the "
              "Azure card (1096) is attached.")
        return

    try:
        print("Hold the robot balanced upright and still...")
        time.sleep(1.0)
        setpoint = calibrate_upright(doublemotor)
        print(f"Calibrated upright pitch: {setpoint:.2f} deg")

        balance(doublemotor, setpoint)
    finally:
        doublemotor.motor_set_end_state(le.MOTOR_END_STATE_BRAKE, motor=le.MOTOR_BOTH)
        doublemotor.disconnect()


if __name__ == "__main__":
    main()
