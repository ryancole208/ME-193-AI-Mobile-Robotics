# tests/t1_imu.py - build step 1: is the MPU-6050 wired up, and do its axes
# match the convention?
#
# Before running: upload config.py and mpu6050_fifo.py to the Pico W.
# To run: open this file in Thonny and press F5 (Run). It does not need to
# be uploaded itself; Thonny sends it to the Pico W and runs it there.
# Stop with Ctrl+C (or Thonny's red Stop button).
#
# Prints acceleration in g and rotation in deg/s about 5 times a second,
# plus which sensor axis is currently pointing up.

import sys
import time
from machine import I2C, Pin

import config
from mpu6050_fifo import MPU6050

print("t1_imu: MPU-6050 wiring and axis check")

i2c = I2C(config.I2C_ID, sda=Pin(config.I2C_SDA_PIN), scl=Pin(config.I2C_SCL_PIN),
          freq=config.I2C_FREQ)
found = i2c.scan()
print("I2C scan found:", [hex(a) for a in found])

if config.MPU_ADDR not in found:
    if not found:
        print("ERROR: no I2C devices found. Check: VCC -> pin 36 3V3(OUT), "
              "GND -> GND, SDA -> GP4 (pin 6), SCL -> GP5 (pin 7). "
              "SDA and SCL swapped is the most common mistake.")
    else:
        print("ERROR: expected the MPU-6050 at 0x%02X but found %s. If it is at "
              "0x69 (AD0 pulled high), set MPU_ADDR = 0x69 in config.py."
              % (config.MPU_ADDR, [hex(a) for a in found]))
    sys.exit()

imu = MPU6050(i2c, config.MPU_ADDR, config.MAX_SAMPLES_PER_PACKET, config.WHO_AM_I_OK)
print("WHO_AM_I = 0x%02X" % imu.who_am_i())
imu.configure(config.DLPF_CFG, config.SMPLRT_DIV, config.ACCEL_RANGE_G, config.GYRO_RANGE_DPS)
print("Configured: +-%d g, +-%d deg/s, %d Hz, DLPF_CFG=%d"
      % (config.ACCEL_RANGE_G, config.GYRO_RANGE_DPS, config.SAMPLE_RATE_HZ, config.DLPF_CFG))
print("At rest, the axis pointing UP reads about +1.00 g and |a| is about 1.00.")
print()

AXES = ("X", "Y", "Z")
errors = 0
while True:
    try:
        ax, ay, az, _temp, gx, gy, gz = imu.read_raw()
    except OSError as e:
        errors += 1
        print("I2C error #%d: %s (check the 4 sensor wires); retrying in 1 s" % (errors, e))
        time.sleep(1)
        continue

    a = (ax / config.ACCEL_LSB_PER_G, ay / config.ACCEL_LSB_PER_G, az / config.ACCEL_LSB_PER_G)
    g = (gx / config.GYRO_LSB_PER_DPS, gy / config.GYRO_LSB_PER_DPS, gz / config.GYRO_LSB_PER_DPS)
    mag = (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5

    # The axis with the largest |a| is the one most nearly vertical.
    # Accelerometers read +1 g on the axis pointing UP, away from the ground.
    i = 0
    for k in (1, 2):
        if abs(a[k]) > abs(a[i]):
            i = k
    up = ("+" if a[i] > 0 else "-") + AXES[i]

    print("acc g  x %+5.2f y %+5.2f z %+5.2f  |a| %4.2f   gyro dps  x %+7.1f y %+7.1f z %+7.1f   up: %s"
          % (a[0], a[1], a[2], mag, g[0], g[1], g[2], up))
    time.sleep_ms(200)
