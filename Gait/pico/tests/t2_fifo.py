# tests/t2_fifo.py - build step 2: does the MPU-6050 really sample itself at
# 100 Hz through its FIFO, without losing samples?
#
# Before running: upload config.py and mpu6050_fifo.py to the Pico W.
# To run: open this file in Thonny and press F5. Takes about 62 s.
#
# Part 1 (60 s): empties the FIFO every ~100 ms, the same way main.py does,
#   and counts the samples. Expect about 6000 samples, 0 overflows.
# Part 2 (2 s): deliberately stops reading for 1.2 s, longer than the FIFO
#   can hold (0.85 s), to prove that overflow detection works.

import struct
import sys
import time
from machine import I2C, Pin

import config
from mpu6050_fifo import MPU6050, BYTES_PER_SAMPLE, FIFO_SIZE

TEST_SECONDS = 60

print("t2_fifo: MPU-6050 FIFO timing test (%d s)" % TEST_SECONDS)

i2c = I2C(config.I2C_ID, sda=Pin(config.I2C_SDA_PIN), scl=Pin(config.I2C_SCL_PIN),
          freq=config.I2C_FREQ)
if config.MPU_ADDR not in i2c.scan():
    print("ERROR: no MPU-6050 at 0x%02X. Run tests/t1_imu.py first to check the wiring."
          % config.MPU_ADDR)
    sys.exit()

imu = MPU6050(i2c, config.MPU_ADDR, config.MAX_SAMPLES_PER_PACKET, config.WHO_AM_I_OK)
imu.configure(config.DLPF_CFG, config.SMPLRT_DIV, config.ACCEL_RANGE_G, config.GYRO_RANGE_DPS)
imu.fifo_start()
print("FIFO running. Move the sensor around if you like; min/max are reported at the end.")

NAMES = ("ax", "ay", "az", "gx", "gy", "gz")
lo = [32767] * 6
hi = [-32768] * 6
samples = 0
overflows = 0
max_fill = 0           # largest FIFO_COUNT seen, in bytes
t_start = time.ticks_ms()
t_last_print = t_start

while time.ticks_diff(time.ticks_ms(), t_start) < TEST_SECONDS * 1000:
    if imu.fifo_overflowed():
        overflows += 1
        print("FIFO overflow! (count %d) - resetting FIFO" % overflows)
        imu.fifo_start()
        continue

    count = imu.fifo_count()
    if count > max_fill:
        max_fill = count
    k = count // BYTES_PER_SAMPLE           # whole samples only
    while k > 0:
        take = min(k, config.MAX_SAMPLES_PER_PACKET)
        view = imu.fifo_read(take)
        for s in range(take):
            vals = struct.unpack_from(">6h", view, s * BYTES_PER_SAMPLE)
            for i in range(6):
                if vals[i] < lo[i]:
                    lo[i] = vals[i]
                if vals[i] > hi[i]:
                    hi[i] = vals[i]
        samples += take
        k -= take

    now = time.ticks_ms()
    if time.ticks_diff(now, t_last_print) >= 5000:
        t_last_print = now
        print("  %2d s: %5d samples" % (time.ticks_diff(now, t_start) // 1000, samples))
    time.sleep_ms(config.LOOP_PERIOD_MS)

elapsed_s = time.ticks_diff(time.ticks_ms(), t_start) / 1000
print()
print("Part 1 result")
print("  samples collected : %d in %.1f s (expect about %d)"
      % (samples, elapsed_s, elapsed_s * config.SAMPLE_RATE_HZ))
print("  effective rate    : %.2f Hz (expect 100 +- 1; the MPU-6050 clock is +-1%%)"
      % (samples / elapsed_s))
print("  overflows         : %d (expect 0)" % overflows)
print("  max FIFO fill     : %d of %d bytes (expect about 120-250)" % (max_fill, FIFO_SIZE))
print("  raw min / max (counts, and converted):")
for i in range(6):
    scale = config.ACCEL_LSB_PER_G if i < 3 else config.GYRO_LSB_PER_DPS
    unit = "g" if i < 3 else "deg/s"
    print("    %s %7d .. %6d   (%+8.2f .. %+8.2f %s)"
          % (NAMES[i], lo[i], hi[i], lo[i] / scale, hi[i] / scale, unit))

print()
print("Part 2: not reading for 1.2 s to force an overflow...")
imu.fifo_start()
time.sleep_ms(1200)
count = imu.fifo_count()
flag = imu.fifo_overflowed()
print("  FIFO_COUNT = %d (expect %d, i.e. full)" % (count, FIFO_SIZE))
print("  overflow detected = %s (expect True)" % flag)
imu.fifo_start()

ok = (abs(samples - elapsed_s * config.SAMPLE_RATE_HZ) < elapsed_s * 2 and overflows == 0 and flag)
print()
print("t2_fifo: PASS" if ok else "t2_fifo: CHECK THE NUMBERS ABOVE")
