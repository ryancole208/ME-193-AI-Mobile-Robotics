# config.py - every setting for the GaitNode ankle sensor in one place.
#
# Upload this file to the Pico W (Thonny: File > Save as > Raspberry Pi Pico,
# name it config.py). The other files read their settings from here, so
# change values here rather than in the other files.

# ---------------------------------------------------------------------------
# Development switches
# ---------------------------------------------------------------------------

# True: print status messages to Thonny's shell.
# False: print nothing (normal battery operation; printing with no USB
# attached is harmless but wastes a little time).
DEBUG = True

# Hardware watchdog: reboots the board if the main loop freezes for
# WATCHDOG_TIMEOUT_MS. Leave this False while developing. Once a watchdog is
# started it CANNOT be stopped, so after Ctrl+C in Thonny the board would
# reboot itself every 8 s and make uploading files painful.
# Turn it on only for the final battery-powered build (see README).
WATCHDOG_ENABLED = False
WATCHDOG_TIMEOUT_MS = 8000      # RP2040 maximum is about 8.3 s

# If main.py hits an unexpected error: True = print it and reboot after 5 s
# (the node recovers on its own on battery); False = stay stopped so you can
# read the error in Thonny.
REBOOT_ON_CRASH = True

# Seconds main.py waits at power-up so Ctrl+C in Thonny can always stop it.
STARTUP_DELAY_S = 3

# ---------------------------------------------------------------------------
# I2C bus and MPU-6050
# ---------------------------------------------------------------------------

I2C_ID = 0             # GP4/GP5 belong to the I2C0 peripheral
I2C_SDA_PIN = 4        # GP4 = physical pin 6
I2C_SCL_PIN = 5        # GP5 = physical pin 7
I2C_FREQ = 400_000     # 400 kHz "fast mode", the MPU-6050's maximum

MPU_ADDR = 0x68        # 0x69 if the AD0 pin on the breakout is pulled high

# Values the WHO_AM_I register (0x75) may return. A genuine MPU-6050 returns
# 0x68. Some cheap GY-521 boards carry clone chips that return something
# else; if t1_imu.py reports a mismatch and the sensor otherwise works,
# add the reported value here.
WHO_AM_I_OK = (0x68,)

# Sample timing. With the digital low-pass filter on (DLPF_CFG 1..6) the
# gyro output rate is 1 kHz, and
#     sample rate = 1000 Hz / (1 + SMPLRT_DIV) = 1000 / 10 = 100 Hz.
# DLPF_CFG = 3 gives about 44 Hz accel / 42 Hz gyro bandwidth.
DLPF_CFG = 3
SMPLRT_DIV = 9
SAMPLE_RATE_HZ = 100
SAMPLE_PERIOD_US = 10_000

# Full-scale ranges and the matching raw-count scale factors
# (register map Rev 4.0, registers 59-64 and 67-72).
ACCEL_RANGE_G = 16
GYRO_RANGE_DPS = 2000
ACCEL_LSB_PER_G = 2048      # at +-16 g
GYRO_LSB_PER_DPS = 16.4     # at +-2000 deg/s

# ---------------------------------------------------------------------------
# Packets (see "Shared wireless protocol" in README.md)
# ---------------------------------------------------------------------------

BYTES_PER_SAMPLE = 12          # ax ay az gx gy gz, int16 each
PACKET_HEADER_BYTES = 7        # uint16 seq + uint32 t0_us + uint8 n
MAX_SAMPLES_PER_PACKET = 10    # 7 + 12*10 = 127 bytes
LOOP_PERIOD_MS = 100           # read the FIFO about this often
RESEND_BUFFER_PACKETS = 20     # last 2 s kept for the RESEND command

# ---------------------------------------------------------------------------
# Battery (VSYS) measurement
# ---------------------------------------------------------------------------

# On the Pico W the VSYS sense pin (GPIO29) is shared with the wireless
# chip's SPI clock. If reading it ever disturbs Bluetooth, set this to
# False: the battery characteristic will then always send 0 (protocol:
# vsys_mV = 0 means "battery reading disabled").
BATTERY_READ_ENABLED = True
BATTERY_PERIOD_MS = 10_000

# ---------------------------------------------------------------------------
# Bluetooth Low Energy (must match the UNO Q side exactly)
# ---------------------------------------------------------------------------

# Heartbeat: the UNO Q writes PING to the control characteristic every 2 s.
# If a central stays connected but sends no PING for this long, the node
# assumes the link is dead (e.g. the UNO Q app restarted while BlueZ kept
# the connection open), disconnects it and advertises again. The timer
# starts at connect, so a central has this long to send its first PING.
# 0 turns the heartbeat off (useful for poking around with nRF Connect).
PING_TIMEOUT_MS = 6000

BLE_NAME = "GaitNode"
PROTOCOL_VERSION = 1
SERVICE_UUID = "6e400000-7a1b-4c2d-9e10-a5c0f0e1d200"
IMU_CHAR_UUID = "6e400001-7a1b-4c2d-9e10-a5c0f0e1d200"      # notify
BATTERY_CHAR_UUID = "6e400002-7a1b-4c2d-9e10-a5c0f0e1d200"  # notify + read
CONTROL_CHAR_UUID = "6e400003-7a1b-4c2d-9e10-a5c0f0e1d200"  # write
INFO_CHAR_UUID = "6e400004-7a1b-4c2d-9e10-a5c0f0e1d200"     # read
