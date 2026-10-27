"""Constants shared by every part of the wired app.

The packet layout is the same as the Bluetooth app's shared protocol (the
sketch builds it from the MPU-6050 FIFO). The display codes must match the
#defines at the top of sketch/sketch.ino.
"""

# ---------------------------------------------------------------- classes
CLASSES = ["still", "walking", "running", "biking"]
CLASS_LETTERS = ["S", "W", "R", "B"]
UNSURE = -1  # pseudo-label: model is unsure, shown as "?"

# ---------------------------------------------------------------- packets (from the sketch)
IMU_HEADER_BYTES = 7          # uint16 seq + uint32 t0_us + uint8 n
SAMPLE_BYTES = 12             # 6 x int16
MAX_SAMPLES_PER_PACKET = 10
SAMPLE_PERIOD_US = 10_000     # 100 Hz
SEQ_MODULO = 1 << 16
T0_MODULO = 1 << 32
MPU6050_WHO_AM_I = 0x68

# ---------------------------------------------------------------- display (== sketch/sketch.ino)
DISPLAY_PROTOCOL_VERSION = 2  # 2 = wired app

MODEL_A = 0
MODEL_B = 1

ST_NO_DATA = 2         # base: "IMU" then "X" (no samples for > 2 s)
ST_UNSURE = 3
ST_CFG_SETUP = 4       # transient: "CFG" then "?" (no usable config_wired.json, starting setup)
ST_LEG_MISMATCH = 5
ST_FLASH3 = 6
ST_CHECK = 7
ST_REDO_WINDOW_ON = 10
ST_REDO_WINDOW_OFF = 11
ST_FLASH1 = 12
ST_IDLE = 13

STEP_STAND = 0
STEP_COUNTDOWN_BASE = 10
STEP_RECORD_BASE = 20
STEP_GESTURE_CHECK = 30
STEP_BUILD = 31
STEP_PARTIAL = 32
STEP_SET_CONFIRM = 33
STEP_DIAG = 34
STEP_DIAG_DONE = 35
STEP_HOLDOUT_DONE = 36

VAL_ACC_A = 1
VAL_ACC_B = 2

# ---------------------------------------------------------------- config
SCHEMA_VERSION = 1
SOURCE = "wired"
