"""Constants shared by every part of the app.

The wireless protocol values come from the shared Pico W / UNO Q brief and must
not change on one side only. The display codes must match
the #defines at the top of sketch/sketch.ino.
"""

# ---------------------------------------------------------------- classes
CLASSES = ["still", "walking", "running", "biking"]
CLASS_LETTERS = ["S", "W", "R", "B"]
UNSURE = -1  # pseudo-label: model is unsure, shown as "?"

# ---------------------------------------------------------------- BLE protocol
DEVICE_NAME = "GaitNode"
SERVICE_UUID = "6e400000-7a1b-4c2d-9e10-a5c0f0e1d200"
IMU_UUID = "6e400001-7a1b-4c2d-9e10-a5c0f0e1d200"
BATTERY_UUID = "6e400002-7a1b-4c2d-9e10-a5c0f0e1d200"
CONTROL_UUID = "6e400003-7a1b-4c2d-9e10-a5c0f0e1d200"
INFO_UUID = "6e400004-7a1b-4c2d-9e10-a5c0f0e1d200"

PROTOCOL_VERSION = 1
IMU_HEADER_BYTES = 7          # uint16 seq + uint32 t0_us + uint8 n
SAMPLE_BYTES = 12             # 6 x int16
MAX_SAMPLES_PER_PACKET = 10
SAMPLE_PERIOD_US = 10_000     # 100 Hz
SEQ_MODULO = 1 << 16
T0_MODULO = 1 << 32
MIN_MTU_FOR_FULL_PACKET = IMU_HEADER_BYTES + SAMPLE_BYTES * MAX_SAMPLES_PER_PACKET + 3  # 130

# ---------------------------------------------------------------- display (== sketch/sketch.ino)
DISPLAY_PROTOCOL_VERSION = 1

MODEL_A = 0
MODEL_B = 1
MODEL_NAMES = {MODEL_A: "A", MODEL_B: "B"}

ST_SEARCHING = 1
ST_LINK_LOST = 2
ST_UNSURE = 3
ST_CFG_ERROR = 4
ST_LEG_MISMATCH = 5
ST_FLASH3 = 6
ST_CHECK = 7
ST_LOW_BATT_ON = 8
ST_LOW_BATT_OFF = 9
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
VAL_BATTERY = 3

# ---------------------------------------------------------------- config
SCHEMA_VERSION = 1
CONFIG_FILENAME = "config.json"
