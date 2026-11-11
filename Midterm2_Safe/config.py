"""Project settings shared by the app, the ML scripts and (later) the UNO Q.

Anything that changes what the model sees (crop, sizes, grayscale) is copied into
models/model_config.json at export time, so deployed code never guesses it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# ---- camera (Stage 1 findings) ---------------------------------------------
# Pick the overhead DICCIK webcam by its Windows name, not by index: Windows can
# swap indexes (on 2026-11-10 it was index 0, earlier index 1). Run
# camera_names.py to see the names.
CAMERA_NAME = "icspring camera"   # the DICCIK USB webcam; laptop cam is "Integrated Webcam"
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# ---- crop: the region the model sees -----------------------------------------
# A rectangle matching the green pad, as fractions of the frame (x0, y0, x1, y1),
# so it survives a resolution change. Measured 2026-11-10: the pad fills the
# frame except a thin dark strip at the bottom and right. The rectangle is then
# LETTERBOXED (scaled to fit, padded top/bottom) into a square, so the whole pad
# is kept (the user wants room to move) and the model input is still square.
# Changing any of this after collecting data does NOT change saved images (they
# are already cropped), so freeze it once collection starts and record any
# change in handoff.md.
CROP_X0 = 0.0
CROP_Y0 = 0.0
CROP_X1 = 0.98
CROP_Y1 = 0.96
LETTERBOX_FILL = 0        # gray value of the padding bands (black)

RAW_SAVE_SIZE = 224       # saved PNG: 224x224 letterboxed square

# ---- model input ------------------------------------------------------------
INPUT_SIZE = 96           # 96x96
# Color (BGR, 3 channels): in grayscale a finger's dark shadow looks like another
# finger (val 83%); in color, skin vs. green-pad shadow is obvious (val 97.9%).
GRAYSCALE = False

# ---- data collection --------------------------------------------------------
BURST_FPS = 8
BURST_SECONDS = 6
COUNTDOWN_SECONDS = 3
TARGET_PER_CLASS = 250    # training images per class (2 sessions x 3 bursts x 48 = 288)
# Sessions whose ID starts with this are the held-out validation set; all other
# sessions are training. The split is by SESSION, never by frame.
VAL_SESSION_PREFIX = "val"

# ---- live inference ---------------------------------------------------------
SMOOTH_FRAMES = 5         # average probabilities over this many frames
HOLD_SECONDS = 1.0        # a count must stay on top this long to register
CONF_THRESHOLD = 0.80     # minimum smoothed probability while holding

# ---- lock demo --------------------------------------------------------------
SECRET_CODE = (3, 1, 4)   # three DIFFERENT numbers 1-5 (lock.validate_code rejects repeats)
RESULT_SECONDS = 4        # how long UNLOCKED / STAYS LOCKED stays on screen
AWAKE_TIMEOUT_SECONDS = 30  # go back to sleep after this long with no digit progress
