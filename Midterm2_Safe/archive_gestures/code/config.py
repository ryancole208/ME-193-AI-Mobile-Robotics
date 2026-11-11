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

# Which way "forward" (away from the user's body) points in the camera image.
# The user's fingers enter from the top edge with the tips pointing down.
FORWARD_DIRECTION = "down"   # one of: up, down, left, right
# The user confirmed on 2026-11-10 that with False the "your left" hint landed on
# their right, so the image is mirrored. Only places the hint on the collect page;
# labels come from the button the user picks, so this cannot corrupt the data.
IMAGE_MIRRORED = True

# ---- crop: the region the model sees -----------------------------------------
# A rectangle matching the green pad, as fractions of the frame (x0, y0, x1, y1),
# so it survives a resolution change. Measured 2026-11-10: the pad fills the
# frame except a thin dark strip at the bottom and right. Changing this after
# collecting data does NOT change saved images (they are already cropped), so
# freeze it once collection starts and record any change in handoff.md.
CROP_X0 = 0.0
CROP_Y0 = 0.0
CROP_X1 = 0.98
CROP_Y1 = 0.96

# Saved PNG size (width, height); about the crop's aspect ratio (1254x691 -> 1.81)
# and larger than the model input so the data stays reusable.
RAW_SAVE_SIZE = (320, 176)

# ---- model input ------------------------------------------------------------
INPUT_SIZE = (160, 88)    # (width, height), same aspect as RAW_SAVE_SIZE
GRAYSCALE = True

# ---- data collection --------------------------------------------------------
BURST_FPS = 8
BURST_SECONDS = 6
COUNTDOWN_SECONDS = 3
TARGET_PER_CLASS = 400

# ---- live inference ---------------------------------------------------------
SMOOTH_FRAMES = 5         # average probabilities over this many frames
HOLD_SECONDS = 1.0        # a gesture must stay on top this long to register
CONF_THRESHOLD = 0.80     # minimum smoothed probability while holding
