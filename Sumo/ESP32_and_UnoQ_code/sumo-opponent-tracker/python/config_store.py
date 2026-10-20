"""Loads and saves the detector's tunable settings (HSV color ranges, the
horizon line, rejection thresholds, distance calibration points, ...) as a
single JSON file at the app's root (config.json - sibling to python/,
sketch/, assets/). This is what the debug page's sliders and calibration
buttons read from and write to, so tuning never requires editing code or
restarting the app.

DEFAULT_CONFIG is the starting point for a brand new install, or for any
key a saved config.json doesn't have yet (so adding a new tunable here
later doesn't break an already-tuned config.json - old files just get the
new key's default merged in).
"""

import copy
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

# Each color entry is an HSV range: h/s/v are each [low, high], 0-179 for
# hue and 0-255 for saturation/value (OpenCV's convention). If h[0] > h[1],
# the range wraps around through 0 (needed for red/pink, which straddles
# hue 0 in OpenCV's 0-179 scale).
DEFAULT_CONFIG = {
    # ----- Region of interest -----
    "horizon_row_frac": 0.35,  # ignore everything above this fraction of the frame height (walls/people/ceiling)
    "own_robot_row_frac": 0.92,  # ignore everything below this fraction (your own robot's body, if visible at the bottom)

    # ----- Opponent color ranges (ASK ME during Step 3 tuning with real LEGO pieces) -----
    "colors": {
        "azure": {"h": [95, 125], "s": [70, 255], "v": [60, 255]},
        "pink": {"h": [155, 10], "s": [70, 255], "v": [60, 255]},  # wraps around hue 0
        "white": {"h": [0, 179], "s": [0, 35], "v": [180, 255]},
        "wheel": {"h": [0, 179], "s": [0, 60], "v": [20, 90]},  # dark gray/black LEGO tires
    },

    # ----- Learned during calibration (not hand-tuned) -----
    "floor_color": None,  # {"h": [...], "s": [...], "v": [...]} once "learn floor color" is run
    "tape_color": None,  # same shape, once "learn tape color" is run
    "floor_fallback_enabled": False,  # opt-in: treat "anything that isn't floor_color" as a candidate too. Off by
    # default because a real floor's lighting varies more than one sampled point covers, which can
    # flag almost the whole image - only turn this on if the named colors alone aren't enough.

    # ----- Guard rails for the hue-based colors (azure/pink) -----
    # Confirmed via the [diag] log on 2026-10-19: a bright/overexposed, nearly
    # colorless patch of floor matched "pink" purely because its hue fell in
    # pink's range, even though saturation was ~0 (gray, not pink) - pink's
    # configured range alone allowed s as low as 0. These two floors are
    # enforced in code on top of whatever range got learned/calibrated, so a
    # future wide calibration click can't reintroduce the same failure. All
    # three are live sliders on the debug page (SIMPLE_SLIDERS in app.js) -
    # tune them there while watching the mask instead of asking for a new
    # default here; 50/50/40 below are just starting points.
    "min_saturation_colored": 50,  # azure/pink pixels must have at least this much saturation to count, regardless of calibrated range
    "min_value_for_hue": 50,  # azure/pink pixels darker than this are excluded - hue is noisy/meaningless near-black
    "mask_saturated_warning_pct": 40.0,  # if any single color mask covers more than this % of the ROI, exclude it from combined and warn

    # ----- Cleanup -----
    "blur_ksize": 5,  # Gaussian blur kernel size (odd number) applied before color masking
    "morph_open_ksize": 3,  # erode+dilate size - removes small color-noise speckles
    "morph_close_ksize": 15,  # dilate+erode size - merges a robot's different-colored parts into one blob

    # ----- Rejection thresholds -----
    "min_area_frac": 0.004,  # a candidate blob must cover at least this fraction of the frame
    "max_area_frac": 0.5,  # ...and at most this fraction (bigger is probably a lighting/color flood, not a robot)
    "min_aspect_ratio": 0.2,  # min(w,h)/max(w,h) of the bounding box - rejects thin/long shapes like tape
    "min_fill_ratio": 0.35,  # contour area / bounding-box area - also rejects thin/diagonal line shapes
    "tape_color_margin": 15,  # how close (in HSV units) a blob's average color must be to tape_color to be rejected as tape

    # ----- Debounce -----
    "debounce_frames": 3,  # consecutive frames needed before flipping visible<->not-visible
    "lost_grace_s": 0.3,  # keep reporting the last known target for this long after losing it

    # ----- Distance calibration: list of {"row_frac": .., "bbox_height_frac": .., "distance_cm": ..} -----
    "distance_calibration_points": [],
}


def _deep_merge_defaults(value, default):
    """Fills in any keys missing from `value` (a loaded config) using
    `default`, recursively for nested dicts. Lets new tunables added to
    DEFAULT_CONFIG show up automatically in an older saved config.json."""
    if isinstance(default, dict):
        if not isinstance(value, dict):
            return copy.deepcopy(default)
        merged = dict(value)
        for key, default_value in default.items():
            merged[key] = _deep_merge_defaults(value.get(key), default_value)
        return merged
    return value if value is not None or default is None else default


def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[config_store] no usable config.json yet ({exc!r}); using defaults")
        saved = {}
    return _deep_merge_defaults(saved, DEFAULT_CONFIG)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config_store] saved settings to {os.path.abspath(CONFIG_PATH)}")
