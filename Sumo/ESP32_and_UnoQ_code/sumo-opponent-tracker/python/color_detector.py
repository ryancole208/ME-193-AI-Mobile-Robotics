"""Classical (no neural network) opponent detector for the sumo ring.

Pipeline, per frame (see _find_target() for the actual order):

  1. REGION OF INTEREST: mask out everything above the horizon line
     (walls, people, ceiling) and everything below the own-robot line
     (your own robot's body, if visible at the bottom of the frame).
  2. COLOR MASKS: convert to HSV and threshold for each configured
     opponent color (azure, pink, white, wheel-gray by default) -
     HSV is used instead of RGB because it separates "what color" (hue)
     from "how bright/washed out" (value/saturation), which stays far more
     stable than RGB under the ring's changing lighting.
  3. CLEAN UP: blur, then morphological opening (erode+dilate, removes
     small speckles) and closing (dilate+erode, merges a robot's
     different-colored parts - e.g. an azure body next to pink wheels -
     into one connected blob instead of several separate ones).
  4. CANDIDATES: find connected components (contours) in the cleaned,
     merged mask. Each one is a candidate "maybe a robot" blob.
  5. REJECT FALSE POSITIVES: drop candidates that are
       - too small or implausibly large,
       - shaped like a thin line (the blue ring tape, not a robot),
       - colored like the calibrated tape color,
       - touching the horizon line (probably a cut-off wall/light, not a
         fully-visible robot).
  6. PICK A TARGET: of the surviving candidates, the one with the lowest
     bottom edge is closest to the camera (see distance_estimator.py for
     why that's true on a flat floor) - sumo matches only ever have one
     opponent at a time, so "closest" is also "the" opponent.

_find_target() above is a plain function with no state - the same frame
and config always give the same answer. ColorDetector wraps it with two
stateful things that need to persist across frames: debouncing/the lost-
target grace period, and the distance estimator's smoothing. Those live
only on ColorDetector.detect()'s path; find_raw_target() (used by distance
calibration capture) deliberately skips them, both because calibration
wants the instantaneous reading and because it can be called from a
different thread (a WebUI message handler) than the main detection loop -
sharing the debounce/smoothing state across threads would be a real (if
rare) race.
"""

import time
from collections import namedtuple

import cv2
import numpy as np

import distance_estimator

DetectionResult = namedtuple(
    "DetectionResult",
    ["visible", "x", "distance_cm", "bbox", "confidence", "colors_found", "processing_ms"],
)

# These two colors are identified by hue, which is noisy/meaningless on
# pixels that are dark or barely saturated (near-gray) - "white" and "wheel"
# are deliberately defined by value/saturation instead of a narrow hue, so
# they don't need (and shouldn't get) this extra floor.
_HUE_DEPENDENT_COLORS = {"azure", "pink"}


def _hsv_mask(hsv, color_range):
    """Thresholds an HSV image for one configured color range. Handles hue
    wraparound (h[0] > h[1]) by OR-ing two ranges together, since OpenCV's
    hue axis is circular (0-179) but inRange() only does simple intervals."""
    h_lo, h_hi = color_range["h"]
    s_lo, s_hi = color_range["s"]
    v_lo, v_hi = color_range["v"]
    if h_lo <= h_hi:
        return cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))
    low_part = cv2.inRange(hsv, (0, s_lo, v_lo), (h_hi, s_hi, v_hi))
    high_part = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (179, s_hi, v_hi))
    return cv2.bitwise_or(low_part, high_part)


def _roi_mask(frame_shape, config):
    """A mask that's 255 between the horizon line and the own-robot line,
    0 everywhere else - multiplying any color mask by this (bitwise_and)
    confines detection to that band of the image."""
    height, width = frame_shape[:2]
    horizon_row = int(height * config["horizon_row_frac"])
    own_robot_row = int(height * config["own_robot_row_frac"])
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[horizon_row:own_robot_row, :] = 255
    return mask, horizon_row, own_robot_row


def _mean_hsv_in_mask(hsv, mask):
    mean = cv2.mean(hsv, mask=mask)  # returns (h, s, v, _)
    return mean[0], mean[1], mean[2]


def _color_distance(hsv_a, hsv_b):
    """Rough HSV distance, treating hue as circular (so 179 and 0 count as
    close together, matching how hue actually wraps around)."""
    dh = min(abs(hsv_a[0] - hsv_b[0]), 180 - abs(hsv_a[0] - hsv_b[0]))
    ds = hsv_a[1] - hsv_b[1]
    dv = hsv_a[2] - hsv_b[2]
    return (dh * dh + ds * ds + dv * dv) ** 0.5


def _roi_coverage_pct(mask, roi_pixel_count):
    """What fraction of the ROI (not the whole frame) this mask covers, as
    a percentage - the number the diagnostic logging is built around."""
    if roi_pixel_count == 0:
        return 0.0
    return 100.0 * cv2.countNonZero(mask) / roi_pixel_count


def _roi_hsv_percentiles(hsv, roi_mask):
    """Median and 10th/90th percentile H/S/V of every pixel inside the ROI
    (regardless of color) - diagnostic context for "what does the scene
    actually look like right now," independent of any configured color."""
    ys, xs = np.nonzero(roi_mask)
    if len(ys) == 0:
        return None
    pixels = hsv[ys, xs]  # shape (N, 3)
    result = {}
    for i, channel in enumerate("hsv"):
        values = pixels[:, i]
        result[channel] = {
            "p10": float(np.percentile(values, 10)),
            "median": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }
    return result


def _find_target(bgr_frame, config):
    """Stateless: runs the full color-mask + cleanup + candidate-rejection
    pipeline on one frame. Returns (target_or_None, horizon_row,
    own_robot_row, cleaned_mask, per_color_masks, diagnostics). target,
    when not None, is a dict with "bbox" (x, y, w, h), "area_frac", and
    "colors_found". per_color_masks is {name: raw HSV mask} before
    cleanup, for the debug page's "show just this color" view.
    diagnostics is {"coverage_pct": {name: pct, ..., "combined_raw": pct,
    "cleaned": pct}, "roi_hsv": {...} or None, "excluded_masks": [name, ...],
    "warnings": [str, ...]} - see _roi_coverage_pct/_roi_hsv_percentiles
    above; logged once a second by main.py, and warnings are also shown on
    the debug page. excluded_masks lists any per-color mask that covered
    more than config["mask_saturated_warning_pct"] of the ROI and was left
    out of combined for that reason."""
    height, width = bgr_frame.shape[:2]
    frame_area = height * width

    blur_ksize = config["blur_ksize"] | 1  # cv2 requires an odd kernel size
    blurred = cv2.GaussianBlur(bgr_frame, (blur_ksize, blur_ksize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    roi_mask, horizon_row, own_robot_row = _roi_mask(bgr_frame.shape, config)
    roi_pixel_count = cv2.countNonZero(roi_mask)

    min_sat_colored = config["min_saturation_colored"]
    min_val_hue = config["min_value_for_hue"]
    hue_floor_mask = cv2.inRange(hsv, (0, min_sat_colored, min_val_hue), (179, 255, 255))

    per_color_masks = {}
    for name, color_range in config["colors"].items():
        mask = cv2.bitwise_and(_hsv_mask(hsv, color_range), roi_mask)
        if name in _HUE_DEPENDENT_COLORS:
            # Enforced on top of the calibrated range itself (see the
            # DEFAULT_CONFIG comment) - a near-gray, overexposed pixel can
            # still fall inside e.g. pink's hue range, so hue alone isn't
            # enough for these two colors.
            mask = cv2.bitwise_and(mask, hue_floor_mask)
        per_color_masks[name] = mask
    if config["floor_fallback_enabled"] and config["floor_color"] is not None:
        # Generic fallback: anything that doesn't match the learned floor
        # color is also a candidate, even if it doesn't match one of the
        # named opponent colors above - catches parts of the opponent
        # that are some other color mixed in. Off by default (opt-in via
        # the "use floor fallback" checkbox) because a real floor usually
        # has lighting gradients/shadows wider than one sampled point
        # covers, which can make this flag almost the whole image.
        per_color_masks["non_floor"] = cv2.bitwise_and(
            cv2.bitwise_not(_hsv_mask(hsv, config["floor_color"])), roi_mask
        )

    coverage_pct = {name: _roi_coverage_pct(mask, roi_pixel_count) for name, mask in per_color_masks.items()}

    # Any single mask covering most of the ROI is almost certainly a
    # lighting/color-floor problem, not a real target that big - exclude it
    # from combined rather than let it swamp everything else, and surface
    # why (both in the console log and the debug page).
    warning_threshold = config["mask_saturated_warning_pct"]
    excluded_masks = sorted(name for name, pct in coverage_pct.items() if pct > warning_threshold)
    warnings = [
        f'"{name}" mask covers {coverage_pct[name]:.0f}% of the ROI (> {warning_threshold:.0f}%) - excluded from detection'
        for name in excluded_masks
    ]

    # combined is only ever OR-ed from masks that are already AND-ed with
    # roi_mask above - the ROI itself is never OR-ed in directly, which
    # would defeat the point of having one (see the module docstring's
    # diagnostic notes for why this matters).
    combined = np.zeros((height, width), dtype=np.uint8)
    for name, mask in per_color_masks.items():
        if name not in excluded_masks:
            combined = cv2.bitwise_or(combined, mask)

    coverage_pct["combined_raw"] = _roi_coverage_pct(combined, roi_pixel_count)

    open_k = config["morph_open_ksize"] | 1
    close_k = config["morph_close_ksize"] | 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    coverage_pct["cleaned"] = _roi_coverage_pct(cleaned, roi_pixel_count)
    diagnostics = {
        "coverage_pct": coverage_pct,
        "roi_hsv": _roi_hsv_percentiles(hsv, roi_mask),
        "excluded_masks": excluded_masks,
        "warnings": warnings,
    }

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        box_area = w * h
        if box_area == 0:
            continue

        area_frac = area / frame_area
        if area_frac < config["min_area_frac"] or area_frac > config["max_area_frac"]:
            continue

        aspect_ratio = min(w, h) / max(w, h)
        fill_ratio = area / box_area
        if aspect_ratio < config["min_aspect_ratio"] or fill_ratio < config["min_fill_ratio"]:
            continue  # thin/long/diagonal - almost certainly the ring's blue tape, not a robot

        if y <= horizon_row + 2:
            continue  # touches the horizon line - likely a cut-off wall/light, not a fully-visible robot

        blob_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)

        if config["tape_color"] is not None:
            mean_hsv = _mean_hsv_in_mask(hsv, blob_mask)
            tape_hsv = config["tape_color"]
            tape_center = (
                sum(tape_hsv["h"]) / 2 % 180,
                sum(tape_hsv["s"]) / 2,
                sum(tape_hsv["v"]) / 2,
            )
            if _color_distance(mean_hsv, tape_center) < config["tape_color_margin"]:
                continue  # matches the learned tape color closely - reject

        colors_found = [
            name for name, mask in per_color_masks.items()
            if name not in excluded_masks and cv2.countNonZero(cv2.bitwise_and(mask, blob_mask)) > 0.05 * area
        ]

        candidates.append({
            "bbox": (x, y, w, h),
            "area_frac": area_frac,
            "colors_found": colors_found,
        })

    # Closest to the robot = lowest bottom edge in the image (see
    # distance_estimator.py for why that holds on a flat floor).
    target = max(candidates, key=lambda c: c["bbox"][1] + c["bbox"][3], default=None)
    return target, horizon_row, own_robot_row, cleaned, per_color_masks, diagnostics


class ColorDetector:
    """Stateful across frames only for debouncing/grace-period smoothing
    and the distance estimator's own smoothing - everything else is a pure
    function of the current frame and the current config."""

    def __init__(self):
        self._visible_state = False
        self._pending_state = None
        self._pending_count = 0
        self._last_seen_time = None
        self._last_target = None  # the last real (non-grace-period) target, held onto during the lost_grace_s window
        self._distance_estimator = distance_estimator.DistanceEstimator()

    def find_raw_target(self, bgr_frame, config):
        """A single-frame reading with no debouncing/smoothing applied -
        see the module docstring for why distance calibration capture uses
        this instead of detect(). Returns the same target dict _find_target
        does (or None)."""
        target, _, _, _, _, _ = _find_target(bgr_frame, config)
        return target

    def detect(self, bgr_frame, config, now):
        """Runs the full pipeline on one BGR frame (as decoded by
        cv2.imdecode), plus this instance's debouncing and distance
        smoothing. `now` is a monotonic timestamp (time.monotonic()), used
        for the lost-target grace period. Only call this from one thread
        at a time (the main detection loop) - see the module docstring.

        Returns (result, debug) where result is a DetectionResult and
        debug is a dict of extra data for the web debug page and console
        diagnostics (not needed for normal operation): {"mask": <uint8 HxW,
        the cleaned/merged mask>, "overlay": <BGR frame with box/line
        drawn>, "per_color_masks": {name: <uint8 HxW, raw mask before
        cleanup>}, "diagnostics": see _find_target's docstring}.
        """
        start = time.monotonic()
        height, width = bgr_frame.shape[:2]

        target, horizon_row, own_robot_row, cleaned, per_color_masks, diagnostics = _find_target(bgr_frame, config)

        if target is not None:
            self._last_target = target

        raw_visible = target is not None
        visible = self._debounce(raw_visible, config, now)
        # During the lost_grace_s window, raw_visible is False (nothing was
        # found THIS frame) but visible is still True - report the last
        # real target's position/size rather than snapping to "nothing"
        # for that brief window.
        effective_target = target if target is not None else self._last_target

        if visible and effective_target is not None:
            x, y, w, h = effective_target["bbox"]
            x_pos = (((x + w / 2) / width) * 2) - 1
            distance_cm = self._distance_estimator.estimate(
                bottom_row_frac=(y + h) / height,
                bbox_height_frac=h / height,
                config=config,
            )
            confidence = min(1.0, 0.4 + 0.3 * len(effective_target["colors_found"]) + 0.3 * min(effective_target["area_frac"] / 0.05, 1.0))
            result = DetectionResult(
                visible=True,
                x=x_pos,
                distance_cm=distance_cm,
                bbox=effective_target["bbox"],
                confidence=confidence,
                colors_found=effective_target["colors_found"],
                processing_ms=(time.monotonic() - start) * 1000,
            )
        else:
            result = DetectionResult(
                visible=False,
                x=0.0,
                distance_cm=None,
                bbox=None,
                confidence=0.0,
                colors_found=[],
                processing_ms=(time.monotonic() - start) * 1000,
            )

        overlay = bgr_frame.copy()
        cv2.line(overlay, (0, horizon_row), (width, horizon_row), (0, 255, 255), 1)
        cv2.line(overlay, (0, own_robot_row), (width, own_robot_row), (255, 0, 255), 1)
        if result.bbox is not None:
            x, y, w, h = result.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{result.distance_cm:.0f}cm" if result.distance_cm is not None else "?"
            cv2.putText(overlay, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return result, {"mask": cleaned, "overlay": overlay, "per_color_masks": per_color_masks, "diagnostics": diagnostics}

    def _debounce(self, raw_visible, config, now):
        """Requires debounce_frames consecutive agreeing frames before
        flipping state, and holds "visible" for lost_grace_s after the
        target disappears, so a single missed frame doesn't cause a blip."""
        if raw_visible:
            self._last_seen_time = now

        candidate = raw_visible or (
            self._visible_state
            and self._last_seen_time is not None
            and (now - self._last_seen_time) < config["lost_grace_s"]
        )

        if candidate == self._visible_state:
            self._pending_state = None
            self._pending_count = 0
            return self._visible_state

        if candidate == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = candidate
            self._pending_count = 1

        if self._pending_count >= config["debounce_frames"]:
            self._visible_state = candidate
            self._pending_state = None
            self._pending_count = 0

        return self._visible_state
