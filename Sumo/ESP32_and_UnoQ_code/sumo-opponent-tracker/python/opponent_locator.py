"""Turns the detector's black-and-white mask into the opponent's position.

The mask (built by color_detector.py - NOT changed here) isn't a clean
outline of the robot: it's a loose collection of separate white clusters
(a pink wheel here, an azure brick there, ...). So, every frame:

  1. GROUP nearby clusters: dilate (grow) a COPY of the mask with a large
     round kernel, so clusters closer than about 2 x GROUP_RADIUS pixels
     touch and become one connected "group". The copy is used ONLY to
     decide which pixels belong together - all measurements below use the
     ORIGINAL mask's white pixels, so dilation never shifts the center.
  2. PICK the group containing the most original white pixels. If even the
     best group has fewer than MIN_PIXELS, it's noise: nothing is seen.
  3. MEASURE that group, using only its original white pixels:
       - center = their average position (center of mass),
       - x = how far left/right the center is, from -1 (left edge) to +1
         (right edge), smoothed over a few frames,
       - distance = from the lowest white pixel (the robot's bottom edge,
         where it touches the floor): lower in the picture = closer.
  4. DEBOUNCE: only report visible=True after the group has been there for
     DEBOUNCE_FRAMES frames in a row; report visible=False as soon as it's
     gone.

find_best_group() is a plain function (same mask in -> same answer out).
OpponentTracker wraps it with the things that need memory across frames:
the smoothing, the debounce streak, and the recent centers used for
confidence.
"""

import time
from collections import deque, namedtuple

import cv2
import numpy as np

# Everything main.py needs about one frame's answer.
OpponentResult = namedtuple(
    "OpponentResult",
    [
        "visible",        # bool - debounced "is the opponent there?"
        "x",              # float -1..+1 (smoothed), or None when not visible
        "distance_cm",    # float, or None when not visible
        "confidence",     # float 0..1
        "group_count",    # how many groups the dilated copy split into (for logging)
        "group_pixels",   # original white pixels in the chosen group (0 if none)
        "center",         # (cx, cy) pixel center of the chosen group, or None
        "outline",        # convex hull of the chosen group's pixels (for drawing), or None
        "processing_ms",  # how long this step took
    ],
)


def find_best_group(mask, group_radius, min_pixels):
    """Stateless grouping. Returns (group_count, group) where group is None
    (nothing big enough) or a dict with "pixels", "center_x", "center_y",
    "bottom_y" and "outline" - all measured from the ORIGINAL mask."""
    # --- 1. Group: grow a copy of the mask so nearby clusters merge ---
    kernel_size = 2 * group_radius + 1  # a circle of radius group_radius
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    grouped = cv2.dilate(mask, kernel)

    # connectedComponents gives every separate white region in the grown
    # copy its own number ("label"): 0 = background, 1..N = groups.
    num_labels, labels = cv2.connectedComponents(grouped)
    group_count = num_labels - 1
    if group_count == 0:
        return 0, None

    # --- 2. Pick: count ORIGINAL white pixels per group ---
    ys, xs = np.nonzero(mask)             # coordinates of every original white pixel
    pixel_labels = labels[ys, xs]         # which group each of them landed in
    counts = np.bincount(pixel_labels, minlength=num_labels)
    counts[0] = 0                         # never pick the background
    best_label = int(np.argmax(counts))
    best_pixels = int(counts[best_label])
    if best_pixels < min_pixels:
        return group_count, None          # biggest group is still just noise

    # --- 3. Measure: only the original white pixels inside that group ---
    in_group = pixel_labels == best_label
    gx, gy = xs[in_group], ys[in_group]
    points = np.column_stack((gx, gy)).astype(np.int32)
    return group_count, {
        "pixels": best_pixels,
        "center_x": float(gx.mean()),     # center of mass, not bounding-box center
        "center_y": float(gy.mean()),
        "bottom_y": int(gy.max()),        # lowest white pixel = robot's bottom edge
        "outline": cv2.convexHull(points),
    }


def distance_from_bottom(bottom_y, horizon_row, own_robot_row, near_cm, far_cm):
    """Simple linear distance: a bottom edge at the own-robot line (bottom of
    the detection area) is near_cm away, one at the horizon line (top of the
    detection area) is far_cm away, and anything between is in proportion."""
    span = own_robot_row - horizon_row
    if span <= 0:
        return float(near_cm)  # lines set in a strange order - avoid dividing by zero
    frac = (own_robot_row - bottom_y) / span  # 0 at the bottom, 1 at the horizon
    frac = max(0.0, min(1.0, frac))
    return float(near_cm + frac * (far_cm - near_cm))


class OpponentTracker:
    """Remembers what it needs across frames (smoothing, debounce, recent
    centers). Use one instance per video source - the live loop has its own,
    and a recording replay makes a fresh one so it can't disturb the live one."""

    def __init__(self, group_radius, min_pixels, x_smoothing, debounce_frames,
                 near_cm, far_cm, full_confidence_pixels, stability_frames, stability_spread):
        self.group_radius = group_radius
        self.min_pixels = min_pixels
        self.x_smoothing = x_smoothing
        self.debounce_frames = debounce_frames
        self.near_cm = near_cm
        self.far_cm = far_cm
        self.full_confidence_pixels = full_confidence_pixels
        self.stability_spread = stability_spread

        self._seen_streak = 0      # frames in a row that had a group
        self._smoothed_x = None    # exponential-smoothing memory (None = start fresh)
        self._recent_x = deque(maxlen=stability_frames)  # raw x of the last few frames

    def update(self, mask, horizon_row, own_robot_row):
        """Processes one frame's mask and returns an OpponentResult."""
        start = time.monotonic()
        width = mask.shape[1]
        group_count, group = find_best_group(mask, self.group_radius, self.min_pixels)

        if group is None:
            # Gone: forget everything, so the next sighting starts fresh
            # (no smoothing "drag" from where it was seconds ago).
            self._seen_streak = 0
            self._smoothed_x = None
            self._recent_x.clear()
            return OpponentResult(False, None, None, 0.0, group_count, 0, None, None,
                                  (time.monotonic() - start) * 1000)

        # x: -1 = left edge, 0 = middle, +1 = right edge
        half_width = width / 2
        raw_x = max(-1.0, min(1.0, (group["center_x"] - half_width) / half_width))

        # Exponential smoothing: new = a * this frame + (1 - a) * previous.
        # a = 1 means no smoothing; smaller a = smoother but slower to follow.
        if self._smoothed_x is None:
            self._smoothed_x = raw_x
        else:
            self._smoothed_x = self.x_smoothing * raw_x + (1 - self.x_smoothing) * self._smoothed_x

        self._seen_streak += 1
        self._recent_x.append(raw_x)
        visible = self._seen_streak >= self.debounce_frames

        # Confidence = (is the group big enough?) x (is its center steady?)
        size_score = min(1.0, group["pixels"] / self.full_confidence_pixels)
        spread = max(self._recent_x) - min(self._recent_x)  # how much x jumped around lately
        stability_score = max(0.0, 1.0 - spread / self.stability_spread)
        confidence = size_score * stability_score

        distance_cm = None
        if visible:
            distance_cm = distance_from_bottom(group["bottom_y"], horizon_row, own_robot_row,
                                               self.near_cm, self.far_cm)

        return OpponentResult(
            visible=visible,
            x=self._smoothed_x if visible else None,
            distance_cm=distance_cm,
            confidence=float(confidence),
            group_count=group_count,
            group_pixels=group["pixels"],
            center=(int(round(group["center_x"])), int(round(group["center_y"]))),
            outline=group["outline"],
            processing_ms=(time.monotonic() - start) * 1000,
        )


def draw_overlay(bgr_frame, result, horizon_row, own_robot_row):
    """A copy of the camera frame with the horizon/own-robot lines, the
    chosen group's outline (green = visible, gray = seen but not confirmed
    yet by the debounce), a red dot at its center, and the numbers."""
    overlay = bgr_frame.copy()
    width = overlay.shape[1]
    cv2.line(overlay, (0, horizon_row), (width, horizon_row), (0, 255, 255), 1)       # yellow
    cv2.line(overlay, (0, own_robot_row), (width, own_robot_row), (255, 0, 255), 1)   # magenta

    if result.outline is not None:
        color = (0, 255, 0) if result.visible else (160, 160, 160)
        cv2.polylines(overlay, [result.outline], True, color, 2)
        cv2.circle(overlay, result.center, 4, (0, 0, 255), -1)

    x_str = f"{result.x:+.2f}" if result.x is not None else "-"
    d_str = f"{result.distance_cm:.0f}cm" if result.distance_cm is not None else "-"
    cv2.putText(overlay, f"vis={result.visible} x={x_str} d={d_str}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return overlay
