"""Turns a detected blob's position and size into a distance estimate, in
centimeters, from calibration data collected with the calibration tool
(see main.py's "capture_calibration_point" handler and the debug page's
calibration panel).

Two independent signals are combined, because each one fails differently:

  - GROUND-PLANE (primary): on a flat floor, an object's bottom edge moves
    further down the image (a larger row fraction) the closer it gets to
    the camera. This only depends on where the object touches the floor,
    so it doesn't care about the robot's orientation - but it breaks if
    the object isn't actually touching the floor (e.g. a raised arm) or
    the floor isn't flat.
  - APPARENT SIZE (secondary): a closer object looks bigger. This doesn't
    care whether the object is touching the floor, but it's thrown off by
    the robot's orientation (a robot facing the camera looks a different
    size than one facing sideways).

Both are fit from the SAME calibration run (the tool records row position
AND bounding-box height at each measured distance), so one calibration
session is enough for both. Until at least 2 calibration points exist,
there's no calibration curve to use, so estimate() returns None rather
than inventing a number - the LED/debug display should show "distance
unknown" in that case, not a misleading guess.
"""

from collections import deque

import numpy as np

# ----- Tuning -----
GROUND_PLANE_WEIGHT = 0.7  # how much to trust the row-based estimate vs. the size-based one (must sum to 1 with below)
APPARENT_SIZE_WEIGHT = 0.3
SMOOTHING_WINDOW = 5  # distance is averaged over this many recent estimates


class DistanceEstimator:
    def __init__(self):
        self._recent = deque(maxlen=SMOOTHING_WINDOW)

    def estimate(self, bottom_row_frac, bbox_height_frac, config):
        points = config["distance_calibration_points"]
        if len(points) < 2:
            return None  # not calibrated yet - see module docstring

        row_estimate = self._interp(points, "row_frac", bottom_row_frac)
        size_estimate = self._interp(points, "bbox_height_frac", bbox_height_frac)

        if row_estimate is None:
            combined = size_estimate
        elif size_estimate is None:
            combined = row_estimate
        else:
            combined = GROUND_PLANE_WEIGHT * row_estimate + APPARENT_SIZE_WEIGHT * size_estimate

        self._recent.append(combined)
        return sum(self._recent) / len(self._recent)

    @staticmethod
    def _interp(points, x_key, x_value):
        """Linear interpolation of distance_cm vs. points[*][x_key],
        clamped to the calibrated range (numpy.interp's default behavior
        for out-of-range x) since extrapolating past measured points is
        unreliable."""
        pairs = sorted((p[x_key], p["distance_cm"]) for p in points)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        if len(set(xs)) < 2:
            return None  # all calibration points had the same x - can't interpolate on this axis
        return float(np.interp(x_value, xs, ys))
