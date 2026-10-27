"""Raw MPU-6050 counts -> physical units in the *right-leg pod frame*.

Order of operations for every sample:
  1. scale:     accel g = raw / 2048 (+-16 g), gyro deg/s = raw / 16.4 (+-2000 dps)
  2. axis map:  config.json "axis_map" relabels/flips sensor axes (identity by default)
  3. leg:       a left-leg pod is mirrored into the right-leg frame

Left-leg transform, derived
---------------------------
Body axes: F = forward, L = to the wearer's left, U = up (F x L = U, right-handed).

Right ankle pod: X = F, Y = U, Z = -L  (Z points outward = to the right).
Left ankle pod:  Y = U, Z = +L (outward = to the left), and because the sensor is
right-handed X = Y x Z = U x L = -F, i.e. +X points BACKWARD (as the brief says).

We want left-leg data to look like the mirror-image movement made by a right
leg. Mirroring across the body midline is the reflection M = diag(1, -1, 1) in
(F, L, U):
  * acceleration is a true (polar) vector:   a' = M a
  * angular velocity is a pseudovector:     w' = det(M) M w = -M w
    (a reflection flips the sense of rotation, so on top of M it gains a minus)

Left sensor reads ax = -aF, ay = aU, az = aL. The mirrored motion seen by a
right sensor is (a'.F, a'.U, a'.(-L)) = (aF, aU, aL) = (-ax, ay, az).

Left gyro reads gx = -wF, gy = wU, gz = wL. Mirrored: w' = (-wF, wL, -wU), so a
right sensor would read (w'.F, w'.U, w'.(-L)) = (-wF, -wU, -wL) = (gx, -gy, -gz).

    accel: (ax, ay, az) -> (-ax,  ay,  az)
    gyro:  (gx, gy, gz) -> ( gx, -gy, -gz)

Sanity check: when either shank swings forward it rotates about the lateral
axis in the same physical sense; a right pod sees that on gz = w.(-L) and a
left pod on gz = w.(+L) = -(right gz). The transform flips gz, so both legs
agree.
"""

from __future__ import annotations

import numpy as np

from .errors import ConfigError

_AXES = {"x": 0, "y": 1, "z": 2}

LEFT_ACCEL_SIGNS = np.array([-1.0, 1.0, 1.0])
LEFT_GYRO_SIGNS = np.array([1.0, -1.0, -1.0])


def axis_map_matrix(axis_map: dict) -> np.ndarray:
    """Build the 3x3 signed permutation P so that out = P @ sensor.

    axis_map = {"x": "+x", "y": "+y", "z": "+z"} means output x is +sensor x, etc.
    Example: {"x": "-z", "y": "+y", "z": "+x"} if the pod is mounted rotated.
    """
    if set(axis_map) != {"x", "y", "z"}:
        raise ConfigError(
            "reading signal.axis_map",
            f"keys are {sorted(axis_map)}, expected exactly x, y, z",
            'use {"x": "+x", "y": "+y", "z": "+z"} unless the pod is mounted differently',
        )
    P = np.zeros((3, 3))
    used = set()
    for out_axis, spec in axis_map.items():
        if not (isinstance(spec, str) and len(spec) == 2 and spec[0] in "+-" and spec[1] in _AXES):
            raise ConfigError(
                "reading signal.axis_map",
                f"entry {out_axis!r}: {spec!r} is not one of +x -x +y -y +z -z",
                'write each entry as a sign and an axis, e.g. "-z"',
            )
        if spec[1] in used:
            raise ConfigError(
                "reading signal.axis_map",
                f"sensor axis {spec[1]!r} is used twice",
                "each sensor axis must appear exactly once",
            )
        used.add(spec[1])
        P[_AXES[out_axis], _AXES[spec[1]]] = 1.0 if spec[0] == "+" else -1.0
    return P


class SignalConverter:
    def __init__(self, accel_lsb_per_g: float, gyro_lsb_per_dps: float, axis_map: dict, leg: str):
        if leg not in ("right", "left"):
            raise ConfigError("setting up unit conversion", f"leg is {leg!r}", 'set leg to "right" or "left" in setup.yaml')
        self.accel_scale = 1.0 / float(accel_lsb_per_g)
        self.gyro_scale = 1.0 / float(gyro_lsb_per_dps)
        self.P = axis_map_matrix(axis_map)
        # A relabelling with det = -1 turns the frame left-handed; gyro readings
        # are pseudovectors and must pick up the extra sign to stay consistent.
        self.gyro_det = float(round(np.linalg.det(self.P)))
        self.leg = leg

    def convert(self, raw: np.ndarray) -> np.ndarray:
        """raw (n, 6) counts -> (n, 6) float [ax, ay, az] g, [gx, gy, gz] deg/s."""
        raw = np.asarray(raw, dtype=float)
        acc = raw[:, 0:3] * self.accel_scale @ self.P.T
        gyr = raw[:, 3:6] * self.gyro_scale @ self.P.T * self.gyro_det
        if self.leg == "left":
            acc, gyr = left_to_right(acc, gyr)
        return np.hstack([acc, gyr])


def left_to_right(acc: np.ndarray, gyr: np.ndarray):
    """Mirror left-leg pod readings into the right-leg frame (see module docstring)."""
    return acc * LEFT_ACCEL_SIGNS, gyr * LEFT_GYRO_SIGNS
