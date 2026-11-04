"""Distance = sum of step lengths, with its uncertainty.

Step length:
  * L0 (default): measured by the user. Toes together at 0, after 4 steps the
    leading toe was at 115 in -> 115 / 4 = 28.75 in = 0.7303 m per step.
    Taken from setup.yaml calib_walks (total distance / total steps).
  * Weinberg model, when the hip is present and model.npz holds a fitted K:
        L_i = K * (a_max - a_min) ** (1/4)
    a_max / a_min = largest and smallest hip vertical acceleration (g) since the
    previous step. Why it works: longer, faster steps make the pelvis bob up
    and down more, and the 4th root (fitted empirically by Weinberg, 2002)
    turns that bounce into a length. Clamped to [0.3, 1.6] m so a stumble or a
    jump can't add a silly length.

Uncertainty:  sigma_D = sqrt( (L_mean * sigma_N)^2 + (r_L * D)^2 )
  first term:  every uncertain step is worth about one mean step length
  second term: the step-length model itself is off by a relative r_L
               (leave-one-out error over calibration walks, from model.npz;
               0.05 if the laptop did not provide one).
"""

from __future__ import annotations

import bisect

import numpy as np

from settings import DT

INCH = 0.0254
FOOT = 0.3048
MILE = 1609.344
L_MIN, L_MAX = 0.3, 1.6
WINDOW_MIN_S, WINDOW_MAX_S = 0.3, 1.0     # where to look for a_max/a_min before a step
R_L_DEFAULT = 0.05


def l0_from_walks(walks: list) -> float:
    steps = sum(w["steps"] for w in walks)
    return sum(w["distance_in"] for w in walks) * INCH / steps


class DistanceTracker:
    def __init__(self, L0_m: float, weinberg_K=None, r_L=None):
        self.L0 = L0_m
        self.K = weinberg_K
        self.r_L = R_L_DEFAULT if r_L is None else r_L
        self.reset()

    def reset(self) -> None:
        self.D = 0.0
        self.n = 0
        self._times: list = []        # sorted event times, for "time since the previous step"
        self.weinberg_steps = 0

    def set_model(self, weinberg_K, r_L) -> None:
        self.K = weinberg_K
        self.r_L = R_L_DEFAULT if r_L is None else r_L

    def add(self, ev, grid) -> float:
        """Assign a length to a newly counted step (any foot) and add it."""
        L = self.L0
        if self.K:
            i = bisect.bisect_left(self._times, ev.t)
            prev = self._times[i - 1] if i > 0 else ev.t - WINDOW_MAX_S
            span = min(max(ev.t - prev, WINDOW_MIN_S), WINDOW_MAX_S)
            k1 = ev.k + 1
            k0 = k1 - int(round(span / DT))
            _, ch, _, _ = grid.rows(k0, k1)
            if len(ch) >= 10 and ch[:, 11].mean() > 0.8:            # hip present over the interval
                hv = ch[:, 7]
                L = float(np.clip(self.K * (hv.max() - hv.min()) ** 0.25, L_MIN, L_MAX))
                self.weinberg_steps += 1
        bisect.insort(self._times, ev.t)
        if len(self._times) > 2000:
            del self._times[:1000]
        ev.length_m = L
        self.D += L
        self.n += 1
        return L

    def remove(self, ev) -> None:
        self.D = max(0.0, self.D - ev.length_m)
        self.n = max(0, self.n - 1)

    def summary(self, sigma_N: float) -> dict:
        L_mean = self.D / self.n if self.n else self.L0
        sigma = float(np.sqrt((L_mean * sigma_N) ** 2 + (self.r_L * self.D) ** 2))
        return {"dist_m": self.D, "dist_sigma_m": sigma, "L_mean_m": L_mean,
                "model": "weinberg" if self.K else "L0"}


def format_distance(d_m: float, units: str) -> str:
    """Matrix text: imperial -> '352FT' below 1000 ft, then '0.27MI'; metric -> '352M', then '1.25KM'."""
    if units == "imperial":
        ft = d_m / FOOT
        return f"{ft:.0f}FT" if ft < 1000 else f"{d_m / MILE:.2f}MI"
    return f"{d_m:.0f}M" if d_m < 1000 else f"{d_m / 1000:.2f}KM"
