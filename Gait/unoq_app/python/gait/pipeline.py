"""The live classification pipeline, shared by live mode, replay and evaluate.

raw counts -> units + leg transform -> (gesture detector on every sample)
                                   -> windows -> features -> kNN A and kNN B
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .constants import CLASSES, UNSURE
from .features import model_a_feature, model_b_features, model_b_subset
from .gesture import GestureDetector, GestureParams
from .training import converter_for, models_from_config
from .windowing import Window, Windower


@dataclass
class WindowResult:
    start: int
    end: int
    valid: bool
    missing_frac: float
    pred_a: int | None = None      # class index, UNSURE (-1), or None for an invalid window
    pred_b: int | None = None
    dist_a: float | None = None
    dist_b: float | None = None
    gesture: bool = False          # overlaps a tap train: logged, not used for smoothing


class Classifier:
    def __init__(self, cfg: dict, leg: str | None = None):
        sig = cfg["signal"]
        fs = sig["sample_rate_hz"]
        self.cfg = cfg
        self.conv = converter_for(sig, leg)
        self.windower = Windower(int(round(sig["window_s"] * fs)), int(round(sig["hop_s"] * fs)),
                                 sig["max_missing_frac"], 6)
        self.knn_a, self.knn_b = models_from_config(cfg)
        self.feature_a = cfg["model_a"]["feature"]
        self.features_b = cfg["model_b"]["features"]
        self.gravity = sig["gravity_unit_vector"]
        unsure_on = bool(cfg.get("display", {}).get("unsure_enabled", False))
        self.unsure_a = cfg["model_a"].get("unsure_distance") if unsure_on else None
        self.unsure_b = cfg["model_b"].get("unsure_distance") if unsure_on else None
        self.gestures = GestureDetector(GestureParams.from_config(cfg))

    def push(self, start_index: int, raw: np.ndarray, reset: bool = False):
        """Returns (window results, gesture events) for this chunk of samples."""
        phys = self.conv.convert(raw)
        events = self.gestures.push(start_index, phys, reset)
        results = [self.classify(w) for w in self.windower.push(start_index, phys, reset)]
        return results, events

    def classify(self, w: Window) -> WindowResult:
        r = WindowResult(w.start_index, w.end_index, w.valid, w.missing_frac)
        r.gesture = self.gestures.overlaps_gesture(w.start_index, w.end_index)
        if not w.valid:
            return r
        fa = model_a_feature(w.data, self.feature_a)
        fb = model_b_subset(model_b_features(w.data, self.gravity), self.features_b)
        (pa,), (da,) = self.knn_a.predict([[fa]])
        (pb,), (db,) = self.knn_b.predict(fb[None, :])
        r.pred_a = UNSURE if self.unsure_a is not None and da > self.unsure_a else int(pa)
        r.pred_b = UNSURE if self.unsure_b is not None and db > self.unsure_b else int(pb)
        r.dist_a, r.dist_b = float(da), float(db)
        return r


class Smoother:
    """Majority vote over the last `n` windows; the shown label only changes
    after the smoothed label has differed from it for `confirm` hops in a row."""

    def __init__(self, n: int = 3, confirm: int = 2):
        self.votes: deque = deque(maxlen=n)
        self.confirm = confirm
        self.shown: int | None = None
        self.smoothed: int | None = None
        self._candidate = None
        self._count = 0

    def majority(self) -> int:
        counts = Counter(self.votes)
        top = max(counts.values())
        tied = [lab for lab, c in counts.items() if c == top]
        if len(tied) == 1:
            return tied[0]
        if self.shown in tied:
            return self.shown
        for lab in reversed(self.votes):  # most recent of the tied labels
            if lab in tied:
                return lab
        return tied[0]

    def push(self, label: int) -> int | None:
        self.votes.append(label)
        self.smoothed = self.majority()
        if self.smoothed == self.shown:
            self._candidate, self._count = None, 0
        else:
            if self.smoothed == self._candidate:
                self._count += 1
            else:
                self._candidate, self._count = self.smoothed, 1
            if self._count >= self.confirm:
                self.shown = self.smoothed
                self._candidate, self._count = None, 0
        return self.shown


def label_name(label) -> str:
    if label is None:
        return ""
    if label == UNSURE:
        return "unsure"
    return CLASSES[label]
