"""
Floor vs. blue tape: two interchangeable classifiers + a debouncer.

config.COLOR_METHOD picks the classifier:
  "knn"  -> KnnColorClassifier (below), trained with calibrate_color.py
  "lego" -> LegoColorClassifier: the sensor's own color name (blue/azure = tape)
Both are used through classify_sensor(classifier, reading), so the rest of the
program doesn't care which one is active.

KNN details. Training data comes from calibrate_color.py: the sensor is placed on each
surface and samples are saved as {"label": "floor"|"blue", "features": [...]}.

Features: (rawRed, rawGreen, rawBlue, reflection). Each feature is divided by its
standard deviation across the training set (z-scoring), so a channel with a big
numeric range doesn't dominate the distance. To classify a reading we take the
K closest training samples and use the majority label. If even the closest one
is farther than KNN_MAX_DISTANCE (in std-devs), the reading is "unknown": not
like anything we trained on (sensor lifted, other robot's paint...). Unknown
never counts as blue, but it doesn't reset the debouncer either.
"""

import json
import math
from collections import Counter
from pathlib import Path

import config

FLOOR, BLUE, UNKNOWN = "floor", "blue", "unknown"
METHODS = ("knn", "lego")      # allowed values of config.COLOR_METHOD


def features_from_sensor(s):
    """Feature vector from a legoeducation ColorSensorNotification (or None if not ready)."""
    vals = (s.rawRed, s.rawGreen, s.rawBlue, s.reflection)
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return None
    return tuple(float(v) for v in vals)


class KnnColorClassifier:
    def __init__(self, samples, *, k=config.KNN_K, max_distance=config.KNN_MAX_DISTANCE):
        """samples: list of (label, features)."""
        if not samples:
            raise ValueError("no training samples")
        labels = {lab for lab, _ in samples}
        if not {FLOOR, BLUE} <= labels:
            raise ValueError(f"need samples of both '{FLOOR}' and '{BLUE}', got {sorted(labels)}")
        self.samples = [(lab, tuple(map(float, f))) for lab, f in samples]
        self.k = min(k, len(self.samples))
        self.max_distance = max_distance
        n = len(self.samples[0][1])
        cols = [[f[i] for _, f in self.samples] for i in range(n)]
        means = [sum(c) / len(c) for c in cols]
        # Floor of 1.0 count: a channel that never varied must not divide by ~0.
        self.scale = [max(1.0, math.sqrt(sum((x - m) ** 2 for x in c) / len(c)))
                      for c, m in zip(cols, means)]
        self._scaled = [(lab, self._norm(f)) for lab, f in self.samples]

    def _norm(self, f):
        return tuple(x / s for x, s in zip(f, self.scale))

    def classify(self, features):
        if features is None:
            return UNKNOWN
        q = self._norm(features)
        dists = sorted((math.dist(q, f), lab) for lab, f in self._scaled)
        if self.max_distance is not None and dists[0][0] > self.max_distance:
            return UNKNOWN
        votes = Counter(lab for _, lab in dists[:self.k])
        best = max(votes.values())
        # Tie: take the label of the single nearest sample among the tied ones.
        for _, lab in dists:
            if votes[lab] == best:
                return lab

    # -- persistence ------------------------------------------------------------
    @staticmethod
    def load_samples(path):
        data = json.loads(Path(path).read_text())
        return [(d["label"], d["features"]) for d in data["samples"]]

    @staticmethod
    def save_samples(path, samples, note=""):
        data = {"features": ["rawRed", "rawGreen", "rawBlue", "reflection"], "note": note,
                "samples": [{"label": lab, "features": list(f)} for lab, f in samples]}
        Path(path).write_text(json.dumps(data, indent=1))


class LegoColorClassifier:
    """COLOR_METHOD = "lego": trust the sensor's own color name.

    The library reports it in reading.color using LEGO's App numbering
    (le.LEGO_COLOR_BLUE = 3, le.LEGO_COLOR_AZURE = 10, ...). Any id in
    tape_colors means tape; any other color means floor; no reading yet = unknown.
    """

    def __init__(self, tape_colors=None):
        self.tape_colors = set(config.TAPE_LEGO_COLORS if tape_colors is None else tape_colors)

    def classify_sensor(self, s):
        c = getattr(s, "color", None)
        if c is None or (isinstance(c, float) and math.isnan(c)):
            return UNKNOWN
        return BLUE if int(c) in self.tape_colors else FLOOR


def samples_needed(speed_mps, n=config.DEBOUNCE_SAMPLES, tape_m=config.TAPE_WIDTH_M,
                   period_s=config.COLOR_SAMPLE_PERIOD_S):
    """How many consecutive readings the debouncer should demand at this speed.

    Normally n. But at speed v the sensor is over the tape for only tape_m / v
    seconds, i.e. tape_m / v / period_s readings. If that's fewer than n + 1, a
    hard shove could carry us across the tape before n readings arrive, so we
    accept a single reading instead.
    """
    expected = tape_m / max(abs(speed_mps), 1e-3) / period_s
    return 1 if expected < n + 1 else n


class Debouncer:
    """Only report floor<->blue after N consecutive identical readings.

    A single noisy reading (dust, a seam) can't trigger the reflex or, worse, a
    false "out". The cost is N sample periods of delay (N=2 at ~30 ms = ~60 ms),
    which shows up as extra overshoot past the tape. N can be lowered per reading
    (see samples_needed) when we move too fast for N readings to fit on the tape.
    """

    def __init__(self, n=config.DEBOUNCE_SAMPLES, initial=FLOOR, log=print):
        self.n = max(1, n)
        self.state = initial
        self._candidate = None
        self._count = 0
        self.log = log

    def update(self, label, required=None):
        """Feed one reading; returns the new state if it just changed, else None."""
        need = max(1, self.n if required is None else required)
        if label == UNKNOWN:
            return None
        if label == self.state:
            if self._candidate is not None:
                self.log(f"[color] ignored {self._count} '{self._candidate}' reading(s) (debounce)")
            self._candidate, self._count = None, 0
            return None
        if label != self._candidate:
            self._candidate, self._count = label, 0
        self._count += 1
        if self._count >= need:
            self.state = label
            self._candidate, self._count = None, 0
            return label
        return None


def load_classifier(method=None, path=None, log=print):
    """The classifier chosen by config.COLOR_METHOD.

    Raises ValueError with an explanation for a bad method or, in "knn" mode,
    a missing calibration file (we never silently switch methods).
    """
    method = config.COLOR_METHOD if method is None else method
    if method not in METHODS:
        raise ValueError(f"COLOR_METHOD must be one of {METHODS}, got {method!r}")
    if method == "lego":
        clf = LegoColorClassifier()
        log(f"[color] method 'lego': the sensor's own color name; tape = color ids {sorted(clf.tape_colors)}")
        return clf
    path = Path(path or Path(__file__).with_name(config.COLOR_SAMPLES_FILE))
    if not path.exists():
        raise ValueError(f"COLOR_METHOD is 'knn' but {path.name} doesn't exist. Run calibrate_color.py, "
                         f"or set COLOR_METHOD = \"lego\" in config.py.")
    clf = KnnColorClassifier(KnnColorClassifier.load_samples(path))
    counts = Counter(lab for lab, _ in clf.samples)
    log(f"[color] method 'knn': loaded {path.name}: {dict(counts)}, k={clf.k}")
    return clf


def classify_sensor(classifier, s):
    """One entry point for both classifier kinds."""
    if isinstance(classifier, LegoColorClassifier):
        return classifier.classify_sensor(s)
    return classifier.classify(features_from_sensor(s))
