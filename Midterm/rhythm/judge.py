"""
Hit judgment. Timing: a swing within GOOD_WINDOW_MS of the cue is a hit (Perfect within
PERFECT_WINDOW_MS). Contact (only in the "aim" game mode): the ball must also actually meet your
paddle -- when the ball reaches you, the drawn blade must overlap the drawn ball
(contact_distance()), else the swing is a Miss ("missed the ball"). An unknown paddle position
counts as no contact. In the "timing" and "follow" modes the caller passes no contact check, so
only timing counts.

All times here are SONG seconds. The caller converts a swing's IMU peak timestamp to song
time and subtracts the calibrated input offset before calling swing(). Each swing goes to
the nearest unjudged cue whose window contains it; each cue can be judged only once.

A cue nobody swung at becomes a Miss only once no late IMU event can still arrive:
cue + good window + miss_wait_s (input offset + peak detection + grace).
"""

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Judgment:
    index: int
    grade: str               # "Perfect", "Good" or "Miss"
    error_ms: float | None   # swing - cue (negative = early); None = no swing
    reason: str              # "", "missed the ball", "no swing", "stray swing"
    t_known: float           # song time the game decided
    source: str = ""

    @property
    def hit(self):
        return self.grade != "Miss"


def grade_for(error_ms, perfect_ms=None, good_ms=None):
    perfect_ms = config.PERFECT_WINDOW_MS if perfect_ms is None else perfect_ms
    good_ms = config.GOOD_WINDOW_MS if good_ms is None else good_ms
    e = abs(error_ms)
    if e <= perfect_ms:
        return "Perfect"
    if e <= good_ms:
        return "Good"
    return None


def contact_distance():
    """Largest |paddle x - ball x| at which the ball meets the paddle: AIM_CONTACT_M, or by default
    exactly what you see -- half the drawn blade plus the drawn ball's radius."""
    if config.AIM_CONTACT_M is not None:
        return config.AIM_CONTACT_M
    return (config.PADDLE_BLADE_W_M * config.PADDLE_DRAW_SCALE / 2
            + config.BALL_RADIUS_M * config.BALL_VISUAL_SCALE)


def paddle_near(paddle_x_at, x, t, tolerance_m=None, lookback_ms=None, lookahead_ms=None, step_ms=10):
    """True if the paddle was within tolerance (default: contact_distance()) of x at any point in
    [t - lookback, t + lookahead] (t = wall time the ball reaches you). paddle_x_at(t) -> x or None
    (unknown: never counts)."""
    tol = contact_distance() if tolerance_m is None else tolerance_m
    back = (config.AIM_LOOKBACK_MS if lookback_ms is None else lookback_ms) / 1000
    ahead = (config.AIM_LOOKAHEAD_MS if lookahead_ms is None else lookahead_ms) / 1000
    n = max(1, int(round((back + ahead) / (step_ms / 1000))))
    for k in range(n + 1):
        px = paddle_x_at(t - back + (back + ahead) * k / n)
        if px is not None and abs(px - x) <= tol + 1e-9:
            return True
    return False


class Judge:
    def __init__(self, flights, *, perfect_ms=None, good_ms=None, miss_wait_s=0.2, stray_penalty=None):
        self.flights = flights
        self.perfect_ms = config.PERFECT_WINDOW_MS if perfect_ms is None else perfect_ms
        self.good_ms = config.GOOD_WINDOW_MS if good_ms is None else good_ms
        self.miss_wait_s = miss_wait_s
        self.stray_penalty = config.STRAY_SWING_PENALTY if stray_penalty is None else stray_penalty
        self.results = {}          # index -> Judgment
        self._first_open = 0       # every cue before this one is judged
        self.strays = 0

    @property
    def done(self):
        return len(self.results) == len(self.flights)

    def _advance(self):
        while self._first_open < len(self.flights) and self._first_open in self.results:
            self._first_open += 1

    def swing(self, t, now, source="", aim_ok=None):
        """A swing at song time t (already offset-corrected). aim_ok(flight) -> bool checks the
        paddle position (None = timing only). Returns the new Judgment or None (stray)."""
        good = self.good_ms / 1000
        best = None
        for fl in self.flights[self._first_open:]:
            if fl.t_arrive - good > t:
                break
            if fl.index in self.results or abs(t - fl.t_arrive) > good:
                continue
            if best is None or abs(t - fl.t_arrive) < abs(t - best.t_arrive):
                best = fl
        if best is None:
            self.strays += 1
            if self.stray_penalty:
                for fl in self.flights[self._first_open:]:
                    if fl.index not in self.results and fl.t_arrive > t:
                        return self._record(Judgment(fl.index, "Miss", None, "stray swing", now, source))
            return None
        err = (t - best.t_arrive) * 1000
        if aim_ok is not None and not aim_ok(best):
            return self._record(Judgment(best.index, "Miss", err, "missed the ball", now, source))
        return self._record(Judgment(best.index, grade_for(err, self.perfect_ms, self.good_ms), err, "", now, source))

    def update(self, now):
        """Call every frame: returns the Misses that just became final."""
        out = []
        deadline = self.good_ms / 1000 + self.miss_wait_s
        for fl in self.flights[self._first_open:]:
            if fl.t_arrive + deadline > now:
                break
            if fl.index not in self.results:
                out.append(self._record(Judgment(fl.index, "Miss", None, "no swing", now)))
        return out

    def _record(self, j):
        self.results[j.index] = j
        self._advance()
        return j
