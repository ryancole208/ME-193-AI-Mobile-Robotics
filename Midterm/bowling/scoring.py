"""
Ten-pin scoring (pure, unit tested): rolls, frames, strikes/spares, the 10th frame's
bonus balls, scorecard marks and running totals.
"""

import config

ALL_PINS = 10


class ScoreSheet:
    def __init__(self, n_frames=config.FRAMES_PER_GAME):
        self.n_frames = n_frames
        self.frames = [[] for _ in range(n_frames)]
        self.gutters = [[] for _ in range(n_frames)]   # parallel to frames: was that roll a gutter ball

    # -- state ------------------------------------------------------------------
    def frame_complete(self, i):
        r = self.frames[i]
        if i < self.n_frames - 1:
            return len(r) == 2 or (len(r) == 1 and r[0] == ALL_PINS)
        if len(r) < 2:
            return False
        if len(r) == 2:
            return r[0] + r[1] < ALL_PINS     # open 10th: no bonus ball
        return True

    @property
    def current_frame(self):
        """Index of the frame being bowled, or None once the game is over."""
        for i in range(self.n_frames):
            if not self.frame_complete(i):
                return i
        return None

    @property
    def game_over(self):
        return self.current_frame is None

    @property
    def roll_in_frame(self):
        i = self.current_frame
        return None if i is None else len(self.frames[i])

    def pins_standing(self):
        """Pins up for the next roll (10 = a fresh rack)."""
        i = self.current_frame
        if i is None:
            return 0
        r = self.frames[i]
        if not r:
            return ALL_PINS
        if i < self.n_frames - 1:
            return ALL_PINS - r[0]
        if len(r) == 1:
            return ALL_PINS if r[0] == ALL_PINS else ALL_PINS - r[0]
        # third ball of the 10th
        if r[0] == ALL_PINS:
            return ALL_PINS if r[1] == ALL_PINS else ALL_PINS - r[1]
        return ALL_PINS  # spare

    def add_roll(self, pins, gutter=False):
        i = self.current_frame
        if i is None:
            raise ValueError("game is over")
        if not 0 <= pins <= self.pins_standing():
            raise ValueError(f"{pins} pins knocked but only {self.pins_standing()} standing")
        self.frames[i].append(pins)
        self.gutters[i].append(bool(gutter))

    # -- scores -----------------------------------------------------------------
    def _flat(self):
        return [p for f in self.frames for p in f]

    def frame_scores(self):
        """Cumulative score per frame, or None where bonus balls are still to come."""
        rolls = self._flat()
        out, total, k = [], 0, 0
        for i, f in enumerate(self.frames):
            if not f:
                out.append(None)
                continue
            if i == self.n_frames - 1:
                score = sum(f) if self.frame_complete(i) else None
            elif f[0] == ALL_PINS:
                bonus = rolls[k + 1:k + 3]
                score = 10 + sum(bonus) if len(bonus) == 2 else None
            elif len(f) == 2 and sum(f) == ALL_PINS:
                bonus = rolls[k + 2:k + 3]
                score = 10 + sum(bonus) if bonus else None
            elif len(f) == 2:
                score = sum(f)
            else:
                score = None
            k += len(f)
            if score is None or (out and out[-1] is None):
                out.append(None)
            else:
                total += score
                out.append(total)
        return out

    def running_total(self):
        """Pins plus the strike/spare bonuses earned so far (equals the final score at the end)."""
        rolls = self._flat()
        total, k = 0, 0
        for i, f in enumerate(self.frames):
            total += sum(f)
            if i < self.n_frames - 1 and f:
                if f[0] == ALL_PINS:
                    total += sum(rolls[k + 1:k + 3])
                elif len(f) == 2 and sum(f) == ALL_PINS:
                    total += sum(rolls[k + 2:k + 3])
            k += len(f)
        return total

    # -- scorecard marks ----------------------------------------------------------
    def marks(self, i):
        """Box contents for frame i: 'X' strike, '/' spare, '-' zero, 'G' gutter ball, digits."""
        r, g = self.frames[i], self.gutters[i]

        def digit(j):
            if r[j] == 0:
                return "G" if g[j] else "-"
            return str(r[j])

        out = []
        if i < self.n_frames - 1:
            if r and r[0] == ALL_PINS:
                return ["", "X"]
            for j in range(len(r)):
                out.append("/" if j == 1 and r[0] + r[1] == ALL_PINS else digit(j))
            return out
        fresh = True  # 10th frame: is this ball thrown at a full rack?
        for j in range(len(r)):
            if fresh:
                out.append("X" if r[j] == ALL_PINS else digit(j))
                fresh = r[j] == ALL_PINS
            else:
                out.append("/" if r[j - 1] + r[j] == ALL_PINS else digit(j))
                fresh = True
        return out
