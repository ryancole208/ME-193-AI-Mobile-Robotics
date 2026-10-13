"""
Accuracy scoring, streaks and saved best results.

Accuracy = average of GRADE_POINTS over every cue judged so far (Perfect 100 %, Good 50 %,
Miss 0 % by default), shown with one decimal. A Miss resets the streak.

Every score change goes through ScoreKeeper.emit(ScoreEvent). That is the ONE place to hook
up MQTT later: add a listener, e.g.

    keeper.listeners.append(lambda ev: mqtt.publish_score(ev.accuracy))

No MQTT is imported here (or anywhere in the rhythm game) for now.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import config

GRADES = ("Perfect", "Good", "Miss")


@dataclass(frozen=True)
class ScoreEvent:
    kind: str                # "cue" after every judged cue, "song_end" when the song finishes
    song_id: str
    accuracy: float          # percent, 0..100
    streak: int
    max_streak: int
    counts: dict
    grade: str | None = None
    error_ms: float | None = None


def rank_for(accuracy, ranks=None):
    for name, threshold in (ranks or config.RANKS):
        if accuracy >= threshold:
            return name
    return (ranks or config.RANKS)[-1][0]


@dataclass
class ScoreKeeper:
    song_id: str = ""
    total_cues: int = 0
    counts: dict = field(default_factory=lambda: {g: 0 for g in GRADES})
    streak: int = 0
    max_streak: int = 0
    points: float = 0.0
    errors_ms: list = field(default_factory=list)   # timing error of each timed swing
    listeners: list = field(default_factory=list)

    @property
    def judged(self):
        return sum(self.counts.values())

    @property
    def accuracy(self):
        """Percent over the cues judged so far (0.0 before the first one)."""
        return 100.0 * self.points / self.judged if self.judged else 0.0

    @property
    def mean_error_ms(self):
        return sum(self.errors_ms) / len(self.errors_ms) if self.errors_ms else None

    def add(self, judgment):
        g = judgment.grade
        self.counts[g] += 1
        self.points += config.GRADE_POINTS[g]
        if judgment.error_ms is not None:
            self.errors_ms.append(judgment.error_ms)
        if judgment.hit:
            self.streak += 1
            self.max_streak = max(self.max_streak, self.streak)
        else:
            self.streak = 0
        self.emit(self._event("cue", g, judgment.error_ms))

    def finish(self):
        self.emit(self._event("song_end"))

    def _event(self, kind, grade=None, error_ms=None):
        return ScoreEvent(kind, self.song_id, round(self.accuracy, 1), self.streak, self.max_streak,
                          dict(self.counts), grade, error_ms)

    def emit(self, event):
        """Score events leave the game here (future MQTT hook)."""
        for fn in list(self.listeners):
            fn(event)


class BestScores:
    """Best accuracy and max streak per song, saved to SCORES_PATH."""

    def __init__(self, path=None):
        self.path = Path(path or config.SCORES_PATH)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.data, dict):
                self.data = {}
        except (OSError, ValueError):
            self.data = {}

    def get(self, song_id):
        return self.data.get(song_id)

    def record(self, song_id, accuracy, max_streak):
        """Store a finished play. Returns (new best accuracy?, new best streak?)."""
        old = self.data.get(song_id) or {}
        accuracy = round(accuracy, 1)                     # what the screen shows
        new_acc = accuracy > old.get("best_accuracy", -1.0)
        new_streak = max_streak > old.get("max_streak", -1)
        self.data[song_id] = {
            "best_accuracy": round(max(accuracy, old.get("best_accuracy", 0.0)), 1),
            "max_streak": max(max_streak, old.get("max_streak", 0)),
            "plays": old.get("plays", 0) + 1,
        }
        try:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"[scores] could not save {self.path}: {e}")
        return new_acc, new_streak
