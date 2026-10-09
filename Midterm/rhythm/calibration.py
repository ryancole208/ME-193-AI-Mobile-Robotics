"""
Latency calibration.

1. Input offset -- you swing along to an audio-only metronome (CALIB_BEATS clicks after a
   CALIB_COUNT_IN_BEATS count-in). For every swing: error = swing time - nearest click.
   Swings more than CALIB_MAX_ABS_MS from any click are ignored, then the ones more than
   CALIB_OUTLIER_MS from the median are dropped; the offset is the median of the rest.
   It covers BLE + swing detection + audio output + your own habit, and is stored per input
   source (imu / keyboard). Every later swing has it subtracted before judging.

2. Visual (A/V) offset -- a ball bounces on every click and you nudge the drawing earlier or
   later until the bounce lands on the click. It only moves the picture, never the judging.

Both are saved to calibration.json and loaded by config.py at start.
"""

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class CalibResult:
    offset_ms: float | None      # what to store (None = not enough good swings)
    mean_ms: float
    std_ms: float
    used: int
    dropped: int
    errors_ms: tuple             # every usable swing's error (before outlier removal)

    @property
    def ok(self):
        return self.offset_ms is not None


def analyze(swing_times, beat_times, *, outlier_ms=None, max_abs_ms=None, min_swings=None):
    """swing_times / beat_times in the same clock (seconds)."""
    outlier_ms = config.CALIB_OUTLIER_MS if outlier_ms is None else outlier_ms
    max_abs_ms = config.CALIB_MAX_ABS_MS if max_abs_ms is None else max_abs_ms
    min_swings = config.CALIB_MIN_SWINGS if min_swings is None else min_swings
    errors, used_beats = [], set()
    for s in swing_times:
        if not beat_times:
            break
        b = min(beat_times, key=lambda bt: abs(s - bt))
        e = (s - b) * 1000
        if abs(e) <= max_abs_ms and b not in used_beats:   # one swing per click
            errors.append(e)
            used_beats.add(b)
    if not errors:
        return CalibResult(None, 0.0, 0.0, 0, 0, ())
    med = statistics.median(errors)
    kept = [e for e in errors if abs(e - med) <= outlier_ms]
    mean = sum(kept) / len(kept)
    std = statistics.pstdev(kept) if len(kept) > 1 else 0.0
    offset = statistics.median(kept) if len(kept) >= min_swings else None
    return CalibResult(offset, mean, std, len(kept), len(errors) - len(kept), tuple(errors))


def save(input_offsets=None, visual_offset_ms=None, path=None):
    """Merge into calibration.json and update the live config values."""
    path = Path(path or config.CALIBRATION_PATH)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if input_offsets:
        data.setdefault("input_offset_ms", {}).update({k: round(v, 1) for k, v in input_offsets.items()})
        config.INPUT_OFFSET_MS.update(input_offsets)
    if visual_offset_ms is not None:
        data["visual_offset_ms"] = round(visual_offset_ms, 1)
        config.VISUAL_OFFSET_MS = visual_offset_ms
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


class InputCalibration:
    """State of one input-calibration run (all times in song seconds of the click track)."""

    def __init__(self, bpm=None, count_in=None, beats=None):
        self.bpm = config.CALIB_BPM if bpm is None else bpm
        self.spb = 60.0 / self.bpm
        self.count_in = config.CALIB_COUNT_IN_BEATS if count_in is None else count_in
        self.beats = config.CALIB_BEATS if beats is None else beats
        self.swings = []          # (song time, source)
        self.result = None
        self.source = None

    @property
    def click_times(self):
        return [k * self.spb for k in range(self.count_in + self.beats)]

    @property
    def measured_times(self):
        return self.click_times[self.count_in:]

    @property
    def length_s(self):
        return (self.count_in + self.beats + 1) * self.spb

    def beat_index(self, song_t):
        return int(song_t // self.spb) if song_t >= 0 else -1

    def add_swing(self, song_t, source):
        if song_t >= self.measured_times[0] - self.spb / 2:
            self.swings.append((song_t, source))

    def finish(self):
        srcs = [s for _, s in self.swings]
        self.source = max(set(srcs), key=srcs.count) if srcs else "keyboard"
        times = [t for t, s in self.swings if s == self.source]
        self.result = analyze(times, self.measured_times)
        return self.result
