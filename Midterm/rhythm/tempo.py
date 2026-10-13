"""
Tempo map: every beat <-> song-time conversion in the game goes through here.

A chart's tempo is a list of segments, each starting on a whole beat:

    "offset_ms": 24.7,                         # song time of beat 0
    "tempo": [{"beat": 0,   "bpm": 120.0},
              {"beat": 255, "bpm": 160.0, "shift_ms": 0.0}]

Inside a segment the beats are evenly spaced at its BPM, so the grid is steady by
construction (no detection jitter). A segment normally starts exactly where the previous one's
beats lead (continuous). shift_ms (optional, default 0) moves that start, and every later beat,
by a few ms when the music's new tempo doesn't begin exactly on the old grid. The one beat
before the segment is then slightly stretched or squeezed to close the gap.

Internally that is a piecewise-linear beat -> time function through breakpoints, extended at
both ends with the first / last segment's tempo. Fractional beats (upbeats, 0.5-8 beat flights)
convert exactly.
"""

import bisect
from dataclasses import dataclass

import config


class TempoError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    beat: int
    bpm: float
    shift_ms: float = 0.0

    @property
    def spb(self):
        return 60.0 / self.bpm


@dataclass(frozen=True)
class TempoChange:
    """A sudden tempo change: the new tempo starts at `beat` (song time `t`)."""
    beat: int
    t: float
    old_bpm: float
    new_bpm: float

    @property
    def faster(self):
        return self.new_bpm > self.old_bpm

    def blink_times(self, count=None):
        """Song times of the LED blinks / ticks: count, ..., 2, 1 NEW-tempo beats before the change."""
        n = config.TEMPO_LED_BLINKS if count is None else count
        spb = 60.0 / self.new_bpm
        return [self.t - k * spb for k in range(n, 0, -1)]


class TempoMap:
    def __init__(self, segments, offset_ms=0.0):
        segs = sorted(segments, key=lambda s: s.beat)
        if not segs:
            raise TempoError("the tempo map needs at least one segment")
        if segs[0].beat != 0:
            raise TempoError("the first tempo segment must start on beat 0")
        for s in segs:
            if not (20 <= s.bpm <= 400):
                raise TempoError(f"bpm {s.bpm:g} (beat {s.beat}) is out of range (20..400)")
        for a, b in zip(segs, segs[1:]):
            if b.beat - a.beat < 1:
                raise TempoError(f"two tempo segments start on beat {a.beat}")
            if abs(b.shift_ms) / 1000 >= 0.5 * a.spb:
                raise TempoError(f"shift_ms {b.shift_ms:g} at beat {b.beat} is more than half a beat")
        self.segments = segs
        self.offset_ms = float(offset_ms)
        # breakpoints (beat, time): collinear within a segment, so interpolation is exact
        beats, times = [0.0], [self.offset_ms / 1000]
        for a, b in zip(segs, segs[1:]):
            t_prev = times[-1] + (b.beat - 1 - a.beat) * a.spb      # the beat before the change
            if b.shift_ms and b.beat - 1 > a.beat:
                beats.append(float(b.beat - 1))
                times.append(t_prev)
            beats.append(float(b.beat))
            times.append(t_prev + a.spb + b.shift_ms / 1000)
        self._beats, self._times = beats, times

    # -- conversions ----------------------------------------------------------------
    def beat_time(self, beat):
        """Song time (s from the start of the audio file) of a beat (fractions allowed)."""
        bs, ts = self._beats, self._times
        if beat <= bs[0]:
            return ts[0] + (beat - bs[0]) * self.segments[0].spb
        if beat >= bs[-1]:
            return ts[-1] + (beat - bs[-1]) * self.segments[-1].spb
        i = bisect.bisect_right(bs, beat) - 1
        f = (beat - bs[i]) / (bs[i + 1] - bs[i])
        return ts[i] + f * (ts[i + 1] - ts[i])

    def time_beat(self, t):
        bs, ts = self._beats, self._times
        if t <= ts[0]:
            return bs[0] + (t - ts[0]) / self.segments[0].spb
        if t >= ts[-1]:
            return bs[-1] + (t - ts[-1]) / self.segments[-1].spb
        i = bisect.bisect_right(ts, t) - 1
        f = (t - ts[i]) / (ts[i + 1] - ts[i])
        return bs[i] + f * (bs[i + 1] - bs[i])

    def segment_at(self, beat):
        keys = [s.beat for s in self.segments]
        return self.segments[max(0, bisect.bisect_right(keys, beat) - 1)]

    def segment_index_at(self, beat):
        keys = [s.beat for s in self.segments]
        return max(0, bisect.bisect_right(keys, beat) - 1)

    def bpm_at(self, beat):
        return self.segment_at(beat).bpm

    def spb_at(self, beat):
        """Seconds per beat at a beat (the music's local beat length)."""
        return self.segment_at(beat).spb

    def duration(self, beat_a, beat_b):
        """Seconds from beat_a to beat_b."""
        return self.beat_time(beat_b) - self.beat_time(beat_a)

    # -- summaries -----------------------------------------------------------------
    def bpm_range(self, beat_a=None, beat_b=None):
        """(slowest, fastest) BPM, optionally only over the segments touching [beat_a, beat_b]."""
        segs = self.segments
        if beat_a is not None and beat_b is not None:
            nxt = [s.beat for s in segs[1:]] + [float("inf")]
            segs = [s for s, end in zip(segs, nxt) if s.beat <= beat_b and end > beat_a] or segs
        bpms = [s.bpm for s in segs]
        return min(bpms), max(bpms)

    def sudden_changes(self, min_jump_bpm=None, within_beats=None):
        """Every sudden tempo change, faster or slower, as TempoChange.

        Rule: at each segment boundary, compare the new tempo with the tempo `within_beats`
        beats earlier (and just before). A rise or a fall of MORE than min_jump_bpm is sudden. A
        single boundary is an instant change, so it counts by itself; several small steps in the
        same direction close together count when they add up within the window. Gradual drift
        (small steps spread further apart) never counts. Once a change is found, boundaries in
        the same direction inside its window are part of it (the change keeps the first boundary
        and the most extreme tempo reached in the window)."""
        jump = config.TEMPO_JUMP_MIN_BPM if min_jump_bpm is None else min_jump_bpm
        within = config.TEMPO_SUDDEN_WITHIN_BEATS if within_beats is None else within_beats
        out = []
        for seg in self.segments[1:]:
            if out and seg.beat - out[-1].beat <= within:
                last = out[-1]
                if (seg.bpm - last.new_bpm) * (last.new_bpm - last.old_bpm) > 0:   # further, same way
                    out[-1] = TempoChange(last.beat, last.t, last.old_bpm, seg.bpm)
                continue
            earlier = (self.bpm_at(seg.beat - within - 1e-6), self.bpm_at(seg.beat - 1e-6))
            if seg.bpm - min(earlier) > jump:
                out.append(TempoChange(seg.beat, self.beat_time(seg.beat), min(earlier), seg.bpm))
            elif max(earlier) - seg.bpm > jump:
                out.append(TempoChange(seg.beat, self.beat_time(seg.beat), max(earlier), seg.bpm))
        return out

    def to_list(self):
        out = []
        for s in self.segments:
            d = {"beat": int(s.beat), "bpm": round(s.bpm, 4)}
            if s.shift_ms:
                d["shift_ms"] = round(s.shift_ms, 2)
            out.append(d)
        return out


@dataclass(frozen=True)
class LedState:
    visible: float           # 0..1: the LED housing fades in before the first blink, out after the change
    lit: float               # 0..1: brightness of the lens (1 at the start of each blink)
    blink: int               # which blink (1..TEMPO_LED_BLINKS), 0 = between blinks
    new_bpm: float
    faster: bool = True      # speed-up (amber) or slowdown (blue)


def led_state(changes, t):
    """What the tempo-change LED shows at song time t (None = hidden). Pure function of t."""
    if t is None:
        return None
    for c in changes:
        blinks = c.blink_times()
        spb = 60.0 / c.new_bpm
        start, end = blinks[0] - config.TEMPO_LED_SHOW_S, c.t + 0.5 * spb
        if not start <= t <= end:
            continue
        visible = min(1.0, (t - start) / max(1e-6, config.TEMPO_LED_SHOW_S), (end - t) / (0.5 * spb))
        lit, which = 0.0, 0
        dur = config.TEMPO_LED_BLINK_BEATS * spb
        for k, b in enumerate(blinks, 1):
            if b <= t < b + dur:
                lit, which = 1.0 - 0.6 * (t - b) / dur, k     # bright at the tick, dimming a little
        return LedState(max(0.0, visible), lit, which, c.new_bpm, c.faster)
    return None


def parse_tempo(raw, offset_ms):
    """TempoMap from a chart's "tempo" list. Raises TempoError with a readable message."""
    if not isinstance(raw, list) or not raw:
        raise TempoError("\"tempo\" must be a non-empty list like [{\"beat\": 0, \"bpm\": 120}]")
    segs = []
    for i, d in enumerate(raw):
        if not isinstance(d, dict) or "beat" not in d or "bpm" not in d:
            raise TempoError(f"tempo segment #{i + 1} needs \"beat\" and \"bpm\"")
        try:
            beat, bpm = float(d["beat"]), float(d["bpm"])
            shift = float(d.get("shift_ms", 0.0))
        except (TypeError, ValueError):
            raise TempoError(f"tempo segment #{i + 1}: beat, bpm and shift_ms must be numbers") from None
        if not beat.is_integer():
            raise TempoError(f"tempo segment #{i + 1}: beat {beat:g} must be a whole beat")
        segs.append(Segment(int(beat), bpm, shift))
    return TempoMap(segs, offset_ms)
