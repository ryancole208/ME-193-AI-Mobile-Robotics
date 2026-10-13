"""Small chart builders shared by the tests (no audio is generated here)."""

import json

from chart import Chart, Cue
from tempo import Segment, TempoMap

ELECTRODOODLE = "Electrodoodle"


def tempo(bpm=120.0, offset_ms=0.0, changes=()):
    """TempoMap: bpm from beat 0, then (beat, bpm[, shift_ms]) changes."""
    segs = [Segment(0, bpm)] + [Segment(c[0], c[1], c[2] if len(c) > 2 else 0.0) for c in changes]
    return TempoMap(segs, offset_ms)


def chart_with_cues(beats, bpm=170.0, offset_ms=0.0, changes=()):
    """A Chart whose hits are exactly these beats (bypasses the bar patterns)."""
    cues = [Cue(float(b)) for b in beats]
    return Chart(None, "x.wav", "x", tempo(bpm, offset_ms, changes), cues=cues)


def bar_chart(start_bar=2, end_bar=9, pattern=(4,), sections=(), bpm=120.0, offset_ms=0.0, downbeat=0,
              changes=()):
    from chart import Section
    secs = [Section(a, b, tuple(p)) for a, b, p in sections]
    return Chart(None, "x.wav", "x", tempo(bpm, offset_ms, changes), downbeat, start_bar, end_bar,
                 tuple(pattern), secs)


def chart_dict(**over):
    d = {"song": "x.wav", "title": "X", "offset_ms": 0.0, "tempo": [{"beat": 0, "bpm": 120}],
         "downbeat": 0, "start_bar": 2, "end_bar": 9, "pattern": [4], "sections": []}
    d.update(over)
    return d


def write_chart(folder, stem, audio=True, **over):
    """songs folder entry: <stem>.chart.json (+ an empty <stem>.wav unless audio=False)."""
    d = chart_dict(song=f"{stem}.wav", title=stem.replace("_", " "), **over)
    (folder / f"{stem}.chart.json").write_text(json.dumps(d), encoding="utf-8")
    if audio:
        (folder / f"{stem}.wav").write_bytes(b"")
    return folder / f"{stem}.chart.json"
