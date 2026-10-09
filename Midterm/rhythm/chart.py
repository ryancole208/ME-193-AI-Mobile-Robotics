"""
Song charts: songs/<name>.chart.json next to songs/<name>.wav|ogg|mp3. See README "Chart format".

    {
      "song": "Electrodoodle.wav",          audio file, relative to the chart
      "title": "Electrodoodle",
      "author": "Kevin MacLeod (Incompetech)",
      "attribution": {...},                 full credit (title, artist, source, license)
      "offset_ms": 24.7,                    song time of beat 0
      "tempo": [{"beat": 0, "bpm": 120.0},  tempo map (see tempo.py)
                {"beat": 255, "bpm": 160.0}],
      "downbeat": 3,                        the beat that is beat 1 of bar 1
      "start_bar": 3,                       first bar with hits
      "end_bar": 80,                        last bar with hits
      "pattern": [4],                       hit pattern of every bar...
      "sections": [                         ...except these bar ranges (inclusive)
        {"bars": [64, 73], "pattern": [2, 4]}
      ]
    }

Every song is 4/4 (BEATS_PER_BAR). Bar n, beat p (1..4) is chart beat downbeat + 4 (n - 1) + (p - 1);
bars before bar 1 (n <= 0) exist too, e.g. for a pickup. A pattern lists the beats of the bar
YOU hit; only [4] and [2, 4] are valid. The opponent hits halfway between your hits (see
scheduler.py). A section may add "warn": true to play the warning cue before its first hit.

HIT_BEATS in config.py, when set, replaces every bar's pattern (the files are never changed).
Everything else in the file is kept as-is when tools re-save it.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import config
from tempo import TempoError, parse_tempo

VALID_PATTERNS = ((4,), (2, 4))
_KNOWN = ("song", "title", "author", "attribution", "offset_ms", "tempo", "downbeat",
          "start_bar", "end_bar", "pattern", "sections")


class ChartError(ValueError):
    pass


@dataclass(frozen=True)
class Cue:
    """One of YOUR hits: the ball reaches your paddle on this chart beat."""
    beat: float
    warn: bool = False

    @property
    def upbeat(self):
        """True when the cue is NOT on a whole beat (e.g. 12.5)."""
        return abs(self.beat - round(self.beat)) > 1e-6


@dataclass(frozen=True)
class Section:
    first_bar: int
    last_bar: int
    pattern: tuple
    warn: bool = False


def check_pattern(value, name="HIT_BEATS"):
    """A valid hit pattern as a sorted tuple, else ChartError. Only [4] and [2, 4] exist."""
    ok = isinstance(value, (list, tuple)) and value and all(
        isinstance(b, (int, float)) and not isinstance(b, bool) for b in value)
    if ok:
        p = tuple(sorted({int(b) for b in value if float(b).is_integer()}))
        if len(p) == len(value) and p in VALID_PATTERNS:
            return p
    raise ChartError(f"{name} must be [4] (you hit on beat 4) or [2, 4] (you hit on beats 2 and 4); "
                     f"got {value!r}")


def check_hit_beats(value=None):
    """Validate config.HIT_BEATS: None (use the charts) or a valid pattern. Returns the tuple or None."""
    value = config.HIT_BEATS if value is None else value
    if value is None:
        return None
    if config.BEATS_PER_BAR != 4:
        raise ChartError(f"BEATS_PER_BAR must be 4 (every song is 4/4), got {config.BEATS_PER_BAR!r}")
    return check_pattern(value, "HIT_BEATS")


def pattern_text(p):
    return "[" + ", ".join(str(b) for b in p) + "]"


@dataclass
class Chart:
    path: Path | None
    song: str
    title: str
    tempo: object                    # tempo.TempoMap
    downbeat: int = 0
    start_bar: int = 1
    end_bar: int = 1
    pattern: tuple = (4,)
    sections: list = field(default_factory=list)
    author: str = ""
    attribution: dict | None = None
    extra: dict = field(default_factory=dict)   # unknown keys, preserved on save
    override: tuple | None = None    # HIT_BEATS, when it replaces the chart's patterns
    cues: list = field(default_factory=list)

    def __post_init__(self):
        if not self.cues:
            self.cues = self.build_cues()

    # -- timing (all through the tempo map) -------------------------------------
    @property
    def offset_ms(self):
        return self.tempo.offset_ms

    def beat_time(self, beat):
        return self.tempo.beat_time(beat)

    def time_beat(self, t):
        return self.tempo.time_beat(t)

    def spb_at(self, beat):
        return self.tempo.spb_at(beat)

    def bar_beat(self, bar, beat_in_bar=1):
        """Chart beat of bar `bar`, beat `beat_in_bar` (1..4; fractions are upbeats)."""
        return self.downbeat + config.BEATS_PER_BAR * (bar - 1) + (beat_in_bar - 1)

    def bar_of(self, beat):
        """(bar, beat in bar 1..4.999) of a chart beat."""
        rel = beat - self.downbeat
        bar = math.floor(rel / config.BEATS_PER_BAR + 1e-9)
        return bar + 1, rel - bar * config.BEATS_PER_BAR + 1

    def bpm_range(self):
        """(slowest, fastest) BPM over the part of the song that has hits."""
        if not self.cues:
            return self.tempo.bpm_range()
        return self.tempo.bpm_range(self.cues[0].beat - 4, self.cues[-1].beat)

    def bpm_label(self):
        lo, hi = self.bpm_range()
        return f"{lo:.0f}" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}"

    def tempo_changes(self):
        """Sudden tempo changes, faster or slower (tempo.TempoMap.sudden_changes)."""
        return self.tempo.sudden_changes()

    # -- hits ---------------------------------------------------------------------
    def pattern_for_bar(self, bar):
        if self.override is not None:
            return self.override
        for s in self.sections:
            if s.first_bar <= bar <= s.last_bar:
                return s.pattern
        return self.pattern

    def build_cues(self):
        """Your hits, bar by bar from start_bar to end_bar."""
        cues = []
        for bar in range(self.start_bar, self.end_bar + 1):
            sec = None if self.override is not None else next(
                (s for s in self.sections if s.first_bar <= bar <= s.last_bar), None)
            warn = sec is not None and sec.warn and bar == sec.first_bar
            for k, b in enumerate(self.pattern_for_bar(bar)):
                cues.append(Cue(float(self.bar_beat(bar, b)), warn=warn and k == 0))
        return cues

    def min_gap_beats(self):
        gaps = [b.beat - a.beat for a, b in zip(self.cues, self.cues[1:])]
        return min(gaps) if gaps else math.inf

    def min_gap_s(self):
        gaps = [self.beat_time(b.beat) - self.beat_time(a.beat) for a, b in zip(self.cues, self.cues[1:])]
        return min(gaps) if gaps else math.inf

    def with_override(self, pattern):
        """A copy whose every bar uses `pattern` (HIT_BEATS)."""
        return Chart(self.path, self.song, self.title, self.tempo, self.downbeat, self.start_bar,
                     self.end_bar, self.pattern, list(self.sections), self.author, self.attribution,
                     dict(self.extra), override=tuple(pattern))

    # -- files ------------------------------------------------------------------
    @property
    def song_path(self):
        base = self.path.parent if self.path else config.SONGS_DIR
        return base / self.song

    @property
    def song_id(self):
        """Stable key for saved best scores."""
        return Path(self.song).stem

    def to_dict(self):
        d = {"song": self.song, "title": self.title}
        if self.author:
            d["author"] = self.author
        if self.attribution:
            d["attribution"] = self.attribution
        d.update({"offset_ms": round(self.offset_ms, 2), "tempo": self.tempo.to_list(),
                  "downbeat": int(self.downbeat), "start_bar": int(self.start_bar),
                  "end_bar": int(self.end_bar), "pattern": list(self.pattern),
                  "sections": [section_to_dict(s) for s in self.sections]})
        d.update({k: v for k, v in self.extra.items() if k not in d})
        return d


def section_to_dict(s):
    d = {"bars": [s.first_bar, s.last_bar], "pattern": list(s.pattern)}
    if s.warn:
        d["warn"] = True
    return d


def _int(data, key, default):
    v = data.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v).is_integer():
        raise ChartError(f"\"{key}\" must be a whole number, got {v!r}")
    return int(v)


def parse_sections(raw):
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ChartError("\"sections\" must be a list like [{\"bars\": [17, 24], \"pattern\": [2, 4]}]")
    out = []
    for i, s in enumerate(raw):
        where = f"section #{i + 1}"
        if not isinstance(s, dict) or "bars" not in s or "pattern" not in s:
            raise ChartError(f"{where} needs \"bars\": [first, last] and \"pattern\"")
        bars = s["bars"]
        if (not isinstance(bars, list) or len(bars) != 2
                or not all(isinstance(b, int) and not isinstance(b, bool) for b in bars) or bars[0] > bars[1]):
            raise ChartError(f"{where}: \"bars\" must be [first, last] whole bar numbers, got {bars!r}")
        out.append(Section(bars[0], bars[1], check_pattern(s["pattern"], f"{where} pattern"),
                           bool(s.get("warn", False))))
    out.sort(key=lambda s: s.first_bar)
    for a, b in zip(out, out[1:]):
        if b.first_bar <= a.last_bar:
            raise ChartError(f"sections overlap: bars {a.first_bar}-{a.last_bar} and {b.first_bar}-{b.last_bar}")
    return out


def parse_chart(data, path=None):
    """Validate a chart dict and build a Chart. Raises ChartError with a readable message."""
    if not isinstance(data, dict):
        raise ChartError("chart must be a JSON object")
    song = data.get("song")
    if not isinstance(song, str) or not song:
        raise ChartError("\"song\" must be the audio file name")
    if Path(song).suffix.lower() not in config.AUDIO_EXTENSIONS:
        raise ChartError(f"unsupported audio type {Path(song).suffix!r} (use {', '.join(config.AUDIO_EXTENSIONS)})")
    try:
        offset = float(data.get("offset_ms", 0.0))
    except (TypeError, ValueError):
        raise ChartError("\"offset_ms\" must be a number") from None
    if "tempo" not in data:
        raise ChartError("missing \"tempo\" (the tempo map, e.g. [{\"beat\": 0, \"bpm\": 120}])")
    try:
        tempo = parse_tempo(data["tempo"], offset)
    except TempoError as e:
        raise ChartError(str(e)) from None
    downbeat = _int(data, "downbeat", 0)
    start_bar = _int(data, "start_bar", 2)
    if "end_bar" not in data:
        raise ChartError("missing \"end_bar\" (the last bar with hits)")
    end_bar = _int(data, "end_bar", 0)
    if end_bar < start_bar:
        raise ChartError(f"end_bar {end_bar} is before start_bar {start_bar}")
    pattern = check_pattern(data.get("pattern", [4]), "\"pattern\"")
    sections = parse_sections(data.get("sections"))
    title = data.get("title") or Path(song).stem.replace("_", " ").title()
    author = data.get("author") or ""
    attribution = data.get("attribution")
    if attribution is not None and not isinstance(attribution, dict):
        raise ChartError("\"attribution\" must be an object (title, artist, source, license, ...)")
    extra = {k: v for k, v in data.items() if k not in _KNOWN}
    chart = Chart(Path(path) if path else None, song, str(title), tempo, downbeat, start_bar, end_bar,
                  pattern, sections, str(author), attribution, extra)
    first = chart.cues[0].beat
    # the opponent serves (half a gap) before your first hit; that must be inside the audio
    serve = first - ((chart.cues[1].beat - first) / 2 if len(chart.cues) > 1 else 2)
    if chart.beat_time(serve) < 0:
        raise ChartError(f"start_bar {start_bar}: the first serve (beat {serve:g}) is before the start of the audio")
    return chart


def load_chart(path):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ChartError(f"not valid JSON (line {e.lineno}: {e.msg})") from None
    except OSError as e:
        raise ChartError(f"can't read the chart: {e}") from None
    chart = parse_chart(data, path)
    if not chart.song_path.exists():
        raise ChartError(f"audio file {chart.song!r} not found next to the chart")
    return chart


def save_chart(chart, path=None):
    """Write the chart in a hand-editable layout: one tempo segment / section per line."""
    path = Path(path or chart.path)
    d = chart.to_dict()
    lines = ["{"]
    items = list(d.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if k in ("tempo", "sections") and v:
            inner = ",\n".join("    " + json.dumps(x) for x in v)
            lines.append(f"  {json.dumps(k)}: [\n{inner}\n  ]{comma}")
        elif not isinstance(v, dict):
            lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}{comma}")
        else:
            body = json.dumps(v, indent=2, ensure_ascii=False).replace("\n", "\n  ")
            lines.append(f"  {json.dumps(k)}: {body}{comma}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    chart.path = path
    return path


@dataclass
class SongEntry:
    path: Path
    chart: Chart


def scan_songs(folder=None, use_hit_beats=True, log=print):
    """Playable charts in the songs folder, sorted by title. A chart that is invalid, or whose
    audio file is missing, is skipped with a warning (never a crash). With use_hit_beats and
    HIT_BEATS set, every chart's bars use the HIT_BEATS pattern (an invalid HIT_BEATS raises
    ChartError: it is a config mistake, not a song problem). Returns (entries, skipped) where
    skipped is a list of (file name, reason)."""
    folder = Path(folder or config.SONGS_DIR)
    override = check_hit_beats() if use_hit_beats else None
    entries, skipped = [], []
    for p in sorted(folder.glob("*" + config.CHART_SUFFIX)):
        try:
            ch = load_chart(p)
            if override is not None:
                ch = ch.with_override(override)
            entries.append(SongEntry(p, ch))
        except ChartError as e:
            skipped.append((p.name, str(e)))
            if log:
                log(f"[songs] skipping {p.name}: {e}")
    entries.sort(key=lambda e: e.chart.title.lower())
    return entries, skipped
