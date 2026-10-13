"""
Suggest hard ([2,4]) sections for the Godot charts from the music's energy.

For each chart: decode the song, cut it into bars with the chart's tempo map / offset / downbeat,
score every bar (loudness + percussive onset density, z-normalised within the song), group the
bars into phrases counted from start_bar and mark the most energetic phrases hard until roughly
TARGET_HARD_FRACTION of the playable bars are hard. The first INTRO_EASY_BARS stay easy; a phrase
whose halves differ a lot can contribute just its louder 4 bars.

Prints a per-bar energy graph and the suggestion, and writes a copy of each chart to
tools/suggested_charts/ in which ONLY the text of "sections" is replaced (every other byte is the
original's), plus summary.txt. The Godot charts are never written.

Run from Midterm/rhythm:
    ..\\..\\my_env\\Scripts\\python.exe -m tools.suggest_sections                 # every chart
    ..\\..\\my_env\\Scripts\\python.exe -m tools.suggest_sections Kick_Shock      # one chart
    ..\\..\\my_env\\Scripts\\python.exe -m tools.suggest_sections --target 0.25 --phrase 4

Tunables: tools/section_config.py.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # rhythm/ (for python tools/...)
from tools import section_config as sc  # noqa: E402


# -- tempo map (beat -> seconds), exactly as the Godot game does it ----------------------------
def tempo_points(tempo, offset_ms):
    """[(beat, time_s, bpm)] for each tempo entry."""
    segs = sorted(tempo, key=lambda s: s["beat"])
    if not segs:
        raise ValueError("empty tempo map")
    pts = [(float(segs[0]["beat"]), offset_ms / 1000, float(segs[0]["bpm"]))]
    for s in segs[1:]:
        pb, pt, pbpm = pts[-1]
        t = pt + (s["beat"] - pb) * 60 / pbpm + s.get("shift_ms", 0) / 1000
        pts.append((float(s["beat"]), t, float(s["bpm"])))
    return pts


def beat_seconds(pts, b):
    seg = pts[0]
    for p in pts:
        if p[0] <= b:
            seg = p
    return seg[1] + (b - seg[0]) * 60 / seg[2]


def bar_to_beat(chart, bar):
    return chart["downbeat"] + (bar - 1) * 4


def bar_edges(chart, start_bar, end_bar):
    """Start time of each bar start_bar..end_bar plus the end time of the last one."""
    pts = tempo_points(chart["tempo"], float(chart.get("offset_ms", 0)))
    return np.array([beat_seconds(pts, bar_to_beat(chart, bar))
                     for bar in range(start_bar, end_bar + 2)])


# -- audio features ----------------------------------------------------------------------------
def load_audio(path, sr=None):
    """Mono float32 at sr (soundfile decodes MP3 natively with libsndfile >= 1.1; no ffmpeg)."""
    import soundfile as sf
    sr = sr or sc.ANALYSIS_SR
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        import soxr
        y = soxr.resample(y, file_sr, sr)
    return np.ascontiguousarray(y, dtype=np.float32), sr


def bar_features(y, sr, edges):
    """(rms_db, onset) per bar; NaN for bars with no audio under them."""
    import librosa
    src = librosa.effects.percussive(y) if sc.USE_PERCUSSIVE else y
    env = librosa.onset.onset_strength(y=src, sr=sr, hop_length=sc.HOP_LENGTH)
    env_t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=sc.HOP_LENGTH)
    dur = len(y) / sr
    n = len(edges) - 1
    rms, onset = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        t0, t1 = max(edges[i], 0.0), min(edges[i + 1], dur)
        if t1 - t0 < 0.05:
            continue
        seg = y[int(t0 * sr):int(t1 * sr)]
        rms[i] = 20 * math.log10(math.sqrt(float(np.mean(seg.astype(np.float64) ** 2))) + 1e-9)
        m = (env_t >= t0) & (env_t < t1)
        if m.any():
            onset[i] = float(env[m].mean())
    return rms, onset


def _z(x):
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    sd = x[ok].std()
    return (x - x[ok].mean()) / (sd if sd > 1e-9 else 1.0)


def combine_scores(rms, onset, w_rms=None, w_onset=None):
    w_rms = sc.W_RMS if w_rms is None else w_rms
    w_onset = sc.W_ONSET if w_onset is None else w_onset
    return w_rms * _z(np.asarray(rms, float)) + w_onset * _z(np.asarray(onset, float))


# -- phrases and selection ---------------------------------------------------------------------
def make_phrases(start_bar, end_bar, phrase_bars, min_bars):
    """[(a, b)] inclusive, from start_bar; a leftover shorter than min_bars joins the last phrase."""
    out = []
    a = start_bar
    while a <= end_bar:
        b = min(a + phrase_bars - 1, end_bar)
        out.append([a, b])
        a = b + 1
    if len(out) > 1 and out[-1][1] - out[-1][0] + 1 < min_bars:
        tail = out.pop()
        out[-1][1] = tail[1]
    return [tuple(p) for p in out]


def _mean(scores, a, b):
    v = [scores[bar] for bar in range(a, b + 1) if not math.isnan(scores.get(bar, math.nan))]
    return sum(v) / len(v) if v else math.nan


def candidates(scores, start_bar, end_bar, phrase_bars=None, cfg=sc):
    """[(a, b, score)] hard-section candidates, one per phrase (or its louder half)."""
    phrase_bars = phrase_bars or cfg.PHRASE_BARS
    first_ok = start_bar + cfg.INTRO_EASY_BARS
    out = []
    for a, b in make_phrases(start_bar, end_bar, phrase_bars, cfg.MIN_SECTION_BARS):
        a = max(a, first_ok)
        n = b - a + 1
        if n < cfg.MIN_SECTION_BARS:
            continue
        whole = _mean(scores, a, b)
        if math.isnan(whole):
            continue
        cand = (a, b, whole)
        if cfg.SPLIT_HALVES and n >= 2 * cfg.MIN_SECTION_BARS:
            mid = a + n // 2
            s1, s2 = _mean(scores, a, mid - 1), _mean(scores, mid, b)
            if not (math.isnan(s1) or math.isnan(s2)) and abs(s1 - s2) > cfg.SPLIT_DIFF:
                cand = (a, mid - 1, s1) if s1 > s2 else (mid, b, s2)
        out.append(cand)
    return out


def merge_ranges(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(r) for r in out]


def select_hard(scores, start_bar, end_bar, target=None, phrase_bars=None, cfg=sc):
    """Highest-energy candidates first, each taken only if it brings the hard-bar count closer to
    target * playable bars. Returns merged inclusive (a, b) ranges."""
    target = cfg.TARGET_HARD_FRACTION if target is None else target
    goal = target * (end_bar - start_bar + 1)
    hard, chosen = 0, []
    for a, b, s in sorted(candidates(scores, start_bar, end_bar, phrase_bars, cfg),
                          key=lambda c: (-c[2], c[0])):
        n = b - a + 1
        if abs(hard + n - goal) < abs(hard - goal):
            chosen.append((a, b))
            hard += n
    return merge_ranges(chosen)


# -- chart text: replace only "sections" ---------------------------------------------------
def _skip_string(text, i):
    """i is at an opening quote; returns the index just past the closing quote."""
    i += 1
    while text[i] != '"':
        i += 2 if text[i] == "\\" else 1
    return i + 1


def _skip_ws(text, i):
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _value_end(text, i):
    """Index just past the JSON value starting at i."""
    if text[i] == '"':
        return _skip_string(text, i)
    if text[i] in "[{":
        depth = 0
        while True:
            c = text[i]
            if c == '"':
                i = _skip_string(text, i)
                continue
            if c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
    while i < len(text) and text[i] not in ",}] \t\r\n":
        i += 1
    return i


def find_top_level_value(text, key):
    """(start, end) of the value of a top-level key, or None."""
    i = _skip_ws(text, 0)
    if text[i] != "{":
        raise ValueError("chart is not a JSON object")
    i += 1
    while True:
        i = _skip_ws(text, i)
        if text[i] == "}":
            return None
        k_end = _skip_string(text, i)
        name = json.loads(text[i:k_end])
        i = _skip_ws(text, k_end)
        i = _skip_ws(text, i + 1)            # past ':'
        v_end = _value_end(text, i)
        if name == key:
            return i, v_end
        i = _skip_ws(text, v_end)
        if text[i] == ",":
            i += 1


def format_sections(sections):
    if not sections:
        return "[]"
    return "[\n" + ",\n".join("\t" + json.dumps(s) for s in sections) + "\n  ]"


def splice_sections(text, sections):
    """The chart text with only the "sections" value replaced (added at the end if missing).
    Checks that the result parses and differs from the original in nothing but sections."""
    new_val = format_sections(sections)
    span = find_top_level_value(text, "sections")
    if span:
        out = text[:span[0]] + new_val + text[span[1]:]
    else:
        close = text.rstrip().rfind("}")
        head = text[:close].rstrip()
        out = head + ',\n  "sections": ' + new_val + "\n" + text[close:]
    expect = json.loads(text)
    expect["sections"] = sections
    if json.loads(out) != expect:
        raise RuntimeError("splicing sections changed something else in the chart")
    return out


# -- per chart ---------------------------------------------------------------------------------
def hard_bars(sections, default_pattern):
    bars = set()
    for s in sections or []:
        if s.get("pattern") != default_pattern:
            a, b = s["bars"]
            bars.update(range(a, b + 1))
    return bars


def analyse(chart_path, music_dir=None, target=None, phrase_bars=None, cfg=sc):
    chart_path = Path(chart_path)
    text = chart_path.read_text(encoding="utf-8", newline="")     # keep the original line endings
    chart = json.loads(text)
    start, end = int(chart.get("start_bar", 1)), int(chart["end_bar"])
    audio = Path(music_dir or cfg.MUSIC_DIR) / chart["song"]
    y, sr = load_audio(audio, cfg.ANALYSIS_SR)
    edges = bar_edges(chart, start, end)
    rms, onset = bar_features(y, sr, edges)
    sc_arr = combine_scores(rms, onset, cfg.W_RMS, cfg.W_ONSET)
    scores = {start + i: float(v) for i, v in enumerate(sc_arr)}
    ranges = select_hard(scores, start, end, target, phrase_bars, cfg)
    sections = [{"bars": [a, b], "pattern": list(cfg.HARD_PATTERN)} for a, b in ranges]
    playable = end - start + 1
    n_hard = sum(b - a + 1 for a, b in ranges)
    old = hard_bars(chart.get("sections"), chart.get("pattern", [4]))
    new = {bar for a, b in ranges for bar in range(a, b + 1)}
    return {
        "path": chart_path, "chart": chart, "text": text, "start": start, "end": end,
        "scores": scores, "ranges": ranges, "sections": sections,
        "hard_fraction": n_hard / playable, "n_hard": n_hard, "playable": playable,
        "old_hard": old, "overlap": len(old & new),
        "audio_s": len(y) / sr, "last_bar_end_s": float(edges[-1]),
    }


def _graph_char():
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        sc.GRAPH_CHAR.encode(enc)
        return sc.GRAPH_CHAR
    except (UnicodeEncodeError, LookupError):
        return sc.GRAPH_FALLBACK_CHAR


def graph_lines(res, phrase_bars):
    ch = _graph_char()
    vals = [v for v in res["scores"].values() if not math.isnan(v)]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    span = hi - lo or 1.0
    hard = {bar for a, b in res["ranges"] for bar in range(a, b + 1)}
    phrase_starts = {a for a, _ in make_phrases(res["start"], res["end"], phrase_bars,
                                                sc.MIN_SECTION_BARS)}
    lines = []
    for bar, v in res["scores"].items():
        sep = "-" if bar in phrase_starts else " "
        mark = "  [2,4]" if bar in hard else ""
        old = "  (hand-made hard)" if bar in res["old_hard"] else ""
        if math.isnan(v):
            lines.append(f"{bar:>4}{sep}|{'(no audio)':<{sc.GRAPH_WIDTH}}|{mark}{old}")
            continue
        n = 1 + round((v - lo) / span * (sc.GRAPH_WIDTH - 1))
        lines.append(f"{bar:>4}{sep}|{ch * n:<{sc.GRAPH_WIDTH}}| {v:+.2f}{mark}{old}")
    return lines


def report_lines(res):
    c = res["chart"]
    out = [f"{res['path'].name}  ({c['song']}, bars {res['start']}-{res['end']}, "
           f"{res['playable']} playable)"]
    if res["sections"]:
        out.append("  sections: " + ", ".join(f"{a}-{b}" for a, b in res["ranges"])
                   + f"  -> {json.dumps(sc.HARD_PATTERN)}")
    else:
        out.append("  sections: none")
    out.append(f"  hard: {res['n_hard']}/{res['playable']} bars = {res['hard_fraction']:.0%}")
    if res["old_hard"]:
        n_new = res["n_hard"]
        out.append(f"  vs hand-made: {len(res['old_hard'])} hand-made hard bars, "
                   f"{res['overlap']} of them also suggested "
                   f"({res['overlap'] / len(res['old_hard']):.0%}); "
                   f"{n_new - res['overlap']} suggested bars are new")
    if res["last_bar_end_s"] > res["audio_s"] + 0.05:
        out.append(f"  warning: end_bar ends at {res['last_bar_end_s']:.1f} s but the audio is "
                   f"{res['audio_s']:.1f} s")
    return out


def resolve_chart(arg, charts_dir=None):
    charts_dir = Path(charts_dir or sc.CHARTS_DIR)
    p = Path(arg)
    if p.is_file():
        return p
    want = arg.lower().replace(" ", "_").removesuffix(".json").removesuffix(".chart")
    for c in sorted(charts_dir.glob("*.chart.json")):
        if c.name.lower().removesuffix(".chart.json") == want:
            return c
    matches = [c for c in sorted(charts_dir.glob("*.chart.json")) if want in c.name.lower()]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"no unique chart matches {arg!r} in {charts_dir}"
                     + (f" (candidates: {', '.join(m.name for m in matches)})" if matches else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("charts", nargs="*", help="chart file or name (default: every chart)")
    ap.add_argument("--target", type=float, default=sc.TARGET_HARD_FRACTION,
                    help=f"hard-bar fraction to aim for (default {sc.TARGET_HARD_FRACTION})")
    ap.add_argument("--phrase", type=int, choices=(4, 8), default=sc.PHRASE_BARS,
                    help=f"phrase length in bars (default {sc.PHRASE_BARS})")
    ap.add_argument("--charts-dir", default=str(sc.CHARTS_DIR))
    ap.add_argument("--music-dir", default=str(sc.MUSIC_DIR))
    ap.add_argument("--out", default=str(sc.OUT_DIR), help="where the suggested copies go")
    ap.add_argument("--no-graph", action="store_true", help="skip the per-bar energy graph")
    args = ap.parse_args(argv)
    if not 0 <= args.target <= 1:
        ap.error("--target must be between 0 and 1")

    out_dir = Path(args.out)
    if out_dir.resolve() == Path(args.charts_dir).resolve():
        ap.error("--out must not be the charts folder (the original charts are never overwritten)")
    paths = ([resolve_chart(a, args.charts_dir) for a in args.charts] if args.charts
             else sorted(Path(args.charts_dir).glob("*.chart.json")))
    if not paths:
        raise SystemExit(f"no charts found in {args.charts_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = [f"Suggested [2,4] sections  (target {args.target:.0%}, phrase {args.phrase} bars, "
               f"intro {sc.INTRO_EASY_BARS} bars easy, min section {sc.MIN_SECTION_BARS}, "
               f"W_RMS {sc.W_RMS}, W_ONSET {sc.W_ONSET})", ""]
    for p in paths:
        print(f"\n=== {p.name} ===", flush=True)
        try:
            res = analyse(p, args.music_dir, args.target, args.phrase)
        except Exception as e:  # one bad chart shouldn't stop the rest
            print(f"  skipped: {e}")
            summary += [f"{p.name}: skipped ({e})", ""]
            continue
        if not args.no_graph:
            print("\n".join(graph_lines(res, args.phrase)))
        rep = report_lines(res)
        print("\n".join(rep))
        dest = out_dir / p.name
        dest.write_text(splice_sections(res["text"], res["sections"]), encoding="utf-8", newline="")
        print(f"  wrote {dest}")
        summary += rep + [""]
    (out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"\nsummary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
