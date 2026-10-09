"""
Chart helper for Concert Rally.

1) Make a chart from a song file (tempo map + bars detected over the whole song):

    python make_chart.py detect "songs/Latin_Industries.wav"
    python make_chart.py detect songs/Electrodoodle.wav --incompetech --force
    python make_chart.py detect "songs/Kick Shock.mp3" --to-wav --incompetech   # -> Kick_Shock.wav + chart
    python make_chart.py detect songs/my_song.wav --bpm 128      # force one fixed tempo

   Writes songs/<name>.chart.json: the tempo map ("tempo"), beat 0 ("offset_ms"), which beat is
   bar 1's downbeat ("downbeat"), the bars with hits ("start_bar" / "end_bar", after the intro
   and before the ending) and the [4] pattern on every bar. --incompetech fills in the author
   ("Kevin MacLeod (Incompetech)") and the full Creative Commons attribution. Re-running with
   --force keeps your title, author, attribution, pattern and sections (the old file is saved
   as .chart.json.bak).

   How the tempo map is found (see README "How tempo detection works"):
     * an onset-strength envelope (librosa when installed, else a built-in numpy spectral flux);
     * the local tempo in overlapping windows (CHART_WINDOW_S, every CHART_WINDOW_HOP_S): the
       best of the autocorrelation peaks inside CHART_BPM_RANGE, refined with a comb filter;
     * runs of windows with the same tempo (within CHART_MERGE_BPM) become segments; a run shorter
       than CHART_MIN_SEGMENT_S is jitter and joins its neighbour;
     * each segment's tempo and beat phase are refitted over its whole span (a steady grid);
     * each change point is where the OLD beat stops fitting the music (the new grid's small
       phase difference is kept as shift_ms);
     * downbeat: the beat of the bar with the strongest bass (kick) onsets.
     * start / end bar: CHART_INTRO_BARS after the music gets loud, CHART_OUTRO_BARS before it
       fades out.

2) Preview a chart by ear -- the song plays with a click on every beat, a louder click on beat 1
   of each bar, and the tempo-change LED + ticks. Nudge and save while it plays:

    python make_chart.py preview songs/Electrodoodle.chart.json
    python make_chart.py preview songs/Electrodoodle.chart.json --at 2:00

   Left/Right offset_ms (5 ms, Shift = 1 ms)    Up/Down downbeat (one beat)
   B          move every beat by half a beat (when the clicks sit on the off-beats)
   [ / ]      shift the tempo segment playing now (5 ms, Shift = 1 ms)
   Space pause    PgUp/PgDn 4 bars back/ahead    J jump to just before the next tempo jump
   Home restart   S save (old file kept as .chart.json.bak)    Esc quit

3) Write the same click track to a file instead (previews/<name>.clicks.wav):

    python make_chart.py clicks songs/Electrodoodle.chart.json
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from chart import Chart, ChartError, load_chart, parse_chart, pattern_text, save_chart  # noqa: E402
from tempo import Segment, TempoMap, led_state  # noqa: E402

DETECTED_KEYS = ("song", "title", "author", "attribution", "offset_ms", "tempo", "downbeat", "start_bar",
                 "end_bar", "bpm", "cues")
INCOMPETECH_AUTHOR = "Kevin MacLeod (Incompetech)"
CC_BY = ("Creative Commons: By Attribution 4.0 License", "http://creativecommons.org/licenses/by/4.0/")


def incompetech_attribution(title):
    return {"title": title, "artist": "Kevin MacLeod", "source": "incompetech.com",
            "license": CC_BY[0], "license_url": CC_BY[1],
            "credit": f"\"{title}\" Kevin MacLeod (incompetech.com)\nLicensed under {CC_BY[0]}\n{CC_BY[1]}"}


# =============================================================================
# Detection
# =============================================================================

def load_mono(path):
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def onset_envelope(y, sr, hop=256, n_fft=1024):
    """Onset strength -> (envelope, frames per second, time of frame 0 in s, method)."""
    try:
        import librosa
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        # its peaks sit ~1.5 frames after the onset (measured on click tracks)
        return np.asarray(env, dtype=float), sr / hop, -1.5 * hop / sr, "librosa"
    except ImportError:
        pass
    win = np.hanning(n_fft)
    frames = 1 + (len(y) - n_fft) // hop
    if frames < 8:
        raise ValueError("audio too short")
    top = int(3000 * n_fft / sr)          # below ~3 kHz: kicks, bass, claps -- not hi-hat noise
    blocks = []
    for a in range(0, frames, 4096):      # in blocks, so a long song doesn't need GBs of RAM
        idx = np.arange(n_fft)[None, :] + hop * np.arange(a, min(frames, a + 4096))[:, None]
        blocks.append(np.log1p(10 * np.abs(np.fft.rfft(y[idx] * win, axis=1)[:, :top])))
    spec = np.vstack(blocks)
    flux = np.maximum(0.0, np.diff(spec, axis=0)).sum(axis=1)
    flux -= np.convolve(flux, np.ones(16) / 16, mode="same")       # remove the slow trend
    # flux[i] compares windows i and i+1; window i+1 is centred at (i+1) * hop + n_fft / 2
    return np.maximum(flux, 0.0), sr / hop, (hop + n_fft / 2) / sr, "numpy"


def comb_mean(env, period, phases):
    """Mean of the (interpolated) envelope on a beat grid, for each phase (all in frames)."""
    n = int((len(env) - 1 - phases.max()) / period)
    if n < 2:
        return np.zeros(len(phases))
    pos = phases[:, None] + period * np.arange(n)[None, :]
    i = pos.astype(int)
    f = pos - i
    return (env[i] * (1 - f) + env[np.minimum(i + 1, len(env) - 1)] * f).mean(axis=1)


def fit_grid(env, fps, bpm_lo, bpm_hi, fine=False):
    """Steady beat grid over a stretch of envelope -> (score, bpm, phase in frames).

    Candidates are the autocorrelation peaks inside [bpm_lo, bpm_hi] (so no half / double
    tempo mistakes); each is refined with a comb filter, and the best comb score wins (so a
    syncopated groove's 4:3 "tempo" loses to the real beat)."""
    x = env - env.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    lag_lo = max(2, int(60 * fps / bpm_hi))
    lag_hi = min(len(ac) - 2, int(60 * fps / bpm_lo) + 1)
    peaks = [(ac[L], L) for L in range(lag_lo, lag_hi + 1) if ac[L] >= ac[L - 1] and ac[L] >= ac[L + 1]]
    if not peaks:
        peaks = [(0.0, 60 * fps / ((bpm_lo + bpm_hi) / 2))]
    best = (-np.inf, 1.0, 0.0)
    for _, lag in sorted(peaks, reverse=True)[:4]:
        for p in np.linspace(lag * 0.985, lag * 1.015, 31):
            p = min(max(p, 60 * fps / bpm_hi), 60 * fps / bpm_lo)
            phases = np.arange(0.0, p, 0.5)
            s = comb_mean(env, p, phases)
            j = int(np.argmax(s))
            if s[j] > best[0]:
                best = (float(s[j]), p, float(phases[j]))
    score, p, ph = best
    if fine:                              # sub-step refinement: a 0.1 % tempo error drifts over a song
        for width, n in ((0.003, 61), (0.0004, 41)):
            periods = np.linspace(p * (1 - width), p * (1 + width), n)
            phases = np.arange(max(0.0, ph - 1.0), ph + 1.0, 0.05)
            scores = [(comb_mean(env, q, phases).max(), q) for q in periods]
            score, p = max(scores)
            ph = float(phases[int(np.argmax(comb_mean(env, p, phases)))])
    return float(score), 60.0 * fps / p, ph


def local_tempos(env, fps, bpm_range=None):
    """[(window centre s, bpm)] over the whole song."""
    lo, hi = bpm_range or config.CHART_BPM_RANGE
    win, hop = config.CHART_WINDOW_S, config.CHART_WINDOW_HOP_S
    dur = len(env) / fps
    if dur <= win:
        return [(dur / 2, fit_grid(env, fps, lo, hi)[1])]
    out = []
    for t0 in np.arange(0.0, dur - win + 1e-9, hop):
        seg = env[int(t0 * fps):int((t0 + win) * fps)]
        out.append((t0 + win / 2, fit_grid(seg, fps, lo, hi)[1]))
    return out


def tempo_groups(bpms):
    """Cluster window tempos (within CHART_MERGE_BPM) -> [(median bpm, count)], biggest first."""
    groups = []
    for x in sorted(bpms):
        if groups and x - groups[-1][-1] <= config.CHART_MERGE_BPM:
            groups[-1].append(x)
        else:
            groups.append([x])
    return sorted(((float(np.median(g)), len(g)) for g in groups), key=lambda g: -g[1])


def run_tempo(run):
    """A run's tempo = the tempo most of its windows agree on (not a blend of two)."""
    return tempo_groups([b for _, b in run])[0][0]


def run_purity(run):
    """Share of the run's windows that agree with its tempo."""
    return tempo_groups([b for _, b in run])[0][1] / len(run)


def tempo_runs(windows, dur):
    """Group windows into runs of one tempo -> [(t_start, t_end, bpm, alternatives)] covering
    [0, dur]. alternatives = [(bpm, share)] of other tempos some windows preferred."""
    merge = config.CHART_MERGE_BPM
    runs = []                                         # [list of (t, bpm)]
    for t, bpm in windows:
        if runs and abs(bpm - run_tempo(runs[-1])) <= merge:
            runs[-1].append((t, bpm))
        else:
            runs.append([(t, bpm)])

    def span(r):
        return r[-1][0] - r[0][0] + config.CHART_WINDOW_HOP_S

    def join(i, j):
        a, b = sorted((i, j))
        runs[a:b + 1] = [runs[a] + runs[b]]
        k = 0                                         # neighbours that now agree merge too
        while k + 1 < len(runs):
            if abs(run_tempo(runs[k]) - run_tempo(runs[k + 1])) <= merge:
                runs[k:k + 2] = [runs[k] + runs[k + 1]]
            else:
                k += 1

    # short runs are jitter: they join the neighbour with the closer tempo (shortest first);
    # then a run whose windows mostly disagree with its tempo (a groove that fits two tempos,
    # e.g. 3:2, flipping back and forth) is not a real tempo change either: it joins a neighbour
    while len(runs) > 1:
        short = [i for i, r in enumerate(runs) if span(r) < config.CHART_MIN_SEGMENT_S]
        mixed = [i for i, r in enumerate(runs) if run_purity(r) < config.CHART_RUN_PURITY]
        if not short and not mixed:
            break
        i = min(short, key=lambda k: span(runs[k])) if short else min(mixed, key=lambda k: run_purity(runs[k]))
        t = run_tempo(runs[i])
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        join(i, min(nbrs, key=lambda k: abs(run_tempo(runs[k]) - t)))
    out = []
    for i, r in enumerate(runs):
        t0 = 0.0 if i == 0 else (runs[i - 1][-1][0] + r[0][0]) / 2
        t1 = dur if i == len(runs) - 1 else (r[-1][0] + runs[i + 1][0][0]) / 2
        groups = tempo_groups([b for _, b in r])
        out.append((t0, t1, groups[0][0], [(g, n / len(r)) for g, n in groups[1:]]))
    return out


def best_change_beat(env, fps, t_env0, tmap, new_bpm, new_phase_t, t_lo, t_hi):
    """The whole beat (of tmap) where the new grid takes over, and its shift (s).

    The old tempo is kept for as long as the music still fits it: the change is the first beat
    from which the old grid's fit stays below CHART_CHANGE_FIT x its usual level for two bars
    while the new grid fits at that level. (A syncopated groove can make the new, faster grid
    fit before the real change too; ears hear the change when the OLD beat stops working.) If
    no beat qualifies, the beat where switching grids scores best overall is used instead."""
    m = float(env[int(max(0, (t_lo - t_env0) * fps)):int((t_hi - t_env0) * fps)].mean())

    def vals(times):
        f = (np.asarray(times, dtype=float) - t_env0) * fps
        f = f[(f >= 0) & (f < len(env) - 1)]
        i = f.astype(int)
        return env[i] * (1 - (f - i)) + env[i + 1] * (f - i) - m

    spb_new = 60.0 / new_bpm
    b_lo, b_hi = math.ceil(tmap.time_beat(t_lo)), math.floor(tmap.time_beat(t_hi))
    old = {b: tmap.beat_time(b) for b in range(b_lo - 8, b_hi + 9)}
    span = config.CHART_WINDOW_S
    lvl_old = max(1e-9, float(vals([t for t in old.values() if t < t_lo + span]).mean()))
    lvl_new = max(1e-9, float(vals(np.arange(new_phase_t + math.ceil((t_hi - span - new_phase_t) / spb_new) * spb_new,
                                             t_hi, spb_new)).mean()))
    cands = []
    for b in range(b_lo + 1, b_hi):
        tb = old[b]
        g = new_phase_t + round((tb - new_phase_t) / spb_new) * spb_new   # new-grid beat nearest b
        shift = g - tb
        if abs(shift) >= 0.45 * tmap.spb_at(b - 1):
            continue
        fit_old = float(vals([old[x] for x in range(b, b + 4)]).mean()) / lvl_old       # one bar
        fit_new = float(vals(g + spb_new * np.arange(4)).mean()) / lvl_new
        total = float(vals([t for t in old.values() if t_lo <= t < tb]).sum()
                      + vals(np.arange(g, t_hi, spb_new)).sum())
        cands.append((b, shift, fit_old, fit_new, total))
    if not cands:
        raise ValueError("could not place the tempo change")
    keep = config.CHART_CHANGE_FIT
    for i, (b, shift, _, fit_new, _) in enumerate(cands):
        ahead = [c for c in cands[i:] if c[0] < b + 8]                  # this beat and the next two bars
        if len(ahead) >= 8 and all(c[2] < keep for c in ahead) and fit_new >= keep:
            # the bar-long fit window reaches past the change, so it fires a beat or two early;
            # music normally switches tempo right where an old beat falls: of the next few
            # beats, take the one the new grid meets with the smallest shift
            near = [c for c in cands[i:] if c[0] <= b + 3]
            b, shift, *_ = min(near, key=lambda c: (abs(c[1]), c[0]))
            return b, shift
    b, shift, *_ = max(cands, key=lambda c: c[4])
    return b, shift


def detect_tempo_map(env, fps, t_env0, dur, bpm=None):
    """TempoMap for the song (and the runs it came from)."""
    if bpm is not None:
        p = 60.0 * fps / bpm
        phases = np.arange(0.0, p, 0.05)
        ph = float(phases[int(np.argmax(comb_mean(env, p, phases)))])
        runs = [(0.0, dur, bpm, [])]
        grids = [(bpm, t_env0 + ph / fps)]
    else:
        runs = tempo_runs(local_tempos(env, fps), dur)
        grids = []
        margin = config.CHART_WINDOW_S / 2
        for i, (t0, t1, b, _alts) in enumerate(runs):
            a = t0 + (margin if i > 0 else 0.0)                    # stay clear of the change zones
            z = t1 - (margin if i < len(runs) - 1 else 0.0)
            if z - a < config.CHART_WINDOW_S:
                a, z = t0, t1
            i0 = int(a * fps)
            _, fb, ph = fit_grid(env[i0:int(z * fps)], fps, b * 0.98, b * 1.02, fine=True)
            grids.append((fb, t_env0 + (i0 + ph) / fps))
    bpm0, t_first = grids[0]
    spb0 = 60.0 / bpm0
    offset = t_first % spb0
    segs = [Segment(0, round(bpm0, 3))]
    tmap = TempoMap(segs, offset * 1000)
    for i in range(1, len(runs)):
        new_bpm, phase_t = grids[i]
        boundary = runs[i][0]
        lo = max(0.0, boundary - 1.5 * config.CHART_WINDOW_S)
        hi = min(dur, boundary + 1.5 * config.CHART_WINDOW_S)
        b, shift = best_change_beat(env, fps, t_env0, tmap, new_bpm, phase_t, lo, hi)
        shift_ms = 0.0 if abs(shift) < 0.002 else round(shift * 1000, 1)
        segs.append(Segment(b, round(new_bpm, 3), shift_ms))
        tmap = TempoMap(segs, offset * 1000)
    return tmap, runs


def snap_changes_to_bars(tmap, downbeat):
    """Music changes tempo on a bar line. A change placed mid-bar (often during a fill whose
    notes already fit the new tempo) moves to the nearest bar line within a bar where the new
    grid meets the old one within CHART_BARLINE_SNAP_MS. The new grid itself doesn't move: the
    beats after the change are renumbered so the bar line lands on it (later segments follow)."""
    segs = list(tmap.segments)
    limit = config.CHART_BARLINE_SNAP_MS / 1000
    for i in range(1, len(segs)):
        prev, seg = segs[i - 1], segs[i]
        if (seg.beat - downbeat) % 4 == 0:
            continue
        m = TempoMap(segs, tmap.offset_ms)
        t_prev = m.beat_time(prev.beat)
        g0, spb = m.beat_time(seg.beat), seg.spb
        nxt = segs[i + 1].beat if i + 1 < len(segs) else math.inf
        best = None
        back = (seg.beat - downbeat) % 4
        for b in (seg.beat - back, seg.beat + 4 - back):          # the bar lines around it
            if not prev.beat < b < nxt:
                continue
            old_t = t_prev + (b - prev.beat) * prev.spb             # the old grid's beat b
            k = round((old_t - g0) / spb)
            shift = g0 + k * spb - old_t                            # the new grid's nearest beat
            if abs(shift) <= limit and (best is None or abs(shift) < abs(best[1])):
                best = (b, shift, k)
        if best is None:
            continue
        b, shift, k = best
        delta = (b - seg.beat) - k
        segs[i] = Segment(b, seg.bpm, round(shift * 1000, 1) if abs(shift) >= 0.002 else 0.0)
        segs[i + 1:] = [Segment(s.beat + delta, s.bpm, s.shift_ms) for s in segs[i + 1:]]
    return TempoMap(segs, tmap.offset_ms)


def bass_downbeat(y, sr, tmap, dur):
    """0..3: the beat of the bar with the strongest low-frequency (kick / bass) onsets."""
    n_fft, hop = 2048, 512
    frames = 1 + (len(y) - n_fft) // hop
    if frames < 8:
        return 0
    top = max(2, int(160 * n_fft / sr))
    win = np.hanning(n_fft)
    low = np.empty(frames)
    for a in range(0, frames, 2048):
        idx = np.arange(n_fft)[None, :] + hop * np.arange(a, min(frames, a + 2048))[:, None]
        low[a:a + len(idx)] = np.log1p(np.abs(np.fft.rfft(y[idx] * win, axis=1)[:, 1:top])).sum(axis=1)
    flux = np.maximum(0.0, np.diff(low, prepend=low[0]))
    t_frame = (np.arange(frames) * hop + n_fft / 2) / sr
    sums = np.zeros(4)
    b = 0
    while True:
        t = tmap.beat_time(b)
        if t >= dur:
            break
        i = int(np.searchsorted(t_frame, t))
        sums[b % 4] += flux[max(0, i - 2):i + 3].max(initial=0.0)
        b += 1
    return int(np.argmax(sums))


def bar_loudness(y, sr, chart, bars):
    out = []
    for n in bars:
        a, b = chart.beat_time(chart.bar_beat(n)), chart.beat_time(chart.bar_beat(n + 1))
        seg = y[int(max(0, a) * sr):int(max(0, b) * sr)]
        out.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    return np.asarray(out)


def hit_bars(y, sr, chart, dur):
    """(start_bar, end_bar): after the intro (CHART_INTRO_BARS after the music gets going) and
    before the ending (CHART_OUTRO_BARS before it fades), with the last return inside the song."""
    first = math.floor((chart.time_beat(0.0) - chart.downbeat) / 4) + 1
    while chart.beat_time(chart.bar_beat(first)) < 0:
        first += 1
    last = first
    while chart.beat_time(chart.bar_beat(last + 1)) <= dur:
        last += 1
    last -= 1                                           # the last bar that ends inside the audio
    bars = list(range(first, last + 1))
    loud = bar_loudness(y, sr, chart, bars)
    ok = loud >= config.CHART_LOUD_FRACTION * float(np.median(loud))
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        raise ValueError("the song seems to be silent")
    start = max(1, bars[idx[0]] + config.CHART_INTRO_BARS)
    end = bars[idx[-1]] - config.CHART_OUTRO_BARS
    # your last return (2 beats after beat 4) must land before the end of the audio
    while end > start and chart.beat_time(chart.bar_beat(end + 1, 2)) > dur:
        end -= 1
    if end < start:
        raise ValueError("the song is too short for any hits")
    return start, end


def detect(path, bpm=None):
    """Everything detect writes -> dict (tempo map, offset, downbeat, bars, duration, method)."""
    y, sr = load_mono(path)
    dur = len(y) / sr
    env, fps, t0, method = onset_envelope(y, sr)
    tmap, runs = detect_tempo_map(env, fps, t0, dur, bpm)
    downbeat = bass_downbeat(y, sr, tmap, dur)
    tmap = snap_changes_to_bars(tmap, downbeat)
    probe = Chart(None, Path(path).name, "", tmap, downbeat, 1, 1)
    start, end = hit_bars(y, sr, probe, dur)
    return {"tempo": tmap, "downbeat": downbeat, "start_bar": start, "end_bar": end,
            "duration": dur, "method": method, "runs": runs}


def to_wav(audio):
    """Decode an MP3 / OGG to songs/<Name_With_Underscores>.wav (16-bit, same sample rate) once, so
    the game plays a sample-exact file. Returns the WAV path (the original is kept)."""
    import soundfile as sf
    wav = audio.with_name(audio.stem.replace(" ", "_") + ".wav")
    if wav.exists():
        print(f"using the existing {wav.name}")
        return wav
    data, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    sf.write(str(wav), data, sr, subtype="PCM_16")
    print(f"decoded {audio.name} -> {wav.name} ({len(data) / sr:.1f} s, {sr} Hz)")
    return wav


def cmd_detect(args):
    audio = Path(args.audio).resolve()
    if not audio.exists():
        sys.exit(f"no such file: {audio}")
    if audio.suffix.lower() not in config.AUDIO_EXTENSIONS:
        sys.exit(f"unsupported audio type {audio.suffix} (use {', '.join(config.AUDIO_EXTENSIONS)})")
    title_from_file = audio.stem.replace("_", " ")
    if args.to_wav and audio.suffix.lower() != ".wav":
        audio = to_wav(audio)
    elif audio.suffix.lower() == ".mp3":
        print("note: MP3 files can start late by an encoder delay (~25-50 ms). Prefer WAV (--to-wav), "
              "or check offset_ms by ear in preview.")
    out = audio.with_name(audio.stem + config.CHART_SUFFIX)
    old = {}
    if out.exists():
        if not args.force:
            sys.exit(f"{out.name} already exists (use --force to re-detect; your title, author, "
                     "attribution, pattern and sections are kept)")
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        shutil.copyfile(out, out.with_name(out.name + ".bak"))
    d = detect(audio, args.bpm)
    title = args.title or old.get("title") or title_from_file
    author = args.author or old.get("author") or ""
    attribution = old.get("attribution")
    if args.incompetech:
        author, attribution = INCOMPETECH_AUTHOR, incompetech_attribution(title)
    # keep pattern, sections and any extra fields; everything detected is replaced (old-format
    # "bpm" / "cues" are dropped)
    keep = {k: v for k, v in old.items() if k not in DETECTED_KEYS}
    rel = audio.name if audio.parent == out.parent else os.path.relpath(audio, out.parent)
    data = {"song": rel, "title": title, "author": author, "attribution": attribution,
            "offset_ms": round(d["tempo"].offset_ms, 1), "tempo": d["tempo"].to_list(),
            "downbeat": d["downbeat"], "start_bar": d["start_bar"], "end_bar": d["end_bar"],
            "pattern": [4], "sections": []}
    data.update(keep)
    data = {k: v for k, v in data.items() if v not in (None, "")}
    try:
        chart = parse_chart(data, out)
    except ChartError as e:
        sys.exit(f"detection gave an unusable chart: {e}")
    save_chart(chart, out)
    print(f"[{d['method']} onsets] {audio.name}: {d['duration']:.1f} s")
    for s in chart.tempo.segments:
        t = chart.beat_time(s.beat)
        extra = f", shift {s.shift_ms:+g} ms" if s.shift_ms else ""
        print(f"  beat {s.beat:>4} ({int(t // 60)}:{t % 60:05.2f})  {s.bpm:g} BPM{extra}")
    for c in chart.tempo_changes():
        bar, beat = chart.bar_of(c.beat)
        print(f"  sudden tempo {'increase' if c.faster else 'decrease'} {c.old_bpm:.0f} -> {c.new_bpm:.0f} BPM "
              f"on bar {bar}, beat {beat:g}")
    print(f"  beat 0 at {chart.offset_ms:.1f} ms, bar 1 starts on beat {chart.downbeat} (strongest bass), "
          f"hits in bars {chart.start_bar}-{chart.end_bar} ({len(chart.cues)} hits)")
    for t0, t1, bpm, alts in d["runs"]:
        for alt, share in alts:
            if share >= 0.15:
                print(f"  note: {share:.0%} of {int(t0 // 60)}:{t0 % 60:04.1f}-{int(t1 // 60)}:{t1 % 60:04.1f} also "
                      f"fits {alt:.1f} BPM (vs {bpm:.1f}). If the clicks feel wrong in preview, re-run with "
                      f"--bpm {alt:.1f} --force")
    print(f"wrote {out}\nCheck it by ear: python make_chart.py preview \"{out}\"")


# =============================================================================
# Preview (live) and clicks (to a file)
# =============================================================================

def click_events(chart, t_from, t_to):
    """[(song time, kind)] for the click track: "accent" on beat 1 of each bar, "beat" on the
    others, "tick" for the tempo-change ticks."""
    out = []
    b = math.ceil(chart.time_beat(t_from) - 1e-9)
    while True:
        t = chart.beat_time(b)
        if t > t_to:
            break
        if t >= t_from:
            out.append((t, "accent" if abs(chart.bar_of(b)[1] - 1) < 1e-6 else "beat"))
        b += 1
    for c in chart.tempo_changes():
        out += [(t, "tick") for t in c.blink_times() if t_from <= t <= t_to]
    return sorted(out)


def cmd_clicks(args):
    """Write previews/<song>.clicks.wav: the song plus the preview click track."""
    from audio_engine import load_audio
    from sounds import click, tempo_tick
    try:
        chart = load_chart(args.chart)
    except ChartError as e:
        sys.exit(f"{Path(args.chart).name}: {e}")
    sr = 44100
    mix = load_audio(chart.song_path, sr) * config.MUSIC_VOLUME
    snd = {"beat": (click(sr), config.PREVIEW_CLICK_VOLUME),
           "accent": (click(sr, accent=True), config.PREVIEW_ACCENT_VOLUME),
           "tick": (tempo_tick(sr), config.TEMPO_TICK_VOLUME)}
    for t, kind in click_events(chart, 0.0, len(mix) / sr):
        data, gain = snd[kind]
        i = int(round(t * sr))
        n = min(len(data), len(mix) - i)
        if n > 0:
            mix[i:i + n] += data[:n] * gain
    out_dir = config.HERE / "previews"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{Path(chart.song).stem}.clicks.wav"
    import soundfile as sf
    sf.write(str(out), np.clip(mix, -1, 1), sr, subtype="PCM_16")
    print(f"wrote {out}\n  click = every beat, loud click = beat 1 of a bar, bright tick = tempo-change warning")


def parse_time(text):
    """'2:08' or '128.5' -> seconds."""
    if text is None:
        return 0.0
    if ":" in text:
        m, s = text.split(":", 1)
        return 60 * float(m) + float(s)
    return float(text)


class Preview:
    """Live chart preview: plays the song with clicks; edits offset / downbeat / segment shifts."""

    def __init__(self, path, engine, sounds, music):
        self.path = Path(path)
        self.engine, self.sounds, self.music = engine, sounds, music
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.chart = load_chart(self.path)
        self.length = len(music) / engine.sr
        self.paused_at = None
        self.dirty = False
        self.backed_up = False
        self.message = ""

    # -- editing ------------------------------------------------------------------
    def edit(self, **changes):
        """Apply changes to the chart dict ("offset_ms", "downbeat", ("shift", index, ms))."""
        data = json.loads(json.dumps(self.data))
        if "offset_ms" in changes:
            data["offset_ms"] = round(float(data.get("offset_ms", 0.0)) + changes["offset_ms"], 2)
        if "downbeat" in changes:
            data["downbeat"] = (int(data.get("downbeat", 0)) + changes["downbeat"]) % 4
        if "shift" in changes:
            i, ms = changes["shift"]
            seg = data["tempo"][i]
            seg["shift_ms"] = round(float(seg.get("shift_ms", 0.0)) + ms, 2)
            if seg["shift_ms"] == 0:
                seg.pop("shift_ms")
        try:
            chart = parse_chart(data, self.path)
        except ChartError as e:
            self.message = f"can't: {e}"
            return False
        self.data, self.chart, self.dirty = data, chart, True
        self.reschedule()
        return True

    def save(self):
        if not self.backed_up:
            shutil.copyfile(self.path, self.path.with_name(self.path.name + ".bak"))
            self.backed_up = True
        save_chart(self.chart, self.path)
        self.dirty = False
        self.message = f"saved {self.path.name}"

    # -- playback ---------------------------------------------------------------------
    def now(self):
        if self.paused_at is not None:
            return self.paused_at
        return self.engine.song_time()

    def play(self, at):
        at = min(max(0.0, at), self.length - 0.5)
        self.paused_at = None
        self.engine.play_song(self.music, preroll_s=0.15, length_s=self.length, start_s=at)
        self.reschedule(at)

    def pause(self):
        if self.paused_at is None:
            self.paused_at = self.engine.song_time() or 0.0
            self.engine.stop_song()
        else:
            self.play(self.paused_at)

    def reschedule(self, t_from=None):
        if self.paused_at is not None or self.engine.song_start is None:
            return
        self.engine.cancel("preview")
        st = self.engine.song_time() if t_from is None else t_from
        sounds = {"beat": (self.sounds.click, config.PREVIEW_CLICK_VOLUME),
                  "accent": (self.sounds.click_accent, config.PREVIEW_ACCENT_VOLUME),
                  "tick": (self.sounds.tempo_tick, config.TEMPO_TICK_VOLUME)}
        for t, kind in click_events(self.chart, max(0.0, st) + 0.03, self.length):
            data, gain = sounds[kind]
            self.engine.schedule(data, t, gain, tag="preview")

    def seek_bars(self, n):
        ch = self.chart
        bar, _ = ch.bar_of(ch.time_beat(max(0.0, self.now() or 0.0)))
        self.play(ch.beat_time(ch.bar_beat(bar + n)) - 0.05)

    def jump_to_change(self):
        st = self.now() or 0.0
        nxt = [c for c in self.chart.tempo_changes() if c.blink_times()[0] > st + 1.0]
        if not nxt:
            nxt = self.chart.tempo_changes()
        if not nxt:
            self.message = "this chart has no sudden tempo change"
            return
        c = nxt[0]
        self.play(self.chart.beat_time(c.beat - 12))


def cmd_preview(args):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        load_chart(args.chart)
    except ChartError as e:
        sys.exit(f"{Path(args.chart).name}: {e}")
    import pygame
    from audio_engine import load_audio, make_engine
    from rhythm_render import draw_tempo_led
    from sounds import SoundBank

    pygame.display.init()
    pygame.font.init()
    engine = make_engine()
    print(f"[audio] {engine.name}: {engine.status}")
    sounds = SoundBank(engine.sr)
    chart = load_chart(args.chart)
    pv = Preview(args.chart, engine, sounds, load_audio(chart.song_path, engine.sr))
    screen = pygame.display.set_mode((960, 520), pygame.RESIZABLE)
    pygame.display.set_caption(f"Preview: {chart.title}")
    font = pygame.font.SysFont(config.FONT_BODY, 19)
    mid = pygame.font.SysFont(config.FONT_BODY, 24, bold=True)
    big = pygame.font.SysFont(config.FONT_TITLE, 46, bold=True)
    c = config.PLAIN_COLORS
    frame = pygame.time.Clock()
    pv.play(parse_time(args.at))
    quit_armed = False
    running = True
    try:
        while running:
            frame.tick(config.FPS)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((max(700, ev.w), max(420, ev.h)), pygame.RESIZABLE)
                elif ev.type == pygame.KEYDOWN:
                    shift = ev.mod & pygame.KMOD_SHIFT
                    step = 1 if shift else config.PREVIEW_STEP_MS
                    k = ev.key
                    if k != pygame.K_ESCAPE:
                        quit_armed = False
                    if k == pygame.K_ESCAPE:
                        if pv.dirty and not quit_armed:
                            pv.message = "unsaved changes: S to save, Esc again to quit without saving"
                            quit_armed = True
                        else:
                            running = False
                    elif k in (pygame.K_LEFT, pygame.K_RIGHT):
                        pv.edit(offset_ms=step if k == pygame.K_RIGHT else -step)
                    elif k == pygame.K_b:
                        pv.edit(offset_ms=500 * pv.chart.tempo.segments[0].spb)
                    elif k in (pygame.K_UP, pygame.K_DOWN):
                        pv.edit(downbeat=1 if k == pygame.K_UP else -1)
                    elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        st = pv.now() or 0.0
                        i = pv.chart.tempo.segment_index_at(pv.chart.time_beat(st))
                        if i == 0:
                            pv.message = "the first segment has no shift -- use Left/Right (offset_ms)"
                        else:
                            pv.edit(shift=(i, step if k == pygame.K_RIGHTBRACKET else -step))
                    elif k == pygame.K_SPACE:
                        pv.pause()
                    elif k in (pygame.K_PAGEUP, pygame.K_COMMA):
                        pv.seek_bars(-config.PREVIEW_SEEK_BARS)
                    elif k in (pygame.K_PAGEDOWN, pygame.K_PERIOD):
                        pv.seek_bars(config.PREVIEW_SEEK_BARS)
                    elif k == pygame.K_HOME:
                        pv.play(0.0)
                    elif k == pygame.K_j:
                        pv.jump_to_change()
                    elif k == pygame.K_s:
                        pv.save()
            st = pv.now()
            if st is not None and st > pv.length + 0.5 and pv.paused_at is None:
                pv.paused_at = pv.length
                engine.stop_song()
            draw_preview(screen, pv, st, font, mid, big, c, draw_tempo_led)
            pygame.display.flip()
    finally:
        engine.close()
        pygame.quit()
    if pv.dirty:
        print("quit without saving")


def draw_preview(screen, pv, st, font, mid, big, c, draw_led):
    import pygame
    w, h = screen.get_size()
    screen.fill(c["wall_top"])
    ch = pv.chart
    st = st if st is not None else 0.0
    beat = ch.time_beat(st)
    bar, in_bar = ch.bar_of(beat)
    seg_i = ch.tempo.segment_index_at(beat)
    seg = ch.tempo.segments[seg_i]

    def text(s, f, col, pos, anchor="topleft"):
        surf = f.render(s, True, col)
        screen.blit(surf, surf.get_rect(**{anchor: pos}))

    title = f"{ch.title}" + (f"  —  {ch.author}" if ch.author else "")
    text(title, mid, c["text"], (24, 18))
    state = "paused" if pv.paused_at is not None else "playing"
    text(f"{int(st // 60)}:{st % 60:05.2f} / {int(pv.length // 60)}:{pv.length % 60:04.1f}   ({state})",
         font, c["dim"], (24, 54))
    text(f"Bar {bar}  ·  beat {int(in_bar)}", big, c["text"], (24, 84))
    # four beat lights, beat 1 brighter
    phase = in_bar - int(in_bar)
    for k in range(4):
        x, y = 34 + k * 46, 160
        on = int(in_bar) == k + 1
        col = c["perfect"] if k == 0 else c["accent"]
        lit = math.exp(-phase / 0.25) if on else 0.0
        pygame.draw.circle(screen, c["button"], (x, y), 15)
        if lit > 0.02:
            pygame.draw.circle(screen, tuple(int(a + (b - a) * lit) for a, b in zip(c["button"], col)), (x, y), 15)
    shift = f", shift {seg.shift_ms:+g} ms" if seg.shift_ms else ""
    hit_mark = ""
    p = ch.pattern_for_bar(bar)
    if ch.start_bar <= bar <= ch.end_bar:
        hit_mark = f"   your hits this bar: {pattern_text(p)}"
    lines = [
        (f"Tempo {seg.bpm:.2f} BPM  (segment {seg_i + 1} of {len(ch.tempo.segments)}, starts on beat {seg.beat}{shift})"
         + hit_mark, c["text"]),
        (f"offset_ms {ch.offset_ms:g}    downbeat {ch.downbeat} (bar 1 = beat {ch.downbeat})    "
         f"hits: bars {ch.start_bar}-{ch.end_bar}", c["text"]),
    ]
    changes = ch.tempo_changes()
    nxt = [x for x in changes if x.t > st]
    if nxt:
        x = nxt[0]
        lines.append((f"next tempo jump: {x.old_bpm:.0f} -> {x.new_bpm:.0f} BPM at bar {ch.bar_of(x.beat)[0]} "
                      f"({int(x.t // 60)}:{x.t % 60:04.1f}, in {x.t - st:.1f} s)", c["accent2"]))
    elif changes:
        lines.append(("no more tempo jumps", c["dim"]))
    else:
        lines.append(("no sudden tempo change in this chart", c["dim"]))
    y = 200
    for s, col in lines:
        text(s, font, col, (24, y))
        y += 28
    keys = ["Left/Right offset (Shift 1 ms)   B half a beat   Up/Down downbeat   [ ] shift this tempo segment",
            "Space pause   PgUp/PgDn 4 bars   J before next tempo jump   Home restart   S save   Esc quit"]
    y = h - 30 * len(keys) - 40
    for s in keys:
        text(s, font, c["dim"], (24, y))
        y += 28
    if pv.message:
        text(pv.message, font, c["warn"], (24, h - 36))
    if pv.dirty:
        text("unsaved", font, c["warn"], (w - 24, 18), "topright")
    draw_led(pygame, screen, (w - 90, 120), config.TEMPO_LED_RADIUS_PX * 1.4, led_state(changes, st), font,
             c["text"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="make a chart (tempo map, bars) from an audio file")
    d.add_argument("audio")
    d.add_argument("--bpm", type=float, default=None, help="force one fixed tempo (no tempo changes)")
    d.add_argument("--title", default=None)
    d.add_argument("--author", default=None)
    d.add_argument("--incompetech", action="store_true",
                   help="author 'Kevin MacLeod (Incompetech)' + the CC BY 4.0 attribution")
    d.add_argument("--force", action="store_true", help="overwrite an existing chart (keeps your edits)")
    d.add_argument("--to-wav", action="store_true",
                   help="decode an MP3 / OGG to songs/<Name_With_Underscores>.wav first and chart that")
    v = sub.add_parser("preview", help="play the song with clicks; nudge offset / downbeat and save")
    v.add_argument("chart")
    v.add_argument("--at", default=None, help="start position, e.g. 2:00 or 120")
    c = sub.add_parser("clicks", help="write the song with the preview clicks to previews/")
    c.add_argument("chart")
    args = p.parse_args(argv)
    {"detect": cmd_detect, "preview": cmd_preview, "clicks": cmd_clicks}[args.cmd](args)


if __name__ == "__main__":
    main()
