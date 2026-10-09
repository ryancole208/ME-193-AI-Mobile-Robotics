"""Chart helper: tempo-map detection on the real songs, the detect command, preview editing,
the click track."""

import json
import shutil
import sys

import numpy as np
import pytest

import config
import make_chart
from audio_engine import NullEngine
from chart import load_chart
from helpers import write_chart
from sounds import SoundBank

SONGS = config.SONGS_DIR


@pytest.fixture(scope="module")
def electrodoodle():
    return make_chart.detect(SONGS / "Electrodoodle.wav")


def test_detects_electrodoodle_speeding_up(electrodoodle):
    tm = electrodoodle["tempo"]
    assert [round(s.bpm) for s in tm.segments] == [120, 160]
    assert tm.segments[0].bpm == pytest.approx(120.0, abs=0.05)
    assert tm.segments[1].bpm == pytest.approx(160.0, abs=0.05)
    assert 129.9 < tm.beat_time(tm.segments[1].beat) < 130.2          # after the fill (ticks ~2:09)
    assert (tm.segments[1].beat - electrodoodle["downbeat"]) % 4 == 0    # on a bar line
    assert abs(tm.segments[1].shift_ms) < 30                          # the new grid starts on a beat
    change = tm.sudden_changes()
    assert len(change) == 1 and change[0].new_bpm == pytest.approx(160.0, abs=0.05)
    assert 1 <= electrodoodle["start_bar"] <= 6
    last_return = tm.beat_time(electrodoodle["downbeat"] + 4 * electrodoodle["end_bar"] + 1)
    assert last_return < electrodoodle["duration"]


def test_detected_beats_sit_on_the_music(electrodoodle):
    """Onsets are much stronger on the detected beats than between them, before AND after the change."""
    y, sr = make_chart.load_mono(SONGS / "Electrodoodle.wav")
    env, fps, t0, _ = make_chart.onset_envelope(y, sr)
    tm = electrodoodle["tempo"]

    def strength(beats):
        f = ((np.array([tm.beat_time(b) for b in beats]) - t0) * fps).astype(int)
        return env[f[(f > 0) & (f < len(env))]].mean()
    for a, b in ((16, 240), (262, 330)):
        assert strength(range(a, b)) > 3 * strength(np.arange(a, b) + 0.25)


@pytest.mark.parametrize("use_librosa", [True, False])
def test_constant_tempo_songs(use_librosa, monkeypatch):
    if not use_librosa:
        monkeypatch.setitem(sys.modules, "librosa", None)       # force the numpy detector
    d = make_chart.detect(SONGS / "Spazzmatica_Polka.wav")
    assert d["method"] == ("librosa" if use_librosa else "numpy")
    assert len(d["tempo"].segments) == 1 and d["tempo"].segments[0].bpm == pytest.approx(140.0, abs=0.2)
    assert d["tempo"].sudden_changes() == []


def test_bpm_override_gives_one_segment():
    d = make_chart.detect(SONGS / "Spazzmatica_Polka.wav", bpm=140.0)
    assert [s.bpm for s in d["tempo"].segments] == [140.0]


def test_detect_command_writes_a_credited_chart_and_keeps_edits(tmp_path):
    wav = tmp_path / "Spazzmatica_Polka.wav"
    shutil.copyfile(SONGS / "Spazzmatica_Polka.wav", wav)
    make_chart.main(["detect", str(wav), "--incompetech"])
    out = tmp_path / ("Spazzmatica_Polka" + config.CHART_SUFFIX)
    ch = load_chart(out)
    assert ch.title == "Spazzmatica Polka" and ch.author == "Kevin MacLeod (Incompetech)"
    assert ch.attribution["license_url"] == "http://creativecommons.org/licenses/by/4.0/"
    assert ch.pattern == (4,) and ch.sections == []
    with pytest.raises(SystemExit):                              # never overwrites without --force
        make_chart.main(["detect", str(wav)])
    d = json.loads(out.read_text(encoding="utf-8"))
    d["sections"] = [{"bars": [10, 12], "pattern": [2, 4]}]
    d["title"] = "Polka!"
    out.write_text(json.dumps(d), encoding="utf-8")
    make_chart.main(["detect", str(wav), "--force"])
    again = load_chart(out)
    assert again.title == "Polka!" and [(s.first_bar, s.last_bar) for s in again.sections] == [(10, 12)]
    assert (tmp_path / (out.name + ".bak")).exists()


def test_click_events_accent_bar_starts_and_include_tempo_ticks():
    ch = load_chart(SONGS / "Electrodoodle.chart.json")
    ev = make_chart.click_events(ch, 0.0, 200.0)
    accents = [t for t, k in ev if k == "accent"]
    beats = [t for t, k in ev if k in ("accent", "beat")]
    assert all(abs(ch.bar_of(ch.time_beat(t))[1] - 1) < 1e-6 for t in accents)
    assert len(accents) == pytest.approx(len(beats) / 4, abs=1)
    ticks = [t for t, k in ev if k == "tick"]
    assert ticks == pytest.approx(ch.tempo_changes()[0].blink_times())


def test_preview_edits_and_saves(tmp_path):
    path = write_chart(tmp_path, "Song", tempo=[{"beat": 0, "bpm": 120}, {"beat": 32, "bpm": 160}])
    eng = NullEngine(clock_fn=lambda: 100.0)
    pv = make_chart.Preview(path, eng, SoundBank(eng.sr), np.zeros((48000 * 30, 2), np.float32))
    pv.play(0.0)
    sched = [t for t, tag in eng.scheduled if tag == "preview"]
    assert sched and sched[0] >= 0
    pv.edit(offset_ms=5)
    pv.edit(downbeat=1)
    pv.edit(shift=(1, -3))
    assert pv.dirty and pv.chart.offset_ms == 5 and pv.chart.downbeat == 1
    assert pv.chart.tempo.segments[1].shift_ms == -3
    assert not pv.edit(shift=(1, -400))                          # refused: more than half a beat
    pv.save()
    saved = load_chart(path)
    assert saved.offset_ms == 5 and saved.downbeat == 1 and saved.tempo.segments[1].shift_ms == -3
    assert (tmp_path / "Song.chart.json.bak").exists() and not pv.dirty


def test_clicks_command_writes_a_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HERE", tmp_path)
    make_chart.main(["clicks", str(SONGS / "Spazzmatica_Polka.chart.json")])
    import soundfile as sf
    info = sf.info(str(tmp_path / "previews" / "Spazzmatica_Polka.clicks.wav"))
    assert info.duration == pytest.approx(sf.info(str(SONGS / "Spazzmatica_Polka.wav")).duration, abs=0.05)


def test_to_wav_decodes_an_mp3_and_charts_it(tmp_path):
    mp3 = tmp_path / "Kick Shock.mp3"
    shutil.copyfile(SONGS / "Kick Shock.mp3", mp3)
    make_chart.main(["detect", str(mp3), "--to-wav", "--incompetech"])
    ch = load_chart(tmp_path / ("Kick_Shock" + config.CHART_SUFFIX))
    assert ch.song == "Kick_Shock.wav" and ch.title == "Kick Shock" and ch.song_path.exists()
    assert ch.tempo.segments[0].bpm == pytest.approx(138.0, abs=0.2)


def test_runs_of_mixed_tempos_are_one_tempo_not_a_change():
    """A groove that fits two tempos (3:2) flips the window tempo back and forth: one tempo,
    the majority, with the other reported. A clean switch stays a change."""
    hop = config.CHART_WINDOW_HOP_S
    flip = [176, 176, 117.3, 117.3, 117.3, 176, 176, 117.3, 176, 176, 117.3, 176, 176, 176, 176, 176]
    runs = make_chart.tempo_runs([(i * hop, b) for i, b in enumerate(flip)], len(flip) * hop)
    assert len(runs) == 1 and runs[0][2] == pytest.approx(176)
    assert runs[0][3][0][0] == pytest.approx(117.3) and runs[0][3][0][1] == pytest.approx(5 / 16)
    clean = [120] * 30 + [160] * 15
    runs = make_chart.tempo_runs([(i * hop, b) for i, b in enumerate(clean)], len(clean) * hop)
    assert [round(r[2]) for r in runs] == [120, 160]


def test_mid_bar_changes_move_to_a_bar_line_where_the_grids_meet():
    from tempo import Segment, TempoMap
    # 120 -> 160 placed on beat 257 (bar 65, beat 2); the grids meet again 3 old beats later
    tm = TempoMap([Segment(0, 120.0), Segment(257, 160.0), Segment(300, 170.0)], 0.0)
    before = [tm.beat_time(b) for b in (262, 300, 310)]
    snapped = make_chart.snap_changes_to_bars(tm, downbeat=0)
    assert [s.beat for s in snapped.segments] == [0, 260, 299]         # later segments renumbered
    assert snapped.beat_time(260) == pytest.approx(tm.beat_time(261))   # same physical grid
    assert [snapped.beat_time(b - 1) for b in (262, 300, 310)] == pytest.approx(before)
    # already on a bar line, or no bar line where the grids meet: unchanged
    on = TempoMap([Segment(0, 120.0), Segment(256, 160.0)], 0.0)
    assert make_chart.snap_changes_to_bars(on, 0).segments == on.segments
    odd = TempoMap([Segment(0, 120.0), Segment(257, 133.0)], 0.0)
    assert make_chart.snap_changes_to_bars(odd, 0).segments == odd.segments
