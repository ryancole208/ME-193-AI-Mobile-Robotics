"""Chart format: tempo map, bars, patterns / sections, HIT_BEATS, saving, scanning the songs folder."""

import json

import pytest

import config
from chart import ChartError, check_hit_beats, load_chart, parse_chart, save_chart, scan_songs
from helpers import ELECTRODOODLE, bar_chart, chart_dict, write_chart


def test_bars_and_the_default_4_pattern():
    ch = bar_chart(start_bar=2, end_bar=4, downbeat=3)
    assert ch.bar_beat(1) == 3 and ch.bar_beat(2, 4) == 10
    assert ch.bar_of(10) == (2, 4) and ch.bar_of(3) == (1, 1) and ch.bar_of(2) == (0, 4)
    assert [c.beat for c in ch.cues] == [10, 14, 18]          # beat 4 of bars 2, 3, 4


def test_sections_switch_patterns_for_ranges_of_bars():
    ch = bar_chart(start_bar=1, end_bar=6, sections=[(3, 4, (2, 4))])
    assert [c.beat for c in ch.cues] == [3, 7, 9, 11, 13, 15, 19, 23]
    assert ch.pattern_for_bar(2) == (4,) and ch.pattern_for_bar(3) == (2, 4) and ch.pattern_for_bar(5) == (4,)


def test_hit_beats_override_and_validation(monkeypatch):
    ch = bar_chart(start_bar=1, end_bar=3, sections=[(2, 2, (2, 4))])
    assert [c.beat for c in ch.with_override((2, 4)).cues] == [1, 3, 5, 7, 9, 11]
    assert [c.beat for c in ch.with_override((4,)).cues] == [3, 7, 11]
    assert check_hit_beats([4]) == (4,) and check_hit_beats([2, 4]) == (2, 4) and check_hit_beats([4, 2]) == (2, 4)
    monkeypatch.setattr(config, "HIT_BEATS", None)
    assert check_hit_beats() is None
    for bad in ([3], [1, 2, 3, 4], [2, 4.5], [4, 4], [], "4", [True]):
        with pytest.raises(ChartError, match=r"must be \[4\] .* or \[2, 4\]"):
            check_hit_beats(bad)


def test_tempo_map_drives_beat_times():
    ch = parse_chart(chart_dict(offset_ms=20.0, tempo=[{"beat": 0, "bpm": 120}, {"beat": 32, "bpm": 160}]))
    assert ch.beat_time(32) == pytest.approx(0.02 + 16.0)
    assert ch.beat_time(36) == pytest.approx(0.02 + 16.0 + 1.5)
    assert ch.bpm_label() == "120–160"                        # hits in bars 2-9 span the change
    ch = parse_chart(chart_dict(tempo=[{"beat": 0, "bpm": 120}, {"beat": 64, "bpm": 160}]))
    assert ch.bpm_label() == "120"                            # the change comes after the last hit
    ch = parse_chart(chart_dict(tempo=[{"beat": 0, "bpm": 120}, {"beat": 16, "bpm": 140.4}]))
    assert ch.bpm_label() == "120–140"


@pytest.mark.parametrize("patch, msg", [
    ({"song": "x.flac"}, "unsupported audio"),
    ({"offset_ms": "soon"}, "must be a number"),
    ({"tempo": [{"beat": 3, "bpm": 120}]}, "beat 0"),
    ({"start_bar": 9, "end_bar": 3}, "before start_bar"),
    ({"start_bar": 1.5}, "whole number"),
    ({"pattern": [3]}, "must be"),
    ({"sections": [{"bars": [3, 4], "pattern": [1, 2, 3, 4]}]}, "section #1 pattern"),
    ({"sections": [{"bars": [3, 6], "pattern": [2, 4]}, {"bars": [5, 7], "pattern": [4]}]}, "overlap"),
    ({"sections": [{"bars": [6, 3], "pattern": [2, 4]}]}, "first, last"),
    ({"start_bar": 1, "offset_ms": -1500}, "before the start of the audio"),
])
def test_invalid_charts_have_readable_errors(patch, msg):
    with pytest.raises(ChartError, match=msg):
        parse_chart(chart_dict(**patch))


def test_missing_keys():
    for key in ("song", "tempo", "end_bar"):
        d = chart_dict()
        del d[key]
        with pytest.raises(ChartError):
            parse_chart(d)


def test_save_round_trips_and_keeps_unknown_keys(tmp_path):
    d = chart_dict(author="Someone", notes="keep me", attribution={"title": "X", "license": "CC BY 4.0"},
                   tempo=[{"beat": 0, "bpm": 120}, {"beat": 32, "bpm": 160, "shift_ms": -4.2}],
                   sections=[{"bars": [3, 4], "pattern": [2, 4], "warn": True}])
    (tmp_path / "x.wav").write_bytes(b"")
    ch = parse_chart(d, tmp_path / "x.chart.json")
    save_chart(ch)
    text = (tmp_path / "x.chart.json").read_text(encoding="utf-8")
    assert '{"beat": 32, "bpm": 160.0, "shift_ms": -4.2}' in text and '"pattern": [4]' in text
    again = load_chart(tmp_path / "x.chart.json")
    assert again.to_dict() == ch.to_dict() and again.extra["notes"] == "keep me"
    assert again.cues[1].warn and not again.cues[2].warn       # warning before the section's first hit


def test_scan_skips_missing_and_invalid_songs_without_crashing(tmp_path):
    write_chart(tmp_path, "Good_Song")
    write_chart(tmp_path, "No_Audio", audio=False)
    write_chart(tmp_path, "Bad_Pattern", pattern=[3])
    (tmp_path / "Broken.chart.json").write_text("{ not json", encoding="utf-8")
    logged = []
    entries, skipped = scan_songs(tmp_path, log=logged.append)
    assert [e.chart.title for e in entries] == ["Good Song"]
    assert sorted(n for n, _ in skipped) == ["Bad_Pattern.chart.json", "Broken.chart.json", "No_Audio.chart.json"]
    assert any("not found" in r for _, r in skipped) and any("not valid JSON" in r for _, r in skipped)
    assert len(logged) == 3 and all(line.startswith("[songs] skipping") for line in logged)


def test_scan_applies_hit_beats_without_changing_files(tmp_path, monkeypatch):
    p = write_chart(tmp_path, "Song", sections=[{"bars": [3, 4], "pattern": [2, 4]}])
    before = p.read_text(encoding="utf-8")
    monkeypatch.setattr(config, "HIT_BEATS", [2, 4])
    ch = scan_songs(tmp_path)[0][0].chart
    assert all(ch.bar_of(c.beat)[1] in (2, 4) for c in ch.cues) and len(ch.cues) == 2 * 8
    monkeypatch.setattr(config, "HIT_BEATS", [3])
    with pytest.raises(ChartError, match="HIT_BEATS"):
        scan_songs(tmp_path)
    assert p.read_text(encoding="utf-8") == before


def test_the_incompetech_charts_are_valid_and_credited():
    entries, skipped = scan_songs(config.SONGS_DIR)
    assert skipped == []
    by_id = {e.chart.song_id: e.chart for e in entries}
    assert set(by_id) == {"Electrodoodle", "Latin_Industries", "Spazzmatica_Polka", "Cut_and_Dry",
                          "Itty_Bitty_8_Bit", "Kick_Shock", "Theme_for_Harold_var_3", "Pinball_Spring_160"}
    for ch in by_id.values():
        assert ch.song_path.suffix == ".wav" and ch.song_path.exists()
        assert ch.author == "Kevin MacLeod (Incompetech)"
        a = ch.attribution
        assert a["artist"] == "Kevin MacLeod" and a["source"] == "incompetech.com"
        assert "Creative Commons" in a["license"] and a["title"] == ch.title
        assert ch.pattern == (4,)
    for song_id, ch in by_id.items():                                 # only Electrodoodle changes tempo
        assert len(ch.tempo_changes()) == (1 if song_id == ELECTRODOODLE else 0), song_id
    e = by_id[ELECTRODOODLE]
    assert e.tempo.bpm_range() == (pytest.approx(120, abs=0.1), pytest.approx(160, abs=0.1))
    assert e.bpm_label() == "120–160"
    change = e.tempo_changes()
    assert len(change) == 1 and 129.9 < change[0].t < 130.2          # the new section, after the fill
    assert e.bar_of(change[0].beat) == (66, 1)                         # on a bar line
    ticks = change[0].blink_times()                                    # ...so the ticks fall around 2:09
    assert 128.8 < ticks[0] < 129.0 and 129.5 < ticks[-1] < 129.8
    assert {(s.first_bar, s.last_bar, s.pattern) for s in e.sections} == {(66, 73, (2, 4))}
    assert 141.9 < e.beat_time(e.bar_beat(74)) < 142.2                # back to [4] at 2:22
    assert json.loads((config.SONGS_DIR / "Electrodoodle.chart.json").read_text(encoding="utf-8"))["song"] == \
        "Electrodoodle.wav"
