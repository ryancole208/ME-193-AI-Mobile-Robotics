"""tools/suggest_sections.py: loud phrases of a synthetic song become [2,4] sections; the chart
copy differs from the original only in "sections"."""

import json

import numpy as np
import pytest
import soundfile as sf

from tools import suggest_sections as ss

SR = 22050
BPM = 120.0                          # 2 s per bar


def synth_song(path, n_bars, loud_bars, offset_s=0.0, seed=0):
    """Kick-like clicks on every beat over noise; loud bars get 10x the level and 8th-note clicks."""
    rng = np.random.default_rng(seed)
    spb = 60 / BPM
    y = np.zeros(int((offset_s + n_bars * 4 * spb + 1) * SR), np.float32)
    click = (np.exp(-np.arange(int(0.03 * SR)) / (0.005 * SR))
             * np.sin(2 * np.pi * 120 * np.arange(int(0.03 * SR)) / SR)).astype(np.float32)
    for bar in range(1, n_bars + 1):
        loud = bar in loud_bars
        amp, step = (0.8, 0.5) if loud else (0.08, 1.0)
        t0 = offset_s + (bar - 1) * 4 * spb
        i0, i1 = int(t0 * SR), int((t0 + 4 * spb) * SR)
        y[i0:i1] += amp * 0.2 * rng.standard_normal(i1 - i0).astype(np.float32)
        for k in np.arange(0, 4, step):
            j = int((t0 + k * spb) * SR)
            y[j:j + len(click)] += amp * click
    sf.write(str(path), np.clip(y, -1, 1), SR)


def write_chart(path, song, end_bar, sections_text='"sections": []'):
    # tabs and odd spacing on purpose: everything but "sections" must survive byte for byte
    path.write_text('{\n  "song": "%s",\n"tag_id": 3,\n  "title": "Synth",\n'
                    '  "attribution": {"credit": "a \\"sections\\" [x]"},\n'
                    '  "offset_ms": 0.0,\n  "tempo": [\n\t{"beat": 0, "bpm": %g}\n  ],\n'
                    '  "downbeat": 0,\n  "start_bar": 1,\n  "end_bar": %d,\n'
                    '  "pattern": [4],\n\t %s\n}\n' % (song, BPM, end_bar, sections_text),
                    encoding="utf-8", newline="")


def test_bar_times_follow_the_tempo_map():
    chart = {"offset_ms": 100, "downbeat": 2,
             "tempo": [{"beat": 2, "bpm": 120}, {"beat": 10, "bpm": 60, "shift_ms": 20}]}
    edges = ss.bar_edges(chart, 1, 3)               # bars 1..3 -> beats 2, 6, 10, 14
    assert edges == pytest.approx([0.1, 2.1, 4.1 + 0.02, 8.1 + 0.02])


def test_loud_phrases_are_selected(tmp_path):
    loud = set(range(17, 25)) | set(range(33, 41))
    write_chart(tmp_path / "Synth.chart.json", "synth.wav", 48)
    synth_song(tmp_path / "synth.wav", 48, loud)
    res = ss.analyse(tmp_path / "Synth.chart.json", tmp_path, target=0.33, phrase_bars=8)
    assert res["ranges"] == [(17, 24), (33, 40)]
    assert res["hard_fraction"] == pytest.approx(16 / 48)
    assert min(res["scores"][b] for b in loud) > max(v for b, v in res["scores"].items()
                                                     if b not in loud)


def test_loud_intro_stays_easy(tmp_path):
    write_chart(tmp_path / "Synth.chart.json", "synth.wav", 40)
    synth_song(tmp_path / "synth.wav", 40, set(range(1, 9)) | set(range(25, 33)))
    res = ss.analyse(tmp_path / "Synth.chart.json", tmp_path, target=0.2, phrase_bars=8)
    assert res["ranges"] == [(25, 32)]


def test_lopsided_phrase_gives_four_bars_and_short_tail_counts():
    scores = {b: 0.0 for b in range(1, 45)}
    for b in range(21, 25):
        scores[b] = 3.0                              # only the second half of phrase 17-24
    for b in range(41, 45):
        scores[b] = 2.0                              # 4-bar tail phrase 41-44
    assert ss.make_phrases(1, 44, 8, 4)[-1] == (41, 44)
    assert ss.make_phrases(1, 42, 8, 4)[-1] == (33, 42)    # a 2-bar leftover joins its neighbour
    assert ss.select_hard(scores, 1, 44, target=8 / 44, phrase_bars=8) == [(21, 24), (41, 44)]


def test_adjacent_phrases_merge_and_target_is_respected():
    scores = {b: (1.0 if 17 <= b <= 32 else 0.0) for b in range(1, 65)}
    assert ss.select_hard(scores, 1, 64, target=0.25, phrase_bars=8) == [(17, 32)]
    assert ss.select_hard(scores, 1, 64, target=0.0, phrase_bars=8) == []


def test_main_writes_copy_with_only_sections_changed(tmp_path):
    charts, out = tmp_path / "charts", tmp_path / "out"
    charts.mkdir()
    old_sections = '"sections": [\n\t{"bars": [9, 16], "pattern": [2, 4]}\n  ]'
    write_chart(charts / "Synth.chart.json", "synth.wav", 48, old_sections)
    synth_song(tmp_path / "synth.wav", 48, set(range(33, 41)))
    original = (charts / "Synth.chart.json").read_bytes()

    ss.main(["--charts-dir", str(charts), "--music-dir", str(tmp_path), "--out", str(out),
             "--target", "0.17", "--no-graph"])

    assert (charts / "Synth.chart.json").read_bytes() == original        # never touched
    new = (out / "Synth.chart.json").read_bytes().decode()
    head, tail = original.decode().split('"sections": ', 1)
    assert new.startswith(head + '"sections": ')
    assert new.endswith("\n}\n")
    new_data, old_data = json.loads(new), json.loads(original)
    assert new_data["sections"] == [{"bars": [33, 40], "pattern": [2, 4]}]
    del new_data["sections"], old_data["sections"]
    assert new_data == old_data
    summary = (out / "summary.txt").read_text(encoding="utf-8")
    assert "33-40" in summary and "8 hand-made hard bars, 0 of them" in summary


def test_splice_adds_missing_sections_key():
    text = '{\n  "song": "x.mp3",\n  "pattern": [4]\n}\n'
    out = ss.splice_sections(text, [{"bars": [9, 16], "pattern": [2, 4]}])
    assert json.loads(out)["sections"] == [{"bars": [9, 16], "pattern": [2, 4]}]
    assert out.startswith('{\n  "song": "x.mp3",\n  "pattern": [4]')
