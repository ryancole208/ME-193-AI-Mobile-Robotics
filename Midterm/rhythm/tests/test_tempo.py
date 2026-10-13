"""Tempo map: beat <-> time through tempo segments, shifts, sudden-change detection, the LED."""

import pytest

import config
from tempo import TempoError, led_state, parse_tempo
from helpers import tempo


def test_constant_tempo_conversions():
    tm = tempo(120.0, offset_ms=50.0)
    assert tm.beat_time(0) == pytest.approx(0.05)
    assert tm.beat_time(4) == pytest.approx(2.05)
    assert tm.beat_time(-2) == pytest.approx(-0.95)            # before beat 0: same tempo
    assert tm.time_beat(2.05) == pytest.approx(4.0)
    assert tm.beat_time(2.5) == pytest.approx(1.3)             # fractions (upbeats) are exact


def test_beats_follow_a_tempo_change():
    tm = tempo(120.0, changes=[(8, 160.0)])
    assert tm.beat_time(8) == pytest.approx(4.0)               # 8 beats at 0.5 s
    assert tm.beat_time(12) == pytest.approx(4.0 + 4 * 0.375)  # then 0.375 s per beat
    assert tm.spb_at(7.9) == pytest.approx(0.5) and tm.spb_at(8) == pytest.approx(0.375)
    for b in (-3.0, 0.0, 5.5, 8.0, 9.25, 40.0):
        assert tm.time_beat(tm.beat_time(b)) == pytest.approx(b)
    assert tm.duration(6, 10) == pytest.approx(2 * 0.5 + 2 * 0.375)
    assert tm.bpm_range() == (120.0, 160.0)


def test_shift_moves_the_new_segment_and_stretches_one_beat():
    tm = tempo(120.0, changes=[(8, 160.0, 40.0)])
    assert tm.beat_time(7) == pytest.approx(3.5)               # unchanged up to the beat before
    assert tm.beat_time(8) == pytest.approx(4.04)              # the change is 40 ms later
    assert tm.beat_time(9) == pytest.approx(4.04 + 0.375)
    assert tm.beat_time(7.5) == pytest.approx(3.5 + 0.54 / 2)  # the stretched beat
    for b in (6.5, 7.25, 7.9, 8.0, 8.4):
        assert tm.time_beat(tm.beat_time(b)) == pytest.approx(b)


@pytest.mark.parametrize("raw, msg", [
    ([], "non-empty"),
    ([{"beat": 4, "bpm": 120}], "beat 0"),
    ([{"beat": 0, "bpm": 500}], "out of range"),
    ([{"beat": 0, "bpm": 120}, {"beat": 0, "bpm": 130}], "two tempo segments"),
    ([{"beat": 0, "bpm": 120}, {"beat": 8.5, "bpm": 130}], "whole beat"),
    ([{"beat": 0, "bpm": 120}, {"beat": 8, "bpm": 130, "shift_ms": 300}], "half a beat"),
    ([{"beat": 0}], "needs"),
])
def test_bad_tempo_maps_are_explained(raw, msg):
    with pytest.raises(TempoError, match=msg):
        parse_tempo(raw, 0.0)


def test_sudden_change_detection():
    # +40 BPM at once: a sudden speed-up
    ch = tempo(120.0, changes=[(64, 160.0)]).sudden_changes()
    assert [(c.beat, c.old_bpm, c.new_bpm, c.faster) for c in ch] == [(64, 120.0, 160.0, True)]
    # exactly +5 is not MORE than 5; +5.5 is
    assert tempo(120.0, changes=[(64, 125.0)]).sudden_changes() == []
    assert len(tempo(120.0, changes=[(64, 125.5)]).sudden_changes()) == 1
    # sudden slowdowns count too (same rule, the other way)
    slow = tempo(140.0, changes=[(64, 120.0)]).sudden_changes()
    assert [(c.old_bpm, c.new_bpm, c.faster) for c in slow] == [(140.0, 120.0, False)]
    assert tempo(140.0, changes=[(64, 135.0)]).sudden_changes() == []
    # gradual drift either way: +/-2 BPM every 8 beats adds up to 10, but never 5 within 2 beats
    for step in (2, -2):
        drift = tempo(120.0, changes=[(8 * k, 120.0 + step * k) for k in range(1, 6)])
        assert drift.sudden_changes() == []
    # two +3 steps one beat apart add up within the window: one change, at the step that
    # makes it more than 5, with the final tempo
    steps = tempo(120.0, changes=[(64, 123.0), (65, 126.0)]).sudden_changes()
    assert [(c.beat, c.old_bpm, c.new_bpm) for c in steps] == [(65, 120.0, 126.0)]
    down = tempo(126.0, changes=[(64, 123.0), (65, 120.0)]).sudden_changes()
    assert [(c.beat, c.old_bpm, c.new_bpm) for c in down] == [(65, 126.0, 120.0)]
    # a speed-up and a slowdown far apart: two changes
    both = tempo(120.0, changes=[(64, 160.0), (128, 120.0)]).sudden_changes()
    assert [c.faster for c in both] == [True, False]


def test_thresholds_come_from_config(monkeypatch):
    tm = tempo(120.0, changes=[(64, 128.0)])
    assert len(tm.sudden_changes()) == 1
    monkeypatch.setattr(config, "TEMPO_JUMP_MIN_BPM", 10.0)
    assert tm.sudden_changes() == []


def test_blinks_land_on_new_tempo_beats_before_the_change():
    tm = tempo(120.0, changes=[(64, 160.0)])
    c = tm.sudden_changes()[0]
    t = tm.beat_time(64)
    assert c.t == pytest.approx(t)
    assert c.blink_times() == pytest.approx([t - 3 * 0.375, t - 2 * 0.375, t - 0.375])


def test_led_blinks_three_times_then_goes_out():
    tm = tempo(120.0, changes=[(64, 160.0)])
    changes = tm.sudden_changes()
    blinks = changes[0].blink_times()
    assert led_state(changes, blinks[0] - config.TEMPO_LED_SHOW_S - 0.01) is None
    lit = [led_state(changes, b + 0.01) for b in blinks]
    assert [s.blink for s in lit] == [1, 2, 3] and all(s.lit > 0.9 for s in lit)
    between = led_state(changes, blinks[0] + 0.3)               # 0.3 s > 0.45 beat at 160 BPM
    assert between.lit == 0.0 and between.visible > 0
    assert led_state(changes, changes[0].t + 0.375) is None
    assert lit[0].new_bpm == 160.0 and lit[0].faster


def test_led_before_a_slowdown_blinks_at_the_slower_tempo():
    tm = tempo(160.0, changes=[(64, 120.0)])
    changes = tm.sudden_changes()
    t = tm.beat_time(64)
    assert changes[0].blink_times() == pytest.approx([t - 1.5, t - 1.0, t - 0.5])
    state = led_state(changes, t - 1.0 + 0.01)
    assert state.blink == 2 and not state.faster and state.new_bpm == 120.0
