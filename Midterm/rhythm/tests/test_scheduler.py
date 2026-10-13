"""Ball scheduling: equal flight time each way (opponent halfway between your hits), positions
from song time only, flight times from the tempo map."""

import pytest

import config
from scheduler import Outcome, ball_at, build_schedule, contacts_from_cues, opponent_x
from helpers import bar_chart, chart_with_cues


def beats_in_bar(ch, beat):
    return ch.bar_of(beat)[1]


def check_equal_flights(ch, flights):
    for f in flights:
        assert f.flight_in_s == pytest.approx(f.t_arrive - f.t_serve)
        assert f.flight_out_s == pytest.approx(f.t_return - f.t_arrive)
    for a, b in zip(flights, flights[1:]):
        assert a.t_return == pytest.approx(b.t_serve)          # the opponent hits your return
        assert a.flight_out_s == pytest.approx(b.flight_in_s)  # equal time each way


@pytest.mark.parametrize("use_override", [False, True])
def test_pattern_4_opponent_on_2_you_on_4(use_override):
    ch = bar_chart(start_bar=2, end_bar=8, downbeat=1)
    if use_override:                                           # same result via HIT_BEATS = [4]
        ch = bar_chart(start_bar=2, end_bar=8, downbeat=1, sections=[(3, 5, (2, 4))]).with_override((4,))
    fl = build_schedule(ch)
    assert len(fl) == 7
    for f in fl:
        assert beats_in_bar(ch, f.beat) == pytest.approx(4)
        assert beats_in_bar(ch, ch.time_beat(f.t_serve)) == pytest.approx(2)
        assert f.flight_in_s == pytest.approx(2 * 0.5) and f.flight_out_s == pytest.approx(2 * 0.5)
    check_equal_flights(ch, fl)


@pytest.mark.parametrize("use_override", [False, True])
def test_pattern_2_4_opponent_on_1_and_3(use_override):
    ch = bar_chart(start_bar=2, end_bar=5, pattern=(2, 4)) if not use_override else \
        bar_chart(start_bar=2, end_bar=5).with_override((2, 4))
    fl = build_schedule(ch)
    assert [beats_in_bar(ch, f.beat) for f in fl] == [2, 4] * 4
    assert {round(beats_in_bar(ch, ch.time_beat(f.t_serve)), 6) for f in fl} == {1, 3}
    assert all(f.flight_in_s == pytest.approx(0.5) for f in fl)
    check_equal_flights(ch, fl)
    assert all(beats_in_bar(ch, f.beat) != 3 for f in fl)      # you never hit on beat 3


def test_switching_patterns_keeps_the_timing_rule():
    ch = bar_chart(start_bar=1, end_bar=6, sections=[(3, 4, (2, 4))])
    fl = build_schedule(ch)
    serves = [round(beats_in_bar(ch, ch.time_beat(f.t_serve)), 6) for f in fl]
    #          bar1 bar2 | bar3   | bar4   | bar5 bar6
    assert serves == [2, 2, 1, 3, 1, 3, 2, 2]
    # [4] -> [2, 4]: you hit 4, the opponent hits the next bar's 1 (halfway to your 2)
    assert fl[1].flight_out_s == pytest.approx(0.5) and fl[2].flight_in_s == pytest.approx(0.5)
    # [2, 4] -> [4]: you hit 4, the opponent hits the next bar's 2 (halfway to your next 4)
    assert fl[5].flight_out_s == pytest.approx(1.0) and fl[6].flight_in_s == pytest.approx(1.0)
    check_equal_flights(ch, fl)


def test_flights_shrink_when_the_tempo_rises():
    ch = bar_chart(start_bar=1, end_bar=12, changes=[(24, 160.0)])          # 120 -> 160 at bar 7
    fl = build_schedule(ch)
    before = [f for f in fl if f.beat < 22]
    after = [f for f in fl if f.beat > 27]
    assert all(f.flight_in_s == pytest.approx(1.0) for f in before)        # 2 beats at 120
    assert all(f.flight_in_s == pytest.approx(0.75) for f in after)        # 2 beats at 160
    for f in fl:
        assert f.t_arrive == pytest.approx(ch.beat_time(f.beat))           # always on the beat


def test_contacts_store_explicit_flights_for_any_gap():
    """Future charts: any gap between hits (0.5 to 8 beats) gives equal halves."""
    ch = chart_with_cues([4, 4.5, 6, 14, 15], bpm=120.0)
    cs = contacts_from_cues(ch.cues)
    assert [(c.serve_beat, c.beat, c.return_beat) for c in cs] == [
        (3.75, 4, 4.25), (4.25, 4.5, 5.25), (5.25, 6, 10), (10, 14, 14.5), (14.5, 15, 15.5)]
    fl = build_schedule(ch)
    check_equal_flights(ch, fl)
    assert fl[3].flight_in_s == pytest.approx(4 * 0.5)                     # an 8-beat gap: 4 each way


def test_ball_arrives_exactly_on_each_cue_and_bounces_before():
    ch = bar_chart(start_bar=2, end_bar=5, sections=[(4, 4, (2, 4))])
    fl = build_schedule(ch)
    for f in fl:
        x, y, z = f.seg_in[1].pos(f.t_arrive)
        assert (x, y, z) == pytest.approx((f.x_arrive, config.PLAYER_HIT_Y_M, config.PADDLE_HEIGHT_M))
        assert f.seg_in[1].pos(f.t_bounce_in)[2] == pytest.approx(0.0)      # on the table
        f_in = ch.time_beat(f.t_arrive) - ch.time_beat(f.t_serve)
        lead = min(config.BOUNCE_LEAD_BEATS, f_in / 2)
        assert ch.time_beat(f.t_arrive) - ch.time_beat(f.t_bounce_in) == pytest.approx(lead)
    assert len({round(f.x_arrive, 3) for f in fl}) > 3                     # varied arrival spots
    assert all(abs(f.x_arrive) <= config.ARRIVAL_X_SPREAD_M for f in fl)


def test_arcs_clear_the_net_and_short_flights_are_flatter():
    ch = bar_chart(start_bar=2, end_bar=5, sections=[(4, 5, (2, 4))])
    fl = build_schedule(ch)
    net_y = config.TABLE_LENGTH_M / 2
    for f in fl:
        for seg in (f.seg_in[0], f.seg_out[0]):
            fn = (net_y - seg.p0[1]) / (seg.p1[1] - seg.p0[1])
            t = seg.t0 + fn * (seg.t1 - seg.t0)
            assert seg.pos(t)[2] > config.NET_HEIGHT_M
    assert fl[-1].seg_in[0].h < fl[0].seg_in[0].h                          # 1-beat flight = flatter


def test_ball_positions_depend_only_on_song_time():
    ch = bar_chart()
    fl = build_schedule(ch)
    t = fl[1].t_serve + 0.1
    a = ball_at(fl, t, {0: Outcome(True, fl[0].t_arrive)})
    b = ball_at(fl, t, {0: Outcome(True, fl[0].t_arrive)})
    assert a == b and a.phase == "in" and a.index == 1


def test_pending_miss_and_late_hit_paths():
    ch = bar_chart()
    fl = build_schedule(ch)
    f0 = fl[0]
    pend = ball_at(fl, f0.t_arrive + 0.05, {})
    assert pend.phase == "pending"
    assert config.PLAYER_HIT_Y_M - 0.6 < pend.pos[1] < config.PLAYER_HIT_Y_M     # eased stop
    t_known = f0.t_arrive + 0.04                     # a late hit blends onto the return (no jump)
    late = {0: Outcome(True, t_known)}
    before = ball_at(fl, t_known - 1e-4, late).pos
    after = ball_at(fl, t_known + 1e-4, late).pos
    assert after == pytest.approx(before, abs=0.02)
    assert ball_at(fl, t_known + config.LATE_HIT_BLEND_S + 0.01, late).phase == "out"
    missed = {0: Outcome(False, f0.t_arrive + 0.2)}
    assert ball_at(fl, f0.t_arrive + 0.3, missed).phase == "fall"
    assert ball_at(fl, f0.t_arrive + 0.2 + config.MISS_FALL_S + 0.01, missed) is None
    serve = ball_at(fl, fl[1].t_serve + 0.01, missed)
    assert serve.phase == "in" and serve.alpha < 1.0                       # next ball fades in


def test_opponent_glides_to_the_return_target():
    fl = build_schedule(bar_chart())
    assert opponent_x(fl, fl[0].t_serve) == pytest.approx(fl[0].x_from)
    assert opponent_x(fl, fl[0].t_return) == pytest.approx(fl[0].x_target, abs=1e-6)
