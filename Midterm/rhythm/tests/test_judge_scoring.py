"""Hit judgment (timing, and aim in aim mode), accuracy scoring, streaks, best results, ranks."""

import pytest

import config
from helpers import chart_with_cues
from judge import Judge, grade_for, paddle_near
from scheduler import build_schedule
from scoring import BestScores, ScoreKeeper, rank_for


def flights(beats=(4, 6, 8, 10.5, 12)):
    return build_schedule(chart_with_cues(beats, bpm=170.0))


def test_grade_windows():
    assert grade_for(0) == "Perfect"
    assert grade_for(-config.PERFECT_WINDOW_MS) == "Perfect"
    assert grade_for(config.PERFECT_WINDOW_MS + 1) == "Good"
    assert grade_for(-config.GOOD_WINDOW_MS) == "Good"
    assert grade_for(config.GOOD_WINDOW_MS + 1) is None


def test_swing_goes_to_the_nearest_open_cue_once():
    fl = flights()
    j = Judge(fl, miss_wait_s=0.2)
    t = fl[1].t_arrive
    a = j.swing(t + 0.03, t + 0.1, "imu")
    assert (a.index, a.grade) == (1, "Perfect") and a.error_ms == pytest.approx(30)
    b = j.swing(t + 0.05, t + 0.1, "imu")              # same cue again: stray
    assert b is None and j.strays == 1
    c = j.swing(fl[2].t_arrive - 0.07, t, "imu")
    assert (c.index, c.grade) == (2, "Good") and c.error_ms == pytest.approx(-70)


def test_aim_check_makes_an_out_of_reach_swing_a_miss():
    fl = flights()
    j = Judge(fl, miss_wait_s=0.2)
    r = j.swing(fl[0].t_arrive, fl[0].t_arrive, "imu", aim_ok=lambda f: False)
    assert r.grade == "Miss" and r.reason == "missed the ball" and not r.hit and r.error_ms == pytest.approx(0)
    r = j.swing(fl[1].t_arrive, fl[1].t_arrive, "imu", aim_ok=lambda f: True)
    assert r.grade == "Perfect"


def test_contact_distance_is_what_you_see(monkeypatch):
    from judge import contact_distance
    drawn = config.PADDLE_BLADE_W_M * config.PADDLE_DRAW_SCALE / 2 + config.BALL_RADIUS_M * config.BALL_VISUAL_SCALE
    assert contact_distance() == pytest.approx(drawn) and contact_distance() < 0.15
    monkeypatch.setattr(config, "AIM_CONTACT_M", 0.1)
    assert contact_distance() == 0.1


def test_paddle_near_uses_the_tolerance_and_the_lag_window():
    xs = {0.90: 0.0, 1.00: 0.5}                                # paddle at 0 until t=1.0, then 0.5
    at = lambda t: xs[0.90] if t < 1.0 else xs[1.00]           # noqa: E731
    assert paddle_near(at, 0.15, 0.95, tolerance_m=0.2, lookback_ms=0, lookahead_ms=0)
    assert not paddle_near(at, 0.25, 0.95, tolerance_m=0.2, lookback_ms=0, lookahead_ms=0)
    assert paddle_near(at, 0.5, 0.95, tolerance_m=0.2, lookback_ms=0, lookahead_ms=60)   # arrives 50 ms later
    assert not paddle_near(lambda t: None, 0.0, 1.0)                                      # no paddle known


def test_unanswered_cue_is_a_miss_only_after_the_late_event_deadline():
    fl = flights()
    j = Judge(fl, miss_wait_s=0.25)
    deadline = fl[0].t_arrive + config.GOOD_WINDOW_MS / 1000 + 0.25
    assert j.update(deadline - 0.01) == []
    # an IMU event can still arrive late and count
    late = j.swing(fl[0].t_arrive + 0.08, deadline - 0.005, "imu")
    assert late.grade == "Good"
    out = j.update(fl[1].t_arrive + config.GOOD_WINDOW_MS / 1000 + 0.26)
    assert [m.index for m in out] == [1] and out[0].reason == "no swing"


def test_stray_penalty_option():
    fl = flights()
    j = Judge(fl, miss_wait_s=0.2, stray_penalty=True)
    r = j.swing(fl[0].t_arrive - 0.25, 0.0, "imu")
    assert r.grade == "Miss" and r.reason == "stray swing" and r.index == 0


def judgment(grade, err=0.0):
    from judge import Judgment
    return Judgment(0, grade, err if grade != "Miss" else None, "", 0.0)


def test_accuracy_streak_and_events():
    events = []
    sk = ScoreKeeper(song_id="s", total_cues=4, listeners=[events.append])
    assert sk.accuracy == 0.0 and sk.judged == 0
    for g in ("Perfect", "Good", "Perfect", "Miss"):
        sk.add(judgment(g))
    assert sk.accuracy == pytest.approx(100 * (1 + 0.5 + 1 + 0) / 4)      # 62.5
    assert sk.streak == 0 and sk.max_streak == 3
    assert [e.grade for e in events] == ["Perfect", "Good", "Perfect", "Miss"]
    assert events[1].accuracy == 75.0 and events[-1].kind == "cue"
    sk.finish()
    assert events[-1].kind == "song_end" and events[-1].counts == {"Perfect": 2, "Good": 1, "Miss": 1}


def test_grade_weights_come_from_config(monkeypatch):
    monkeypatch.setattr(config, "GRADE_POINTS", {"Perfect": 1.0, "Good": 0.8, "Miss": 0.0})
    sk = ScoreKeeper()
    sk.add(judgment("Good"))
    assert sk.accuracy == pytest.approx(80.0)


def test_best_scores_persist_per_song(tmp_path):
    path = tmp_path / "s.json"
    b = BestScores(path)
    assert b.record("song", 72.04, 10) == (True, True)
    assert b.record("song", 72.0, 9) == (False, False)          # same shown accuracy: not a new best
    assert b.record("song", 80.0, 3) == (True, False)
    again = BestScores(path).get("song")
    assert again == {"best_accuracy": 80.0, "max_streak": 10, "plays": 3}


def test_rank_thresholds():
    assert rank_for(97) == "S" and rank_for(95) == "S" and rank_for(90) == "A"
    assert rank_for(60) == "C" and rank_for(0) == "D"
