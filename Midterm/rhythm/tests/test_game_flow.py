"""Whole-game flow in keyboard fallback mode with a fake clock (no audio device, no hardware),
played on the real Electrodoodle chart (120 -> 160 BPM, a [2, 4] section), in all three modes."""

import subprocess
import sys

import pytest

import config
from audio_engine import NullEngine
from helpers import ELECTRODOODLE, write_chart
from player_input import KeyboardPaddle, PlayerInput
from rhythm_game import RhythmGame
from scheduler import ball_at
from judge import contact_distance
from scoring import BestScores
from tempo import led_state


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def new_game(tmp_path, songs_dir=None, mode="aim"):
    clk = Clock()
    eng = NullEngine(clock_fn=clk)
    kb = KeyboardPaddle()
    game = RhythmGame(eng, PlayerInput(keyboard=kb), best=BestScores(tmp_path / "s.json"),
                      songs_dir=songs_dir, clock_fn=clk)
    game.mode = mode
    return game, eng, kb, clk


def place(kb, x, t):
    """Put the arrow-key paddle at x at wall time t (as if you had glided it there)."""
    kb.x = x
    kb._history.append((t, x))


def start_song(game, song_id=ELECTRODOODLE):
    game.on_enter()                    # title: Play
    assert game.state == "select"
    game.sel_index = [i for i, e in enumerate(game.songs) if e.chart.song_id == song_id][0]
    game.on_enter()
    assert game.state == "play"


def goto(game, eng, clk, song_t, step=1 / 60):
    """Advance in frame steps to song time song_t (like the real loop)."""
    target = eng.song_start / eng.sr + song_t
    while clk.t < target:
        clk.t = min(target, clk.t + step)
        game.update(clk.t)


@pytest.mark.parametrize("mode", ["aim", "timing", "follow"])
def test_perfect_keyboard_run_scores_100_and_saves_best(tmp_path, mode):
    game, eng, kb, clk = new_game(tmp_path, mode=mode)
    start_song(game)
    accs = []
    for fl in game.flights:
        goto(game, eng, clk, fl.t_arrive - 0.2, step=1 / 30)
        if mode == "aim":
            place(kb, fl.x_arrive + 0.1, clk.t)                 # the blade overlaps the ball
        goto(game, eng, clk, fl.t_arrive + 0.01, step=1 / 30)
        j = game.swing(clk.t, "keyboard")
        assert j is not None and j.grade == "Perfect", (fl.index, j)
        accs.append(game.score.accuracy)
    assert all(a == 100.0 for a in accs)
    goto(game, eng, clk, game.end_song_t + 0.1, step=0.05)
    assert game.state == "results"
    assert game.results["accuracy"] == 100.0 and game.results["rank"] == "S"
    assert game.results["max_streak"] == len(game.flights)
    saved = BestScores(tmp_path / "s.json")
    assert saved.get(f"{game.chart.song_id}@{mode}")["best_accuracy"] == 100.0       # per mode
    assert saved.get(game.chart.song_id) is None


def test_electrodoodle_hits_follow_its_bars_and_tempo(tmp_path):
    game, eng, kb, clk = new_game(tmp_path)
    start_song(game)
    ch = game.chart
    for fl in game.flights:
        bar, beat = ch.bar_of(fl.beat)
        serve_beat = ch.bar_of(ch.time_beat(fl.t_serve))[1]
        if 66 <= bar <= 73:
            assert beat in (2, 4) and round(serve_beat, 6) in (1, 3)       # [2, 4] section
        else:
            assert beat == 4 and round(serve_beat, 6) == 2                 # [4], also right after [2, 4]
        assert fl.t_arrive == pytest.approx(ch.beat_time(fl.beat))
    fast = [f for f in game.flights if ch.bar_of(f.beat)[0] >= 75]
    assert fast and all(f.flight_in_s == pytest.approx(2 * 60 / 160, abs=1e-3) for f in fast)


def test_aim_mode_needs_the_paddle_at_the_ball(tmp_path):
    game, eng, kb, clk = new_game(tmp_path, mode="aim")
    start_song(game)
    for fl in game.flights[:3]:
        goto(game, eng, clk, fl.t_arrive - 0.3)
        place(kb, fl.x_arrive, clk.t)
        goto(game, eng, clk, fl.t_arrive)
        assert game.swing(clk.t, "keyboard").grade == "Perfect"
    assert game.score.streak == 3
    fl = game.flights[3]
    goto(game, eng, clk, fl.t_arrive - 0.3)
    place(kb, fl.x_arrive + contact_distance() + 0.02, clk.t)             # ball passes beside the blade
    goto(game, eng, clk, fl.t_arrive)
    j = game.swing(clk.t, "keyboard")
    assert j.grade == "Miss" and j.reason == "missed the ball" and game.score.streak == 0
    assert game.score.accuracy == pytest.approx(75.0)
    # contact is checked when the ball ARRIVES, not when you swing: an early swing with the paddle
    # still on its way counts if the paddle is there when the ball is
    fl = game.flights[4]
    goto(game, eng, clk, fl.t_arrive - 0.3)
    place(kb, fl.x_arrive + 0.6, clk.t)
    goto(game, eng, clk, fl.t_arrive - 0.06)
    t_swing = clk.t                                                       # 60 ms early
    goto(game, eng, clk, fl.t_arrive - 0.01)
    place(kb, fl.x_arrive, clk.t)
    goto(game, eng, clk, fl.t_arrive + 0.02)
    assert game.swing(t_swing, "keyboard").hit
    # ...and the paddle sweeping past the ball well before it arrives does NOT count
    fl = game.flights[5]
    goto(game, eng, clk, fl.t_arrive - 0.3)
    place(kb, fl.x_arrive, clk.t)
    goto(game, eng, clk, fl.t_arrive - 0.15)
    place(kb, fl.x_arrive - 0.6, clk.t)
    goto(game, eng, clk, fl.t_arrive)
    assert game.swing(clk.t, "keyboard").reason == "missed the ball"


def test_aim_mode_unknown_webcam_position_is_no_contact(tmp_path):
    """While the webcam is tracking you, a gap in its samples must not fall back to the
    arrow-key paddle (which sits wherever you last left it)."""
    from types import SimpleNamespace
    game, eng, kb, clk = new_game(tmp_path, mode="aim")
    pose = SimpleNamespace(tracking=True, paddle_x_at=lambda t: None,
                           snapshot=lambda: SimpleNamespace(paddle_x=None))
    game.player.pose = pose
    start_song(game)
    fl = game.flights[0]
    place(kb, fl.x_arrive, clk.t)                                          # the keyboard paddle IS at the ball
    goto(game, eng, clk, fl.t_arrive)
    assert game.swing(clk.t, "keyboard").reason == "missed the ball"


@pytest.mark.parametrize("mode", ["timing", "follow"])
def test_paddle_position_does_not_affect_scoring(tmp_path, mode):
    game, eng, kb, clk = new_game(tmp_path, mode=mode)
    start_song(game)
    for fl in game.flights[:4]:
        goto(game, eng, clk, fl.t_arrive - 0.1)
        place(kb, -0.7 if fl.x_arrive > 0 else 0.7, clk.t)                 # far from the planned spot
        goto(game, eng, clk, fl.t_arrive)
        assert game.swing(clk.t, "keyboard").grade == "Perfect"


def test_follow_mode_ball_goes_to_the_paddle(tmp_path):
    game, eng, kb, clk = new_game(tmp_path, mode="follow")
    start_song(game)
    fl = game.flights[2]
    goto(game, eng, clk, fl.t_serve + 0.05)
    place(kb, -0.55, clk.t)
    start = ball_at(game.flights, fl.t_serve + 0.05, game.outcomes, game.ball_x_at())
    plain = ball_at(game.flights, fl.t_serve + 0.05, game.outcomes)
    assert start.pos[0] == pytest.approx(plain.pos[0], abs=0.05)          # leaves the opponent as planned
    goto(game, eng, clk, fl.t_arrive)
    at = ball_at(game.flights, fl.t_arrive, game.outcomes, game.ball_x_at())
    assert at.pos[0] == pytest.approx(-0.55)                              # ...and reaches your paddle
    game.swing(clk.t, "keyboard")
    place(kb, 0.6, clk.t + 0.01)                                          # moving away after the hit
    goto(game, eng, clk, fl.t_arrive + 0.05)
    out = ball_at(game.flights, fl.t_arrive + 0.05, game.outcomes, game.ball_x_at())
    assert out.phase == "out" and out.pos[0] < 0                          # the return starts where you hit
    assert game.ball_x_at() is not None
    game.mode = "aim"
    assert game.ball_x_at() is None


def test_m_cycles_the_mode_on_song_select_and_bests_are_per_mode(tmp_path):
    game, eng, kb, clk = new_game(tmp_path)
    game.cycle_mode()
    assert game.mode == "aim"                                             # only on song select
    game.on_enter()
    seen = [game.mode]
    for _ in range(3):
        game.cycle_mode()
        seen.append(game.mode)
    assert seen == ["aim", "timing", "follow", "aim"] and "Mode: Aim" in game.message
    ch = game.songs[0].chart
    game.best.record(f"{ch.song_id}@timing", 90.0, 5)
    assert game.best_for(ch) is None
    game.mode = "timing"
    assert game.best_for(ch)["best_accuracy"] == 90.0


def test_no_input_misses_everything_but_the_song_keeps_going(tmp_path):
    game, eng, kb, clk = new_game(tmp_path)
    start_song(game)
    goto(game, eng, clk, game.flights[10].t_arrive + 1.0, step=0.05)
    assert game.state == "play" and game.score.counts["Miss"] >= 10 and game.score.streak == 0
    assert game.score.accuracy == 0.0


def test_input_offset_is_subtracted(tmp_path, monkeypatch):
    monkeypatch.setitem(config.INPUT_OFFSET_MS, "keyboard", 80.0)
    game, eng, kb, clk = new_game(tmp_path, mode="timing")
    start_song(game)
    fl = game.flights[0]
    goto(game, eng, clk, fl.t_arrive + 0.080)               # 80 ms "late" = on time after calibration
    j = game.swing(clk.t, "keyboard")
    assert j.grade == "Perfect" and abs(j.error_ms) < 1


def test_judgment_does_not_depend_on_frame_rate(tmp_path):
    results = []
    for step in (1 / 60, 1 / 9):                             # 60 FPS vs badly dropped frames
        game, eng, kb, clk = new_game(tmp_path, mode="timing")
        start_song(game)
        for fl in game.flights[:8]:
            goto(game, eng, clk, fl.t_arrive - 0.25, step)
            t_swing = eng.song_start / eng.sr + fl.t_arrive + 0.03
            goto(game, eng, clk, fl.t_arrive + 0.2, step)    # the game notices the swing late
            game.swing(t_swing, "keyboard")                  # ...but judges its own timestamp
        results.append([(j.index, j.grade, round(j.error_ms, 3)) for j in game.judge.results.values()])
    assert results[0] == results[1]


def test_tempo_ticks_and_led_before_the_speed_up(tmp_path):
    game, eng, kb, clk = new_game(tmp_path)
    start_song(game)
    assert len(game.tempo_changes) == 1
    change = game.tempo_changes[0]
    ticks = sorted(t for t, tag in eng.scheduled if tag == "tempo")
    spb = 60 / change.new_bpm
    assert ticks == pytest.approx([change.t - 3 * spb, change.t - 2 * spb, change.t - spb])
    blinks = [led_state(game.tempo_changes, t + 0.01) for t in ticks]
    assert [b.blink for b in blinks] == [1, 2, 3]
    assert led_state(game.tempo_changes, change.t - 10) is None


def test_tempo_ticks_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TEMPO_TICK_ENABLED", False)
    game, eng, kb, clk = new_game(tmp_path)
    start_song(game)
    assert not [t for t, tag in eng.scheduled if tag == "tempo"]


def test_countdown_labels_follow_the_beat(tmp_path):
    game, eng, kb, clk = new_game(tmp_path)
    start_song(game)
    ch = game.chart
    c = game.first_cue_beat
    labels = [game.countdown_label(ch.beat_time(c - 4 + k + 0.1)) for k in range(5)]
    assert labels == ["3", "2", "1", "GO!", None]


def test_escape_leaves_the_song_and_menus_navigate(tmp_path):
    game, eng, kb, clk = new_game(tmp_path)
    game.on_down()
    assert game.menu_index == 1
    game.on_up()
    start_song(game)
    game.on_escape()
    assert game.state == "select" and eng.song_start is None
    game.on_escape()
    assert game.state == "title"


def test_missing_or_broken_songs_are_skipped(tmp_path):
    folder = tmp_path / "songs"
    folder.mkdir()
    write_chart(folder, "Playable")
    write_chart(folder, "Missing_Audio", audio=False)
    (folder / "Broken.chart.json").write_text("{", encoding="utf-8")
    game, eng, kb, clk = new_game(tmp_path, songs_dir=folder)
    assert [e.chart.title for e in game.songs] == ["Playable"]
    assert len(game.skipped) == 2
    game.on_enter()
    game.on_retry()                                          # R rescans: still fine
    assert "skipped 2" in game.message


def test_input_calibration_flow_saves_offset(tmp_path):
    game, eng, kb, clk = new_game(tmp_path)
    game.on_calibrate()
    assert game.state == "calib_input"
    cal = game.calib
    for t in cal.click_times:
        goto(game, eng, clk, t + 0.07)
        game.swing(clk.t, "keyboard")
    goto(game, eng, clk, cal.length_s + 0.05, step=0.05)
    assert cal.result is not None and cal.result.offset_ms == pytest.approx(70, abs=17)
    game.on_enter()
    assert game.state == "title" and config.INPUT_OFFSET_MS["keyboard"] == pytest.approx(cal.result.offset_ms)


def test_visual_calibration_adjusts_only_drawing(tmp_path):
    game, eng, kb, clk = new_game(tmp_path)
    game.on_calibrate(visual=True)
    goto(game, eng, clk, 1.0)
    t0 = game.draw_time()
    game.on_adjust(+20)
    assert game.draw_time() == pytest.approx(t0 - 0.020)
    game.on_enter()
    assert config.VISUAL_OFFSET_MS == 20 and game.state == "title"


def test_play_by_name_matches_title_or_file(tmp_path):
    for stem in ("Song_One", "Polka_Two", "polka_three"):
        write_chart(tmp_path, stem)
    game, eng, kb, clk = new_game(tmp_path, songs_dir=tmp_path)
    assert game.find_song("song one")[0].chart.title == "Song One"
    assert game.find_song("POLKA_TWO")[0].chart.title == "Polka Two"     # file name, any case
    assert game.find_song("one")[0].chart.title == "Song One"             # unique partial match
    entry, err = game.find_song("polka")
    assert entry is None and "several" in err and "Polka Two" in err
    entry, err = game.find_song("waltz")
    assert entry is None and "no song matches" in err and "Song One" in err


def test_keyboard_paddle_glides_like_ping_pong():
    """Holding Right moves the paddle smoothly through every position (no lanes, no jumps),
    stopping wherever you let go -- the same KeyboardPaddle the ping-pong game uses."""
    kb = KeyboardPaddle()
    t, dt, xs = 0.0, 1 / 60, []
    for _ in range(90):                                  # hold Right for 1.5 s
        t += dt
        kb.update(t, dt, False, True)
        xs.append(kb.x)
    steps = [b - a for a, b in zip(xs, xs[1:])]
    assert max(steps) <= config.KEYBOARD_PADDLE_SPEED_M_S * dt + 1e-9      # small, even steps
    assert xs[-1] == pytest.approx(config.TABLE_WIDTH_M / 2)                # stops at the table edge
    for _ in range(10):                                  # let go: it stays put
        t += dt
        kb.update(t, dt, False, False)
    assert kb.x == xs[-1]


def test_rhythm_game_never_imports_mqtt():
    code = ("import sys; sys.path.insert(0, r'%s'); import config, rhythm_main, rhythm_game, rhythm_render, "
            "make_chart, rhythm_motor, scoring, tempo; "
            "bad = [m for m in sys.modules if 'mqtt' in m.lower() or m.startswith('paho')]; "
            "print(bad); sys.exit(1 if bad else 0)") % str(config.HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_bad_hit_beats_or_mode_stops_the_game_with_a_clear_message(monkeypatch):
    import rhythm_main
    monkeypatch.setattr(config, "HIT_BEATS", [3])
    with pytest.raises(SystemExit) as e:
        rhythm_main.check_config()
    assert "HIT_BEATS must be [4]" in str(e.value)
    monkeypatch.setattr(config, "HIT_BEATS", None)
    monkeypatch.setattr(config, "GAME_MODE", "lanes")
    with pytest.raises(SystemExit) as e:
        rhythm_main.check_config()
    assert "GAME_MODE must be one of aim, timing, follow" in str(e.value)
