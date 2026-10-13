"""Headless rendering of every screen at several window sizes (SDL dummy video driver)."""

import pytest

from audio_engine import NullEngine
from player_input import KeyboardPaddle, PlayerInput
from rhythm_game import RhythmGame
from scoring import BestScores


@pytest.fixture(scope="module")
def pg():
    import pygame
    pygame.display.init()
    pygame.font.init()
    yield pygame
    pygame.quit()


@pytest.mark.parametrize("size", [(1100, 720), (700, 440), (1600, 900)])
def test_every_screen_draws(pg, size, tmp_path):
    from rhythm_render import PlayerView, RhythmRenderer
    now = [500.0]
    eng = NullEngine(clock_fn=lambda: now[0])
    kb = KeyboardPaddle()
    game = RhythmGame(eng, PlayerInput(keyboard=kb), best=BestScores(tmp_path / "s.json"),
                      clock_fn=lambda: now[0])
    screen = pg.display.set_mode(size)
    r = RhythmRenderer(screen)
    st = [("Motor", "disabled"), ("Pose", "disabled"), ("Audio", eng.status)]

    def draw():
        r.draw(now[0], game, PlayerView(x=0.3, roll_deg=20, swing_t=now[0] - 0.05), st, None,
               ["line"] if size[0] < 800 else None, ["Space swing", "hold Left/Right aim", "H hide controls"])

    draw()                                              # title
    game.on_enter(); draw()                             # select
    game.sel_index = [i for i, e in enumerate(game.songs) if e.chart.song_id == "Electrodoodle"][0]
    draw()
    game.on_enter()
    assert game.state == "play"
    change = game.tempo_changes[0]                      # Electrodoodle: around its speed-up (LED)
    picks = game.flights[:6] + [f for f in game.flights if abs(f.t_arrive - change.t) < 3]
    for fl in picks:                                    # countdown, flights, popups, tempo LED
        for dt in (-0.3, -0.05, 0.05, 0.2):
            now[0] = eng.song_start / eng.sr + fl.t_arrive + dt
            if dt == 0.05:
                game.swing(now[0], "keyboard")
            game.update(now[0])
            draw()
    now[0] = eng.song_start / eng.sr + game.end_song_t + 1
    game.update(now[0]); draw()                         # results
    game.on_calibrate(); now[0] += 2; game.update(now[0]); draw()
    game.on_escape(); game.on_calibrate(visual=True); now[0] += 2; game.update(now[0]); draw()


def test_controls_bar_wraps_to_the_window(pg):
    from rhythm_main import CONTROLS
    from rhythm_render import RhythmRenderer
    screen = pg.display.set_mode((640, 420))
    r = RhythmRenderer(screen)
    for segments in CONTROLS.values():
        for line in r.wrap_hint(segments, 600):
            assert r.f_small.size(line)[0] <= 600


def test_tempo_led_draws_at_any_level(pg):
    from rhythm_render import draw_tempo_led
    from tempo import LedState
    screen = pg.display.set_mode((400, 300))
    draw_tempo_led(pg, screen, (200, 100), 18, None)
    for lit in (0.0, 0.5, 1.0):
        draw_tempo_led(pg, screen, (200, 100), 18, LedState(1.0, lit, 1, 160.0), pg.font.SysFont(None, 18))
    assert screen.get_at((200, 100))[:3] != (0, 0, 0)


@pytest.mark.parametrize("mode", ["aim", "timing", "follow"])
def test_play_screen_draws_in_every_mode(pg, mode, tmp_path):
    from rhythm_render import PlayerView, RhythmRenderer
    now = [500.0]
    eng = NullEngine(clock_fn=lambda: now[0])
    game = RhythmGame(eng, PlayerInput(keyboard=KeyboardPaddle()), best=BestScores(tmp_path / "s.json"),
                      clock_fn=lambda: now[0])
    game.on_enter()
    while game.mode != mode:
        game.cycle_mode()
    r = RhythmRenderer(pg.display.set_mode((1100, 720)))
    r.draw(now[0], game, PlayerView(), [], None, None, None)            # select, with the mode line
    game.on_enter()
    for fl in game.flights[:4]:
        for dt in (-0.4, 0.0, 0.1):
            now[0] = eng.song_start / eng.sr + fl.t_arrive + dt
            if dt == 0.0:
                game.swing(now[0], "keyboard")
            game.update(now[0])
            r.draw(now[0], game, PlayerView(x=-0.3), [], None, None, None)


def test_led_colour_says_faster_or_slower(pg):
    import config
    from rhythm_render import draw_tempo_led
    from tempo import LedState
    for faster, col in ((True, config.TEMPO_LED_COLOR), (False, config.TEMPO_LED_SLOW_COLOR)):
        screen = pg.display.set_mode((200, 200))
        draw_tempo_led(pg, screen, (100, 100), 18, LedState(1.0, 1.0, 1, 120.0, faster))
        px = screen.get_at((106, 106))[:3]                               # the lit lens, off the highlight
        assert max(abs(a - b) for a, b in zip(px, col)) < 40, (faster, px)
