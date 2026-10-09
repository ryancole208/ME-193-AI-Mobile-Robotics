"""Headless smoke tests of the pygame renderer (SDL dummy video driver)."""

import os
import random
import time

import numpy as np
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
pygame = pytest.importorskip("pygame")

import config  # noqa: E402
from events import SwingEvent  # noqa: E402
from game import GameLogic  # noqa: E402
from opponent import SimulatedOpponent  # noqa: E402
from player_input import PlayerInput  # noqa: E402
from pose_tracker import PoseSnapshot  # noqa: E402
from renderer import PlayerView, Renderer  # noqa: E402
from theme import get_theme  # noqa: E402

STATUSES = [("Motor", "motor", "scanning"), ("MQTT", "mqtt", "connected"), ("Pose", "pose", "disabled")]


@pytest.fixture
def screen():
    pygame.init()
    yield pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.quit()


@pytest.fixture
def renderer(screen):
    return Renderer(screen)


def rally_frames(renderer, g, snap=None):
    renderer.draw(0.0, g, PlayerView(), STATUSES, snap, ["debug line"], "hint")
    assert {"start", "difficulty:Easy", "difficulty:Medium", "difficulty:Hard"} <= set(renderer.buttons)

    g.start(0.0)
    t = config.SERVE_DELAY_S
    g.update(t)
    for k in range(0, 60):   # the ball along its whole flight, both sides of the net
        renderer.draw(t + k * 0.03, g, PlayerView(x=0.1, roll_deg=k * 7), STATUSES, snap, None, "")
    assert renderer.buttons == {}
    arrive = g.flight.t_end
    g.input.keyboard.x = g.flight.end_x
    g.add_swing(SwingEvent(arrive, 1.2, "keyboard", spin_side=0.8, spin_top=0.4))
    g.update(arrive + 0.01)
    assert g.state == "to_opponent"
    view = PlayerView(x=0.0, roll_deg=-30, swing_t=arrive, direction="left")
    for k in range(12):
        renderer.draw(arrive + k * 0.03, g, view, STATUSES, snap, ["x"], "")
    return g


def test_menu_and_play_frames_render(renderer):
    g = GameLogic(SimulatedOpponent(rng=random.Random(0)), PlayerInput(), rng=random.Random(0))
    preview = np.zeros((180, config.PREVIEW_WIDTH, 3), dtype=np.uint8)
    rally_frames(renderer, g, PoseSnapshot(preview_rgb=preview))
    assert g.spin_text == "CURVE RIGHT!"


def test_plain_paddle_x_still_accepted(renderer):
    g = GameLogic(SimulatedOpponent(rng=random.Random(0)), PlayerInput(), rng=random.Random(0))
    renderer.draw(0.0, g, 0.25, STATUSES)


def test_skeleton_reactions_render(renderer):
    g = GameLogic(SimulatedOpponent(rng=random.Random(0)), PlayerInput(), rng=random.Random(0))
    g.start(0.0)
    g.update(config.SERVE_DELAY_S)
    for kind in ("opponent_hit", "opponent_miss", "player_miss"):
        g.event_times = {kind: 5.0}
        moods = set()
        for k in range(10):
            now = 5.0 + k * 0.05
            moods.add(renderer.skeleton.pose(now, g)["mood"])
            renderer.draw(now, g, PlayerView(), STATUSES)
        assert moods & {"swing", "oops", "dance"}


def test_classic_theme_renders(screen):
    r = Renderer(screen, theme=get_theme("classic"))
    g = GameLogic(SimulatedOpponent(rng=random.Random(1)), PlayerInput(), rng=random.Random(1))
    rally_frames(r, g)


def test_frame_time_budget(renderer):
    """Rendering must leave room for the 60 FPS target (generous bound for slow CI)."""
    g = GameLogic(SimulatedOpponent(rng=random.Random(0)), PlayerInput(), rng=random.Random(0))
    g.start(0.0)
    g.update(config.SERVE_DELAY_S)
    t0 = time.perf_counter()
    n = 30
    for k in range(n):
        renderer.draw(config.SERVE_DELAY_S + k * 0.02, g, PlayerView(roll_deg=k * 5, swing_t=config.SERVE_DELAY_S),
                      STATUSES, None, None, "hint")
    per_frame = (time.perf_counter() - t0) / n
    assert per_frame < 1.0 / config.FPS


# =============================================================================
# Controls bar, 4-button menu, Dynamic panel
# =============================================================================

def _dynamic_game():
    from opponent import ModeOpponent
    from rl_opponent import RLOpponent
    inp = PlayerInput()
    rl = RLOpponent(inp.current_paddle_x, persist=False, rng=random.Random(0))
    g = GameLogic(ModeOpponent(SimulatedOpponent(rng=random.Random(0)), {"Dynamic": rl}), inp,
                  rng=random.Random(0))
    g.set_difficulty("Dynamic")
    return g


@pytest.mark.parametrize("size", [(1100, 720), (800, 600), (640, 480)])
def test_controls_bar_and_menu_fit_any_window(size):
    from main import CONTROLS
    pygame.init()
    screen = pygame.display.set_mode(size)
    r = Renderer(screen)
    lines = r.wrap_hint(CONTROLS, size[0] - 40)
    assert all(r.f_small.size(line)[0] <= size[0] - 40 for line in lines)
    assert "".join(lines).replace(" ", "") == "".join(CONTROLS).replace(" ", "")   # nothing dropped
    assert any("4 = Dynamic" in line for line in lines) and any("R reset AI" in line for line in lines)
    assert any("H hide controls" in line for line in lines)
    top = r._draw_hint(CONTROLS)
    assert 0 < top < size[1]
    g = _dynamic_game()
    r.draw(0.0, g, PlayerView(), STATUSES, None, None, CONTROLS)
    names = {f"difficulty:{n}" for n in config.DIFFICULTIES}
    assert names | {"start"} <= set(r.buttons)
    for rect in r.buttons.values():
        assert screen.get_rect().contains(rect)
    pygame.quit()


def test_hidden_controls_bar_leaves_a_hint(renderer):
    g = _dynamic_game()
    renderer.draw(0.0, g, PlayerView(), STATUSES, None, None, ["H show controls"])
    assert renderer._draw_hint(["H show controls"]) > renderer.h - 60


def test_dynamic_panel_renders_during_play(renderer):
    g = _dynamic_game()
    g.start(0.0)
    t = config.SERVE_DELAY_S
    g.update(t)
    info = g.opponent.panel_info()
    lines = Renderer.dynamic_lines(info)
    text = "\n".join(lines)
    for word in ("eps", "gamma", "EXPLORE", "difficulty", "reward", "mostly", "return rate", "target 70%-85%"):
        assert word in text or (word == "EXPLORE" and "EXPLOIT" in text)
    for k in range(40):
        renderer.draw(t + k * 0.03, g, PlayerView(), STATUSES, None, ["debug"], ["H show controls"])


def test_dynamic_frame_time_budget(renderer):
    from main import CONTROLS
    g = _dynamic_game()
    g.start(0.0)
    g.update(config.SERVE_DELAY_S)
    t0 = time.perf_counter()
    n = 30
    for k in range(n):
        now = config.SERVE_DELAY_S + k * 0.02
        g.update(now)
        renderer.draw(now, g, PlayerView(roll_deg=k * 5), STATUSES, None, None, CONTROLS)
    assert (time.perf_counter() - t0) / n < 1.0 / config.FPS
