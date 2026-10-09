"""Game rules: serve, hit window, lateral tolerance, streak/score, returns."""

import random

import pytest

import config
from events import SwingEvent
from camera3d import Camera
from game import Flight, GameLogic, ball_height, return_angle_deg, return_speed, status_color
from opponent import MISS, Opponent, Shot, SimulatedOpponent
from player_input import KeyboardPaddle, PlayerInput


class FixedInput:
    """Player input whose paddle/direction we control directly."""

    def __init__(self, x=0.0, direction="center"):
        self.x, self.direction = x, direction

    def paddle_x_at(self, t):
        return self.x

    def direction_at(self, t):
        return self.direction

    def current_paddle_x(self):
        return self.x


class ScriptedOpponent(Opponent):
    """Serves straight down the middle; always returns to the middle."""

    def __init__(self, result="shot"):
        self.result = result
        self.incoming = []
        self.resets = 0

    def serve(self, now):
        return Shot(0.0, 0.0, self.ball_speed)

    def ball_incoming(self, ball):
        self.incoming.append(ball)

    def poll(self, now):
        if not self.incoming or now < self.incoming[-1].arrival_time:
            return None
        self.incoming.clear()
        return MISS if self.result == "miss" else Shot(0.0, 0.0, self.ball_speed)

    def reset(self):
        self.resets += 1


def make_game(x=0.0, direction="center", opponent=None):
    scores, hits = [], []
    opp = opponent or ScriptedOpponent()
    inp = FixedInput(x, direction)
    g = GameLogic(opp, inp, on_score=scores.append, on_hit=lambda: hits.append(1), rng=random.Random(1))
    return g, inp, opp, scores, hits


def serve(g, t=0.0):
    g.start(t)
    t += config.SERVE_DELAY_S
    g.update(t)
    assert g.state == "to_player"
    return t


def test_menu_difficulty_and_start():
    g, *_ = make_game()
    assert g.state == "menu"
    assert g.set_difficulty("Hard") and g.difficulty == "Hard"
    assert g.opponent.ball_speed == config.DIFFICULTIES["Hard"]["ball_speed"]
    with pytest.raises(ValueError):
        g.set_difficulty("Impossible")
    g.start(0)
    assert g.state == "serve_wait"
    assert not g.set_difficulty("Easy")   # locked during play
    g.update(config.SERVE_DELAY_S - 0.01)
    assert g.state == "serve_wait"
    g.update(config.SERVE_DELAY_S)
    assert g.state == "to_player"


def test_serve_speed_matches_difficulty():
    for name, d in config.DIFFICULTIES.items():
        g, *_ = make_game()
        g.set_difficulty(name)
        serve(g)
        assert g.flight.speed == d["ball_speed"]


def test_on_time_in_position_swing_is_a_hit():
    g, inp, opp, scores, hits = make_game(x=0.05)
    serve(g)
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(t=arrive - 0.05, strength=1.0))
    g.update(arrive - 0.04)
    assert g.state == "to_opponent"
    assert g.streak == 1 and scores == [1.0] and hits == [1]
    assert isinstance(scores[0], float)
    assert opp.incoming and opp.incoming[0].x == pytest.approx(g.flight.end_x)
    # early swing: return flight starts when the ball actually arrives
    assert g.flight.t0 == pytest.approx(arrive)


@pytest.mark.parametrize("offset", [-config.HIT_WINDOW_BEFORE_S + 0.01, config.HIT_WINDOW_AFTER_S - 0.01])
def test_window_edges_hit(offset):
    g, *_ = make_game()
    serve(g)
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(t=arrive + offset, strength=1.0))
    g.update(max(arrive + offset, arrive) + 0.01)
    assert g.state == "to_opponent"


def test_too_early_swing_is_ignored_then_miss():
    g, _, _, scores, _ = make_game()
    serve(g)
    g._set_streak(3); scores.clear()
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(t=arrive - config.HIT_WINDOW_BEFORE_S - 0.1, strength=2.0))
    g.update(arrive - 0.1)
    assert g.state == "to_player"
    g.update(arrive + config.HIT_WINDOW_AFTER_S + config.SWING_PEAK_WINDOW_S + 0.1)
    assert g.state == "miss"
    assert g.streak == 0 and scores == [0.0]
    assert g.best == 3


def test_too_late_swing_is_a_miss():
    g, *_ = make_game()
    serve(g)
    arrive = g.flight.t_end
    late = arrive + config.HIT_WINDOW_AFTER_S + 0.05
    g.add_swing(SwingEvent(t=late, strength=1.0))
    g.update(late + config.SWING_PEAK_WINDOW_S + 0.1)
    assert g.state == "miss"


def test_out_of_position_is_a_miss():
    g, inp, opp, scores, _ = make_game(x=config.LATERAL_HIT_TOLERANCE_M + 0.05)
    serve(g)
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(t=arrive, strength=1.0))
    g.update(arrive + 0.01)
    assert g.state == "miss"
    assert "out of position" in g.last_result["reason"]
    assert opp.resets >= 1


def test_no_swing_is_a_miss_and_next_serve_follows():
    g, *_ = make_game()
    t = serve(g)
    end = g.flight.t_end + config.HIT_WINDOW_AFTER_S + config.SWING_PEAK_WINDOW_S + 0.1
    g.update(end)
    assert g.state == "miss"
    g.update(end + config.MISS_DISPLAY_S)
    assert g.state == "serve_wait"
    g.update(end + config.MISS_DISPLAY_S + config.SERVE_DELAY_S)
    assert g.state == "to_player"


def test_rally_back_and_forth_counts_streak():
    g, _, _, scores, _ = make_game()
    serve(g)
    for i in range(1, 4):
        arrive = g.flight.t_end
        g.add_swing(SwingEvent(t=arrive, strength=1.0))
        g.update(arrive + 0.01)
        assert g.state == "to_opponent"
        g.update(g.flight.t_end + 0.01)   # opponent returns
        assert g.state == "to_player"
    assert g.streak == 3 and scores == [1.0, 2.0, 3.0]


def test_opponent_miss_keeps_streak():
    g, *_ = make_game(opponent=ScriptedOpponent(result="miss"))
    serve(g)
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(t=arrive, strength=1.0))
    g.update(arrive + 0.01)
    g.update(g.flight.t_end + 0.01)
    assert g.state == "miss" and g.message.startswith("Opponent")
    assert g.streak == 1


def test_silent_opponent_times_out():
    class Silent(ScriptedOpponent):
        def poll(self, now):
            return None
    g, *_ = make_game(opponent=Silent())
    serve(g)
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(t=arrive, strength=1.0))
    g.update(arrive + 0.01)
    g.update(g.flight.t_end + config.OPPONENT_TIMEOUT_S + 0.01)
    assert g.state == "miss"


@pytest.mark.parametrize("direction,sign", [("left", -1), ("right", 1)])
def test_swing_direction_angles_return(direction, sign):
    g, *_ = make_game(direction=direction)
    serve(g)
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(t=arrive, strength=1.0))
    g.update(arrive + 0.01)
    assert (g.flight.end_x - g.flight.start_x) * sign > 0.2
    assert g.last_result["direction"] == direction


def test_strength_changes_return_speed_within_limits():
    base = 3.0
    assert return_speed(base, 1.0) == pytest.approx(base)
    assert return_speed(base, 100.0) == pytest.approx(base * config.RETURN_SPEED_MAX_MULT)
    assert return_speed(base, 0.0) >= base * config.RETURN_SPEED_MIN_MULT
    assert return_speed(base, 2.0) > return_speed(base, 1.2)


def test_return_angle_limits():
    rng = random.Random(0)
    assert return_angle_deg("left", rng) == -config.RETURN_ANGLE_LEFT_RIGHT_DEG
    assert return_angle_deg("right", rng) == config.RETURN_ANGLE_LEFT_RIGHT_DEG
    for _ in range(50):
        assert abs(return_angle_deg("center", rng)) <= config.RETURN_ANGLE_CENTER_JITTER_DEG


def test_return_lands_on_table():
    g, *_ = make_game(x=0.7, direction="right")
    g.opponent.serve = lambda now: Shot(0.7, 0.7, g.ball_speed)
    serve(g)
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(t=arrive, strength=1.0))
    g.update(arrive + 0.01)
    assert abs(g.flight.end_x) <= config.TABLE_WIDTH_M / 2


def test_esc_to_menu_mid_rally():
    g, *_ = make_game()
    serve(g)
    g.to_menu()
    assert g.state == "menu" and g.ball_position(0) is None


def test_flight_and_ball_height():
    f = Flight(0, 3.0, 0.5, -0.1, t0=1.0, speed=3.1)
    assert f.duration == pytest.approx(1.0)
    assert f.position(1.0)[:2] == (0, 3.0)
    x, y, _ = f.position(2.0)
    assert (x, y) == pytest.approx((0.5, -0.1))
    assert f.position(5.0, clamp_end=True)[:2] == pytest.approx((0.5, -0.1))
    assert ball_height(0.0) == 0
    assert ball_height(0.7) == pytest.approx(0, abs=1e-9)   # bounce
    assert ball_height(0.35) == pytest.approx(config.BALL_ARC_HEIGHT_M)


def test_projection_is_front_angled_perspective():
    cam = Camera(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    project = cam.project
    near_l, near_y, _ = project(-0.7, 0.0, 0.0)
    near_r, _, _ = project(0.7, 0.0, 0.0)
    far_l, far_y, _ = project(-0.7, config.TABLE_LENGTH_M, 0.0)
    far_r, _, _ = project(0.7, config.TABLE_LENGTH_M, 0.0)
    assert near_r - near_l > far_r - far_l       # far end is narrower
    assert near_y > far_y                        # far end is higher on screen
    assert near_y < config.WINDOW_HEIGHT         # whole table top on screen
    assert 0 < cam.horizon_y < far_y             # sky visible above the far end
    _, raised, _ = project(0, 1.0, 0.3)
    _, ground, _ = project(0, 1.0, 0.0)
    assert raised < ground                       # height moves the ball up the screen
    _, _, s_near = project(0, 0.0, 0.0)
    _, _, s_far = project(0, config.TABLE_LENGTH_M, 0.0)
    assert s_near > s_far                        # perspective scaling (ball gets smaller far away)
    # my paddle (below/behind) is on screen and the camera is behind it
    px, py, _ = project(0, config.PLAYER_HIT_Y_M, config.PADDLE_HEIGHT_M)
    assert 0 < px < config.WINDOW_WIDTH and 0 < py < config.WINDOW_HEIGHT
    assert cam.facing((0, 0, 0), (0, -1, 0)) and not cam.facing((0, 0, 0), (0, 1, 0))


def test_status_colors():
    assert status_color("motor", "connected: Double Motor") != status_color("motor", "scanning")
    assert status_color("pose", "tracking") == status_color("mqtt", "connected")
    assert status_color("mqtt", "disabled") == status_color("pose", "disabled")


def test_keyboard_fallback_paddle_and_swing():
    kb = KeyboardPaddle(speed=1.0)
    inp = PlayerInput(pose=None, keyboard=kb)
    kb.update(0.0, 0.2, left=False, right=True)
    assert inp.current_paddle_x() == pytest.approx(0.2)
    assert inp.direction_at(0.0) == "right"
    assert inp.source == "keyboard"
    for i in range(100):
        kb.update(0.0, 0.1, left=True, right=False)
    assert kb.x == pytest.approx(-config.TABLE_WIDTH_M / 2)

    g = GameLogic(ScriptedOpponent(), PlayerInput(keyboard=KeyboardPaddle()))
    serve(g)
    arrive = g.flight.t_end
    g.add_swing(SwingEvent(t=arrive, strength=config.KEYBOARD_SWING_STRENGTH, source="keyboard"))
    g.update(arrive + 0.01)
    assert g.state == "to_opponent"


def test_full_rally_with_simulated_opponent():
    g = GameLogic(SimulatedOpponent(rng=random.Random(3)), FixedInput(), rng=random.Random(3))
    serve(g)
    for _ in range(5):
        g.input.x = g.flight.end_x   # perfect positioning
        arrive = g.flight.t_end
        g.add_swing(SwingEvent(t=arrive, strength=1.3))
        g.update(arrive + 0.01)
        assert g.state == "to_opponent"
        g.update(g.flight.t_end + config.OPPONENT_REACTION_S + 0.01)
        assert g.state == "to_player"
    assert g.streak == 5
