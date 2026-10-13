"""Game flow with scripted putts: holes, intro, score names, penalties, pick-ups, MQTT strokes."""

import random

import pytest

import config
from course import generate_course
from golf_game import GolfGame, score_name
from golfer_input import GolferInput
from putt_detector import PuttEvent


class ScriptedSim:
    """Stands in for PuttSim: each putt ends with the next scripted outcome."""

    def __init__(self, script):
        self.script = list(script)
        self.launches = []

    def __call__(self, hole, start, heading, speed, t0, *, cup_radius, capture_speed):
        self.launches.append(dict(hole=hole, start=start, heading=heading, speed=speed,
                                  cup_radius=cup_radius, capture_speed=capture_speed))
        return _Sim(start, t0, self.script.pop(0))


class _Ball:
    def __init__(self, x, y):
        self.x, self.y, self.z, self.speed = x, y, 0.0, 0.0


class _Sim:
    def __init__(self, start, t0, outcome):
        self.start, self.t0, self.final = tuple(start), t0, outcome
        self.ball = _Ball(start[0], start[1] + 1.0)
        self.outcome = None
        self.finished = False
        self.lipped = outcome == "lip"
        self.holed_t = None
        self.wall_hits = 0

    @property
    def holed(self):
        return self.outcome == "holed"

    def advance_to(self, t):
        if t - self.t0 >= 0.5 and self.outcome is None:
            self.outcome = "stopped" if self.final == "lip" else self.final
            self.holed_t = t if self.final == "holed" else None
        self.finished = t - self.t0 >= 1.0


def putt(t, strength=2.0):
    return PuttEvent(t_onset=t, t_impact=t, strength=strength, source="keyboard")


def new_game(script, difficulty="Easy"):
    scores, holed = [], []
    sim = ScriptedSim(script)
    g = GolfGame(GolferInput(), on_score=scores.append, on_holed=lambda: holed.append(1),
                 rng=random.Random(0), sim_factory=sim)
    g.set_difficulty(difficulty)
    g.start(0.0)
    return g, sim, scores, holed


def wait_for_aim(g, now):
    while g.state != "aiming":
        now += 0.1
        g.update(now)
    return now


def stroke(g, now):
    """Putt once and run the clock until the game wants the next putt (or the hole/game is over)."""
    now = wait_for_aim(g, now) + config.AIM_ARM_DELAY_S + 0.01
    assert g.add_putt(putt(now), now)
    msg = ""
    while g.state in ("rolling", "result"):
        now += 0.1
        g.update(now)
        msg = msg or g.message
    if g.state == "holed":
        msg = g.message
    return now, msg


def finish_hole(g, now):
    while g.state == "holed":
        now += 0.5
        g.update(now)
    return now


def test_score_names():
    assert score_name(1, 3) == "HOLE IN ONE!"
    assert score_name(2, 3) == "Birdie!" and score_name(2, 4) == "Eagle!"
    assert score_name(3, 3) == "Par" and score_name(4, 3) == "Bogey" and score_name(5, 3) == "Double Bogey"
    assert score_name(8, 3) == "+5"


def test_start_generates_five_holes_and_plays_intro():
    g, *_ = new_game([])
    assert len(g.course) == 5 and g.state == "intro"
    assert g.ball == g.course[0].tee
    assert g.message.startswith("Hole 1")
    assert not g.add_putt(putt(1.0), 1.0)                   # no putting during the flyover
    g.update(config.INTRO_S + 0.01)
    assert g.state == "aiming"


def test_enter_skips_the_intro():
    g, *_ = new_game([])
    g.start(0.5)
    assert g.state == "aiming" and g.aim_since == 0.5


def test_full_round_publishes_running_strokes_and_ends():
    script = ["holed"] + ["stopped", "holed"] * 2 + ["stopped", "stopped", "holed"] * 2
    g, sim, scores, holed = new_game(script)
    now, msgs = 0.0, []
    for _ in range(5):
        while g.state != "holed":
            now, msg = stroke(g, now)
        msgs.append(g.message)
        now = finish_hole(g, now)
    assert g.state == "game_over"
    assert g.strokes == [1, 2, 2, 3, 3] and g.total_strokes == 11
    assert msgs[0] == "HOLE IN ONE!"
    assert scores[0] == 0.0 and scores[-1] == 11.0 and scores == sorted(scores)
    assert len(holed) == 5
    assert g.best == 11


def test_ball_moves_to_where_the_putt_stopped_and_aim_resets():
    g, sim, *_ = new_game(["stopped", "holed"])
    now, _ = stroke(g, 0.0)
    assert g.state == "aiming"
    assert g.ball == (g.course[0].tee[0], g.course[0].tee[1] + 1.0)
    assert sim.launches[0]["start"] == g.course[0].tee
    now, _ = stroke(g, now)
    assert sim.launches[1]["start"] == g.ball


def test_pit_costs_a_stroke_and_returns_the_ball():
    g, sim, scores, _ = new_game(["hazard", "holed"])
    now, msg = stroke(g, 0.0)
    assert msg.startswith("Splash")
    assert g.hole_strokes == 1 + config.PIT_PENALTY_STROKES
    assert g.ball == g.course[0].tee
    assert scores[-1] == float(1 + config.PIT_PENALTY_STROKES)


def test_lip_out_callout():
    g, *_ = new_game(["lip", "holed"])
    _, msg = stroke(g, 0.0)
    assert msg == "Lipped out!"


def test_pick_up_after_max_strokes():
    g, sim, scores, _ = new_game(["stopped"] * config.MAX_STROKES)
    now = 0.0
    for _ in range(config.MAX_STROKES):
        now, msg = stroke(g, now)
    assert g.state == "holed" and msg == "Picked up"
    assert g.strokes == [config.MAX_STROKES]
    now = finish_hole(g, now)
    assert g.hole_index == 1 and g.state == "intro"


def test_penalty_past_max_is_capped():
    g, sim, scores, _ = new_game(["stopped"] * (config.MAX_STROKES - 2) + ["hazard"])
    now = 0.0
    for _ in range(config.MAX_STROKES - 1):
        now, msg = stroke(g, now)
    assert g.strokes == [config.MAX_STROKES] and scores[-1] == float(config.MAX_STROKES)


def test_putts_ignored_unless_aiming_and_armed():
    g, sim, *_ = new_game(["stopped"] * 3)
    now = wait_for_aim(g, 0.0)
    assert not g.add_putt(putt(now + 0.1), now + 0.1)         # too soon after "ready"
    t = now + config.AIM_ARM_DELAY_S + 0.01
    assert g.add_putt(putt(t), t)
    assert not g.add_putt(putt(t + 0.05), t + 0.05)           # already rolling
    g.to_menu()
    assert not g.add_putt(putt(t + 5), t + 5)
    assert len(sim.launches) == 1


def test_putt_uses_aim_at_onset_push_and_strength():
    g, sim, *_ = new_game(["stopped"])
    now = wait_for_aim(g, 0.0)
    g.golfer.set_aim(12.0, now)
    t = now + config.AIM_ARM_DELAY_S + 0.1
    g.golfer.update(t, 0.1)
    g.add_putt(putt(t, strength=3.0), t)
    launch = sim.launches[0]
    assert launch["heading"] == pytest.approx(12.0)
    assert launch["speed"] > config.PUTT_SPEED_AT_THRESHOLD
    assert g.last_putt["aim_deg"] == pytest.approx(12.0) and g.last_putt["push_deg"] == 0.0


def test_difficulty_sets_cup_and_is_locked_during_a_round():
    g, sim, *_ = new_game(["stopped"], difficulty="Hard")
    assert not g.set_difficulty("Easy")
    stroke(g, 0.0)
    assert sim.launches[0]["cup_radius"] == config.DIFFICULTIES["Hard"]["cup_radius_m"]
    assert sim.launches[0]["capture_speed"] == config.DIFFICULTIES["Hard"]["capture_speed"]


def test_play_again_makes_a_new_course():
    g, *_ = new_game(["holed"] * 5)
    first = [h.path for h in g.course]
    now = 0.0
    for _ in range(5):
        now, _ = stroke(g, now)
        now = finish_hole(g, now)
    assert g.state == "game_over"
    g.start(now)
    assert g.state == "intro" and g.strokes == [0] and g.hole_index == 0
    assert [h.path for h in g.course] != first


def test_real_physics_round_runs_to_completion():
    g = GolfGame(GolferInput(), rng=random.Random(2))
    g.start(0.0)
    now = 0.0
    while g.state != "game_over":
        now = wait_for_aim(g, now) + config.AIM_ARM_DELAY_S + 0.01
        hole = g.hole
        d = ((hole.cup[0] - g.ball[0]) ** 2 + (hole.cup[1] - g.ball[1]) ** 2) ** 0.5
        strength = 1.0 + (min(4.6, (2 * 0.7 * d) ** 0.5 + 0.25) - config.PUTT_SPEED_AT_THRESHOLD) \
            / config.PUTT_SPEED_PER_STRENGTH
        g.add_putt(putt(now, strength), now)
        while g.state in ("rolling", "result", "holed"):
            now += 0.05
            g.update(now)
    assert len(g.strokes) == 5
    assert all(1 <= s <= config.MAX_STROKES for s in g.strokes)
