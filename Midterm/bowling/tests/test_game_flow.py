"""Game flow with a scripted roll factory: frames, callouts, rack resets, arming, MQTT score."""

import random

import config
from bowler_input import BowlerInput
from bowling_game import BowlingGame
from lane import ALL_PINS, RollSim
from throw_detector import ThrowEvent


class ScriptedRoll:
    """Stands in for RollSim: knocks down a scripted number of the standing pins."""

    def __init__(self, script):
        self.script = list(script)
        self.launches = []

    def __call__(self, launch, standing, bumpers, rng):
        self.launches.append((launch, set(standing), bumpers))
        n, gutter = self.script.pop(0)
        roll = _Done(set(sorted(standing)[:n]), gutter)
        return roll


class _Done:
    def __init__(self, knocked, gutter):
        self.knocked, self.gutter = knocked, gutter
        self.first_contact_t = None if gutter else 0.5
        self.finished = False
        self.t = 0.0
        self.pins, self.standing = [], set()

    def advance_to(self, t):
        self.t = t
        self.finished = t >= 1.0


def throw(t, strength=2.0, spin=0.0, source="keyboard"):
    return ThrowEvent(t_onset=t, t_release=t, strength=strength, spin=spin, source=source)


def bowl(game, now, script_len=1):
    """Throw once and run the clock until the game is waiting for the next throw."""
    now += config.AIM_ARM_DELAY_S + 0.01
    assert game.add_throw(throw(now), now)
    while game.state == "rolling":
        now += 0.1
        game.update(now)
    messages = game.message
    while game.state == "result":
        now += 0.5
        game.update(now)
    return now, messages


def new_game(script, difficulty="No Bumpers"):
    scores, impacts = [], []
    factory = ScriptedRoll(script)
    g = BowlingGame(BowlerInput(), on_score=scores.append, on_impact=lambda: impacts.append(1),
                    rng=random.Random(0), roll_factory=factory)
    g.set_difficulty(difficulty)
    g.start(0.0)
    return g, factory, scores, impacts


def test_perfect_game_callouts_and_mqtt_scores():
    g, factory, scores, impacts = new_game([(10, False)] * 12)
    now, msgs = 0.0, []
    for _ in range(12):
        now, msg = bowl(g, now)
        msgs.append(msg)
    assert g.state == "game_over"
    assert g.sheet.running_total() == 300 and g.best == 300
    assert msgs[0] == "STRIKE!" and "Double" in msgs[1] and "Turkey" in msgs[2] and "4-Bagger" in msgs[3]
    assert scores[0] == 0.0 and scores[-1] == 300.0
    assert len(impacts) == 12
    assert all(standing == set(ALL_PINS) for _, standing, _ in factory.launches)


def test_spare_respots_only_the_remaining_pins():
    g, factory, scores, _ = new_game([(7, False), (3, False)])
    now, msg = bowl(g, 0.0)
    assert msg == "7 pins"
    assert len(factory.launches[-1][1]) == 10
    now, msg = bowl(g, now)
    assert msg == "SPARE!"
    assert len(factory.launches[-1][1]) == 3
    assert g.standing == set(ALL_PINS) and g.sheet.current_frame == 1
    assert scores[-1] == 10.0


def test_gutter_ball_callout_and_mark():
    g, *_ = new_game([(0, True), (0, True)])
    _, msg = bowl(g, 0.0)
    assert msg == "Gutter..."
    assert g.sheet.marks(0) == ["G"]


def test_bumpers_choice_reaches_the_roll():
    g, factory, *_ = new_game([(1, False)], difficulty="Bumpers")
    bowl(g, 0.0)
    assert factory.launches[0][2] is True


def test_throws_ignored_unless_aiming_and_armed():
    g, factory, *_ = new_game([(1, False)] * 3)
    assert not g.add_throw(throw(0.1), 0.1)                 # too soon after "ready"
    g.to_menu()
    assert not g.add_throw(throw(5.0), 5.0)                 # menu
    g.start(6.0)
    assert g.add_throw(throw(7.0), 7.0)
    assert not g.add_throw(throw(7.1), 7.1)                 # already rolling
    assert len(factory.launches) == 1


def test_difficulty_locked_during_game_and_restart_after_game_over():
    g, *_ = new_game([(0, False)] * 40)
    assert not g.set_difficulty("Bumpers")
    now = 0.0
    for _ in range(20):
        now, _ = bowl(g, now)
    assert g.state == "game_over" and g.sheet.running_total() == 0
    g.start(now)
    assert g.state == "aiming" and g.sheet.running_total() == 0 and g.sheet.current_frame == 0


def test_launch_uses_bowler_position_aim_and_strength():
    g, factory, *_ = new_game([(1, False)])
    g.bowler.keyboard_x = -0.2
    g.bowler.preset_aim_deg = 1.5
    t = config.AIM_ARM_DELAY_S + 0.1
    g.add_throw(throw(t, strength=3.0, spin=-0.6), t)
    launch = factory.launches[0][0]
    assert launch.x == -0.2 and launch.angle_deg == 1.5 and launch.spin == -0.6
    assert launch.speed > config.BALL_SPEED_AT_THRESHOLD


def test_real_physics_game_runs_to_completion():
    g = BowlingGame(BowlerInput(), rng=random.Random(3), roll_factory=RollSim)
    g.start(0.0)
    now = 0.0
    while g.state != "game_over":
        now += config.AIM_ARM_DELAY_S + 0.01
        g.bowler.keyboard_x = 0.07
        g.add_throw(throw(now), now)
        while g.state in ("rolling", "result"):
            now += 0.05
            g.update(now)
    assert 0 < g.sheet.running_total() <= 300
