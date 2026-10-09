"""
Game flow (no pygame -- unit tested).

    menu -> intro -> aiming -> (putt) rolling -> aiming ... -> holed -> intro (next hole) ... -> game_over
                                       `-> result ("Splash! +1") -> aiming
    game_over -> (Enter) intro (new course)  or  (Esc) menu

Like Wii Sports putting: choose Easy / Hard, then play 5 freshly generated Halloween
holes. Each hole opens with a flyover; while aiming you turn the putt (twist the motor
or ←/→). A putt (quick IMU swing or Space) goes at the aim you had when the swing fired, plus
any push/pull from your wrist around impact, with speed from swing strength. The
roll is simulated live. A cauldron costs a penalty stroke and puts the ball back;
after MAX_STROKES you pick up. The total stroke count goes out over MQTT.
"""

import random

import config
from course import generate_course
from physics import PuttSim
from putt_detector import strength_to_speed

SCORE_NAMES = {-3: "Albatross!", -2: "Eagle!", -1: "Birdie!", 0: "Par", 1: "Bogey",
               2: "Double Bogey", 3: "Triple Bogey"}


def score_name(strokes, par):
    if strokes == 1:
        return "HOLE IN ONE!"
    d = strokes - par
    return SCORE_NAMES.get(d, f"{d:+d}" if d > 0 else "Condor!")


class GolfGame:
    def __init__(self, golfer, *, on_score=None, on_holed=None, rng=None,
                 course_factory=generate_course, sim_factory=PuttSim):
        self.golfer = golfer
        self.on_score = on_score or (lambda score: None)
        self.on_holed = on_holed or (lambda: None)
        self.rng = rng or random.Random()
        self.course_factory = course_factory
        self.sim_factory = sim_factory
        self.state = "menu"
        self.difficulty = config.DEFAULT_DIFFICULTY
        self.course = []
        self.hole_index = 0
        self.strokes = []          # per hole played so far
        self.ball = (0.0, 0.0)
        self.sim = None
        self.state_since = 0.0
        self.aim_since = 0.0
        self.best = None
        self.message = ""
        self.message_until = 0.0
        self.last_putt = None      # dict for the HUD / debug overlay
        self._holed_sent = False

    @property
    def settings(self):
        return config.DIFFICULTIES[self.difficulty]

    @property
    def hole(self):
        return self.course[self.hole_index] if self.course else None

    @property
    def total_strokes(self):
        return sum(self.strokes)

    @property
    def total_par_played(self):
        return sum(h.par for h in self.course[:len(self.strokes)])

    @property
    def hole_strokes(self):
        return self.strokes[-1] if self.strokes else 0

    def _say(self, text, now, duration):
        self.message, self.message_until = text, now + duration

    def _set_state(self, state, now):
        self.state, self.state_since = state, now

    # -- commands ---------------------------------------------------------------
    def set_difficulty(self, name):
        if name not in config.DIFFICULTIES:
            raise ValueError(f"unknown difficulty {name!r}")
        if self.state != "menu":
            return False
        self.difficulty = name
        return True

    def start(self, now):
        """Enter in the menu / game over starts a new course; in a hole intro it skips the flyover."""
        if self.state == "intro":
            self._begin_aiming(now)
            return
        if self.state not in ("menu", "game_over"):
            return
        self.course = self.course_factory(self.rng, self.difficulty)
        self.hole_index = 0
        self.strokes = []
        self.last_putt = None
        self.on_score(0.0)
        self._begin_hole(now)

    def to_menu(self):
        self.state = "menu"
        self.sim = None

    def add_putt(self, ev, now):
        """A PuttEvent from the IMU or keyboard. Ignored unless the golfer is ready."""
        if self.state != "aiming" or ev.t_onset < self.aim_since + config.AIM_ARM_DELAY_S:
            return False
        aim = self.golfer.aim_at(ev.t_onset)
        push = self.golfer.push_deg(ev.t_impact)
        heading = self.golfer.putt_heading_deg(ev.t_onset, ev.t_impact)
        speed = strength_to_speed(ev.strength)
        self.last_putt = {"aim_deg": aim, "push_deg": push, "heading_deg": heading, "speed": speed,
                          "strength": ev.strength, "source": ev.source}
        s = self.settings
        self.sim = self.sim_factory(self.hole, self.ball, heading, speed, now,
                                    cup_radius=s["cup_radius_m"], capture_speed=s["capture_speed"])
        self.strokes[-1] += 1
        self.on_score(float(self.total_strokes))
        self._holed_sent = False
        self._set_state("rolling", now)
        self.message = ""
        return True

    # -- per-frame update ---------------------------------------------------------
    def update(self, now):
        if self.state == "intro" and now - self.state_since >= config.INTRO_S:
            self._begin_aiming(now)
        elif self.state == "rolling":
            self.sim.advance_to(now)
            if self.sim.holed and not self._holed_sent:
                self._holed_sent = True
                self.on_holed()
            if self.sim.finished:
                self._finish_putt(now)
        elif self.state == "result" and now >= self.message_until:
            self._begin_aiming(now)
        elif self.state == "holed" and now >= self.message_until:
            if self.hole_index + 1 < len(self.course):
                self.hole_index += 1
                self._begin_hole(now)
            else:
                total = self.total_strokes
                self.best = total if self.best is None else min(self.best, total)
                self._set_state("game_over", now)
                self.message = ""

    def _begin_hole(self, now):
        self.strokes.append(0)
        self.ball = self.hole.tee
        self.sim = None
        self._set_state("intro", now)
        self.golfer.set_aim(self.hole.default_aim_deg(self.ball), now)
        self._say(f"Hole {self.hole.number}   Par {self.hole.par}", now, config.INTRO_S)

    def _begin_aiming(self, now):
        self._set_state("aiming", now)
        self.aim_since = now
        self.golfer.set_aim(self.hole.default_aim_deg(self.ball), now)
        if self.message_until > now and self.message.startswith("Hole "):
            self.message = ""

    def _finish_putt(self, now):
        sim = self.sim
        hole = self.hole
        if sim.outcome == "holed":
            n = self.hole_strokes
            self._say(score_name(n, hole.par), now, config.HOLED_DISPLAY_S)
            self._set_state("holed", now)
            return
        if sim.outcome == "hazard":
            self.strokes[-1] += config.PIT_PENALTY_STROKES
            self.on_score(float(self.total_strokes))
            self.ball = sim.start
            text = f"Splash!  +{config.PIT_PENALTY_STROKES}"
        else:
            self.ball = (sim.ball.x, sim.ball.y)
            text = "Lipped out!" if sim.lipped else ""
        if self.hole_strokes >= config.MAX_STROKES:
            self.on_score(float(self.total_strokes - self.hole_strokes + config.MAX_STROKES))
            self.strokes[-1] = config.MAX_STROKES
            self._say("Picked up", now, config.HOLED_DISPLAY_S)
            self._set_state("holed", now)
        elif text:
            self._say(text, now, config.RESULT_DISPLAY_S)
            self._set_state("result", now)
        else:
            self._begin_aiming(now)

    # -- for rendering --------------------------------------------------------------
    def distance_to_cup(self):
        if not self.hole:
            return 0.0
        (bx, by), (cx, cy) = self.ball, self.hole.cup
        return ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
