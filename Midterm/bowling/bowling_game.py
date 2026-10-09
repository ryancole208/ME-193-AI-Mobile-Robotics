"""
Game flow (no pygame -- unit tested).

    menu -> aiming -> (throw) rolling -> result -> aiming ... -> game_over
    game_over -> (Enter) aiming (new game)  or  (Esc) menu

Like Wii Sports Bowling: one bowler, 10 frames, choose Bumpers / No Bumpers first.
While aiming you set where you stand (wrist / ←→) and turn the aim (A/D). A throw
(IMU or Space) launches the ball from where you stood when the swing started, at the
preset aim plus any sideways swing, with speed from swing strength and hook from
the twist at release. The roll is simulated live; when the pins settle they're
counted, the scorecard updates, and the pinsetter respots or resets the rack.
"""

import random

import config
from lane import ALL_PINS, Launch, RollSim, is_split
from scoring import ScoreSheet
from throw_detector import strength_to_speed

STREAK_NAMES = {2: "Double!", 3: "Turkey!"}


class BowlingGame:
    def __init__(self, bowler, *, on_score=None, on_impact=None, rng=None, roll_factory=RollSim):
        self.bowler = bowler
        self.on_score = on_score or (lambda score: None)
        self.on_impact = on_impact or (lambda: None)
        self.rng = rng or random.Random()
        self.roll_factory = roll_factory
        self.state = "menu"
        self.difficulty = config.DEFAULT_DIFFICULTY
        self.sheet = ScoreSheet()
        self.standing = set(ALL_PINS)
        self.roll = None
        self.roll_t0 = 0.0
        self.aim_since = 0.0
        self.strike_streak = 0
        self.best = 0
        self.message = ""
        self.message_until = 0.0
        self.last_throw = None     # dict for the HUD / debug overlay
        self._impact_sent = False

    @property
    def bumpers(self):
        return config.DIFFICULTIES[self.difficulty]["bumpers"]

    def _say(self, text, now, duration):
        self.message, self.message_until = text, now + duration

    # -- commands ---------------------------------------------------------------
    def set_difficulty(self, name):
        if name not in config.DIFFICULTIES:
            raise ValueError(f"unknown difficulty {name!r}")
        if self.state != "menu":
            return False
        self.difficulty = name
        return True

    def start(self, now):
        if self.state not in ("menu", "game_over"):
            return
        self.sheet = ScoreSheet()
        self.strike_streak = 0
        self.last_throw = None
        self.on_score(0.0)
        self._new_rack()
        self._begin_aiming(now)
        self._say(f"Frame 1  -  {self.difficulty}", now, 1.5)

    def to_menu(self):
        self.state = "menu"
        self.roll = None

    def add_throw(self, ev, now):
        """A ThrowEvent from the IMU or keyboard. Ignored unless the bowler is ready."""
        if self.state != "aiming" or ev.t_onset < self.aim_since + config.AIM_ARM_DELAY_S:
            return False
        x = self.bowler.position_at(ev.t_onset)
        angle = self.bowler.launch_angle_deg(ev.t_release)
        speed = strength_to_speed(ev.strength)
        launch = Launch(x=x, angle_deg=angle, speed=speed, spin=ev.spin)
        self.last_throw = {"x": x, "angle_deg": angle, "speed": speed, "spin": ev.spin,
                           "strength": ev.strength, "source": ev.source}
        self.roll = self.roll_factory(launch, self.standing, self.bumpers, self.rng)
        self.roll_t0 = now
        self._impact_sent = False
        self.state = "rolling"
        self.message = ""
        return True

    # -- per-frame update ---------------------------------------------------------
    def update(self, now):
        if self.state == "rolling":
            self.roll.advance_to(now - self.roll_t0)
            hit_t = self.roll.first_contact_t
            if not self._impact_sent and hit_t is not None and now - self.roll_t0 >= hit_t:
                self._impact_sent = True
                self.on_impact()
            if self.roll.finished:
                self._finish_roll(now)
        elif self.state == "result" and now >= self.message_until:
            if self.sheet.game_over:
                self.best = max(self.best, self.sheet.running_total())
                self.state = "game_over"
                self.message = ""
            else:
                if self.sheet.pins_standing() == 10:
                    self._new_rack()
                self._begin_aiming(now)

    def _new_rack(self):
        self.standing = set(ALL_PINS)

    def _begin_aiming(self, now):
        self.state = "aiming"
        self.aim_since = now
        self.roll = None

    def _finish_roll(self, now):
        roll = self.roll
        up_before = len(self.standing)
        knocked = roll.knocked & self.standing
        pins = len(knocked)
        first_ball = up_before == 10
        self.sheet.add_roll(pins, gutter=roll.gutter and pins == 0)
        self.standing -= knocked
        self.on_score(float(self.sheet.running_total()))

        duration = config.RESULT_DISPLAY_S
        if first_ball and pins == 10:
            self.strike_streak += 1
            n = self.strike_streak
            text = "STRIKE!" if n == 1 else f"STRIKE!  {STREAK_NAMES.get(n, f'{n}-Bagger!')}"
            duration = config.STRIKE_DISPLAY_S
        else:
            self.strike_streak = 0
            if not first_ball and pins == up_before:
                text = "SPARE!"
                duration = config.STRIKE_DISPLAY_S
            elif roll.gutter and pins == 0:
                text = "Gutter..."
            elif first_ball and is_split(self.standing):
                text = f"{pins}  -  Split!"
            elif pins == 0:
                text = "Miss"
            else:
                text = f"{pins} pin{'s' if pins != 1 else ''}"
        self._say(text, now, duration)
        self.state = "result"

    # -- for rendering --------------------------------------------------------------
    @property
    def frame_number(self):
        i = self.sheet.current_frame
        return None if i is None else i + 1
