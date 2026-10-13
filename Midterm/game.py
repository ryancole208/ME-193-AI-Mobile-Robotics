"""
Game rules/physics (GameLogic, no pygame -- unit tested). Drawing is in renderer.py.

Coordinates (metres): x lateral (left -, right +), y along the table (0 = my end,
TABLE_LENGTH = opponent end), z height above the table. Ball flights go between the
two hit planes (PLAYER_HIT_Y / OPPONENT_HIT_Y) at a constant speed along y (changed
at the bounce by top/backspin); z is an arc with a bounce on the receiver's side.
Sidespin adds a constant sideways (Magnus) push, so the ball curves. Every flight's
arrival time and landing x are known at launch, which keeps the hit window and the
Opponent interface exact.

Rally flow:  menu -> serve_wait -> to_player -> (hit) to_opponent -> to_player ...
                                       to_player -> (miss) miss -> serve_wait
A hit needs BOTH: a swing whose onset is within [arrival - HIT_WINDOW_BEFORE,
arrival + HIT_WINDOW_AFTER], and the paddle (at swing time) within
LATERAL_HIT_TOLERANCE of the ball's lateral position at arrival.
"""

import math
import random

import config
from opponent import MISS, IncomingBall, Shot, clamp_to_table
from spin import (BOUNCE_FRACTION, NO_SPIN, SECOND_ARC_RATIO, flight_duration, flight_params,
                  magnus_curve_m, spin_for_swing, spin_label)


class Flight:
    """One ball flight from start to end (hit plane to hit plane).

    end_x is where the ball actually arrives; curve_x is how much of that came from
    sidespin (the ball leaves aimed at end_x - curve_x and bends onto end_x).
    """

    def __init__(self, start_x, start_y, end_x, end_y, t0, speed, spin=NO_SPIN, curve_x=0.0):
        self.start_x, self.start_y = start_x, start_y
        self.end_x, self.end_y = end_x, end_y
        self.t0 = t0
        self.speed = speed
        self.spin = spin
        self.curve_x = curve_x
        self.params = p = flight_params(spin)
        self.distance = abs(end_y - start_y)
        self.d_bounce = self.distance * p.bounce_fraction
        self.t_bounce = self.d_bounce / speed
        self.speed_after = speed * p.post_speed_mult
        self.duration = flight_duration(self.distance, speed, p)
        self.t_end = t0 + self.duration

    def fraction(self, t):
        """Fraction of the distance covered at time t (not clamped)."""
        dt = t - self.t0
        if dt <= self.t_bounce:
            d = self.speed * dt
        else:
            d = self.d_bounce + self.speed_after * (dt - self.t_bounce)
        return d / self.distance

    def height(self, f):
        p = self.params
        return ball_height(f, config.BALL_ARC_HEIGHT_M * p.arc_mult, p.bounce_fraction, p.second_arc_ratio)

    def position(self, t, clamp_end=False):
        f = max(0.0, self.fraction(t))
        if clamp_end:
            f = min(f, 1.0)
        aim = self.end_x - self.curve_x
        x = self.start_x + (aim - self.start_x) * f + self.curve_x * f * f
        y = self.start_y + (self.end_y - self.start_y) * f
        return x, y, self.height(f)


def ball_height(f, h=config.BALL_ARC_HEIGHT_M, bounce=BOUNCE_FRACTION, second=SECOND_ARC_RATIO):
    if f <= bounce:
        u = f / bounce
        return 4 * h * u * (1 - u)
    u = (f - bounce) / (2 * (1 - bounce))  # receiver strikes at the arc's apex
    return max(0.0, 4 * h * second * u * (1 - u))


def make_flight(start_x, start_y, aim_x, end_y, t0, speed, spin=NO_SPIN):
    """Flight aimed at aim_x; sidespin bends it, but never off the table."""
    duration = flight_duration(abs(end_y - start_y), speed, flight_params(spin))
    end_x = clamp_to_table(aim_x + magnus_curve_m(spin.side, speed, duration))
    return Flight(start_x, start_y, end_x, end_y, t0, speed, spin, curve_x=end_x - aim_x)


def return_angle_deg(direction, rng):
    a = config.RETURN_ANGLE_LEFT_RIGHT_DEG
    j = config.RETURN_ANGLE_CENTER_JITTER_DEG
    angle = {"left": -a, "right": a}.get(direction)
    if angle is None:
        angle = rng.uniform(-j, j)
    m = config.RETURN_ANGLE_MAX_DEG
    return max(-m, min(m, angle))


def return_speed(base_speed, strength):
    mult = 1 + config.RETURN_STRENGTH_GAIN * (strength - 1)
    mult = max(config.RETURN_SPEED_MIN_MULT, min(config.RETURN_SPEED_MAX_MULT, mult))
    return base_speed * mult


class GameLogic:
    def __init__(self, opponent, player_input, *, on_score=None, on_hit=None, rng=None):
        self.opponent = opponent
        self.input = player_input
        self.on_score = on_score or (lambda score: None)
        self.on_hit = on_hit or (lambda: None)
        self.rng = rng or random.Random()
        self.state = "menu"
        self.difficulty = config.DEFAULT_DIFFICULTY
        self.streak = 0
        self.best = 0
        self.flight = None
        self.prev_flight = None      # shown until a delayed flight (t0 in the future) starts
        self.swings = []
        self.serve_at = 0.0
        self.message = ""
        self.message_until = 0.0
        self.last_result = None      # dict describing the last hit/miss decision (debug overlay)
        self.opponent_x = 0.0
        self.event_times = {}        # kind -> time, for animations: serve, player_hit, player_miss,
                                     # opponent_hit, opponent_miss
        self.spin_text = ""
        self.spin_text_until = 0.0
        self.opponent.set_difficulty(self.difficulty, self.ball_speed)

    # -- helpers ----------------------------------------------------------------
    @property
    def ball_speed(self):
        return config.DIFFICULTIES[self.difficulty]["ball_speed"]

    def _set_streak(self, value):
        if value != self.streak:
            self.streak = value
            self.best = max(self.best, value)
            self.on_score(float(value))

    def _say(self, text, now, duration=1.0):
        self.message, self.message_until = text, now + duration

    def _event(self, kind, t):
        self.event_times[kind] = t

    # -- commands ---------------------------------------------------------------
    def set_difficulty(self, name):
        if name not in config.DIFFICULTIES:
            raise ValueError(f"unknown difficulty {name!r}")
        if self.state != "menu":
            return False
        self.difficulty = name
        self.opponent.set_difficulty(name, self.ball_speed)
        return True

    def start(self, now):
        if self.state != "menu":
            return
        self._set_streak(0)
        self.opponent.reset()
        self.flight = self.prev_flight = None
        self.swings.clear()
        self.state = "serve_wait"
        self.serve_at = now + config.SERVE_DELAY_S
        self._say("Get ready!", now, config.SERVE_DELAY_S)

    def to_menu(self):
        self.state = "menu"
        self.flight = self.prev_flight = None
        self.swings.clear()
        self.opponent.reset()

    def add_swing(self, swing):
        if self.state in ("to_player", "serve_wait", "to_opponent"):
            self.swings.append(swing)

    # -- per-frame update -------------------------------------------------------
    def update(self, now):
        if self.state == "serve_wait" and now >= self.serve_at:
            self._serve(now)
        elif self.state == "to_player":
            self._resolve_player(now)
        elif self.state == "to_opponent":
            self._resolve_opponent(now)
        elif self.state == "miss" and now >= self.message_until:
            self.state = "serve_wait"
            self.serve_at = now + config.SERVE_DELAY_S

    def _serve(self, now):
        shot = self.opponent.serve(now)
        self._launch_to_player(shot, now)
        self._event("serve", now)

    def _launch_to_player(self, shot, t0):
        self.prev_flight = self.flight
        self.opponent_x = shot.start_x
        self.flight = make_flight(shot.start_x, config.OPPONENT_HIT_Y_M, shot.target_x,
                                  config.PLAYER_HIT_Y_M, t0, shot.speed, getattr(shot, "spin", NO_SPIN))
        self._event("opponent_hit", t0)
        self.swings = [s for s in self.swings if s.t >= self.flight.t_end - config.HIT_WINDOW_BEFORE_S]
        self.state = "to_player"

    def _resolve_player(self, now):
        f = self.flight
        lo = f.t_end - config.HIT_WINDOW_BEFORE_S
        hi = f.t_end + config.HIT_WINDOW_AFTER_S
        self.swings = [s for s in sorted(self.swings, key=lambda s: s.t) if s.t >= lo]
        in_window = [s for s in self.swings if s.t <= hi]
        if in_window:
            swing = in_window[0]
            self.swings.remove(swing)
            paddle = self.input.paddle_x_at(swing.t)
            if paddle is None:
                paddle = self.input.current_paddle_x()
            err = paddle - f.end_x
            result = {"timing_s": swing.t - f.t_end, "lateral_err_m": err, "strength": swing.strength,
                      "source": swing.source}
            if abs(err) <= config.LATERAL_HIT_TOLERANCE_M:
                self._player_hit(swing, now, result)
            else:
                side = "left" if err > 0 else "right"
                self._player_miss(now, f"out of position (ball to your {side})", result)
            return
        # Swing events arrive SWING_PEAK_WINDOW_S after onset; wait that long past the window.
        if now > hi + config.SWING_PEAK_WINDOW_S + 0.05:
            self._player_miss(now, "no swing in time", {"timing_s": None})

    def _player_hit(self, swing, now, result):
        f = self.flight
        direction = self.input.direction_at(swing.t)
        angle = return_angle_deg(direction, self.rng)
        speed = return_speed(self.ball_speed, swing.strength)
        dist = config.OPPONENT_HIT_Y_M - config.PLAYER_HIT_Y_M
        aim = clamp_to_table(f.end_x + math.tan(math.radians(angle)) * dist)
        spin = spin_for_swing(swing)

        self.prev_flight = f
        t0 = max(now, f.t_end)  # an early swing returns the ball when it actually arrives
        self.flight = make_flight(f.end_x, config.PLAYER_HIT_Y_M, aim, config.OPPONENT_HIT_Y_M, t0, speed, spin)
        target = self.flight.end_x
        result.update(hit=True, direction=direction, angle_deg=angle, speed=speed,
                      spin_side=spin.side, spin_top=spin.top, curve_m=self.flight.curve_x)
        self.last_result = result
        self.opponent.shot_result(True, now)
        self.opponent.ball_incoming(IncomingBall(x=target, arrival_time=self.flight.t_end, speed=speed,
                                                 spin=spin))
        self.state = "to_opponent"
        self._set_streak(self.streak + 1)
        self._say(f"HIT! {direction}", now, 0.6)
        label = spin_label(spin)
        if label:
            self.spin_text, self.spin_text_until = label, now + config.SPIN_LABEL_S
        self._event("player_hit", t0)
        self.on_hit()

    def _player_miss(self, now, reason, result):
        result.update(hit=False, reason=reason)
        self.last_result = result
        self.opponent.shot_result(False, now)
        self.opponent.reset()
        self._set_streak(0)
        self.state = "miss"
        self._say(f"MISS - {reason}", now, config.MISS_DISPLAY_S)
        self._event("player_miss", now)

    def _resolve_opponent(self, now):
        res = self.opponent.poll(now)
        if isinstance(res, Shot):
            self._launch_to_player(res, max(now, self.flight.t_end))
        elif res is MISS or now > self.flight.t_end + config.OPPONENT_TIMEOUT_S:
            self.opponent.reset()
            self.state = "miss"
            self._say("Opponent missed!", now, config.MISS_DISPLAY_S)
            self._event("opponent_miss", now)

    # -- for rendering ----------------------------------------------------------
    def ball_position(self, now):
        if self.flight is None or self.state in ("menu", "serve_wait"):
            return None
        if now < self.flight.t0 and self.prev_flight is not None:
            return self.prev_flight.position(now, clamp_end=True)
        # Hold the ball at my paddle plane while a late swing is still being resolved.
        return self.flight.position(now, clamp_end=self.state == "to_player")

    def ball_spin(self, now):
        """Spin of the ball currently shown (for the spinning marking and trail tint)."""
        if self.flight is None:
            return NO_SPIN
        if now < self.flight.t0 and self.prev_flight is not None:
            return self.prev_flight.spin
        return self.flight.spin

    def time_to_arrival(self, now):
        if self.state != "to_player" or self.flight is None:
            return None
        return self.flight.t_end - now


# =============================================================================
# Status lights (also used by ../bowling and ../minigolf)
# =============================================================================

GREEN, YELLOW, RED, GREY = (70, 200, 110), (235, 190, 60), (225, 80, 70), (130, 130, 140)


def status_color(kind, status):
    s = (status or "").lower()
    if s.startswith("disabled") or s in ("idle", "stopped"):
        return GREY
    good = {"motor": ("connected",), "mqtt": ("connected",), "pose": ("tracking",)}[kind]
    if s.startswith(good):
        return GREEN
    if any(w in s for w in ("connecting", "scanning", "calibrating", "loading", "opening", "no person",
                            "reconnecting", "retrying")):
        return YELLOW
    return RED
