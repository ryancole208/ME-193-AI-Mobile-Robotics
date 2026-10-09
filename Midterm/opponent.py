"""
The opponent sits behind a small interface so the simulated player can later be
swapped for the physical Arduino UNO Q robot (over MQTT) or an AI player without
touching game.py.

Contract (all times are time.monotonic() seconds, positions in table metres):
  1. set_difficulty(name, ball_speed)   -- before a rally
  2. serve(now) -> Shot                 -- opponent puts a ball in play toward me
  3. ball_incoming(IncomingBall)        -- I returned the ball; tells the opponent where/when
                                           it will reach the opponent's hit plane
  4. poll(now) -> Shot | MISS | None    -- called every frame while the ball travels to the
                                           opponent; return a Shot when it hits back, MISS if
                                           it missed, None while still deciding
A remote/robot opponent would publish in ball_incoming() and answer in poll() from
whatever its MQTT subscription received.

Optional hooks (no-ops by default, so existing opponents need no changes):
  shot_result(returned, now)  -- did I return the opponent's last shot? (learning opponents)
  checkpoint() / reset_learning() -- save / forget learned state
  panel_info() -> dict | None -- extra HUD panel (the Dynamic RL panel)
  planned_shot -> (target_x, Spin) | None -- the reply already chosen, so the skeleton can
                                 wind up toward it before contact

Both Shot and IncomingBall carry an optional spin (spin.Spin, default no spin), so an
opponent that ignores spin needs no changes.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import config
from spin import NO_SPIN, Spin


@dataclass(frozen=True)
class Shot:
    """A ball struck toward the other player."""
    start_x: float   # lateral position where the ball was struck (m)
    target_x: float  # lateral position where it reaches the receiver's hit plane (m)
    speed: float     # speed along the table (m/s)
    spin: Spin = NO_SPIN


@dataclass(frozen=True)
class IncomingBall:
    """My return, as seen by the opponent."""
    x: float              # lateral position at the opponent's hit plane (m)
    arrival_time: float   # when it reaches the opponent's hit plane
    speed: float          # speed along the table (m/s)
    spin: Spin = NO_SPIN  # spin I put on it


class _Miss:
    def __repr__(self):
        return "MISS"


MISS = _Miss()


def clamp_to_table(x, margin=0.08, table_width=config.TABLE_WIDTH_M):
    half = table_width / 2 - margin
    return max(-half, min(half, x))


class Opponent(ABC):
    name = "opponent"

    def start(self):
        """Open connections (e.g. MQTT for a robot). Simulated: nothing to do."""

    def stop(self):
        """Release resources."""

    @property
    def status(self):
        return "ready"

    def set_difficulty(self, name, ball_speed):
        self.difficulty = name
        self.ball_speed = ball_speed

    def reset(self):
        """Forget any in-flight ball (called when a rally ends)."""

    def shot_result(self, returned, now):
        """I returned (True) or missed (False) the opponent's last shot."""

    def checkpoint(self):
        """Persist anything learned (called on Esc-to-menu and at shutdown)."""

    def reset_learning(self):
        """Forget anything learned."""

    def panel_info(self):
        return None

    @property
    def planned_shot(self):
        return None

    @abstractmethod
    def serve(self, now) -> Shot: ...

    @abstractmethod
    def ball_incoming(self, ball: IncomingBall) -> None: ...

    @abstractmethod
    def poll(self, now): ...


class SimulatedOpponent(Opponent):
    """Always reaches the ball; places returns randomly across the table.

    Heavy spin makes it less reliable: extra miss chance and placement scatter, scaled
    by the spin's magnitude and OPPONENT_SPIN_PENALTY[difficulty].
    """

    name = "simulated"

    def __init__(self, rng=None, spread=config.OPPONENT_PLACEMENT_SPREAD_M,
                 reaction_s=config.OPPONENT_REACTION_S, miss_probability=config.OPPONENT_MISS_PROBABILITY):
        self.rng = rng or random.Random()
        self.spread = spread
        self.reaction_s = reaction_s
        self.miss_probability = miss_probability
        self.difficulty = config.DEFAULT_DIFFICULTY
        self.ball_speed = config.DIFFICULTIES[self.difficulty]["ball_speed"]
        self._incoming = None

    def _placement(self):
        return clamp_to_table(self.rng.uniform(-self.spread, self.spread))

    def reset(self):
        self._incoming = None

    def serve(self, now):
        return Shot(start_x=clamp_to_table(self.rng.uniform(-0.2, 0.2)),
                    target_x=self._placement(), speed=self.ball_speed)

    def ball_incoming(self, ball):
        self._incoming = ball

    def poll(self, now):
        ball = self._incoming
        if ball is None or now < ball.arrival_time + self.reaction_s:
            return None
        self._incoming = None
        missed, penalty = spin_misses(self.difficulty, ball.spin, self.rng, self.miss_probability)
        if missed:
            return MISS
        target = self._placement()
        if penalty > 0:
            scatter = penalty * config.OPPONENT_SPIN_SCATTER_M
            target = clamp_to_table(target + self.rng.uniform(-scatter, scatter))
        return Shot(start_x=ball.x, target_x=target, speed=self.ball_speed)

    def spin_penalty(self, spin):
        return spin_penalty(self.difficulty, spin)


def spin_penalty(difficulty, spin):
    """0 (no effect) .. OPPONENT_SPIN_PENALTY[difficulty] (full spin)."""
    factor = config.OPPONENT_SPIN_PENALTY.get(difficulty, 0.0)
    return factor * min(1.0, spin.magnitude / config.SPIN_MAX)


def spin_misses(difficulty, spin, rng, base_miss=0.0):
    """Does the opponent miss a ball with this spin? Returns (missed, penalty)."""
    penalty = spin_penalty(difficulty, spin)
    return rng.random() < base_miss + penalty * config.OPPONENT_SPIN_MISS_MAX, penalty


class ModeOpponent(Opponent):
    """Routes every call to the opponent for the current difficulty (e.g. the RL opponent
    for "Dynamic", the simulated one otherwise). GameLogic only ever sees this one object."""

    def __init__(self, default, by_difficulty=None):
        self.default = default
        self.by_difficulty = dict(by_difficulty or {})
        self.active = default
        self.difficulty = getattr(default, "difficulty", config.DEFAULT_DIFFICULTY)

    @property
    def all(self):
        out = [self.default]
        for o in self.by_difficulty.values():
            if o not in out:
                out.append(o)
        return out

    @property
    def name(self):
        return self.active.name

    @property
    def status(self):
        return self.active.status

    def start(self):
        for o in self.all:
            o.start()

    def stop(self):
        for o in self.all:
            o.stop()

    def set_difficulty(self, name, ball_speed):
        self.difficulty, self.ball_speed = name, ball_speed
        new = self.by_difficulty.get(name, self.default)
        if new is not self.active:
            self.active.reset()
            self.active = new
        new.set_difficulty(name, ball_speed)

    def reset(self):
        self.active.reset()

    def serve(self, now):
        return self.active.serve(now)

    def ball_incoming(self, ball):
        self.active.ball_incoming(ball)

    def poll(self, now):
        return self.active.poll(now)

    def shot_result(self, returned, now):
        self.active.shot_result(returned, now)

    def checkpoint(self):
        for o in self.all:
            o.checkpoint()

    def reset_learning(self):
        for o in self.all:
            o.reset_learning()

    def panel_info(self):
        return self.active.panel_info()

    @property
    def planned_shot(self):
        return self.active.planned_shot
