"""
Dynamic difficulty: a reinforcement-learning shot selector for the opponent.

GOAL: not to beat the player, but to keep them in a challenging-but-winnable zone -- find
the hardest shots they can still return, so the game gets harder as they improve while
rallies keep going.

Pure Python + numpy (no pygame, no ML framework) so it is fully unit-tested and each
decision/update costs well under a millisecond.

Pieces
  * ShotCommand  -- placement x speed x spin labels (+ numbers). The same command could drive
                    the physical robot later.
  * State        -- paddle zone x previous shot x rally length x return-rate band (72 states).
  * RewardModel  -- reward = w . phi, with phi the per-shot component vector
                    (returned, D^p if returned, missed, rally term) and w the band-adjusted
                    weights from config.REWARD.
  * DynamicAgent -- successor-feature Q-learning with factored backoff:
        psi(s, a)   discounted expected sum of each reward component (TD-learned, gamma > 0)
        Q(s, a)   = w . psi(s, a), with the CURRENT weights, so weight changes act at once
        psi(s, a) = beta * psi_joint(s, a) + (1 - beta) * psi_factored(c, a),
                    beta = n / (n + K), n = visits of the joint entry
        psi_factored = psi_P[c, placement] + psi_V[c, speed] + psi_S[c, spin] - 2 psi_0[c]
                    (main effects over a coarse state c = paddle zone x previous shot)
"""

import json
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from spin import NO_SPIN, Spin

SAVE_VERSION = "dynamic-sf-v1"

# -- actions -------------------------------------------------------------------------------
PLACEMENTS = ("left", "center", "right")
SPEEDS = ("slow", "medium", "fast")
SPINS = ("none", "side_left", "side_right", "top", "back")
ACTIONS = tuple((p, v, s) for p in PLACEMENTS for v in SPEEDS for s in SPINS)
N_ACTIONS = len(ACTIONS)
_A_P = np.array([PLACEMENTS.index(a[0]) for a in ACTIONS])
_A_V = np.array([SPEEDS.index(a[1]) for a in ACTIONS])
_A_S = np.array([SPINS.index(a[2]) for a in ACTIONS])
SPIN_LABELS = {"none": "no spin", "side_left": "curve L", "side_right": "curve R", "top": "topspin",
               "back": "backspin"}

# -- reward components ---------------------------------------------------------------------
COMPONENTS = ("returned", "hard_return", "missed", "rally")
N_COMP = len(COMPONENTS)

# -- states --------------------------------------------------------------------------------
ZONES = ("L", "C", "R")                  # my paddle zone when the agent decides
PREVS = ("serve", "L", "C", "R")         # where the agent's previous shot in this rally went
RALLIES = ("short", "long")
RATES = ("below", "in", "above")         # my rolling return rate vs. the target band
N_STATES = len(ZONES) * len(PREVS) * len(RALLIES) * len(RATES)   # 72
N_COARSE = len(ZONES) * len(PREVS)                                # 12


def _clamp01(v):
    return max(0.0, min(1.0, v))


@dataclass(frozen=True)
class ShotCommand:
    """One opponent shot: where it lands (my side), how fast, what spin."""
    placement: str
    speed: str
    spin: str

    @classmethod
    def from_index(cls, i):
        return cls(*ACTIONS[i])

    @property
    def index(self):
        return ACTIONS.index((self.placement, self.speed, self.spin))

    @property
    def target_x(self):
        return config.RL_PLACEMENT_X_M[self.placement]

    @property
    def speed_m_s(self):
        return config.RL_SPEEDS_M_S[self.speed]

    @property
    def spin_value(self):
        k = config.RL_SPIN_AMOUNT
        return {"none": NO_SPIN, "side_left": Spin(side=-k), "side_right": Spin(side=k),
                "top": Spin(top=k), "back": Spin(top=-k)}[self.spin]

    def to_dict(self):
        s = self.spin_value
        return {"placement": self.placement, "speed": self.speed, "spin": self.spin, "target_x": self.target_x,
                "speed_m_s": self.speed_m_s, "spin_side": s.side, "spin_top": s.top}

    def label(self):
        return f"{self.speed} · {self.placement} · {SPIN_LABELS[self.spin]}"


@dataclass(frozen=True)
class State:
    zone: str
    prev: str
    rally: str
    rate: str

    @property
    def index(self):
        i = ZONES.index(self.zone)
        i = i * len(PREVS) + PREVS.index(self.prev)
        i = i * len(RALLIES) + RALLIES.index(self.rally)
        return i * len(RATES) + RATES.index(self.rate)

    @property
    def coarse(self):
        return ZONES.index(self.zone) * len(PREVS) + PREVS.index(self.prev)

    def label(self):
        return f"paddle {self.zone}, prev {self.prev}, {self.rally} rally, {self.rate} band"


def paddle_zone(x, edge=None):
    edge = config.RL_PADDLE_ZONE_M if edge is None else edge
    return "L" if x < -edge else "R" if x > edge else "C"


def placement_zone(placement):
    return {"left": "L", "center": "C", "right": "R"}[placement]


def rally_bucket(rally_len):
    return "long" if rally_len >= config.RL_RALLY_LONG else "short"


def make_state(paddle_x, prev, rally_len, rate):
    return State(paddle_zone(paddle_x), prev, rally_bucket(rally_len), rate)


# =============================================================================
# Reward
# =============================================================================

def difficulty_score(speed_m_s, land_x, paddle_x, spin_magnitude, reward=None):
    """0..1: weighted mix of speed, distance from my paddle and spin strength."""
    r = reward or config.REWARD
    speeds = list(config.RL_SPEEDS_M_S.values())
    lo, hi = min(speeds), max(speeds)
    sn = _clamp01((speed_m_s - lo) / (hi - lo)) if hi > lo else 0.0
    dn = _clamp01(abs(land_x - paddle_x) / r["distance_full_m"])
    pn = _clamp01(spin_magnitude / config.SPIN_MAX)
    w = r["difficulty_weights"]
    total = w["speed"] + w["distance"] + w["spin"]
    return (w["speed"] * sn + w["distance"] * dn + w["spin"] * pn) / total if total > 0 else 0.0


def components(returned, difficulty, rally_len, reward=None):
    """phi: the immediate reward-component vector of one shot (see COMPONENTS)."""
    r = reward or config.REWARD
    if not returned:
        return np.array([0.0, 0.0, 1.0, 0.0])
    cap = max(1, r["rally_cap"])
    return np.array([1.0, _clamp01(difficulty) ** r["difficulty_exponent"], 0.0, min(rally_len, cap) / cap])


class RewardModel:
    """Weights w (reward = w . phi) plus the target-band feedback that adjusts them."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config.REWARD
        self.reset()

    def reset(self):
        self.bonus = float(self.cfg["hard_return_bonus"])
        self.penalty = float(self.cfg["miss_penalty"])
        self.outcomes = deque(maxlen=int(self.cfg["window"]))
        self.since_adjust = 0
        self.last_adjust = "none"

    def weights(self):
        c = self.cfg
        return np.array([c["base_return"], self.bonus, -self.penalty, c["rally_weight"]])

    def reward(self, phi):
        return float(np.dot(self.weights(), phi))

    @property
    def return_rate(self):
        if len(self.outcomes) < self.cfg["min_samples"]:
            return None
        return sum(self.outcomes) / len(self.outcomes)

    def band(self):
        """'below', 'in' or 'above' the target band ('in' until there are enough samples)."""
        rate = self.return_rate
        lo, hi = self.cfg["target_band"]
        if rate is None or lo <= rate <= hi:
            return "in"
        return "above" if rate > hi else "below"

    def record(self, returned):
        self.outcomes.append(1 if returned else 0)
        self.since_adjust += 1
        if self.since_adjust >= self.cfg["adjust_every"]:
            self.since_adjust = 0
            self.adjust()

    def adjust(self):
        """Above the band -> bonus up; below -> penalty up; inside -> both relax to defaults."""
        c = self.cfg
        if self.return_rate is None:
            self.last_adjust = "none"
            return
        lo_m, hi_m = c["weight_limits"]
        b0, p0 = c["hard_return_bonus"], c["miss_penalty"]
        band = self.band()
        if band == "above":
            self.bonus *= 1 + c["adjust_rate"]
        elif band == "below":
            self.penalty *= 1 + c["adjust_rate"]
        else:
            self.bonus += (b0 - self.bonus) * c["relax_rate"]
            self.penalty += (p0 - self.penalty) * c["relax_rate"]
        self.bonus = min(hi_m * b0, max(lo_m * b0, self.bonus))
        self.penalty = min(hi_m * p0, max(lo_m * p0, self.penalty))
        self.last_adjust = band

    def to_dict(self):
        return {"bonus": self.bonus, "penalty": self.penalty, "outcomes": list(self.outcomes),
                "since_adjust": self.since_adjust}

    def load_dict(self, d):
        self.bonus, self.penalty = float(d["bonus"]), float(d["penalty"])
        self.outcomes.clear()
        self.outcomes.extend(int(v) for v in d["outcomes"])
        self.since_adjust = int(d["since_adjust"])


# =============================================================================
# Agent
# =============================================================================

def allowed_actions(epsilon, progressive=None):
    """Action indices playable at this epsilon (harder speeds/spins unlock as it falls)."""
    progressive = config.RL_PROGRESSIVE_UNLOCK if progressive is None else progressive
    if not progressive:
        return np.arange(N_ACTIONS)
    ok = []
    for i, (_, v, s) in enumerate(ACTIONS):
        if v == "fast" and epsilon >= config.RL_UNLOCK_FAST_EPSILON:
            continue
        if s in ("top", "back") and epsilon >= config.RL_UNLOCK_TOPBACK_EPSILON:
            continue
        ok.append(i)
    return np.array(ok)


def fingerprint(gamma, reward=None):
    """Everything that changes what psi means. A saved model with another fingerprint is reset."""
    r = reward or config.REWARD
    return {"actions": ["/".join(a) for a in ACTIONS], "components": list(COMPONENTS),
            "states": [list(ZONES), list(PREVS), list(RALLIES), list(RATES)], "gamma": gamma,
            "difficulty_exponent": r["difficulty_exponent"], "difficulty_weights": dict(r["difficulty_weights"]),
            "distance_full_m": r["distance_full_m"], "rally_cap": r["rally_cap"],
            "paddle_zone_m": config.RL_PADDLE_ZONE_M, "rally_long": config.RL_RALLY_LONG,
            "speeds": dict(config.RL_SPEEDS_M_S), "placements": dict(config.RL_PLACEMENT_X_M),
            "spin_amount": config.RL_SPIN_AMOUNT}


def _canon(obj):
    return json.dumps(obj, sort_keys=True)


@dataclass(frozen=True)
class Decision:
    state: State
    action: int
    explored: bool
    beta: float          # weight of the joint estimate (0 = all separate tables)
    q: float

    @property
    def command(self):
        return ShotCommand.from_index(self.action)


class DynamicAgent:
    def __init__(self, *, gamma=None, alpha=None, backoff_k=None, rng=None, reward=None, allowed_fn=None,
                 epsilon=None):
        self.gamma = config.RL_GAMMA if gamma is None else gamma
        self.alpha = config.RL_ALPHA if alpha is None else alpha
        self.k = config.RL_BACKOFF_K if backoff_k is None else backoff_k
        self.bound = 1.0 / (1.0 - self.gamma) if self.gamma < 1 else 1e6
        self.rng = rng or random.Random()
        self.reward = reward or RewardModel()
        self.allowed_fn = allowed_fn          # state -> action indices (tests); default: unlock rule
        self._epsilon_start = config.RL_EPSILON_START if epsilon is None else epsilon
        self.reset_learning()

    def reset_learning(self):
        z = np.zeros
        self.joint, self.joint_n = z((N_STATES, N_ACTIONS, N_COMP)), z((N_STATES, N_ACTIONS))
        self.head_p, self.head_p_n = z((N_COARSE, len(PLACEMENTS), N_COMP)), z((N_COARSE, len(PLACEMENTS)))
        self.head_v, self.head_v_n = z((N_COARSE, len(SPEEDS), N_COMP)), z((N_COARSE, len(SPEEDS)))
        self.head_s, self.head_s_n = z((N_COARSE, len(SPINS), N_COMP)), z((N_COARSE, len(SPINS)))
        self.base, self.base_n = z((N_COARSE, N_COMP)), z(N_COARSE)
        self.epsilon = self._epsilon_start
        self.hit_counter = 0
        self.total_hits = 0
        self.updates = 0
        self.reward.reset()

    # -- exploration schedule -------------------------------------------------------------
    def on_player_hit(self):
        """Count one of my hits; every RL_HITS_PER_STEP hits epsilon steps toward exploitation."""
        self.total_hits += 1
        self.hit_counter += 1
        if self.hit_counter >= config.RL_HITS_PER_STEP:
            self.hit_counter = 0
            self.epsilon = max(config.RL_EPSILON_MIN, self.epsilon * config.RL_EPSILON_DECAY)
            return True
        return False

    def on_player_miss(self):
        if config.RL_HIT_COUNT_MODE == "rally":
            self.hit_counter = 0

    def allowed(self, state):
        if self.allowed_fn is not None:
            return np.asarray(self.allowed_fn(state))
        return allowed_actions(self.epsilon)

    # -- value estimates --------------------------------------------------------------------
    def factored(self, state):
        """psi from the separate tables for every action: (N_ACTIONS, N_COMP)."""
        c = state.coarse
        b = self.base[c]
        if self.base_n[c] == 0:
            return np.zeros((N_ACTIONS, N_COMP))

        def eff(table, counts):   # an unvisited choice has no main effect: it equals the baseline
            return np.where(counts[c][:, None] > 0, table[c], b)

        P, V, S = eff(self.head_p, self.head_p_n), eff(self.head_v, self.head_v_n), eff(self.head_s, self.head_s_n)
        return np.clip(P[_A_P] + V[_A_V] + S[_A_S] - 2 * b, 0.0, self.bound)

    def psi_hat(self, state):
        """Blended psi for every action and the joint weight beta: ((N_ACTIONS, N_COMP), (N_ACTIONS,))."""
        n = self.joint_n[state.index]
        beta = n / (n + self.k) if self.k > 0 else (n > 0).astype(float)
        psi = beta[:, None] * self.joint[state.index] + (1 - beta[:, None]) * self.factored(state)
        return psi, beta

    def q_values(self, state):
        psi, beta = self.psi_hat(state)
        return psi @ self.reward.weights(), beta

    def greedy(self, state, allowed=None):
        allowed = self.allowed(state) if allowed is None else np.asarray(allowed)
        q, _ = self.q_values(state)
        qa = q[allowed]
        best = np.flatnonzero(qa >= qa.max() - 1e-9)
        return int(allowed[best[self.rng.randrange(len(best))]])

    def choose(self, state):
        allowed = self.allowed(state)
        q, beta = self.q_values(state)
        if self.rng.random() < self.epsilon:
            a, explored = int(allowed[self.rng.randrange(len(allowed))]), True
        else:
            a, explored = self.greedy(state, allowed), False
        return Decision(state, a, explored, float(beta[a]), float(q[a]))

    # -- learning -------------------------------------------------------------------------------
    def update(self, state, action, phi, next_state=None):
        """TD update of psi for every table: target = phi + gamma psi(s', a*) (phi alone if terminal)."""
        phi = np.asarray(phi, dtype=float)
        if next_state is None:
            target = phi.copy()
        else:
            a_star = self.greedy(next_state)
            psi_next, _ = self.psi_hat(next_state)
            target = phi + self.gamma * psi_next[a_star]
        target = np.clip(target, 0.0, self.bound)
        i, c = state.index, state.coarse
        p, v, s = _A_P[action], _A_V[action], _A_S[action]
        for table, counts, idx in ((self.joint, self.joint_n, (i, action)), (self.head_p, self.head_p_n, (c, p)),
                                   (self.head_v, self.head_v_n, (c, v)), (self.head_s, self.head_s_n, (c, s)),
                                   (self.base, self.base_n, c)):
            counts[idx] += 1
            a = max(self.alpha, 1.0 / counts[idx])
            table[idx] += a * (target - table[idx])
        self.updates += 1
        return target

    # -- persistence ------------------------------------------------------------------------------
    def to_dict(self):
        visited = np.argwhere(self.joint_n > 0)
        joint = [[int(si), int(ai), int(self.joint_n[si, ai]), [round(float(x), 6) for x in self.joint[si, ai]]]
                 for si, ai in visited]

        def table(t, n):
            return {"psi": np.round(t, 6).tolist(), "n": n.astype(int).tolist()}

        return {"version": SAVE_VERSION, "fingerprint": fingerprint(self.gamma, self.reward.cfg),
                "epsilon": self.epsilon, "hit_counter": self.hit_counter, "total_hits": self.total_hits,
                "updates": self.updates, "reward": self.reward.to_dict(), "joint": joint,
                "head_p": table(self.head_p, self.head_p_n), "head_v": table(self.head_v, self.head_v_n),
                "head_s": table(self.head_s, self.head_s_n), "base": table(self.base, self.base_n)}

    def load_dict(self, d):
        self.reset_learning()
        for si, ai, n, psi in d["joint"]:
            self.joint_n[si, ai] = n
            self.joint[si, ai] = psi
        for name in ("head_p", "head_v", "head_s", "base"):
            t = getattr(self, name)
            t[...] = np.array(d[name]["psi"], dtype=float).reshape(t.shape)
            n = getattr(self, name + "_n")
            n[...] = np.array(d[name]["n"], dtype=float).reshape(n.shape)
        self.epsilon = float(d["epsilon"])
        self.hit_counter = int(d["hit_counter"])
        self.total_hits = int(d["total_hits"])
        self.updates = int(d["updates"])
        self.reward.load_dict(d["reward"])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load_or_new(cls, path, **kw):
        """(agent, notice). An incompatible/older saved model is never loaded: it is renamed to
        *.bak, a fresh agent is returned, and notice says why."""
        agent = cls(**kw)
        path = Path(path)
        if not path.exists():
            return agent, None
        reason = None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("version") != SAVE_VERSION:
                reason = f"saved version {d.get('version')!r}, this game uses {SAVE_VERSION!r}"
            elif _canon(d.get("fingerprint")) != _canon(fingerprint(agent.gamma, agent.reward.cfg)):
                reason = "it was trained with different settings (gamma / actions / states / difficulty score)"
            else:
                agent.load_dict(d)
                return agent, None
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            reason = f"unreadable ({exc.__class__.__name__})"
            agent.reset_learning()
        bak = path.with_name(path.name + ".bak")
        os.replace(path, bak)
        notice = f"Old Dynamic model not loaded: {reason}. Learning starts fresh (backup: {bak.name})."
        print(f"[dynamic] {notice}")
        return agent, notice

