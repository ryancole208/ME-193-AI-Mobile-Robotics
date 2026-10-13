"""
The Dynamic-difficulty opponent: DynamicAgent (rl_agent.py) behind the Opponent interface.

Timeline of one of its shots (s = state, a = action, phi = reward components):
  serve() / poll()        -- choose a in s, play it          (pending: s, a, D)
  shot_result(False)      -- I missed: terminal update        psi(s,a) <- phi_miss
  shot_result(True)       -- I returned it: phi known; the next state s' is built from where
                             my paddle is now and where this shot went
  ball_incoming()         -- choose the reply a' in s' (so the skeleton can wind up toward it)
  poll() hits it back     -- bootstrap update  psi(s,a) <- phi + gamma psi(s', a*)
  poll() misses (my spin) -- the rally ended: terminal update psi(s,a) <- phi
  reset() mid-rally (Esc) -- the unfinished shot is dropped (no update)

Placement is where the ball LANDS on my side: the aim is offset so sidespin curves it onto
the target. The ShotCommand (placement, speed, spin) is what a physical robot would get.
"""

import random
from pathlib import Path

import config
from opponent import MISS, Opponent, Shot, clamp_to_table, spin_misses
from rl_agent import DynamicAgent, components, difficulty_score, make_state, placement_zone
from spin import flight_duration, flight_params, magnus_curve_m


def aim_for_landing(land_x, speed, spin):
    """Aim x so that a ball with this sidespin curves onto land_x."""
    dist = config.OPPONENT_HIT_Y_M - config.PLAYER_HIT_Y_M
    duration = flight_duration(dist, speed, flight_params(spin))
    return clamp_to_table(land_x - magnus_curve_m(spin.side, speed, duration))


class RLOpponent(Opponent):
    name = "dynamic"

    def __init__(self, paddle_x=None, *, agent=None, rng=None, reaction_s=config.OPPONENT_REACTION_S,
                 persist=None, save_path=None):
        self.rng = rng or random.Random()
        self.persist = config.RL_PERSIST if persist is None else persist
        self.save_path = Path(save_path or config.RL_SAVE_PATH)
        self.notice = None
        if agent is None:
            if self.persist:
                agent, self.notice = DynamicAgent.load_or_new(self.save_path, rng=self.rng)
            else:
                agent = DynamicAgent(rng=self.rng)
        self.agent = agent
        self.paddle_x = paddle_x or (lambda: 0.0)
        self.reaction_s = reaction_s
        self.difficulty = config.DYNAMIC_DIFFICULTY
        self.ball_speed = config.DIFFICULTIES[self.difficulty]["ball_speed"]
        self.rally_hits = 0
        self._incoming = None    # IncomingBall: my return on its way
        self._shot = None        # (Decision, D) of the shot flying toward me
        self._awaiting = None    # (s, a, phi, s') after I returned it; finished in poll()
        self._next = None        # Decision for the reply to the incoming ball
        self.last = {}           # for the Dynamic panel

    # -- helpers ------------------------------------------------------------------------------
    def _state(self, prev):
        return make_state(self.paddle_x(), prev, self.rally_hits, self.agent.reward.band())

    def _play(self, decision, start_x):
        cmd = decision.command
        spin = cmd.spin_value
        paddle = self.paddle_x()
        D = difficulty_score(cmd.speed_m_s, cmd.target_x, paddle, spin.magnitude)
        self._shot = (decision, D)
        self.last.update(action=cmd.label(), command=cmd.to_dict(), explored=decision.explored,
                         difficulty=D, beta=decision.beta, q=decision.q)
        return Shot(start_x=start_x, target_x=aim_for_landing(cmd.target_x, cmd.speed_m_s, spin),
                    speed=cmd.speed_m_s, spin=spin)

    def _finish_awaiting(self, bootstrap):
        if self._awaiting is None:
            return
        s, a, phi, s_next = self._awaiting
        self._awaiting = None
        self.agent.update(s, a, phi, s_next if bootstrap else None)

    # -- Opponent interface ----------------------------------------------------------------------
    def reset(self):
        self._incoming = self._shot = self._awaiting = self._next = None

    def serve(self, now):
        self.reset()
        self.rally_hits = 0
        decision = self.agent.choose(self._state("serve"))
        return self._play(decision, clamp_to_table(self.rng.uniform(-0.2, 0.2)))

    def shot_result(self, returned, now):
        if self._shot is None:
            return
        decision, D = self._shot
        self._shot = None
        s, a = decision.state, decision.action
        reward = self.agent.reward
        if returned:
            self.rally_hits += 1
            phi = components(True, D, self.rally_hits)
            reward.record(True)
            self.agent.on_player_hit()
            s_next = self._state(placement_zone(decision.command.placement))
            self._awaiting = (s, a, phi, s_next)
        else:
            phi = components(False, D, self.rally_hits)
            reward.record(False)
            self.agent.on_player_miss()
            self.agent.update(s, a, phi, None)
        self.last.update(returned=returned, reward=reward.reward(phi))

    def ball_incoming(self, ball):
        self._incoming = ball
        if self._awaiting is not None:
            self._next = self.agent.choose(self._awaiting[3])

    def poll(self, now):
        ball = self._incoming
        if ball is None or now < ball.arrival_time + self.reaction_s:
            return None
        self._incoming = None
        missed, _ = spin_misses(self.difficulty, ball.spin, self.rng)
        if missed:
            self._finish_awaiting(bootstrap=False)   # the rally ended: terminal
            self._next = None
            return MISS
        decision = self._next
        if decision is None:   # no plan (e.g. a stray ball): decide now
            decision = self.agent.choose(self._state(placement_zone(self.last.get("command", {})
                                                                    .get("placement", "center"))))
        self._next = None
        self._finish_awaiting(bootstrap=True)
        return self._play(decision, ball.x)

    @property
    def planned_shot(self):
        if self._next is None:
            return None
        cmd = self._next.command
        return cmd.target_x, cmd.spin_value

    # -- learning management --------------------------------------------------------------------
    def checkpoint(self):
        if self.persist:
            try:
                self.agent.save(self.save_path)
            except OSError as exc:
                print(f"[dynamic] could not save the model: {exc}")

    def stop(self):
        self.checkpoint()

    def reset_learning(self):
        self.reset()
        self.agent.reset_learning()
        self.last = {}
        self.notice = "Learning reset: the Dynamic opponent starts from scratch."
        print(f"[dynamic] {self.notice}")
        self.checkpoint()

    def panel_info(self):
        a, r = self.agent, self.agent.reward
        return {"epsilon": a.epsilon, "gamma": a.gamma, "explored": self.last.get("explored"),
                "action": self.last.get("action"), "difficulty": self.last.get("difficulty"),
                "reward": self.last.get("reward"), "beta": self.last.get("beta"),
                "return_rate": r.return_rate, "band": tuple(r.cfg["target_band"]), "band_pos": r.band(),
                "bonus": r.bonus, "penalty": r.penalty, "hits": a.total_hits, "notice": self.notice}
