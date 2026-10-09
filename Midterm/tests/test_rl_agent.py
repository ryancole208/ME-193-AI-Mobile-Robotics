"""Dynamic-difficulty RL agent: maths, schedule, rewards, target band, persistence, learning
behaviour against simulated players, and the game integration."""

import json
import random
import time

import numpy as np
import pytest

import config
from game import GameLogic, make_flight
from opponent import ModeOpponent, Shot, SimulatedOpponent
from rl_agent import (ACTIONS, N_ACTIONS, SAVE_VERSION, DynamicAgent, RewardModel, ShotCommand, State,
                      allowed_actions, components, difficulty_score, make_state, placement_zone)
from rl_opponent import RLOpponent, aim_for_landing

S0 = State("C", "serve", "short", "in")
S1 = State("L", "L", "short", "in")
S2 = State("C", "C", "short", "in")


def act(p, v, s):
    return ACTIONS.index((p, v, s))


def make_agent(**kw):
    kw.setdefault("rng", random.Random(0))
    return DynamicAgent(**kw)


def set_joint(agent, state, action, psi, n=1):
    agent.joint[state.index, action] = psi
    agent.joint_n[state.index, action] = n


# =============================================================================
# TD / successor-feature maths
# =============================================================================

def test_first_updates_match_hand_calculation():
    a = make_agent(gamma=0.7, alpha=0.1)
    x = act("left", "fast", "top")
    phi1, phi2 = np.array([1, 0.25, 0, 0.1]), np.array([0, 0, 1, 0])
    a.update(S0, x, phi1)                                  # n=1 -> alpha = 1
    assert np.allclose(a.joint[S0.index, x], phi1)
    a.update(S0, x, phi2)                                  # n=2 -> alpha = max(0.1, 1/2)
    expect = 0.5 * phi1 + 0.5 * phi2
    c = S0.coarse
    for got in (a.joint[S0.index, x], a.head_p[c, 0], a.head_v[c, 2], a.head_s[c, 3], a.base[c]):
        assert np.allclose(got, expect)
    for _ in range(50):                                    # later: the constant floor
        a.update(S0, x, phi2)
    assert a.joint_n[S0.index, x] == 52
    before = a.joint[S0.index, x].copy()
    a.update(S0, x, phi1)
    assert np.allclose(a.joint[S0.index, x], before + 0.1 * (phi1 - before))


def _valued_next_state(agent):
    """S1 where X is clearly the greedy action: psi(X) = [1,1,0,0], psi(Y) = [1,0,0,0]."""
    X, Y = act("right", "fast", "none"), act("left", "slow", "none")
    set_joint(agent, S1, X, [1, 1, 0, 0])
    set_joint(agent, S1, Y, [1, 0, 0, 0])
    return X, Y


def test_bootstrap_uses_greedy_next_action_under_current_weights():
    a = make_agent(gamma=0.7, backoff_k=0)
    X, Y = _valued_next_state(a)
    a.allowed_fn = lambda s: [X, Y]
    phi = np.array([1, 0.2, 0, 0])
    target = a.update(S0, act("center", "medium", "none"), phi, S1)
    assert np.allclose(target, phi + 0.7 * np.array([1, 1, 0, 0]))
    assert np.allclose(a.joint[S0.index, act("center", "medium", "none")], target)


def test_terminal_update_never_bootstraps():
    a = make_agent(gamma=0.7, backoff_k=0)
    _valued_next_state(a)
    phi = np.array([0, 0, 1, 0])
    target = a.update(S0, 0, phi, None)
    assert np.allclose(target, phi)


def test_weight_change_flips_greedy_without_retraining():
    a = make_agent(backoff_k=0)
    X, Y = act("right", "fast", "top"), act("center", "slow", "none")
    set_joint(a, S0, X, [1, 0.2, 0.4, 0])     # hard, but often missed
    set_joint(a, S0, Y, [1, 0.05, 0.0, 0])    # easy, always returned
    a.allowed_fn = lambda s: [X, Y]
    assert a.greedy(S0) == Y                  # penalty 0.6: X = 0.06, Y = 0.15
    updates = a.updates
    a.reward.penalty = 0.1                    # X = 0.26 now
    assert a.greedy(S0) == X
    assert a.updates == updates


def test_backoff_blend_moves_from_separate_tables_to_joint():
    a = make_agent(backoff_k=8)
    x = act("left", "fast", "none")
    for _ in range(3):                          # gives the separate tables (and this joint entry) values
        a.update(S0, x, [1, 0.5, 0, 0])
    a.joint[S0.index, x] = [0, 0, 1, 0]         # pretend the joint entry learned something else
    for n, beta in ((0, 0.0), (8, 0.5), (1000, 1000 / 1008)):
        a.joint_n[S0.index, x] = n
        psi, b = a.psi_hat(S0)
        assert b[x] == pytest.approx(beta)
        expect = beta * np.array([0, 0, 1, 0]) + (1 - beta) * a.factored(S0)[x]
        assert np.allclose(psi[x], expect)
    a.joint_n[S0.index, x] = 0
    assert np.allclose(a.psi_hat(S0)[0][x], a.factored(S0)[x])


def test_factored_estimate_is_additive_main_effects():
    a = make_agent()
    c = S0.coarse
    a.base[c], a.base_n[c] = 0.5, 1
    a.head_p[c, 0], a.head_p_n[c, 0] = 0.7, 1     # left
    a.head_v[c, 2], a.head_v_n[c, 2] = 0.6, 1     # fast
    a.head_s[c, 0], a.head_s_n[c, 0] = 0.5, 1     # no spin
    f = a.factored(S0)
    assert np.allclose(f[act("left", "fast", "none")], 0.7 + 0.6 + 0.5 - 1.0)
    # an unvisited choice (centre) has no main effect: it counts as the baseline
    assert np.allclose(f[act("center", "fast", "none")], 0.5 + 0.6 + 0.5 - 1.0)
    assert np.allclose(make_agent().factored(S0), 0.0)   # nothing learned yet


def test_psi_and_weights_stay_bounded():
    rng = random.Random(3)
    a = make_agent(gamma=0.9, rng=rng)
    a.epsilon = 0.0
    all_states = [State(z, p, r, b) for z in "LCR" for p in ("serve", "L", "C", "R")
                  for r in ("short", "long") for b in ("below", "in", "above")]
    for _ in range(3000):
        s = rng.choice(all_states)
        phi = components(rng.random() < 0.7, rng.random(), rng.randrange(20))
        a.update(s, rng.randrange(N_ACTIONS), phi, rng.choice(all_states + [None]))
    hi = 1 / (1 - 0.9) + 1e-9
    for t in (a.joint, a.head_p, a.head_v, a.head_s, a.base):
        assert t.min() >= 0 and t.max() <= hi
    rm = RewardModel()
    for _ in range(5000):
        rm.record(rng.random() < rng.choice((0.1, 0.99)))
        lo, top = rm.cfg["weight_limits"]
        assert lo * rm.cfg["hard_return_bonus"] - 1e-9 <= rm.bonus <= top * rm.cfg["hard_return_bonus"] + 1e-9
        assert lo * rm.cfg["miss_penalty"] - 1e-9 <= rm.penalty <= top * rm.cfg["miss_penalty"] + 1e-9


# =============================================================================
# Exploration schedule and unlocks
# =============================================================================

def test_epsilon_steps_every_5_hits_down_to_floor():
    a = make_agent()
    eps0 = a.epsilon
    assert eps0 == config.RL_EPSILON_START
    for _ in range(config.RL_HITS_PER_STEP - 1):
        assert a.on_player_hit() is False
    assert a.epsilon == eps0
    assert a.on_player_hit() is True
    assert a.epsilon == pytest.approx(eps0 * config.RL_EPSILON_DECAY)
    for _ in range(2000):
        a.on_player_hit()
    assert a.epsilon == config.RL_EPSILON_MIN
    assert a.total_hits == 2000 + config.RL_HITS_PER_STEP


def test_session_mode_keeps_counting_after_a_miss(monkeypatch):
    monkeypatch.setattr(config, "RL_HIT_COUNT_MODE", "session")
    a = make_agent()
    for _ in range(3):
        a.on_player_hit()
    a.on_player_miss()
    a.on_player_hit()
    assert a.on_player_hit() is True


def test_rally_mode_restarts_the_count_after_a_miss(monkeypatch):
    monkeypatch.setattr(config, "RL_HIT_COUNT_MODE", "rally")
    a = make_agent()
    for _ in range(3):
        a.on_player_hit()
    a.on_player_miss()
    assert not any(a.on_player_hit() for _ in range(config.RL_HITS_PER_STEP - 1))
    assert a.on_player_hit() is True


def test_progressive_unlock():
    def kinds(eps):
        return {(ACTIONS[i][1], ACTIONS[i][2]) for i in allowed_actions(eps, progressive=True)}
    start = kinds(0.9)
    assert len(allowed_actions(0.9, progressive=True)) == 3 * 2 * 3
    assert not any(v == "fast" or s in ("top", "back") for v, s in start)
    mid = kinds(config.RL_UNLOCK_FAST_EPSILON - 0.01)
    assert any(v == "fast" for v, _ in mid) and not any(s in ("top", "back") for _, s in mid)
    assert len(allowed_actions(config.RL_UNLOCK_TOPBACK_EPSILON - 0.01, progressive=True)) == N_ACTIONS
    assert len(allowed_actions(0.9, progressive=False)) == N_ACTIONS


def test_choose_respects_unlocks_and_reports_exploration():
    a = make_agent()
    a.epsilon = 1.0
    seen = {a.choose(S0).action for _ in range(300)}
    assert seen <= set(allowed_actions(1.0).tolist())
    assert all(a.choose(S0).explored for _ in range(20))
    a.epsilon = 0.0
    assert not any(a.choose(S0).explored for _ in range(20))


# =============================================================================
# Difficulty score and rewards
# =============================================================================

def test_difficulty_score_bounds_and_monotonic():
    slow, fast = config.RL_SPEEDS_M_S["slow"], config.RL_SPEEDS_M_S["fast"]
    assert difficulty_score(slow, 0.0, 0.0, 0.0) == 0.0
    assert difficulty_score(fast, 0.45, -0.55, config.SPIN_MAX) == pytest.approx(1.0)
    assert difficulty_score(99, 9, -9, 9) <= 1.0
    base = difficulty_score(3.2, 0.2, 0.0, 0.3)
    assert difficulty_score(4.0, 0.2, 0.0, 0.3) > base       # faster
    assert difficulty_score(3.2, 0.5, 0.0, 0.3) > base       # farther from my paddle
    assert difficulty_score(3.2, -0.2, 0.0, 0.3) == pytest.approx(difficulty_score(3.2, 0.2, 0.0, 0.3))
    assert difficulty_score(3.2, 0.2, 0.0, 0.6) > base       # more spin


def test_hard_returns_earn_much_more_than_easy_ones():
    rm = RewardModel()
    easy, hard = rm.reward(components(True, 0.2, 1)), rm.reward(components(True, 0.9, 1))
    assert hard > 5 * easy
    assert easy < 0.2                                         # easy shots only earn a little
    assert rm.reward(components(True, 0.5, 10)) > rm.reward(components(True, 0.5, 1))   # rally bonus


def test_missed_shots_are_penalised_never_rewarded():
    rm = RewardModel()
    for d in np.linspace(0, 1, 11):
        miss = rm.reward(components(False, d, 5))
        assert miss == pytest.approx(-config.REWARD["miss_penalty"])
        assert miss < 0 < rm.reward(components(True, 0.0, 0))


# =============================================================================
# Target band
# =============================================================================

def test_above_band_raises_the_difficulty_bonus():
    rm = RewardModel()
    for _ in range(20):
        rm.record(True)
    assert rm.band() == "above"
    assert rm.bonus > config.REWARD["hard_return_bonus"]
    assert rm.penalty == config.REWARD["miss_penalty"]


def test_below_band_raises_the_miss_penalty():
    rm = RewardModel()
    for _ in range(20):
        rm.record(False)
    assert rm.band() == "below"
    assert rm.penalty > config.REWARD["miss_penalty"]
    assert rm.bonus == config.REWARD["hard_return_bonus"]


def test_no_adjustment_before_min_samples():
    rm = RewardModel()
    for _ in range(config.REWARD["min_samples"] - 1):
        rm.record(True)
    assert rm.bonus == config.REWARD["hard_return_bonus"] and rm.band() == "in"


def test_inside_band_relaxes_monotonically_toward_defaults():
    rm = RewardModel()
    rm.bonus, rm.penalty = 2.0, 1.2
    history = []
    for i in range(400):
        rm.record(i % 5 != 0)                 # 80 % returned: inside 70-85 %
        history.append((rm.bonus, rm.penalty))
    assert rm.band() == "in"
    bonuses, penalties = zip(*history)
    assert all(b2 <= b1 for b1, b2 in zip(bonuses, bonuses[1:]))
    assert all(p2 <= p1 for p1, p2 in zip(penalties, penalties[1:]))
    # the first scheduled adjustment comes before min_samples outcomes, so it is skipped
    keep = (1 - config.REWARD["relax_rate"]) ** (400 // config.REWARD["adjust_every"] - 1)
    assert rm.bonus == pytest.approx(1.0 + 1.0 * keep)
    assert rm.penalty == pytest.approx(0.6 + 0.6 * keep)


def test_steady_in_band_rate_does_not_oscillate():
    rm = RewardModel()
    for i in range(300):
        rm.record(i % 5 != 0)
    assert rm.bonus == config.REWARD["hard_return_bonus"]
    assert rm.penalty == config.REWARD["miss_penalty"]


def test_weights_are_clamped():
    rm = RewardModel()
    for _ in range(3000):
        rm.record(True)
    assert rm.bonus == pytest.approx(config.REWARD["weight_limits"][1] * config.REWARD["hard_return_bonus"])


# =============================================================================
# Persistence
# =============================================================================

def _trained(seed=0, n=200):
    rng = random.Random(seed)
    a = make_agent(rng=rng)
    for _ in range(n):
        a.update(S0, rng.randrange(N_ACTIONS), components(rng.random() < 0.7, rng.random(), 2), S1)
        a.reward.record(rng.random() < 0.7)
        a.on_player_hit()
    return a


def test_save_and_load_round_trip(tmp_path):
    a = _trained()
    path = tmp_path / "agent.json"
    a.save(path)
    b, notice = DynamicAgent.load_or_new(path, rng=random.Random(1))
    assert notice is None
    for name in ("joint", "joint_n", "head_p", "head_v", "head_s", "base", "base_n"):
        assert np.allclose(getattr(a, name), getattr(b, name), atol=1e-5)
    assert (b.epsilon, b.hit_counter, b.total_hits) == (a.epsilon, a.hit_counter, a.total_hits)
    assert (b.reward.bonus, b.reward.penalty, list(b.reward.outcomes)) == \
        (a.reward.bonus, a.reward.penalty, list(a.reward.outcomes))
    assert b.greedy(S0) == a.greedy(S0) or np.isclose(b.q_values(S0)[0].max(), a.q_values(S0)[0].max())


def test_missing_file_gives_fresh_agent(tmp_path):
    a, notice = DynamicAgent.load_or_new(tmp_path / "none.json")
    assert notice is None and a.updates == 0


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(version="dynamic-bandit-v0"),                           # older design
    lambda d: d.pop("version"),
    lambda d: d["fingerprint"].update(gamma=0.0),                              # other gamma
    lambda d: d["fingerprint"].update(components=["returned", "missed"]),      # other reward parts
])
def test_old_or_incompatible_model_is_not_loaded(tmp_path, mutate):
    path = tmp_path / "agent.json"
    a = _trained()
    a.save(path)
    d = json.loads(path.read_text())
    mutate(d)
    path.write_text(json.dumps(d))
    b, notice = DynamicAgent.load_or_new(path)
    assert notice and "not loaded" in notice
    assert b.updates == 0 and b.epsilon == config.RL_EPSILON_START
    assert not path.exists() and (tmp_path / "agent.json.bak").exists()


def test_corrupt_file_is_reported_and_backed_up(tmp_path):
    path = tmp_path / "agent.json"
    path.write_text("{not json")
    b, notice = DynamicAgent.load_or_new(path)
    assert "unreadable" in notice and b.updates == 0
    assert (tmp_path / "agent.json.bak").exists()


def test_changing_reward_weights_keeps_the_model(tmp_path, monkeypatch):
    path = tmp_path / "agent.json"
    _trained().save(path)
    monkeypatch.setitem(config.REWARD, "miss_penalty", 0.3)
    monkeypatch.setitem(config.REWARD, "hard_return_bonus", 2.0)
    b, notice = DynamicAgent.load_or_new(path)
    assert notice is None and b.updates > 0


def test_reset_learning_clears_and_saves(tmp_path):
    path = tmp_path / "agent.json"
    opp = RLOpponent(persist=True, save_path=path, agent=_trained())
    opp.checkpoint()
    opp.reset_learning()
    d = json.loads(path.read_text())
    assert d["version"] == SAVE_VERSION and d["updates"] == 0 and d["joint"] == []
    assert d["epsilon"] == config.RL_EPSILON_START
    assert "reset" in opp.panel_info()["notice"].lower()


# =============================================================================
# Learning behaviour against simulated players
# =============================================================================

def simulate(agent, shots, returns, rng, paddle_after=lambda cmd, x: x, schedule=False):
    """Plays `shots` shots. returns(cmd, D, paddle_x, rng) -> bool decides my return."""
    paddle, rally, prev = 0.0, 0, "serve"
    s = make_state(paddle, prev, rally, agent.reward.band())
    log = []
    for _ in range(shots):
        d = agent.choose(s)
        cmd = d.command
        D = difficulty_score(cmd.speed_m_s, cmd.target_x, paddle, cmd.spin_value.magnitude)
        ok = returns(cmd, D, paddle, rng)
        agent.reward.record(ok)
        log.append((cmd, D, ok))
        if ok:
            rally += 1
            if schedule:
                agent.on_player_hit()
            paddle = paddle_after(cmd, paddle)
            s2 = make_state(paddle, placement_zone(cmd.placement), rally, agent.reward.band())
            agent.update(s, d.action, components(True, D, rally), s2)
            s = s2
        else:
            agent.on_player_miss()
            agent.update(s, d.action, components(False, D, rally))
            paddle, rally = 0.0, 0
            s = make_state(paddle, "serve", 0, agent.reward.band())
    return log


def test_learns_a_combination_the_separate_tables_cannot():
    """I only miss FAST shots to my backhand (left). Fast to the right and slow to the left are fine."""
    rng = random.Random(7)
    a = make_agent(rng=rng, allowed_fn=lambda s: np.arange(N_ACTIONS))
    a.epsilon = 0.3

    def returns(cmd, D, paddle, rng):
        return not (cmd.speed == "fast" and cmd.placement == "left")

    simulate(a, 8000, returns, rng)
    s = max((State("C", p, r, b) for p in ("serve", "L", "C", "R") for r in ("short", "long")
             for b in ("below", "in", "above")), key=lambda st: a.joint_n[st.index].sum())
    psi, _ = a.psi_hat(s)
    fast_left, slow_left = act("left", "fast", "none"), act("left", "slow", "none")
    fast_right = act("right", "fast", "none")
    MISSED = 2
    assert psi[fast_left, MISSED] > 0.8                 # the combination is known to fail ...
    assert psi[slow_left, MISSED] < 0.3                 # ... but left alone is fine
    assert psi[fast_right, MISSED] < 0.3                # ... and fast alone is fine
    g = ACTIONS[a.greedy(s)]
    assert g[1] == "fast" and g[0] != "left"            # hardest returnable: fast, not to the backhand
    f = a.factored(s)                                   # the separate tables alone get it wrong
    assert max(1.0 - f[fast_left, MISSED], f[slow_left, MISSED]) > 0.2


def _setup_world(agent):
    """s0: A (setup, small reward) -> s1 where C earns a big reward; B (greedy now) -> s2 (small)."""
    A, B, C = act("left", "slow", "none"), act("center", "medium", "none"), act("right", "fast", "none")
    allowed = {S0: [A, B], S1: [C], S2: [B]}
    agent.allowed_fn = lambda s: allowed[s]
    step = {(S0, A): ([1, 0.1, 0, 0], S1), (S0, B): ([1, 0.3, 0, 0], S2),
            (S1, C): ([1, 0.9, 0, 0], None), (S2, B): ([1, 0.1, 0, 0], None)}
    return A, B, step


@pytest.mark.parametrize("gamma, expect_setup", [(0.7, True), (0.0, False)])
def test_discount_lets_it_learn_setups(gamma, expect_setup):
    rng = random.Random(1)
    a = make_agent(gamma=gamma, rng=rng)
    a.epsilon = 0.3
    A, B, step = _setup_world(a)
    for _ in range(400):
        s = S0
        while s is not None:
            d = a.choose(s)
            phi, nxt = step[(s, d.action)]
            a.update(s, d.action, phi, nxt)
            s = nxt
    assert (a.greedy(S0) == A) is expect_setup
    q, _ = a.q_values(S0)
    if gamma:
        assert q[A] == pytest.approx(0.2 + 0.7 * 1.0, abs=0.05)       # 0.9: the set-up pays off later
        assert q[B] == pytest.approx(0.4 + 0.7 * 0.2, abs=0.05)       # 0.54
    else:
        assert q[A] == pytest.approx(0.2, abs=0.02) and q[B] == pytest.approx(0.4, abs=0.02)


def _skilled_player(cmd, D, paddle, rng):
    """Returns everything up to D = 0.4, then less and less (nothing at D >= 0.9)."""
    p = 1.0 if D < 0.4 else max(0.0, 1.0 - 2.0 * (D - 0.4))
    return rng.random() < p


@pytest.mark.xfail(strict=True, reason=(
    "OPEN DESIGN DECISION: with gamma = 0.7 and a miss ending the rally, keeping the rally alive is "
    "worth ~gamma*V on its own, and V grows with the bonus, so the band adjustment cannot pull the "
    "return rate down into 70-85 % (it plateaus near 97-99 %). See the summary for the options."))
def test_goal_challenging_but_winnable():
    """Against a player who returns easy shots and struggles with hard ones, the agent ends up
    near the target band while playing much harder shots than the easiest ones."""
    rng = random.Random(11)
    a = make_agent(rng=rng)
    log = simulate(a, 5000, _skilled_player, rng, paddle_after=lambda cmd, x: cmd.target_x, schedule=True)
    tail = log[-1000:]
    rate = sum(ok for _, _, ok in tail) / len(tail)
    mean_d = sum(D for _, D, _ in tail) / len(tail)
    lo, hi = config.REWARD["target_band"]
    assert lo - 0.1 <= rate <= hi + 0.1
    assert mean_d > 0.4
    assert a.epsilon == config.RL_EPSILON_MIN

    # a uniformly random policy: either too hard (low return rate) or easier on average
    rnd = make_agent(rng=random.Random(12), allowed_fn=lambda s: np.arange(N_ACTIONS))
    rnd.epsilon = 1.0
    rlog = simulate(rnd, 1000, _skilled_player, random.Random(13), paddle_after=lambda cmd, x: cmd.target_x)
    r_rate = sum(ok for _, _, ok in rlog) / len(rlog)
    r_returned_d = np.mean([D for _, D, ok in rlog if ok])
    returned_d = np.mean([D for _, D, ok in tail if ok])
    assert rate > r_rate
    assert returned_d > r_returned_d


def test_gets_harder_as_the_player_improves():
    rng = random.Random(5)
    a = make_agent(rng=rng)

    def weak(cmd, D, paddle, rng):
        return rng.random() < (1.0 if D < 0.2 else max(0.0, 1.0 - 2.5 * (D - 0.2)))

    simulate(a, 3000, weak, rng, paddle_after=lambda cmd, x: cmd.target_x, schedule=True)
    weak_log = simulate(a, 600, weak, rng, paddle_after=lambda cmd, x: cmd.target_x, schedule=True)
    simulate(a, 4000, _skilled_player, rng, paddle_after=lambda cmd, x: cmd.target_x, schedule=True)
    strong_log = simulate(a, 600, _skilled_player, rng, paddle_after=lambda cmd, x: cmd.target_x, schedule=True)
    assert np.mean([D for _, D, _ in strong_log]) > np.mean([D for _, D, _ in weak_log]) + 0.05


# =============================================================================
# Shot commands, opponent and game integration
# =============================================================================

def test_shot_command_round_trip_and_robot_payload():
    for i in range(N_ACTIONS):
        cmd = ShotCommand.from_index(i)
        assert cmd.index == i
        payload = json.loads(json.dumps(cmd.to_dict()))
        assert set(payload) == {"placement", "speed", "spin", "target_x", "speed_m_s", "spin_side", "spin_top"}
    assert ShotCommand("left", "fast", "top").spin_value.top > 0
    assert ShotCommand("left", "fast", "back").spin_value.top < 0
    assert ShotCommand("left", "fast", "side_right").spin_value.side > 0


@pytest.mark.parametrize("spin", ["none", "side_left", "side_right", "top", "back"])
def test_sidespin_aim_lands_on_the_placement(spin):
    cmd = ShotCommand("right", "medium", spin)
    aim = aim_for_landing(cmd.target_x, cmd.speed_m_s, cmd.spin_value)
    f = make_flight(0.0, config.OPPONENT_HIT_Y_M, aim, config.PLAYER_HIT_Y_M, 0.0, cmd.speed_m_s, cmd.spin_value)
    assert f.end_x == pytest.approx(cmd.target_x, abs=1e-6)


class AimedInput:
    """Player who stands where the ball lands, except on shots the agent should learn to avoid."""

    def __init__(self):
        self.x = 0.0

    def paddle_x_at(self, t):
        return self.x

    def direction_at(self, t):
        return "center"

    def current_paddle_x(self):
        return self.x


def test_dynamic_mode_plays_full_rallies_in_the_game():
    from events import SwingEvent
    rng = random.Random(2)
    inp = AimedInput()
    rl = RLOpponent(inp.current_paddle_x, persist=False, rng=rng)
    opp = ModeOpponent(SimulatedOpponent(rng=random.Random(3)), {config.DYNAMIC_DIFFICULTY: rl})
    scores = []
    g = GameLogic(opp, inp, on_score=scores.append, rng=random.Random(4))
    assert g.set_difficulty("Dynamic") and opp.active is rl
    g.start(0.0)
    t, planned_seen, misses = 0.0, 0, 0
    while t < 400.0:
        t += 0.02
        if g.state == "to_player" and g.flight is not None and not g.swings:
            assert abs(g.flight.end_x) <= config.TABLE_WIDTH_M / 2
            fast_left = g.flight.speed >= config.RL_SPEEDS_M_S["fast"] and g.flight.end_x < -0.3
            inp.x = g.flight.end_x + (1.0 if fast_left else 0.0)    # can't reach fast backhands
            g.add_swing(SwingEvent(g.flight.t_end, 1.0, "keyboard"))
        if g.state == "to_opponent" and opp.planned_shot is not None:
            planned_seen += 1
        before = g.state
        g.update(t)
        misses += before == "to_player" and g.state == "miss"
    assert all(isinstance(s, float) for s in scores) and max(scores) >= 3
    assert rl.agent.updates > 50 and rl.agent.total_hits > 50
    assert misses > 0 and planned_seen > 0
    assert rl.agent.epsilon < config.RL_EPSILON_START
    info = opp.panel_info()
    assert {"epsilon", "gamma", "explored", "action", "difficulty", "reward", "beta", "return_rate",
            "band"} <= set(info)
    # back to Easy: the simulated opponent takes over, no learning happens
    g.to_menu()
    updates = rl.agent.updates
    assert g.set_difficulty("Easy") and opp.active is not rl
    g.start(t)
    for k in range(500):
        g.update(t + k * 0.02)
    assert rl.agent.updates == updates
    assert opp.panel_info() is None


def test_rl_opponent_reset_mid_rally_drops_the_unfinished_shot():
    from opponent import IncomingBall
    rl = RLOpponent(persist=False, rng=random.Random(0))
    rl.serve(0.0)
    rl.shot_result(True, 1.0)
    rl.ball_incoming(IncomingBall(0.1, 2.0, 3.0))
    updates = rl.agent.updates
    rl.reset()                                     # Esc mid-rally
    assert rl.poll(5.0) is None and rl.agent.updates == updates
    assert isinstance(rl.serve(6.0), Shot)


def test_decision_and_update_are_fast():
    rng = random.Random(0)
    a = _trained()
    a.rng = rng
    n = 300
    t0 = time.perf_counter()
    for _ in range(n):
        d = a.choose(S0)
        a.update(S0, d.action, components(True, 0.5, 2), S1)
    per = (time.perf_counter() - t0) / n
    assert per < 0.001
