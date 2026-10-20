"""
End-to-end: run every simulator scenario through the REAL control loop
(KNN or LEGO color, debouncer, motion filter, line tracker, behavior tree, link logic) and
check what the robot did against the scenario's expectations.
"""

import pytest

import config
from simulator import SCENARIOS, SimClock, SimHardware, SimLink, SimWorld
from color_classifier import KnnColorClassifier, LegoColorClassifier
from sumo import run, shutdown


def run_scenario(name, method="knn"):
    world = SimWorld(SCENARIOS[name])
    clock = SimClock(realtime=False)
    hw = SimHardware(world)
    link = SimLink(world, clock=clock.now)
    clf = KnnColorClassifier(world.training_samples()) if method == "knn" else LegoColorClassifier()
    hw.connect_all()
    try:
        result = run(hw, link, clf, clock)
    finally:
        shutdown(hw, link)
    return world, link, result


@pytest.mark.parametrize("method", ["knn", "lego"])   # both COLOR_METHOD settings
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_expectations(name, method, capsys):
    sc = SCENARIOS[name]
    world, link, result = run_scenario(name, method)
    seen = result.branches_seen
    assert (result.exit_cause == "lost") == sc.expect_out, f"exit={result.exit_cause} reason={result.lost_reason}"
    if sc.expect_reason:
        assert sc.expect_reason in (result.lost_reason or "")
    for b in sc.expect_branches:
        assert b in seen, f"branch {b} never ran; saw {seen}"
    for b in sc.forbid_branches:
        assert b not in seen, f"branch {b} must not run"


def test_no_motion_while_armed():
    """Before the start push, every command must be 0 (we only log commands once running)."""
    world, link, result = run_scenario("edge_forward")
    first_t = result.command_log[0][0]
    assert first_t >= 0.5            # the tree only ran after the start push
    published = [p for t, p in link.sent if t == config.T_ROBOT_STATE]
    assert published[0] == '{"phase": "armed", "branch": null}'


def test_start_push_does_not_drive():
    """Teleop is locked until the levers return to center after the start push."""
    world, link, result = run_scenario("edge_forward")
    for t, branch, cmd in result.command_log:
        if t < 0.7:
            assert cmd == (0, 0), (t, branch, cmd)


def test_reverse_runs_full_time_after_blue_disappears():
    world, link, result = run_scenario("edge_forward")
    log = result.command_log
    starts = [i for i, (t, b, c) in enumerate(log) if b == "reverse" and (i == 0 or log[i - 1][1] != "reverse")]
    assert starts
    i = starts[0]
    t_start = log[i][0]
    j = i
    while j < len(log) and log[j][1] == "reverse":
        j += 1
    duration = log[j - 1][0] - t_start
    assert duration == pytest.approx(config.REVERSE_TIME_S, abs=0.05)
    # and blue had already disappeared well before the reverse ended
    assert world.max_tape_overshoot_m < 0.05


def test_overshoot_estimate_matches_simulated_truth():
    """The printed overshoot estimate (+ the debounce delay) should match how far
    the sensor really went past the tape's inner edge in the simulator."""
    world, link, result = run_scenario("edge_forward")
    assert len(result.overshoots_m) >= 1          # the levers stay pushed, so it may hit the edge twice
    est = max(result.overshoots_m)
    print(f"estimate {est * 100:.1f} cm, true {world.max_tape_overshoot_m * 100:.1f} cm")
    assert est == pytest.approx(world.max_tape_overshoot_m, abs=0.01)   # within 1 cm


def test_color_dropout_stops_motors():
    world, link, result = run_scenario("color_dropout")
    safety = [c for t, b, c in result.command_log if b == "color_safety"]
    assert safety and all(c == (0, 0) for c in safety)
    # and it drove again afterwards
    after = [c for t, b, c in result.command_log if t > 2.1 and b == "teleop"]
    assert after and all(c != (0, 0) for c in after)


def test_lost_sequence_published_in_order():
    world, link, result = run_scenario("pushed_out_backward")
    i_state = link.sent.index((config.T_ROBOT_STATE, '{"phase": "lost", "branch": null}'))
    i_event = [t for t, _ in link.sent].index(config.T_EVENT_LOST)
    assert i_state < i_event
    assert result.exit_cause == "lost"
    # song_done arrives 1.5 s after event/lost: we must exit then, not at the 8 s timeout
    t_lost = next(t for t, b in result.branch_log if b == "out_of_ring")
    t_exit = result.command_log[-1][0]
    assert t_exit - t_lost == pytest.approx(SCENARIOS["pushed_out_backward"].song_delay_s, abs=0.1)


def test_chase_steers_toward_opponent():
    world, link, result = run_scenario("chase")
    assert world.min_opp_dist <= 0.13          # it reached (and pushed) the opponent


def test_uncertain_direction_leans_in():
    world, link, result = run_scenario("pivot_onto_tape")
    assert result.uncertain_events >= 1
    assert result.exit_cause != "lost"
