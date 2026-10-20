"""
Sumo robot brain (runs on the laptop). Running this program = one round.

    python sumo.py                 real robot + UNO Q over MQTT
    python sumo.py --no-motors     dry run: real sensors, wheels never commanded
    python sumo.py --no-unoq       LEGO only, no MQTT (chase branch never runs)
    python sumo.py --sim [--scenario NAME] [--fast]
                                   simulated sensors / UNO Q (no hardware needed)
    python sumo.py --list-scenarios

Each tick (TICK_HZ):
  1. read inputs   : new color readings -> KNN -> debouncer -> line tracker
                     (with the motion direction at that moment); levers; vision
  2. round phase   : armed -> running (start push) ; lost ends the round
  3. tick the tree : the root selector picks the highest-priority branch that applies
  4. outputs       : wheel command (sent by a background thread), robot/state on change
Whatever happens (normal end, Ctrl+C, crash), the motors are stopped first.
"""

import argparse
import atexit
import signal
import sys
import time
from dataclasses import dataclass, field

import config
from behavior_tree import render
from color_classifier import (BLUE, FLOOR, Debouncer, KnnColorClassifier, classify_sensor,
                              load_classifier, samples_needed)
from line_tracker import LineTracker, Zone
from motion import MotionEstimator
from robot_tree import CONTRACT_BRANCH, Blackboard, active_branch, build_tree
from round_start import RUNNING, RoundStart
from watchdogs import DirectDriveWatchdog, WheelSignWatchdog


@dataclass
class RunResult:
    exit_cause: str = ""                 # "lost", "sim finished", "ctrl+c", "error"
    lost_reason: str | None = None
    branch_log: list = field(default_factory=list)    # [(t, branch)] on every change
    command_log: list = field(default_factory=list)   # [(t, branch, (left, right))] every tick
    uncertain_events: int = 0
    overshoots_m: list = field(default_factory=list)  # OvershootMeter estimates, one per reflex
    ticks: int = 0

    @property
    def branches_seen(self):
        return {b for _, b in self.branch_log}


class OvershootMeter:
    """How far does the robot keep rolling FORWARD after the edge is detected?

    Started when the line tracker reports "blue while moving forward". The sensor
    actually reached the tape a little earlier: the debouncer waited for N blue
    readings first, so we start the count at  speed x N x sample period  (lead_m).
    Then it adds up the fused forward speed x time until the speed drops to zero
    (the reverse reflex has stopped us) and prints the total: an estimate of how
    far the sensor got past the tape's INNER edge. Compare it with the ring.
    """

    def __init__(self, log=print):
        self.log = log
        self.active = False
        self.dist_m = 0.0
        self._last_t = None
        self.results = []          # metres, one per reflex

    def start(self, now, lead_m=0.0):
        self.active, self.dist_m, self._last_t = True, lead_m, now

    def update(self, now, v_mps):
        if not self.active:
            return
        self.dist_m += max(0.0, v_mps) * (now - self._last_t)
        self._last_t = now
        if v_mps <= 0:
            self.active = False
            self.results.append(self.dist_m)
            self.log(f"[edge] overshoot: the sensor went about {self.dist_m * 100:.1f} cm past the "
                     f"tape's inner edge before stopping (estimate)")


class RealClock:
    def now(self):
        return time.monotonic()

    def sleep(self, dt):
        if dt > 0:
            time.sleep(dt)


def status_line(now, phase, branch, levers, start, color_fresh, drive, link, motion=None, sample=None,
                zone=None, motor_states=None, color_info=None):
    """Everything that decides whether the wheels move, plus the current state."""
    lv = "none (Controller stale)" if levers is None else f"{levers[0]:+.0f}/{levers[1]:+.0f}"
    motor = ""
    if hasattr(drive, "sends"):
        motor = f" sent={drive.sends} last_frame={drive.last_sent} errors={drive.errors}"
    lines = [f"[status] t={now:6.1f}s  STATE: phase={phase} branch={branch} "
             f"(robot/state branch={CONTRACT_BRANCH.get(branch) if phase == RUNNING else None})"
             + (f" zone={zone}" if zone else ""),
             f"         inputs: levers={lv} teleop={'on' if start.teleop_unlocked else 'LOCKED'} "
             f"color={'ok' if color_fresh else 'STALE'} mqtt={'up' if link.connected else 'down'}",
             f"         color: method={config.COLOR_METHOD} {color_info or 'no reading yet'}",
             f"         motors: cmd(robot L,R)={drive.commanded}{motor}"
             + (f" motor state L/R={motor_states}" if motor_states else "")]
    if motion is not None:
        enc = "stale" if sample is None or sample.left_speed is None else \
            f"{sample.left_speed:+.0f}/{sample.right_speed:+.0f}"
        lines.append(f"         motion: {motion.direction.value} {motion.describe()} encoder speeds L/R={enc}")
    return "\n".join(lines)


def color_detail(label, s):
    """Latest color reading for the status line: what it was classified as, and the raw values."""
    if s is None:
        return None
    raw = " ".join(f"{n}={getattr(s, n, '?')}" for n in ("rawRed", "rawGreen", "rawBlue", "reflection", "color"))
    return f"last reading = {label}  ({raw})"


def run(hw, link, classifier, clock, *, max_s=None, status_s=None):
    """The control loop. Returns a RunResult. Callers must call shutdown() afterwards.
    status_s: print a status line this often (None = never)."""
    result = RunResult()
    tracker = LineTracker()
    motion = MotionEstimator()
    debouncer = Debouncer()
    overshoot = OvershootMeter()
    start = RoundStart()
    tree = build_tree()
    dd_watch, sign_watch = DirectDriveWatchdog(), WheelSignWatchdog()
    bb = Blackboard(tracker=tracker, drive=hw.drive, link=link)
    period = 1.0 / config.TICK_HZ
    t0 = clock.now()
    next_tick = t0
    last_branch = "<none>"
    next_status = t0
    last_color, last_raw = None, None

    print("\nBehavior tree:\n" + render(tree) + "\n")
    print(f"[armed] Center both levers, then push either lever past {config.START_THRESHOLD}% to start.")

    while True:
        now = clock.now()
        hw.poll(now)

        # 1. inputs ------------------------------------------------------------
        sample = hw.motion_sample(now)
        direction = motion.update(sample, now)
        for _t, reading in hw.drain_color():
            needed = samples_needed(motion.v)
            label = classify_sensor(classifier, reading)
            if (label == BLUE) != (last_color == BLUE):
                if label == BLUE:
                    print(f"[color] t={now - t0:.2f}s sees BLUE  ({color_detail(label, reading)})")
                else:
                    print(f"[color] t={now - t0:.2f}s blue gone -> {label}")
            last_color, last_raw = label, reading
            change = debouncer.update(label, required=needed)
            if change is not None:
                print(f"[color] t={now - t0:.2f}s confirmed (debounced): {change.upper()}"
                      + ("" if start.phase == RUNNING else "  (round not running: ignored)"))
            if change is None or start.phase != RUNNING:
                continue
            detail = f"(t={now - t0:.2f}s, {motion.describe()})"
            if change == BLUE:
                was = tracker.zone
                if tracker.on_blue(motion.direction, detail, now=now) is Zone.ON_EDGE and was is Zone.INSIDE:
                    overshoot.start(now, lead_m=max(0.0, motion.v) * needed * config.COLOR_SAMPLE_PERIOD_S)
            elif change == FLOOR:
                tracker.on_floor(motion.direction, detail, now=now)
        if start.phase == RUNNING:
            tracker.update(now, f"(t={now - t0:.2f}s, {motion.describe()})")
        overshoot.update(now, motion.v)
        levers = hw.levers(now)
        phase = start.update(now, levers)

        # 2. round phase -------------------------------------------------------------
        if phase != RUNNING:
            hw.drive.stop()
            link.publish_state("armed", None)        # countdown also shows as "armed"
            dd_watch.update(now, hw.drive.commanded, sample, levers)
        else:
            # 3. tick the tree -------------------------------------------------------
            bb.now = now
            bb.color_fresh = hw.color_fresh(now)
            bb.direction = direction
            bb.opponent = link.store.fresh_opponent(now, link.connected)
            bb.levers = levers
            bb.teleop_unlocked = start.teleop_unlocked
            tree.tick(bb)
            branch = active_branch(tree)
            if branch != last_branch:
                print(f"[tree] t={now - t0:6.2f}s  branch: {last_branch} -> {branch}")
                result.branch_log.append((now - t0, branch))
                last_branch = branch
            # 4. outputs ---------------------------------------------------------------
            if branch == "out_of_ring":
                link.publish_state("lost", None)
            else:
                link.publish_state("running", CONTRACT_BRANCH[branch])
            result.command_log.append((now - t0, branch, hw.drive.commanded))
            sign_watch.update(now, hw.drive.commanded, sample)

        if status_s is not None and now >= next_status:
            next_status = now + status_s
            print(status_line(now - t0, phase, last_branch if phase == RUNNING else "-", levers,
                              start, hw.color_fresh(now), hw.drive, link, motion, sample,
                              tracker.zone.value,
                              hw.motor_states() if hasattr(hw, "motor_states") else None,
                              color_detail(last_color, last_raw)))

        result.ticks += 1
        if bb.exit_requested:
            result.exit_cause = "lost"
            result.lost_reason = bb.lost_reason
            break
        if getattr(hw, "finished", False) or (max_s is not None and now - t0 >= max_s):
            result.exit_cause = "sim finished"
            break

        # fixed rate: sleep until the next tick (skip ahead if we fell behind)
        next_tick += period
        delay = next_tick - clock.now()
        if delay < -period:
            next_tick = clock.now()
        clock.sleep(max(0.0, delay))

    result.uncertain_events = tracker.uncertain_events
    result.overshoots_m = list(overshoot.results)
    return result


def shutdown(hw, link):
    """Motors first, then MQTT, then Bluetooth. Safe to call more than once."""
    try:
        hw.emergency_stop()
    except Exception as e:
        print(f"[shutdown] motor stop failed: {e}")
    try:
        link.close()
    except Exception as e:
        print(f"[shutdown] MQTT close failed: {e}")
    try:
        hw.close()
    except Exception as e:
        print(f"[shutdown] LEGO close failed: {e}")


def build(args):
    """Create (hw, link, classifier, clock) for the chosen mode."""
    if args.sim:
        from simulator import SCENARIOS, SimClock, SimHardware, SimLink, SimWorld
        world = SimWorld(SCENARIOS[args.scenario])
        clock = SimClock(realtime=not args.fast)
        hw = SimHardware(world)
        if args.no_unoq:
            from mqtt_link import NullLink
            link = NullLink(clock=clock.now)
        else:
            link = SimLink(world, clock=clock.now)
        # Same COLOR_METHOD as on the robot. "knn" trains on the simulator's own
        # synthetic samples, so no calibration file is needed for --sim.
        if config.COLOR_METHOD == "knn":
            classifier = KnnColorClassifier(world.training_samples())
            print("[color] method 'knn' (trained on simulated samples)")
        else:
            classifier = load_classifier()
        return hw, link, classifier, clock

    from lego_io import LegoHardware
    from mqtt_link import MqttLink, NullLink
    clock = RealClock()
    try:
        classifier = load_classifier()
    except ValueError as e:
        sys.exit(f"[color] {e}")
    hw = LegoHardware(motors_enabled=not args.no_motors, clock=clock.now)
    link = NullLink(clock=clock.now) if args.no_unoq else MqttLink(clock=clock.now)
    return hw, link, classifier, clock


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LEGO sumo robot brain (one round per run)")
    p.add_argument("--no-motors", action="store_true", help="dry run: never command the wheels")
    p.add_argument("--no-unoq", action="store_true", help="run without MQTT / UNO Q")
    p.add_argument("--sim", action="store_true", help="use the sensor simulator instead of hardware")
    p.add_argument("--scenario", default="full_round", help="simulator scenario (see --list-scenarios)")
    p.add_argument("--fast", action="store_true", help="--sim: run as fast as possible")
    p.add_argument("--list-scenarios", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    # LEGO device names contain emoji; don't let a cp1252 console crash a print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    if args.list_scenarios:
        from simulator import SCENARIOS
        for sc in SCENARIOS.values():
            print(f"{sc.name:26s} {sc.description}")
        return 0
    if args.sim:
        from simulator import SCENARIOS
        if args.scenario not in SCENARIOS:
            print(f"unknown scenario {args.scenario!r}; choose from: {', '.join(SCENARIOS)}")
            return 2

    hw, link, classifier, clock = build(args)
    # Backstops: interpreter exit, and Ctrl+Break / console close on Windows.
    atexit.register(hw.emergency_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    code = 0
    result = None
    try:
        link.start()
        hw.connect_all()
        result = run(hw, link, classifier, clock, status_s=None if args.sim else config.STATUS_PRINT_S)
        print(f"[end] {result.exit_cause}" + (f": {result.lost_reason}" if result.lost_reason else ""))
    except KeyboardInterrupt:
        print("\n[end] Ctrl+C: stopping")
        code = 130
    except Exception:
        import traceback
        traceback.print_exc()
        print("[end] crashed: motors stopped")
        code = 1
    finally:
        shutdown(hw, link)
    return code


if __name__ == "__main__":
    sys.exit(main())
