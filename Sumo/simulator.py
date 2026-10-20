"""
Sensor simulator for --sim mode and for the automated tests.

A small 2-D model of the ring produces the SAME raw signals the real hardware
does: Color Sensor readings (RGB + reflection), Double Motor wheel speeds (with
config.py's signs/units) and IMU acceleration (milli-g on config.IMU_FORWARD_AXIS),
Controller lever %, plus the UNO Q's MQTT messages (vision with latency, status,
song_done). So the whole real pipeline (KNN, debouncer, motion filter, line
tracker, behavior tree, link) runs unchanged; only the bottom layer is fake.

Geometry (top view, metres). Ring centre at (0, 0); blue tape from RING_RADIUS to
RING_RADIUS + TAPE_WIDTH; the floor outside is the same color as inside.

        back --- [axle] --------- [sensor] ---> heading
                         SENSOR_AHEAD

Opponent pushes are scripted as an extra world velocity. With slip=True our tyres
slip (the encoders keep reporting what the motors do); with slip=False they grip
and are back-driven (the encoders report the real ground speed).

Scenarios (pick one with --scenario): see SCENARIOS at the bottom. Each one
also lists what the program is expected to do, which tests/test_scenarios.py checks.
"""

import json
import math
import random
import time
from dataclasses import dataclass, field

import config
from color_classifier import BLUE, FLOOR
from lego_io import Drive
from motion import MotionSample
from mqtt_link import LinkBase

RING_RADIUS = 0.50
TAPE_WIDTH = config.TAPE_WIDTH_M
SENSOR_AHEAD = 0.07
WHEEL_BASE = 0.10
MPS_PER_PERCENT = 0.004       # 100 % wheel command = 0.4 m/s
MOTOR_TAU = 0.08              # motor speed response time
NOTIFY_S = 0.030              # LEGO notification period (measured ~30 ms)
STEP_S = 0.005                # physics step
PUSH_RAMP_S = 0.1             # pushes ramp up/down over this time (finite acceleration)
FOV_HALF_DEG = 30.0
VISION_PERIOD_S = 0.066
CONTACT_M = 0.12              # centre distance at which we touch the opponent

FLOOR_READING = (195, 190, 192, 45)     # measured on the real table: R, G, B, reflection
BLUE_READING = (45, 85, 170, 14)


@dataclass
class SimColor:
    """Looks like a legoeducation ColorSensorNotification."""
    rawRed: float
    rawGreen: float
    rawBlue: float
    reflection: float
    color: int
    hue: float = 0
    saturation: float = 0
    value: float = 0


# Every scenario starts with the round-start gesture: levers centered, a hard push
# at 0.5 s (starts the round), back to center at 0.7 s (unlocks teleop).
START = [(0.0, 0, 0), (0.5, 85, 85), (0.7, 0, 0)]


@dataclass
class Scenario:
    name: str
    description: str
    duration_s: float
    start: tuple = (0.0, 0.0, 0.0)            # x, y, heading (degrees, 0 = +x)
    levers: list = field(default_factory=lambda: list(START))  # [(t, left %, right %)] steps
    pushes: list = field(default_factory=list)  # [(t0, t1, vx, vy, slip)]
    opponent: tuple | None = None             # (x, y) or None
    opponent_vel: tuple = (0.0, 0.0)
    color_dropouts: list = field(default_factory=list)   # [(t0, t1)]
    broker_down: list = field(default_factory=list)
    unoq_offline: list = field(default_factory=list)
    vision_latency_s: float = 0.15
    vision_frame_age_ms: int = 40
    song_delay_s: float = 1.5
    seed: int = 1
    # expectations (checked by tests/test_scenarios.py)
    expect_out: bool = False
    expect_reason: str | None = None
    expect_branches: tuple = ()
    forbid_branches: tuple = ()


class SimWorld:
    def __init__(self, sc: Scenario, *, quiet=False):
        self.sc = sc
        self.rng = random.Random(sc.seed)
        self.t = 0.0
        self.x, self.y, self.th = sc.start[0], sc.start[1], math.radians(sc.start[2])
        self.wl = self.wr = 0.0                 # actual wheel ground speeds, m/s
        self.cmd = (0.0, 0.0)
        self.opp = list(sc.opponent) if sc.opponent else None
        self.link = None                        # SimLink, if any
        self._next_notify = 0.0
        self._next_vision = 0.0
        self._vision_queue = []                 # (deliver_at, payload)
        self._seq = 0
        self._v_fwd_prev = 0.0
        self.color_q = []
        self.last_color_t = -math.inf
        self.motor_sample = MotionSample(None, None, None)
        self.motor_t = -math.inf
        self.lever_values = (0.0, 0.0)
        self.min_opp_dist = math.inf
        self.max_tape_overshoot_m = 0.0         # how far the sensor got past the tape's inner edge
        self.truth_out_at = None                # when the axle first went past the tape
        self.quiet = quiet

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _active(windows, t):
        return any(a <= t < b for a, b in windows)

    def heading(self):
        return math.cos(self.th), math.sin(self.th)

    def sensor_pos(self):
        hx, hy = self.heading()
        return self.x + SENSOR_AHEAD * hx, self.y + SENSOR_AHEAD * hy

    def sensor_on_tape(self):
        r = math.hypot(*self.sensor_pos())
        return RING_RADIUS <= r <= RING_RADIUS + TAPE_WIDTH

    def _push(self, t):
        """Total scripted push velocity at time t (with ramps) and whether tyres slip."""
        vx = vy = 0.0
        slip = False
        for t0, t1, px, py, s in self.sc.pushes:
            if t0 <= t < t1:
                k = min(1.0, (t - t0) / PUSH_RAMP_S, (t1 - t) / PUSH_RAMP_S)
                vx += k * px
                vy += k * py
                slip = slip or s
        return vx, vy, slip

    def _levers_at(self, t):
        l = r = 0.0
        for t0, a, b in self.sc.levers:
            if t >= t0:
                l, r = a, b
        return float(l), float(r)

    def training_samples(self, n=40):
        """Synthetic calibration data from the same noise model as the readings."""
        rng = random.Random(99)
        out = []
        for label, base in ((FLOOR, FLOOR_READING), (BLUE, BLUE_READING)):
            for _ in range(n):
                out.append((label, tuple(v + rng.gauss(0, 3) for v in base)))
        return out

    # -- simulation ------------------------------------------------------------------
    def step(self, now):
        while self.t + STEP_S <= now:
            self._physics(STEP_S)
            self.t += STEP_S
            if self.t >= self._next_notify:
                self._notify()
                self._next_notify += NOTIFY_S
            if self.t >= self._next_vision:
                self._vision()
                self._next_vision += VISION_PERIOD_S
            self._deliver_vision()

    def _physics(self, dt):
        cl, cr = self.cmd
        self.wl += (cl * MPS_PER_PERCENT - self.wl) * dt / MOTOR_TAU
        self.wr += (cr * MPS_PER_PERCENT - self.wr) * dt / MOTOR_TAU
        hx, hy = self.heading()
        px, py, _ = self._push(self.t)
        v_drive = 0.5 * (self.wl + self.wr)
        self.x += (hx * v_drive + px) * dt
        self.y += (hy * v_drive + py) * dt
        self.th += (self.wr - self.wl) / WHEEL_BASE * dt
        if self.opp is not None:
            self.opp[0] += self.sc.opponent_vel[0] * dt
            self.opp[1] += self.sc.opponent_vel[1] * dt
            dx, dy = self.opp[0] - self.x, self.opp[1] - self.y
            d = math.hypot(dx, dy)
            if 1e-6 < d < CONTACT_M:               # we push the opponent along
                self.opp[0] = self.x + dx / d * CONTACT_M
                self.opp[1] = self.y + dy / d * CONTACT_M
                d = CONTACT_M
            self.min_opp_dist = min(self.min_opp_dist, d)
        r_s = math.hypot(*self.sensor_pos())
        if r_s >= RING_RADIUS and self.truth_out_at is None:
            self.max_tape_overshoot_m = max(self.max_tape_overshoot_m, r_s - RING_RADIUS)
        if self.truth_out_at is None and math.hypot(self.x, self.y) > RING_RADIUS + TAPE_WIDTH:
            self.truth_out_at = self.t

    def _notify(self):
        t, n = self.t, self.rng.gauss
        hx, hy = self.heading()
        px, py, slip = self._push(t)
        v_body_fwd = 0.5 * (self.wl + self.wr) + px * hx + py * hy
        # Color Sensor
        if not self._active(self.sc.color_dropouts, t):
            base, col = (BLUE_READING, 3) if self.sensor_on_tape() else (FLOOR_READING, 0)
            vals = [v + n(0, 3) for v in base]
            self.color_q.append((t, SimColor(*vals, color=col)))
            self.last_color_t = t
        # Double Motor: encoders + IMU
        if (px or py) and not slip:
            # Tyres grip and get back-driven: encoders measure the real ground speed.
            turn = 0.5 * (self.wr - self.wl)
            ml, mr = v_body_fwd - turn, v_body_fwd + turn
        else:
            ml, mr = self.wl, self.wr
        to_raw = 1.0 / config.WHEEL_SPEED_TO_MPS
        left_raw = round(config.LEFT_WHEEL_SIGN * ml * to_raw + n(0, 0.5))
        right_raw = round(config.RIGHT_WHEEL_SIGN * mr * to_raw + n(0, 0.5))
        a_fwd = (v_body_fwd - self._v_fwd_prev) / NOTIFY_S
        self._v_fwd_prev = v_body_fwd
        acc = [n(0, 3), n(0, 3), 1000 + n(0, 3)]
        axis = "xyz".index(config.IMU_FORWARD_AXIS)
        acc[axis] += config.IMU_FORWARD_SIGN * a_fwd / 9.81 * config.IMU_RAW_PER_G
        self.motor_sample = MotionSample(left_raw, right_raw, tuple(round(a) for a in acc), t=t)
        self.motor_t = t
        # Controller
        self.lever_values = self._levers_at(t)

    def _vision(self):
        if self.opp is None or self.link is None:
            return
        dx, dy = self.opp[0] - self.x, self.opp[1] - self.y
        dist = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dy, dx) - self.th)
        bearing = (bearing + 180) % 360 - 180            # + = to our LEFT (counter-clockwise)
        visible = abs(bearing) <= FOV_HALF_DEG and dist < 2.0
        self._seq += 1
        msg = {"seq": self._seq, "visible": visible,
               "x": round(-bearing / FOV_HALF_DEG, 3) if visible else None,   # + = to our RIGHT
               "distance_cm": round(dist * 100, 1) if visible else None,
               "confidence": 0.9 if visible else 0.0,
               "frame_age_ms": self.sc.vision_frame_age_ms}
        self._vision_queue.append((self.t + self.sc.vision_latency_s, json.dumps(msg)))

    def _deliver_vision(self):
        while self._vision_queue and self._vision_queue[0][0] <= self.t:
            _, payload = self._vision_queue.pop(0)
            if self.link is not None and self.link.connected:     # QoS 0: lost while broker is down
                self.link.handle_message(config.T_VISION, payload)
        if self.link is not None:
            self.link.poll_sim(self.t)


class SimDrive(Drive):
    def __init__(self, world):
        super().__init__()
        self.world = world

    def _apply(self, left, right):
        self.world.cmd = (left, right)


class SimHardware:
    """Same interface as lego_io.LegoHardware, backed by SimWorld."""

    def __init__(self, world):
        self.world = world
        self.drive = SimDrive(world)

    @property
    def finished(self):
        return self.world.t >= self.world.sc.duration_s

    def connect_all(self):
        print(f"[sim] scenario '{self.world.sc.name}': {self.world.sc.description}")

    def poll(self, now):
        self.world.step(now)

    def drain_color(self):
        q, self.world.color_q = self.world.color_q, []
        return q

    def color_fresh(self, now):
        return now - self.world.last_color_t <= config.COLOR_STALE_S

    def levers(self, now):
        return self.world.lever_values

    def motion_sample(self, now):
        if now - self.world.motor_t > config.MOTOR_STALE_S:
            return MotionSample(None, None, None)
        return self.world.motor_sample

    def emergency_stop(self):
        self.drive.commanded = (0, 0)
        self.world.cmd = (0, 0)

    def close(self):
        self.emergency_stop()


class SimLink(LinkBase):
    """Plays the UNO Q side over a simulated broker."""

    def __init__(self, world, clock):
        super().__init__(clock)
        self.world = world
        world.link = self
        self._song_at = None
        self._status_sent = None

    @property
    def connected(self):
        return not SimWorld._active(self.world.sc.broker_down, self.world.t)

    def _send(self, topic, payload, qos=0, retain=False):
        self._record(topic, payload)
        print(f"[sim-mqtt] -> {topic.removeprefix(config.TOPIC_PREFIX)} {payload}")
        if topic == config.T_EVENT_LOST:
            self._song_at = self.world.t + self.world.sc.song_delay_s

    def poll_sim(self, t):
        if not self.connected:
            return
        offline = SimWorld._active(self.world.sc.unoq_offline, t)
        state = "offline" if offline else "online"
        if state != self._status_sent:                 # retained status, re-sent on change
            self._status_sent = state
            self.handle_message(config.T_UNOQ_STATUS,
                                json.dumps({"state": state, "camera": "ok", "fps": 15.0}))
        if self._song_at is not None and t >= self._song_at:
            self._song_at = None
            self.handle_message(config.T_SONG_DONE, "{}")


class SimClock:
    """Simulated time. realtime=True paces it with the wall clock (for watching)."""

    def __init__(self, realtime=False):
        self.t = 0.0
        self.realtime = realtime

    def now(self):
        return self.t

    def sleep(self, dt):
        if self.realtime:
            time.sleep(dt)
        self.t += dt


def _drive(t0, t1, l, r):
    return [(t0, l, r), (t1, 0, 0)]


SCENARIOS = {sc.name: sc for sc in [
    Scenario("full_round", "drive to the edge (reflex), chase the opponent, get pushed out backward, lose",
             duration_s=16.0, start=(0.0, 0.0, 0.0),
             # forward to the edge (reflex fires), then spin left until the opponent is in view
             levers=START + _drive(1.0, 3.8, 60, 60) + _drive(4.3, 5.4, -45, 45),
             opponent=(-0.30, 0.10),          # behind us at first; comes into view as we turn
             # while we chase (facing roughly -x) the opponent shoves us back toward +x
             pushes=[(8.0, 12.0, 0.6, 0.0, True)],
             expect_out=True, expect_reason="backward",
             expect_branches=("teleop", "reverse", "chase", "out_of_ring")),
    Scenario("edge_forward", "teleop drives forward onto the tape: blue while moving forward -> reverse 1 s",
             duration_s=7.0, levers=START + _drive(1.0, 6.0, 60, 60),
             expect_branches=("teleop", "reverse"), forbid_branches=("out_of_ring",)),
    Scenario("pushed_out_backward", "opponent shoves us backward while our wheels SLIP forward -> OUT",
             duration_s=5.0, start=(-0.38, 0.0, 0.0),        # facing the centre, back to the edge
             levers=START + _drive(1.0, 5.0, 50, 50),
             pushes=[(1.2, 4.0, -0.45, 0.0, True)],
             expect_out=True, expect_reason="backward", expect_branches=("teleop", "out_of_ring")),
    Scenario("fast_shove_out", "violent 0.9 m/s shove backward: the tape is under the sensor for ~1 reading -> still OUT",
             duration_s=4.0, start=(-0.20, 0.0, 0.0),
             pushes=[(1.0, 2.5, -0.9, 0.0, True)],
             expect_out=True, expect_reason="backward"),
    Scenario("pushed_out_backward_grip", "same shove, but our tyres grip and get back-driven -> OUT",
             duration_s=5.0, start=(-0.38, 0.0, 0.0),
             levers=START + _drive(1.0, 5.0, 50, 50),
             pushes=[(1.2, 4.0, -0.45, 0.0, False)],
             expect_out=True, expect_reason="backward"),
    Scenario("crossed_forward", "shoved from behind across the tape while reversing: blue then floor moving forward -> "
             "OUT (after the overshoot grace: it never comes back over the tape)",
             duration_s=6.0, start=(0.30, 0.0, 0.0),         # facing the edge
             pushes=[(1.0, 3.0, 0.5, 0.0, True)],
             expect_out=True, expect_reason="crossed", expect_branches=("reverse", "out_of_ring")),
    Scenario("pivot_onto_tape", "spin in place until the sensor sweeps onto the tape: direction UNCERTAIN -> lean IN, reverse",
             duration_s=4.0, start=(0.45, 0.0, 90.0),
             levers=START + _drive(1.0, 3.0, 50, -50),
             expect_branches=("teleop", "reverse"), forbid_branches=("out_of_ring",)),
    Scenario("chase", "opponent ahead and to the right: steer toward it and push",
             duration_s=4.0, opponent=(0.30, -0.12),
             expect_branches=("chase",), forbid_branches=("out_of_ring",)),
    Scenario("color_dropout", "Color Sensor goes silent for 0.5 s while driving: safety stop, then resume",
             duration_s=3.0, levers=START + _drive(1.0, 3.0, 40, 40),
             color_dropouts=[(1.5, 2.0)],
             expect_branches=("teleop", "color_safety"), forbid_branches=("out_of_ring",)),
    Scenario("vision_stale", "camera frames are 900 ms old: chase must NOT run, teleop still works",
             duration_s=3.0, levers=START + _drive(1.0, 3.0, 30, 30), opponent=(0.4, 0.0),
             vision_frame_age_ms=900, expect_branches=("teleop",), forbid_branches=("chase",)),
    Scenario("broker_down", "broker unreachable for 1.5 s mid-chase: chase pauses, controller still drives",
             duration_s=4.0, levers=START + [(1.0, 25, 25)], opponent=(1.2, 0.0),
             broker_down=[(1.5, 3.0)], expect_branches=("chase", "teleop")),
    Scenario("unoq_offline", "UNO Q reports offline: chase must not run",
             duration_s=3.0, levers=START + [(1.0, 25, 25)], opponent=(1.2, 0.0),
             unoq_offline=[(0.0, 99.0)], expect_branches=("teleop",), forbid_branches=("chase",)),
]}
