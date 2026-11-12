"""
PascalNN: the Pascal Silly Walk, with a neural network instead of a Q-table
==========================================================================

Same robot, actions, rewards, steps and epsilon schedule as PascalQTable.py,
but Q is a small neural network instead of a 3x2 table:

    input   the yaw itself (a number, not a state bin), scaled to about -1..1
    hidden  16 tanh -> 16 tanh
    output  Q(yaw, Move to LEFT), Q(yaw, Move to RIGHT)

The rewards still come from the three states (Drifted Left / Straight /
Drifted Right). The network has to discover where the boundaries are and what
to do at every yaw in between.

How it learns (a "DQN", written in plain numpy so you can read it):
    * every step (yaw, action, reward, next yaw) goes into a REPLAY BUFFER
    * after each step it does a few gradient-descent updates on random
      batches from the buffer, shrinking
          loss = ( Q(yaw, a) - [ r + gamma * max Q_target(next yaw, a') ] )^2
    * Q_target is a copy of the network that is only refreshed every
      TARGET_SYNC updates, so the target doesn't move while it chases it

Run it:
    python3 -m pip install "legoeducation>=1.2.0" numpy
    python3 PascalNN.py
(on Windows: py -m pip install ... and py PascalNN.py). Turn the Double Motor on
first; it connects to the first one it finds (set SERIAL to pick one). The
learning is saved next to this file.

Buttons: Train (10 episodes, keeps the saved network), Run (100 greedy steps),
Clear & train (new random network). Pause = space bar, Stop = Esc.
"""

import time
import threading
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
import legoeducation as le

HERE = Path(__file__).resolve().parent


class doubleMotor(le.DoubleMotor):
    """The few helpers from the ME193 lelib library that this program uses,
    built in so this one file runs on its own."""

    def connect(self, card_serial=None, card_color=None):
        for attempt in range(5):          # a hub can say "not ready" at first
            try:
                super().connect(card_color=card_color, card_serial=card_serial)
                break
            except Exception as e:
                if "not ready" in str(e).lower() and attempt < 4:
                    time.sleep(1)
                else:
                    raise
        if not self.connected:
            raise ConnectionError("Error connecting to Double Motor.")

    def reset_heading(self):
        """Zero the yaw at the current orientation."""
        self.imu_reset_yaw_axis(0)

    def yaw(self):
        """Raw hub yaw (tenths of a degree) since the last reset_heading()."""
        return float(self.imu_device.yaw)

# ── Settings (robot, steps and rewards the same as PascalQTable) ──────────────
SERIAL = None          # Connection Card serial (e.g. 2279), or None = first hub found

TURN_DEGREES = 135     # how far one motor turns per step
TURN_WAIT    = 0.3     # s to wait for the turn to finish
SETTLE       = 0.2     # s extra so the yaw reading settles

NUM_STEPS    = 10      # steps per episode
NUM_EPISODES = 10      # episodes per Train click
POLICY_STEPS = 100     # steps per Run click

GAMMA     = 0.9        # discount factor
EPS_START = 1.0        # exploring falls in a straight line from start to end
EPS_END   = 0.05

YAW_LEFT  = 10         # yaw >  YAW_LEFT  -> Drifted Left
YAW_RIGHT = -10        # yaw <  YAW_RIGHT -> Drifted Right
STATE_NAMES  = ["Drifted Left", "Straight", "Drifted Right"]
ACTION_NAMES = ["Move to LEFT", "Move to RIGHT"]
REWARDS = [-1, 10, -1]          # per state, any action
REWARD_NEXT_STATE = True        # True = reward for the state the action led to (standard);
                                # False = for the state it acted from (like the web app)

# Neural network
HIDDEN        = (16, 16)   # tanh units in each hidden layer
LEARNING_RATE = 0.01       # Adam step size (this replaces the table's alpha)
BATCH         = 32         # transitions per gradient update
UPDATES_PER_STEP = 20      # gradient updates after every robot step
TARGET_SYNC   = 50         # copy the network into the target network this often
BUFFER_SIZE   = 2000       # replay buffer length
YAW_INPUT_SCALE = 45.0     # network input = yaw / this
Q_SCALE       = 10.0       # network output * this = Q (keeps the numbers small inside)

YAW_SCALE   = 0.1      # hub reports yaw in tenths of a degree
SAMPLE_DT   = 0.02     # s between yaw reads while waiting
NET_FILE    = HERE / "pascal_nn.npz"
PLOT_RANGE  = 45       # degrees each side on the plots

N_ACTIONS = len(ACTION_NAMES)


# ── State and reward (same as PascalQTable) ──────────────────────────────────
def to_state(yaw):
    if yaw > YAW_LEFT:
        return 0
    if yaw >= YAW_RIGHT:
        return 1
    return 2


def state_range(s):
    return [f"yaw > {YAW_LEFT}°", f"{YAW_RIGHT}° to {YAW_LEFT}°", f"yaw < {YAW_RIGHT}°"][s]


def eps_at(progress):
    progress = min(max(progress, 0.0), 1.0)
    return EPS_START + (EPS_END - EPS_START) * progress


# ── The neural network ────────────────────────────────────────────────────────
class QNet:
    """Fully connected net: yaw -> tanh layers -> one Q per action.
    Forward pass, backpropagation and the Adam optimizer, all in numpy."""

    def __init__(self, rng=None):
        rng = rng or np.random.default_rng()
        sizes = (1,) + HIDDEN + (N_ACTIONS,)
        self.W = [rng.normal(0, 1 / np.sqrt(i), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]
        self._adam_reset()

    def _adam_reset(self):
        self.t = 0
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]

    @staticmethod
    def inputs(yaws):
        return (np.asarray(yaws, dtype=float) / YAW_INPUT_SCALE).reshape(-1, 1)

    def _forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(self.W) - 1 else np.tanh(z))   # last layer linear
        return acts

    def q(self, yaws):
        """Q values (n x actions) for an array of yaws in degrees."""
        return self._forward(self.inputs(yaws))[-1] * Q_SCALE

    def train_step(self, yaws, actions, targets):
        """One gradient-descent step on the squared error of the chosen
        actions' Q values. Returns the loss (in Q units squared)."""
        acts = self._forward(self.inputs(yaws))
        n = len(actions)
        rows = np.arange(n)
        err = acts[-1][rows, actions] - np.asarray(targets) / Q_SCALE
        grad = np.zeros_like(acts[-1])
        grad[rows, actions] = 2 * err / n                  # d loss / d output
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):             # backpropagation
            gW[i] = acts[i].T @ grad
            gb[i] = grad.sum(axis=0)
            if i > 0:
                grad = (grad @ self.W[i].T) * (1 - acts[i] ** 2)   # tanh' = 1 - tanh^2
        self.t += 1                                         # Adam update
        b1, b2, e = 0.9, 0.999, 1e-8
        for P, G, M, V in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
            for i in range(len(P)):
                M[i] = b1 * M[i] + (1 - b1) * G[i]
                V[i] = b2 * V[i] + (1 - b2) * G[i] ** 2
                mhat, vhat = M[i] / (1 - b1 ** self.t), V[i] / (1 - b2 ** self.t)
                P[i] -= LEARNING_RATE * mhat / (np.sqrt(vhat) + e)
        return float(np.mean(err ** 2)) * Q_SCALE ** 2

    def copy(self):
        c = QNet.__new__(QNet)
        c.W = [w.copy() for w in self.W]
        c.b = [b.copy() for b in self.b]
        c._adam_reset()
        return c

    def to_dict(self, prefix):
        d = {f"{prefix}W{i}": w for i, w in enumerate(self.W)}
        d.update({f"{prefix}b{i}": b for i, b in enumerate(self.b)})
        return d

    @classmethod
    def from_dict(cls, data, prefix):
        c = cls.__new__(cls)
        n = len(HIDDEN) + 1
        c.W = [data[f"{prefix}W{i}"].copy() for i in range(n)]
        c.b = [data[f"{prefix}b{i}"].copy() for i in range(n)]
        c._adam_reset()
        return c


def choose(net, yaw, eps):
    if np.random.random() < eps:
        return np.random.randint(N_ACTIONS)
    return int(np.argmax(net.q([yaw])[0]))


# ── Robot helpers ─────────────────────────────────────────────────────────────
def take_action(dm, a):
    """Same motor commands as the web app's take_action()."""
    if a == 0:   # Move to LEFT: right leg steps
        dm.motor_run_for_degrees(TURN_DEGREES, direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
                                 motor=le.MOTOR_RIGHT, blocking=False)
    else:        # Move to RIGHT: left leg steps
        dm.motor_run_for_degrees(TURN_DEGREES, direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE,
                                 motor=le.MOTOR_LEFT, blocking=False)


def stop_motors(dm):
    for m in (le.MOTOR_LEFT, le.MOTOR_RIGHT):
        try:
            dm.motor_stop(motor=m)
        except Exception:
            pass


def read_yaw(dm):
    return dm.yaw() * YAW_SCALE


def wait(dm, seconds, abort, on_sample):
    """Sleep for `seconds`, reading the yaw for the gauge. False if aborted."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        if abort():
            return False
        on_sample(read_yaw(dm))
        time.sleep(SAMPLE_DT)
    return True


# ── Trainer: talks to the robot in a background thread ──────────────────────
class Trainer:
    def __init__(self):
        self.lock = threading.Lock()
        self.rng = np.random.default_rng()
        self.net = QNet(self.rng)
        self.target = self.net.copy()
        self.buffer = deque(maxlen=BUFFER_SIZE)   # (yaw, action, reward, next yaw)
        self.updates = 0
        self.episodes_total = 0
        self.history = []        # total reward per training episode
        self.losses = []         # mean loss per training episode
        self.load_note = self._load()

        self.state = "connecting"
        self.message = "Connecting to the Double Motor..."
        self.mode, self.train = None, True
        self.ep, self.step, self.eps = 0, 0, 0.0
        self.yaw, self.live = None, None
        self.cur_state, self.last_action, self.last_reward = None, None, None
        self.ep_reward = 0.0
        self.run_rewards = []
        self.readout = None
        self.log = deque(maxlen=4)
        self.version = 0

        self.start_evt = threading.Event()
        self.pause_evt = threading.Event()
        self.stop_evt = threading.Event()
        self.quit_evt = threading.Event()
        self.dm = None
        self.thread = threading.Thread(target=self._main, daemon=True)

    # -- saving ---------------------------------------------------------------
    def _load(self):
        if not NET_FILE.exists():
            return "No saved network yet (starting from a random one)."
        try:
            data = np.load(NET_FILE)
            self.net = QNet.from_dict(data, "net_")
            self.target = QNet.from_dict(data, "tgt_")
        except Exception:
            return "Saved network doesn't match the current settings; starting from a random one."
        self.episodes_total = int(data["episodes"])
        self.updates = int(data["updates"])
        self.history, self.losses = list(data["history"]), list(data["losses"])
        for row in data["buffer"]:
            self.buffer.append((float(row[0]), int(row[1]), float(row[2]), float(row[3])))
        return f"Loaded the saved network ({self.episodes_total} episodes of training)."

    def save(self):
        with self.lock:
            buf = np.array(list(self.buffer), dtype=float).reshape(-1, 4)
            np.savez(NET_FILE, episodes=self.episodes_total, updates=self.updates,
                     history=np.array(self.history), losses=np.array(self.losses),
                     buffer=buf, **self.net.to_dict("net_"), **self.target.to_dict("tgt_"))

    # -- called by the dashboard ----------------------------------------------
    def snapshot(self):
        with self.lock:
            return dict(
                net=self.net.copy(), buffer_yaws=[t[0] for t in self.buffer],
                buffer_len=len(self.buffer), updates=self.updates,
                episodes_total=self.episodes_total, history=list(self.history),
                losses=list(self.losses), run_rewards=list(self.run_rewards),
                train=self.train, state=self.state, message=self.message,
                ep=self.ep, step=self.step, eps=self.eps, yaw=self.yaw, live=self.live,
                cur_state=self.cur_state, last_action=self.last_action,
                last_reward=self.last_reward, ep_reward=self.ep_reward,
                readout=self.readout, log=list(self.log),
                version=self.version, paused=self.pause_evt.is_set())

    def begin(self, mode):          # "T", "R" or "C"
        if self.state == "ready":
            self.mode = mode
            self.stop_evt.clear()
            self.pause_evt.clear()
            self.start_evt.set()

    def toggle_pause(self):
        if self.state in ("running", "paused"):
            if self.pause_evt.is_set():
                self.pause_evt.clear()
            else:
                self.pause_evt.set()

    def stop(self):
        self.stop_evt.set()
        self.pause_evt.clear()

    def quit(self):
        self.quit_evt.set()
        self.stop()
        self.start_evt.set()

    # -- the robot thread -------------------------------------------------------
    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)
            self.version += 1

    def _main(self):
        try:
            dm = doubleMotor()
            dm.connect(SERIAL)
            try:
                dm.device_notification_request(15)   # ~66 Hz IMU, not 10 Hz
            except Exception:
                pass
            self.dm = dm
        except Exception as e:
            self._set(state="error", message=f"Couldn't connect to the Double Motor: {e}")
            return
        self._set(state="ready", message=(
            f"Connected. {self.load_note} Aim the robot, then click Train, Run or Clear & train."))
        try:
            while not self.quit_evt.is_set():
                self.start_evt.wait()
                self.start_evt.clear()
                if self.quit_evt.is_set():
                    break
                self._session()
        except Exception as e:
            self._set(state="error", message=f"Robot error: {e}")
        finally:
            try:
                stop_motors(dm)
                dm.disconnect()
            except Exception:
                pass

    def _reset_yaw(self):
        self.dm.reset_heading()
        time.sleep(0.3)          # let a fresh IMU reading arrive

    def _abort(self):
        return self.stop_evt.is_set() or self.pause_evt.is_set()

    def _on_sample(self, y):
        self.live = y

    def _pause_if_needed(self, walking):
        """Called after a step was cut short. Returns False if we should stop."""
        stop_motors(self.dm)
        if self.stop_evt.is_set():
            return False
        self._set(state="paused", live=None, message=(
            "PAUSED. Move the robot, aim it, then Resume (space). Esc = stop."))
        while self.pause_evt.is_set() and not self.stop_evt.is_set():
            time.sleep(0.05)
        if self.stop_evt.is_set():
            return False
        self._reset_yaw()        # it was moved, so its new direction is straight
        self._set(state="running", message=walking)
        return True

    def _session(self):
        dm = self.dm
        if self.mode == "C":
            with self.lock:
                self.net = QNet(self.rng)
                self.target = self.net.copy()
                self.buffer.clear()
                self.updates, self.episodes_total = 0, 0
                self.history, self.losses = [], []
        train = self.mode != "R"
        walking = ("TRAINING" if train else "RUNNING the learned policy") + \
                  ". Space = pause, Esc = stop."
        with self.lock:
            self.train, self.run_rewards, self.readout = train, [], None
            self.log.clear()
        self._set(state="running", message=walking)
        finished = False
        try:
            finished = (self._train_loop if train else self._run_loop)(dm, walking)
        finally:
            stop_motors(dm)
            if train:
                self.save()
                done = (f"Done: trained {NUM_EPISODES} episodes" if finished else "Stopped")
                done += f". Saved ({self.episodes_total} episodes in total)."
            else:
                done = f"Done: ran {POLICY_STEPS} steps." if finished else "Stopped."
            self._set(state="ready", live=None, message=(
                done + " Aim the robot and click Train, Run or Clear & train."))

    def _learn(self, yaw, a, r, yaw2):
        """Store the step, then do UPDATES_PER_STEP gradient steps on random
        batches from the replay buffer. Returns the readout for the dashboard."""
        with self.lock:
            self.buffer.append((yaw, a, r, yaw2))
            q_before = float(self.net.q([yaw])[0, a])
            target_q = float(self.target.q([yaw2])[0].max())
            data = np.array(self.buffer)
            losses = []
            for _ in range(UPDATES_PER_STEP):
                idx = self.rng.integers(len(data), size=min(BATCH, len(data)))
                b = data[idx]
                targets = b[:, 2] + GAMMA * self.target.q(b[:, 3]).max(axis=1)
                losses.append(self.net.train_step(b[:, 0], b[:, 1].astype(int), targets))
                self.updates += 1
                if self.updates % TARGET_SYNC == 0:
                    self.target = self.net.copy()
            q_after = float(self.net.q([yaw])[0, a])
        return dict(yaw=yaw, a=a, yaw2=yaw2, r=r, target_q=target_q,
                    target=r + GAMMA * target_q, q_before=q_before, q_after=q_after,
                    loss=float(np.mean(losses)))

    def _do_step(self, dm, yaw, eps, learn):
        """One act-wait-observe(-learn) step. Returns (next yaw, reward, loss)
        or None if Pause/Stop cut it short."""
        a = choose(self.net, yaw, eps)
        take_action(dm, a)
        if not wait(dm, TURN_WAIT + SETTLE, self._abort, self._on_sample):
            return None
        yaw2 = read_yaw(dm)
        s, s2 = to_state(yaw), to_state(yaw2)
        r = float(REWARDS[s2] if REWARD_NEXT_STATE else REWARDS[s])
        info = self._learn(yaw, a, r, yaw2) if learn else None
        with self.lock:
            if info:
                self.readout = info
            self.yaw, self.cur_state = yaw2, s2
            self.last_action, self.last_reward = a, r
            where = f"ep {self.ep:>2}" if learn else "run  "
            self.log.append(f"{where} step {self.step + 1:>3}  yaw {yaw:+6.1f}° -> "
                            f"{ACTION_NAMES[a]:<13} -> yaw {yaw2:+6.1f}° ({STATE_NAMES[s2]})  "
                            f"r {r:+.0f}" + (f"  loss {info['loss']:.2f}" if info else ""))
            self.version += 1
        return yaw2, r, (info["loss"] if info else None)

    def _train_loop(self, dm, walking):
        total_steps = NUM_STEPS * NUM_EPISODES
        counter = 0
        for ep in range(NUM_EPISODES):
            self._reset_yaw()                       # every episode starts "straight"
            yaw = read_yaw(dm)
            self._set(ep=ep + 1, step=0, ep_reward=0.0, yaw=yaw, cur_state=to_state(yaw))
            step, ep_reward, ep_losses = 0, 0.0, []
            while step < NUM_STEPS:
                if self.stop_evt.is_set():
                    return False
                eps = eps_at(counter / total_steps)
                self._set(eps=eps)
                result = self._do_step(dm, yaw, eps, learn=True)
                if result is None:                  # interrupted: discard this step
                    if not self._pause_if_needed(walking):
                        return False
                    yaw = read_yaw(dm)
                    self._set(yaw=yaw, cur_state=to_state(yaw))
                    continue
                yaw, r, loss = result
                step += 1
                counter += 1
                ep_reward += r
                ep_losses.append(loss)
                self._set(step=step, ep_reward=ep_reward)
            with self.lock:
                self.episodes_total += 1
                self.history.append(ep_reward)
                self.losses.append(float(np.mean(ep_losses)))
                self.log.append(f"── episode {ep + 1}/{NUM_EPISODES} done: "
                                f"total reward {ep_reward:+.0f}, resetting yaw")
                self.version += 1
            self.save()
        return True

    def _run_loop(self, dm, walking):
        self._reset_yaw()
        yaw = read_yaw(dm)
        self._set(ep=0, step=0, eps=0.0, yaw=yaw, cur_state=to_state(yaw), ep_reward=0.0)
        step, total = 0, 0.0
        while step < POLICY_STEPS:
            if self.stop_evt.is_set():
                return False
            result = self._do_step(dm, yaw, 0.0, learn=False)
            if result is None:
                if not self._pause_if_needed(walking):
                    return False
                yaw = read_yaw(dm)
                self._set(yaw=yaw, cur_state=to_state(yaw))
                continue
            yaw, r, _ = result
            step += 1
            total += r
            with self.lock:
                self.run_rewards.append(r)
                self.step, self.ep_reward = step, total
                self.version += 1
        return True


# ── Dashboard ─────────────────────────────────────────────────────────────────
FONT, MONO = "Helvetica", "Menlo"
SURFACE, CARD, BORDER = "#fcfcfb", "#ffffff", "#e4e3df"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
ZONE = "#f0efec"                       # shading for the two "drifted" zones
ACTION_COLORS = ["#2a78d6", "#eb6834"]  # Move to LEFT = blue, Move to RIGHT = orange
LINE = "#3a3a38"                       # single-series charts
BTN_BLUE, BTN_AMBER, BTN_RED, BTN_OFF = "#2a78d6", "#fab219", "#d03b3b", "#e4e3df"
STATE_COLORS = {"connecting": MUTED, "ready": INK2, "running": "#2a78d6",
                "paused": "#c98a00", "error": BTN_RED}


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def nice_ticks(lo, hi, n=4):
    """About n round-number ticks covering lo..hi."""
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    return list(np.arange(np.ceil(lo / step) * step, hi + step / 2, step))


class FlatButton(tk.Label):
    """A colored button (macOS ignores colors on real tk/ttk buttons)."""

    def __init__(self, parent, text, color, command, fg="white"):
        super().__init__(parent, text=text, bg=color, fg=fg, padx=16, pady=8,
                         font=(FONT, 14, "bold"), cursor="hand2")
        self.color, self.fg, self.command, self.enabled = color, fg, command, True
        self.bind("<Button-1>", lambda e: self.enabled and self.command())
        self.bind("<Enter>", lambda e: self.enabled and
                  self.config(bg=mix(self.color, "#000000", 0.15)))
        self.bind("<Leave>", lambda e: self.config(
            bg=self.color if self.enabled else BTN_OFF))

    def set_enabled(self, on):
        if on != self.enabled:
            self.enabled = on
            self.config(bg=self.color if on else BTN_OFF, fg=self.fg if on else MUTED,
                        cursor="hand2" if on else "arrow")


def card(parent, title):
    f = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1,
                 padx=12, pady=8)
    tk.Label(f, text=title, bg=CARD, fg=INK, font=(FONT, 14, "bold"),
             anchor="w").pack(fill="x")
    return f


def eq_line(parent, label, text, mono=True):
    row = tk.Frame(parent, bg=CARD)
    row.pack(fill="x", pady=(3, 0))
    tk.Label(row, bg=CARD, fg=INK2, font=(FONT, 13), text=label, width=9,
             anchor="w").pack(side="left")
    tk.Label(row, bg=CARD, fg=INK, font=(MONO if mono else FONT, 13),
             text=text).pack(side="left")
    return row


class LineChart:
    """Small single-series chart (reward per episode, loss per episode)."""

    def __init__(self, parent, title, w=440, h=124):
        self.f = card(parent, title)
        self.title = self.f.winfo_children()[0]
        self.c = tk.Canvas(self.f, width=w, height=h, bg=CARD, highlightthickness=0)
        self.c.pack(fill="x", pady=(2, 0))
        self.hover = tk.Label(self.f, text="", bg=CARD, fg=INK, font=(FONT, 12), anchor="w")
        self.hover.pack(fill="x")
        self.c.bind("<Motion>", self._hover)
        self.c.bind("<Leave>", lambda e: self.hover.config(text=""))
        self.x0, self.x1, self.y0, self.y1 = 48, w - 10, 10, h - 22
        self.data, self.first, self.what, self.fmt = [], 0, "", "{:+g}"

    def draw(self, data, lo, hi, first, what, unit, fmt="{:+g}", title=None):
        c = self.c
        c.delete("all")
        if title:
            self.title.config(text=title)
        self.data, self.first, self.what, self.fmt = data, first, what, fmt
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        Y = lambda v: y1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (y1 - y0)
        for v in nice_ticks(lo, hi, 3):
            c.create_line(x0, Y(v), x1, Y(v), fill=INK2 if v == 0 else BORDER)
            c.create_text(x0 - 6, Y(v), text=f"{v:g}", anchor="e", fill=INK2, font=(FONT, 11))
        if not data:
            c.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=f"No {unit}s yet",
                          fill=MUTED, font=(FONT, 12))
            return
        n = len(data)
        X = lambda i: x0 + (i / max(n - 1, 1)) * (x1 - x0)
        if n > 1:
            c.create_line(*[v for i, d in enumerate(data) for v in (X(i), Y(d))],
                          fill=LINE, width=2)
        if n <= 40:
            for i, d in enumerate(data):
                c.create_oval(X(i) - 4, Y(d) - 4, X(i) + 4, Y(d) + 4, fill=LINE,
                              outline=CARD, width=2)
        c.create_text(x0, y1 + 12, text=f"{unit} {first}", anchor="w", fill=INK2,
                      font=(FONT, 11))
        c.create_text(x1, y1 + 12, text=f"{first + n - 1}", anchor="e", fill=INK2,
                      font=(FONT, 11))

    def _hover(self, e):
        if not self.data:
            return
        n = len(self.data)
        i = int(round((min(max(e.x, self.x0), self.x1) - self.x0)
                      / (self.x1 - self.x0) * (n - 1)))
        self.hover.config(text=f"{self.what} {self.first + i}: "
                               + self.fmt.format(self.data[i]))


# Q-curve plot geometry
QW, QH = 640, 250
QX0, QX1, QY0, QY1 = 56, QW - 14, 28, 188     # plot area
STRIP_Y = 198                                  # policy strip


class Dashboard:
    def __init__(self, root, trainer):
        self.root, self.tr = root, trainer
        self.seen_version = -1
        self.snap = None
        self.curve = None
        root.title("Pascal neural net")
        root.configure(bg=SURFACE)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<space>", lambda e: self.tr.toggle_pause())
        root.bind("<Escape>", lambda e: self.tr.stop())

        # top bar
        top = tk.Frame(root, bg=SURFACE, padx=14, pady=8)
        top.pack(fill="x")
        tk.Label(top, text="Pascal neural net", bg=SURFACE, fg=INK,
                 font=(FONT, 20, "bold")).pack(side="left", padx=(0, 18))
        self.b_train = FlatButton(top, "Train", BTN_BLUE, lambda: self.tr.begin("T"))
        self.b_run = FlatButton(top, "Run", BTN_BLUE, lambda: self.tr.begin("R"))
        self.b_clear = FlatButton(top, "Clear & train", BTN_BLUE, self.clear_and_train)
        for b in (self.b_train, self.b_run, self.b_clear):
            b.pack(side="left", padx=4)
        self.b_stop = FlatButton(top, "Stop  (Esc)", BTN_RED, self.tr.stop)
        self.b_stop.pack(side="right", padx=4)
        self.b_pause = FlatButton(top, "Pause  (space)", BTN_AMBER,
                                  self.tr.toggle_pause, fg=INK)
        self.b_pause.pack(side="right", padx=4)

        # status line
        st = tk.Frame(root, bg=SURFACE, padx=14)
        st.pack(fill="x")
        self.state_pill = tk.Label(st, text="", fg="white", font=(FONT, 12, "bold"),
                                   padx=8, pady=2)
        self.state_pill.pack(side="left")
        self.status = tk.Label(st, text=self.tr.load_note, bg=SURFACE, fg=INK,
                               font=(FONT, 14), anchor="w", width=1)
        self.status.pack(side="left", padx=10, fill="x", expand=True)

        # the math and the settings
        eq = tk.Frame(root, bg=CARD, highlightbackground=BORDER, highlightthickness=1,
                      padx=12, pady=5)
        eq.pack(fill="x", padx=14, pady=(8, 0))
        which = "the state it moved to" if REWARD_NEXT_STATE else "the state it was in"
        eq_line(eq, "Reward:", "   ".join(
            f"{n} r = {REWARDS[i]:+d}" for i, n in enumerate(STATE_NAMES))
            + f"     (any action; for {which})")
        eq_line(eq, "Learning:", "loss = ( Q(yaw,a) − [ r + γ·max Q_target(yaw′,a′) ] )²"
                "   → gradient descent")
        eq_line(eq, "Network:", f"yaw/{YAW_INPUT_SCALE:g} → "
                + " → ".join(f"{h} tanh" for h in HIDDEN)
                + " → Q(LEFT), Q(RIGHT)      "
                f"replay batches of {BATCH}, {UPDATES_PER_STEP} updates per step, "
                f"target copied every {TARGET_SYNC} updates", mono=False)
        row = eq_line(eq, "Settings:", f"learning rate (Adam) = {LEARNING_RATE}      "
                      f"γ (discount) = {GAMMA}      ", mono=False)
        self.eps_label = tk.Label(row, bg=CARD, fg=INK, font=(FONT, 13, "bold"),
                                  width=20, anchor="w")
        self.eps_label.pack(side="left")
        tk.Label(row, bg=CARD, fg=INK2, font=(FONT, 13), text=(
            f"1 = always random, 0 = always the best-known action.   Each step one leg "
            f"turns {TURN_DEGREES}°, wait {TURN_WAIT} s;  Train = {NUM_EPISODES}×{NUM_STEPS} "
            f"steps")).pack(side="left")

        body = tk.Frame(root, bg=SURFACE, padx=14, pady=8)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=SURFACE)
        left.pack(side="left", fill="y")
        right = tk.Frame(body, bg=SURFACE)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.build_qplot(left)
        self.build_table(left)
        self.build_now(right)
        self.build_gauge(right)
        self.reward_chart = LineChart(right, "Total reward per training episode")
        self.reward_chart.f.pack(fill="x", pady=(10, 0))
        self.loss_chart = LineChart(right, "Training loss per episode (lower = fits better)")
        self.loss_chart.f.pack(fill="x", pady=(10, 0))
        self.build_log(left)
        self.tick()

    # -- buttons --------------------------------------------------------------
    def clear_and_train(self):
        if self.tr.episodes_total and not messagebox.askyesno(
                "Start a new network?",
                f"Throw away {self.tr.episodes_total} episodes of training and start "
                "from a random network?"):
            return
        self.tr.begin("C")

    def close(self):
        self.tr.quit()
        self.tr.thread.join(timeout=3)
        self.root.destroy()

    # -- Q-curve plot: what the network thinks at every yaw -------------------
    def qx(self, yaw):   # positive yaw = drifted LEFT, so it's drawn on the left
        return QX0 + (PLOT_RANGE - np.clip(yaw, -PLOT_RANGE, PLOT_RANGE)) \
            / (2 * PLOT_RANGE) * (QX1 - QX0)

    def build_qplot(self, parent):
        f = card(parent, "What the network has learned: Q at every yaw")
        f.pack(fill="x")
        self.qc = tk.Canvas(f, width=QW, height=QH, bg=CARD, highlightthickness=0)
        self.qc.pack(pady=(2, 0))
        self.q_hover_hint = ("Hover over the plot to read the network.   "
                             "Gray ticks at the bottom = yaws it has experienced.")
        self.qhover = tk.Label(f, text=self.q_hover_hint, bg=CARD, fg=INK2,
                               font=(FONT, 12), anchor="w")
        self.qhover.pack(fill="x")
        self.qc.bind("<Motion>", self.hover_qplot)
        self.qc.bind("<Leave>", lambda e: (self.qc.delete("cross"),
                                           self.qhover.config(text=self.q_hover_hint, fg=INK2)))

    def draw_qplot(self, snap):
        c = self.qc
        c.delete("all")
        yaws = np.linspace(PLOT_RANGE, -PLOT_RANGE, 181)
        Q = snap["net"].q(yaws)
        self.curve = (yaws, Q)
        lo, hi = float(min(Q.min(), 0)), float(max(Q.max(), 1))
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        self.qY = Y = lambda v: QY1 - (v - lo) / (hi - lo) * (QY1 - QY0)
        # state zones
        for a_, b_ in ((PLOT_RANGE, YAW_LEFT), (YAW_RIGHT, -PLOT_RANGE)):
            c.create_rectangle(self.qx(a_), QY0, self.qx(b_), QY1, fill=ZONE, outline="")
        for txt, mid in (("Drifted Left", (PLOT_RANGE + YAW_LEFT) / 2),
                         ("Straight", 0), ("Drifted Right", (YAW_RIGHT - PLOT_RANGE) / 2)):
            c.create_text(self.qx(mid), QY0 - 14, text=txt, fill=INK2, font=(FONT, 12))
        # grid
        for v in nice_ticks(lo, hi, 4):
            c.create_line(QX0, Y(v), QX1, Y(v), fill=INK2 if v == 0 else BORDER)
            c.create_text(QX0 - 6, Y(v), text=f"{v:g}", anchor="e", fill=INK2, font=(FONT, 11))
        c.create_text(14, (QY0 + QY1) / 2, text="Q", fill=INK2, font=(FONT, 13, "bold"))
        # experience rug
        for y in snap["buffer_yaws"][-400:]:
            if abs(y) <= PLOT_RANGE:
                x = self.qx(y)
                c.create_line(x, QY1 - 7, x, QY1, fill="#b9b8b3")
        # the two Q curves, with direct labels at the left end
        ends = []
        for a in range(N_ACTIONS):
            pts = [v for yv, q in zip(yaws, Q[:, a]) for v in (self.qx(yv), Y(q))]
            c.create_line(*pts, fill=ACTION_COLORS[a], width=2)
            ends.append(Y(Q[0, a]))
        if abs(ends[0] - ends[1]) < 16:          # nudge apart if they would collide
            mid = (ends[0] + ends[1]) / 2
            up = 0 if ends[0] <= ends[1] else 1
            ends[up], ends[1 - up] = mid - 8, mid + 8
        for a in range(N_ACTIONS):
            c.create_text(QX0 + 6, ends[a] - 10, text=ACTION_NAMES[a], anchor="w",
                          fill=INK, font=(FONT, 12, "bold"))
            c.create_line(QX0 + 6, ends[a] - 2, QX0 + 26, ends[a] - 2,
                          fill=ACTION_COLORS[a], width=3)
        # policy strip: best action at each yaw
        best = np.argmax(Q, axis=1)
        for i in range(len(yaws) - 1):
            c.create_rectangle(self.qx(yaws[i]), STRIP_Y, self.qx(yaws[i + 1]) + 1,
                               STRIP_Y + 12, fill=ACTION_COLORS[best[i]], outline="")
        c.create_text(QX0 - 6, STRIP_Y + 6, text="best", anchor="e", fill=INK2,
                      font=(FONT, 11))
        # x axis
        for d in (PLOT_RANGE, 30, YAW_LEFT, 0, YAW_RIGHT, -30, -PLOT_RANGE):
            c.create_text(self.qx(d), STRIP_Y + 26, text=f"{d:+d}°" if d else "0°",
                          fill=INK2, font=(FONT, 11))
        c.create_text((QX0 + QX1) / 2, STRIP_Y + 44, text="yaw since the episode started",
                      fill=INK2, font=(FONT, 11))
        # where the robot is now
        if snap["yaw"] is not None and snap["state"] in ("running", "paused"):
            x = self.qx(snap["yaw"])
            c.create_line(x, QY0, x, STRIP_Y + 12, fill=INK, width=2, dash=(4, 3))
            c.create_text(x + 4, STRIP_Y - 4, text="now", anchor="sw",
                          fill=INK, font=(FONT, 11, "bold"))

    def hover_qplot(self, e):
        if self.curve is None or not (QX0 <= e.x <= QX1):
            return
        yaw = PLOT_RANGE - (e.x - QX0) / (QX1 - QX0) * 2 * PLOT_RANGE
        q = self.snap["net"].q([yaw])[0]
        best = int(np.argmax(q))
        self.qc.delete("cross")
        self.qc.create_line(e.x, QY0, e.x, QY1, fill=MUTED, tags="cross")
        for a in range(N_ACTIONS):
            y = self.qY(q[a])
            self.qc.create_oval(e.x - 5, y - 5, e.x + 5, y + 5, fill=ACTION_COLORS[a],
                                outline=CARD, width=2, tags="cross")
        self.qhover.config(fg=INK, text=(
            f"yaw {yaw:+.1f}° ({STATE_NAMES[to_state(yaw)]}):   "
            f"Q(LEFT) = {q[0]:.2f}   Q(RIGHT) = {q[1]:.2f}   → best: {ACTION_NAMES[best]}"))

    # -- the network summarized as a Q-table, plus the last update -----------
    def build_table(self, parent):
        f = card(parent, "The network as a Q-table (average Q over each state's yaws)")
        f.pack(fill="x", pady=(10, 0))
        self.tc = tk.Canvas(f, width=QW, height=112, bg=CARD, highlightthickness=0)
        self.tc.pack(pady=(2, 0))
        self.readout = tk.Label(f, text="Start training to see each step's update here.\n",
                                bg=CARD, fg=INK, font=(MONO, 12), justify="left", anchor="w")
        self.readout.pack(fill="x", pady=(4, 0))

    def draw_table(self, snap):
        c = self.tc
        c.delete("all")
        LW, CWd, RH, HEAD = 220, 200, 28, 26
        for a in range(N_ACTIONS):
            x = LW + a * CWd
            c.create_rectangle(x + 30, HEAD / 2 - 2, x + 46, HEAD / 2 + 2,
                               fill=ACTION_COLORS[a], outline="")
            c.create_text(x + 52, HEAD / 2, text=ACTION_NAMES[a], anchor="w", fill=INK,
                          font=(FONT, 13, "bold"))
        net = snap["net"]
        ranges = [(YAW_LEFT, PLOT_RANGE), (YAW_RIGHT, YAW_LEFT), (-PLOT_RANGE, YAW_RIGHT)]
        for s, (a_, b_) in enumerate(ranges):
            y = HEAD + s * RH
            c.create_text(8, y + RH / 2, anchor="w", fill=INK, font=(FONT, 13, "bold"),
                          text=STATE_NAMES[s])
            c.create_text(120, y + RH / 2, anchor="w", fill=INK2, font=(FONT, 12),
                          text=state_range(s))
            q = net.q(np.linspace(a_, b_, 40)).mean(axis=0)
            best = int(np.argmax(q))
            for a in range(N_ACTIONS):
                x = LW + a * CWd
                fill = mix(CARD, ACTION_COLORS[a], 0.22) if a == best else "#f7f6f3"
                c.create_rectangle(x + 2, y + 2, x + CWd - 2, y + RH - 2, fill=fill,
                                   outline="")
                c.create_text(x + CWd / 2, y + RH / 2, fill=INK,
                              text=("★ " if a == best else "") + f"{q[a]:.2f}",
                              font=(MONO, 14, "bold" if a == best else "normal"))
        r = snap["readout"]
        if r is None:
            self.readout.config(text=(
                "Running the learned policy: the network is not changed.\n"
                if not snap["train"] and snap["state"] in ("running", "paused")
                else "Start training to see each step's update here.\n"))
        else:
            self.readout.config(text=(
                f"Last step: yaw {r['yaw']:+.1f}° → {ACTION_NAMES[r['a']]} → yaw {r['yaw2']:+.1f}°,"
                f"  r = {r['r']:+.0f},  target = {r['r']:+.0f} + {GAMMA}·{r['target_q']:.2f}"
                f" = {r['target']:.2f}\n"
                f"Q(yaw, {ACTION_NAMES[r['a']]}): {r['q_before']:.2f} → {r['q_after']:.2f}"
                f" after {UPDATES_PER_STEP} batch updates   (mean loss {r['loss']:.2f},"
                f" buffer {snap['buffer_len']}, updates {snap['updates']})"))

    # -- current numbers ------------------------------------------------------
    def build_now(self, parent):
        f = card(parent, "Now")
        f.pack(fill="x")
        grid = tk.Frame(f, bg=CARD)
        grid.pack(fill="x", pady=(2, 0))
        self.now_vals = {}
        names = ["Episode", "Step", "Yaw", "State", "Action", "Reward"]
        for i, name in enumerate(names):
            col = tk.Frame(grid, bg=CARD)
            col.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 24), pady=0)
            tk.Label(col, text=name, bg=CARD, fg=INK2, font=(FONT, 11)).pack(anchor="w")
            v = tk.Label(col, text="–", bg=CARD, fg=INK, font=(FONT, 16, "bold"))
            v.pack(anchor="w")
            self.now_vals[name] = v

    def update_now(self, snap):
        active = snap["state"] in ("running", "paused")
        dash = "–"
        if active and snap["train"]:
            episode, step = f"{snap['ep']} / {NUM_EPISODES}", f"{snap['step']} / {NUM_STEPS}"
        elif active:
            episode, step = "Run", f"{snap['step']} / {POLICY_STEPS}"
        else:
            episode, step = f"{snap['episodes_total']} total", dash
        vals = {
            "Episode": episode, "Step": step,
            "Yaw": f"{snap['yaw']:+.1f}°" if active and snap["yaw"] is not None else dash,
            "State": STATE_NAMES[snap["cur_state"]] if active and snap["cur_state"] is not None else dash,
            "Action": ACTION_NAMES[snap["last_action"]] if active and snap["last_action"] is not None else dash,
            "Reward": f"{snap['last_reward']:+.0f}" if active and snap["last_reward"] is not None else dash,
        }
        for k, v in vals.items():
            self.now_vals[k].config(text=v)
        if snap["last_action"] is not None and active:
            self.now_vals["Action"].config(fg=ACTION_COLORS[snap["last_action"]])
        else:
            self.now_vals["Action"].config(fg=INK)

    # -- yaw gauge (redrawn every tick) ---------------------------------------
    def build_gauge(self, parent):
        f = card(parent, "Yaw since the episode started")
        f.pack(fill="x", pady=(10, 0))
        w = 440
        self.gc = tk.Canvas(f, width=w, height=84, bg=CARD, highlightthickness=0)
        self.gc.pack(fill="x", pady=(2, 0))
        c = self.gc
        self.gx = lambda d: w / 2 - np.clip(d, -PLOT_RANGE, PLOT_RANGE) / PLOT_RANGE * (w / 2 - 16)
        for a_, b_ in ((PLOT_RANGE, YAW_LEFT), (YAW_RIGHT, -PLOT_RANGE)):
            c.create_rectangle(self.gx(a_), 26, self.gx(b_), 50, fill=ZONE, outline="")
        c.create_line(self.gx(PLOT_RANGE), 38, self.gx(-PLOT_RANGE), 38, fill=BORDER, width=2)
        for d in (PLOT_RANGE, YAW_LEFT, 0, YAW_RIGHT, -PLOT_RANGE):
            c.create_line(self.gx(d), 30, self.gx(d), 46, fill=MUTED)
            c.create_text(self.gx(d), 60, text=f"{d:+d}°" if d else "0°", fill=INK2,
                          font=(FONT, 11))
        mid_l = (self.gx(PLOT_RANGE) + self.gx(YAW_LEFT)) / 2
        mid_r = (self.gx(YAW_RIGHT) + self.gx(-PLOT_RANGE)) / 2
        c.create_text(mid_l, 76, text="Drifted Left", fill=INK2, font=(FONT, 12))
        c.create_text(self.gx(0), 76, text="Straight", fill=INK2, font=(FONT, 12))
        c.create_text(mid_r, 76, text="Drifted Right", fill=INK2, font=(FONT, 12))

    def draw_gauge(self, snap):
        c = self.gc
        c.delete("dyn")
        if snap["state"] not in ("running", "paused"):
            return
        if snap["live"] is not None:
            x = self.gx(snap["live"])
            c.create_line(x, 26, x, 50, fill=MUTED, width=2, tags="dyn")
        if snap["yaw"] is not None:
            x = self.gx(snap["yaw"])
            c.create_polygon(x - 8, 4, x + 8, 4, x, 20, fill=INK, tags="dyn")
            c.create_line(x, 20, x, 50, fill=INK, width=2, tags="dyn")
            right_side = x < 300
            c.create_text(x + 12 if right_side else x - 12, 11, text=f"{snap['yaw']:+.1f}°",
                          anchor="w" if right_side else "e", fill=INK,
                          font=(FONT, 13, "bold"), tags="dyn")

    # -- charts and log -------------------------------------------------------
    def draw_charts(self, snap):
        if snap["train"]:
            data = snap["history"][-200:]
            first = len(snap["history"]) - len(data)
            self.reward_chart.draw(data, min(REWARDS) * NUM_STEPS, max(REWARDS) * NUM_STEPS,
                                   first, "Episode", "episode", "total reward {:+g}",
                                   title="Total reward per training episode")
        else:
            data = list(np.cumsum(snap["run_rewards"]))
            self.reward_chart.draw(data, min([0.0] + data), max([100.0] + data), 1,
                                   "Step", "step", "total reward so far {:+g}",
                                   title="Total reward so far (Run)")
        losses = snap["losses"][-200:]
        first = len(snap["losses"]) - len(losses)
        self.loss_chart.draw(losses, 0.0, max([1.0] + losses) * 1.1, first, "Episode",
                             "episode", "mean loss {:.2f}")

    def build_log(self, parent):
        f = card(parent, "Recent steps")
        f.pack(fill="both", expand=True, pady=(10, 0))
        self.logbox = tk.Text(f, height=4, width=80, bg=CARD, fg=INK, bd=0, wrap="none",
                              font=(MONO, 11), highlightthickness=0)
        self.logbox.pack(fill="both", expand=True, pady=(2, 0))
        self.logbox.config(state="disabled")

    def update_log(self, snap):
        self.logbox.config(state="normal")
        self.logbox.delete("1.0", "end")
        self.logbox.insert("end", "\n".join(snap["log"]))
        self.logbox.config(state="disabled")

    # -- refresh loop ---------------------------------------------------------
    def tick(self):
        snap = self.snap = self.tr.snapshot()
        state = snap["state"]
        ready, active = state == "ready", state in ("running", "paused")
        for b in (self.b_train, self.b_run, self.b_clear):
            b.set_enabled(ready)
        self.b_pause.set_enabled(active)
        self.b_stop.set_enabled(active)
        self.b_pause.config(text="Resume  (space)" if snap["paused"] else "Pause  (space)")
        label = {"running": "TRAINING" if snap["train"] else "RUNNING"}.get(
            state, state.upper())
        self.state_pill.config(text=label, bg=STATE_COLORS[state])
        self.eps_label.config(text=f"ε (exploring) = {snap['eps']:.2f}" if active
                              else f"ε (exploring) = {EPS_START:g} → {EPS_END:g}")
        if snap["version"] != self.seen_version:
            if snap["version"] > 0:
                self.status.config(text=snap["message"])
            self.seen_version = snap["version"]
            self.draw_qplot(snap)
            self.draw_table(snap)
            self.update_now(snap)
            self.draw_charts(snap)
            self.update_log(snap)
        self.draw_gauge(snap)
        self.root.after(50, self.tick)


def main():
    trainer = Trainer()
    root = tk.Tk()
    dash = Dashboard(root, trainer)
    trainer.thread.start()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        dash.close()


if __name__ == "__main__":
    main()
