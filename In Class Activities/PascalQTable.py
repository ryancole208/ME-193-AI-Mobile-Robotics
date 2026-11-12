"""
PascalQTable: the QLearn web app's Silly Walk Q-table, as a desktop dashboard
============================================================================

Same robot, states, actions, rewards and settings as the "Silly Walk" tab of
the QLearn web app (app.html in this folder), but run from Python on your
computer, with a live dashboard window.

    States  (from the hub yaw, re-zeroed at the start of every episode)
        Drifted Left    yaw >  10°
        Straight        -10° .. 10°
        Drifted Right   yaw < -10°
    Actions
        Move to LEFT    right motor turns 135°
        Move to RIGHT   left  motor turns 135°
    Each step: act, wait 0.3 s (+0.2 s to settle), read yaw, update Q.
    Reward depends only on the state the step ends in (any action): -1, +10, -1
    (set REWARD_NEXT_STATE = False to reward the starting state, like the web app).
    10 steps per episode, 10 episodes. Exploring (epsilon) falls in a
    straight line from 1 to 0.05 over all 100 steps.

Run it:
    python3 -m pip install "legoeducation>=1.2.0" numpy
    python3 PascalQTable.py
(on Windows: py -m pip install ... and py PascalQTable.py). Turn the Double Motor on
first; it connects to the first one it finds (set SERIAL to pick one). The
learning is saved next to this file.

Buttons: Train (10 episodes, keeps the saved table), Run (100 greedy steps),
Clear & train (forget the table first). Pause = space bar, Stop = Esc.
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

# ── Settings (same as the QLearn web app) ─────────────────────────────────────
SERIAL = None          # Connection Card serial (e.g. 2279), or None = first hub found

# Turn settings
TURN_DEGREES = 135     # how far one motor turns per step
TURN_WAIT    = 0.3     # s to wait for the turn to finish
SETTLE       = 0.2     # s extra so the yaw reading settles (the web app does this too)

# Training settings
NUM_STEPS    = 10      # steps per episode
NUM_EPISODES = 10      # episodes per Train click
POLICY_STEPS = 100     # steps per Run click

# Q-learning parameters
ALPHA = 0.1            # learning rate
GAMMA = 0.9            # discount factor

# Exploration: falls in a straight line from start to end over all training steps
EPS_START = 1.0
EPS_END   = 0.05

# States (yaw thresholds, degrees). Positive yaw = drifted left, as in the web app.
YAW_LEFT  = 10         # yaw >  YAW_LEFT  -> Drifted Left
YAW_RIGHT = -10        # yaw <  YAW_RIGHT -> Drifted Right
STATE_NAMES  = ["Drifted Left", "Straight", "Drifted Right"]
ACTION_NAMES = ["Move to LEFT", "Move to RIGHT"]

# Rewards per state (any action)
REWARDS = [-1, 10, -1]
# True = reward for the state the action led to (standard Q-learning: the
# action gets credit for its result straight away).
# False = reward for the state the robot was in when it acted (what the QLearn
# web app does; same best policy in the end, but slower to learn).
REWARD_NEXT_STATE = True

YAW_SCALE   = 0.1      # hub reports yaw in tenths of a degree
SAMPLE_DT   = 0.02     # s between yaw reads while waiting
QTABLE_FILE = HERE / "pascal_qtable.npz"

N_STATES, N_ACTIONS = len(STATE_NAMES), len(ACTION_NAMES)


# ── Q-learning helpers ────────────────────────────────────────────────────────
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


def choose(Q, s, eps):
    if np.random.random() < eps:
        return np.random.randint(N_ACTIONS)
    best = np.flatnonzero(Q[s] == Q[s].max())   # break ties at random
    return int(np.random.choice(best))


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
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.N = np.zeros((N_STATES, N_ACTIONS), dtype=int)   # tries per cell
        self.episodes_total = 0
        self.history = []        # total reward per training episode (saved)
        self.load_note = self._load()

        self.state = "connecting"   # connecting, ready, running, paused, error
        self.message = "Connecting to the Double Motor..."
        self.mode, self.train = None, True
        self.ep, self.step, self.eps = 0, 0, 0.0
        self.yaw, self.live = None, None
        self.cur_state, self.last_action, self.last_reward = None, None, None
        self.ep_reward = 0.0
        self.run_rewards = []    # reward each step in Run mode
        self.updated, self.bellman = None, None
        self.log = deque(maxlen=8)
        self.version = 0

        self.start_evt = threading.Event()
        self.pause_evt = threading.Event()
        self.stop_evt = threading.Event()
        self.quit_evt = threading.Event()
        self.dm = None
        self.thread = threading.Thread(target=self._main, daemon=True)

    # -- saving ---------------------------------------------------------------
    def _load(self):
        if not QTABLE_FILE.exists():
            return "No saved Q-table yet."
        data = np.load(QTABLE_FILE)
        if data["Q"].shape != self.Q.shape:
            return "Saved Q-table doesn't match the current settings; ignoring it."
        self.Q, self.N = data["Q"].copy(), data["N"].copy()
        self.episodes_total = int(data["episodes"])
        self.history = list(data["history"])
        return f"Loaded the saved Q-table ({self.episodes_total} episodes of training)."

    def save(self):
        with self.lock:
            np.savez(QTABLE_FILE, Q=self.Q, N=self.N, episodes=self.episodes_total,
                     history=np.array(self.history))

    # -- called by the dashboard ----------------------------------------------
    def snapshot(self):
        with self.lock:
            return dict(
                Q=self.Q.copy(), N=self.N.copy(), episodes_total=self.episodes_total,
                history=list(self.history), run_rewards=list(self.run_rewards),
                train=self.train, state=self.state, message=self.message,
                ep=self.ep, step=self.step, eps=self.eps, yaw=self.yaw, live=self.live,
                cur_state=self.cur_state, last_action=self.last_action,
                last_reward=self.last_reward, ep_reward=self.ep_reward,
                updated=self.updated, bellman=self.bellman, log=list(self.log),
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
        time.sleep(0.3)          # let a fresh IMU reading arrive (as the web app does)

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
                self.Q[:], self.N[:] = 0, 0
                self.episodes_total, self.history = 0, []
        train = self.mode != "R"
        walking = ("TRAINING" if train else "RUNNING the learned policy") + \
                  ". Space = pause, Esc = stop."
        with self.lock:
            self.train, self.run_rewards, self.bellman, self.updated = train, [], None, None
            self.log.clear()
        self._set(state="running", message=walking)
        finished = False
        try:
            if train:
                finished = self._train_loop(dm, walking)
            else:
                finished = self._run_loop(dm, walking)
        finally:
            stop_motors(dm)
            if train:
                self.save()
            if train:
                done = (f"Done: trained {NUM_EPISODES} episodes" if finished else "Stopped")
                done += f". Saved ({self.episodes_total} episodes in total)."
            else:
                done = f"Done: ran {POLICY_STEPS} steps." if finished else "Stopped."
            self._set(state="ready", live=None, message=(
                done + " Aim the robot and click Train, Run or Clear & train."))

    def _do_step(self, dm, s, eps, learn):
        """One act-wait-observe(-learn) step. Returns (next_state, yaw, reward)
        or None if Pause/Stop cut it short."""
        a = choose(self.Q, s, eps)
        take_action(dm, a)
        if not wait(dm, TURN_WAIT + SETTLE, self._abort, self._on_sample):
            return None
        yaw = read_yaw(dm)
        s2 = to_state(yaw)
        r = float(REWARDS[s2] if REWARD_NEXT_STATE else REWARDS[s])
        with self.lock:
            if learn:
                old, nxt = self.Q[s, a], self.Q[s2].max()
                self.Q[s, a] += ALPHA * (r + GAMMA * nxt - self.Q[s, a])
                self.N[s, a] += 1
                self.bellman = (s, a, s2, old, r, nxt, self.Q[s, a])
                self.updated = (s, a)
            self.yaw, self.cur_state = yaw, s2
            self.last_action, self.last_reward = a, r
            where = f"ep {self.ep:>2}" if learn else "run  "
            self.log.append(f"{where} step {self.step + 1:>3}  "
                            f"{STATE_NAMES[s]:<13} -> {ACTION_NAMES[a]:<13} -> "
                            f"{STATE_NAMES[s2]:<13} yaw {yaw:+6.1f}°  r {r:+.0f}")
            self.version += 1
        return s2, yaw, r

    def _train_loop(self, dm, walking):
        total_steps = NUM_STEPS * NUM_EPISODES
        counter = 0
        for ep in range(NUM_EPISODES):
            self._reset_yaw()                       # every episode starts "straight"
            yaw = read_yaw(dm)
            s = to_state(yaw)
            self._set(ep=ep + 1, step=0, ep_reward=0.0, yaw=yaw, cur_state=s)
            step, ep_reward = 0, 0.0
            while step < NUM_STEPS:
                if self.stop_evt.is_set():
                    return False
                eps = eps_at(counter / total_steps)
                self._set(eps=eps)
                result = self._do_step(dm, s, eps, learn=True)
                if result is None:                  # interrupted: discard this step
                    if not self._pause_if_needed(walking):
                        return False
                    yaw = read_yaw(dm)
                    s = to_state(yaw)
                    self._set(yaw=yaw, cur_state=s)
                    continue
                s, yaw, r = result
                step += 1
                counter += 1
                ep_reward += r
                self._set(step=step, ep_reward=ep_reward)
            with self.lock:
                self.episodes_total += 1
                self.history.append(ep_reward)
                self.log.append(f"── episode {ep + 1}/{NUM_EPISODES} done: "
                                f"total reward {ep_reward:+.0f}, resetting yaw")
                self.version += 1
            self.save()
        return True

    def _run_loop(self, dm, walking):
        self._reset_yaw()
        yaw = read_yaw(dm)
        s = to_state(yaw)
        self._set(ep=0, step=0, eps=0.0, yaw=yaw, cur_state=s, ep_reward=0.0)
        step, total = 0, 0.0
        while step < POLICY_STEPS:
            if self.stop_evt.is_set():
                return False
            result = self._do_step(dm, s, 0.0, learn=False)
            if result is None:
                if not self._pause_if_needed(walking):
                    return False
                yaw = read_yaw(dm)
                s = to_state(yaw)
                self._set(yaw=yaw, cur_state=s)
                continue
            s, yaw, r = result
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
# Q values: diverging blue (good) <-> red (bad) around a neutral gray at 0
NEUTRAL, GOOD, BAD = "#f0efec", "#2a78d6", "#e34948"
STRAIGHT_ZONE = "#cde2fb"
BTN_BLUE, BTN_AMBER, BTN_RED, BTN_OFF = "#2a78d6", "#fab219", "#d03b3b", "#e4e3df"
STATE_COLORS = {"connecting": MUTED, "ready": INK2, "running": GOOD,
                "paused": "#c98a00", "error": BTN_RED}

# Q-table geometry: label column, reward column, two action columns
LBL_W, REW_W, ARW, CW, RH, HEAD = 190, 100, 26, 170, 84, 56
TABLE_W = LBL_W + REW_W + ARW + N_ACTIONS * CW + 4
TABLE_H = HEAD + N_STATES * RH + 70
GAUGE_RANGE = 45       # degrees each side on the yaw gauge


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def q_colors(q, vmax):
    """(fill, text) for a Q value, scaled so |q| = vmax gets the full color."""
    t = min(1.0, abs(q) / vmax)
    return mix(NEUTRAL, GOOD if q > 0 else BAD, t), ("white" if t > 0.55 else INK)


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
                 padx=12, pady=10)
    tk.Label(f, text=title, bg=CARD, fg=INK, font=(FONT, 14, "bold"),
             anchor="w").pack(fill="x")
    return f


def eq_line(parent, label, text, mono=True):
    row = tk.Frame(parent, bg=CARD)
    row.pack(fill="x", pady=(3, 0))
    tk.Label(row, bg=CARD, fg=INK2, font=(FONT, 13), text=label, width=9,
             anchor="w").pack(side="left")
    lab = tk.Label(row, bg=CARD, fg=INK, font=(MONO if mono else FONT, 13), text=text)
    lab.pack(side="left")
    return row


class Dashboard:
    def __init__(self, root, trainer):
        self.root, self.tr = root, trainer
        self.seen_version = -1
        self.snap = None
        root.title("Pascal Q-table")
        root.configure(bg=SURFACE)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<space>", lambda e: self.tr.toggle_pause())
        root.bind("<Escape>", lambda e: self.tr.stop())

        # top bar
        top = tk.Frame(root, bg=SURFACE, padx=14, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="Pascal Q-table", bg=SURFACE, fg=INK,
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
                               font=(FONT, 14), anchor="w", width=1)  # width=1: never widens the window
        self.status.pack(side="left", padx=10, fill="x", expand=True)

        # the math and the settings
        eq = tk.Frame(root, bg=CARD, highlightbackground=BORDER, highlightthickness=1,
                      padx=12, pady=6)
        eq.pack(fill="x", padx=14, pady=(8, 0))
        which = "the state it moved to" if REWARD_NEXT_STATE else "the state it was in"
        eq_line(eq, "Reward:", "   ".join(
            f"{n} r = {REWARDS[i]:+d}" for i, n in enumerate(STATE_NAMES))
            + f"     (any action; for {which})")
        eq_line(eq, "Learning:", "Q(s,a) ← Q(s,a) + α·[ r + γ·max Q(s′,a′) − Q(s,a) ]")
        row = eq_line(eq, "Settings:", f"α (learning rate) = {ALPHA}      "
                      f"γ (discount) = {GAMMA}      ", mono=False)
        self.eps_label = tk.Label(row, bg=CARD, fg=INK, font=(FONT, 13, "bold"),
                                  width=20, anchor="w")
        self.eps_label.pack(side="left")
        tk.Label(row, bg=CARD, fg=INK2, font=(FONT, 13), text=(
            f"    1 = always random, 0 = always the best-known action   "
            f"(falls {EPS_START:g} → {EPS_END:g} while training)")).pack(side="left")
        eq_line(eq, "Steps:", f"each step one leg turns {TURN_DEGREES}°, then wait "
                f"{TURN_WAIT} s (+{SETTLE} s settle).   Train = {NUM_EPISODES} episodes × "
                f"{NUM_STEPS} steps,  Run = {POLICY_STEPS} steps.", mono=False)

        body = tk.Frame(root, bg=SURFACE, padx=14, pady=10)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=SURFACE)
        left.pack(side="left", fill="y")
        right = tk.Frame(body, bg=SURFACE)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.build_table(left)
        self.build_now(right)
        self.build_gauge(right)
        self.build_chart(right)
        self.build_log(left)
        self.tick()

    # -- buttons --------------------------------------------------------------
    def clear_and_train(self):
        if self.tr.episodes_total and not messagebox.askyesno(
                "Clear the Q-table?",
                f"Throw away {self.tr.episodes_total} episodes of training and start over?"):
            return
        self.tr.begin("C")

    def close(self):
        self.tr.quit()
        self.tr.thread.join(timeout=3)   # let it stop the legs and save
        self.root.destroy()

    # -- Q-table --------------------------------------------------------------
    def cell_xy(self, s, a):
        return LBL_W + REW_W + ARW + a * CW, HEAD + s * RH

    def build_table(self, parent):
        f = card(parent, "Q-table")
        f.pack(fill="x")
        c = self.table = tk.Canvas(f, width=TABLE_W, height=TABLE_H, bg=CARD,
                                   highlightthickness=0)
        c.pack(pady=(4, 0))
        c.create_text(LBL_W / 2, HEAD / 2, text="State (yaw)", fill=INK2, font=(FONT, 13))
        c.create_text(LBL_W + REW_W / 2, HEAD / 2, text="Reward", fill=INK2, font=(FONT, 13))
        x0 = LBL_W + REW_W + ARW
        for a, name in enumerate(ACTION_NAMES):
            self_x = x0 + a * CW + CW / 2
            c.create_text(self_x, HEAD / 2, text=name, fill=INK, font=(FONT, 15, "bold"))
        self.cells = {}
        for s, name in enumerate(STATE_NAMES):
            y = HEAD + s * RH
            c.create_rectangle(2, y + 2, LBL_W - 4, y + RH - 2, fill=NEUTRAL, outline="")
            c.create_text(LBL_W / 2, y + RH / 2 - 11, text=name, fill=INK,
                          font=(FONT, 16, "bold"))
            c.create_text(LBL_W / 2, y + RH / 2 + 14, text=state_range(s), fill=INK2,
                          font=(FONT, 13))
            c.create_text(LBL_W + REW_W / 2, y + RH / 2, text=f"{REWARDS[s]:+d}",
                          fill=INK, font=(MONO, 18))
            for a in range(N_ACTIONS):
                x, _ = self.cell_xy(s, a)
                r = c.create_rectangle(x + 2, y + 2, x + CW - 2, y + RH - 2,
                                       fill=NEUTRAL, outline="")
                tx = c.create_text(x + CW / 2, y + RH / 2 - 6, text="0.00", fill=INK,
                                   font=(MONO, 20))
                sub = c.create_text(x + CW / 2, y + RH / 2 + 22, text="", fill=INK2,
                                    font=(FONT, 11))
                self.cells[s, a] = (r, tx, sub)
        self.updated_box = c.create_rectangle(0, 0, 0, 0, outline=INK, width=3,
                                              state="hidden")
        self.now_arrow = c.create_text(0, 0, text="▶", fill=GOOD, font=(FONT, 20),
                                       state="hidden")
        ky = HEAD + N_STATES * RH + 14
        self.key_text = c.create_text(0, ky, anchor="w", fill=INK2, font=(FONT, 12), text="")
        c.create_text(0, ky + 24, anchor="w", fill=INK2, font=(FONT, 12),
                      text="★ best choice     ▶ robot is here     "
                           "black box = just updated     number under = times tried")
        self.bellman_label = tk.Label(f, text="Start training to see each step's update here.",
                                      bg=CARD, fg=INK, font=(MONO, 13), justify="left",
                                      anchor="w")
        self.bellman_label.pack(fill="x", pady=(4, 0))

    def update_table(self, snap):
        c, Q, N = self.table, snap["Q"], snap["N"]
        vmax = max(1.0, float(np.abs(Q).max()))
        for s in range(N_STATES):
            best = int(np.argmax(Q[s])) if N[s].any() else -1
            for a in range(N_ACTIONS):
                r, tx, sub = self.cells[s, a]
                fill, ink = q_colors(Q[s, a], vmax)
                star = "★ " if a == best else ""
                c.itemconfig(r, fill=fill)
                c.itemconfig(tx, text=f"{star}{Q[s, a]:.2f}", fill=ink,
                             font=(MONO, 20, "bold" if star else "normal"))
                c.itemconfig(sub, text=f"tried {N[s, a]}×" if N[s, a] else "never tried",
                             fill="white" if ink == "white" else INK2)
        c.itemconfig(self.key_text, text=(
            f"Color: blue = good, red = bad, full color at ±{vmax:.1f} (the biggest |Q| so far)"))
        if snap["updated"] is not None:
            x, y = self.cell_xy(*snap["updated"])
            c.coords(self.updated_box, x + 2, y + 2, x + CW - 2, y + RH - 2)
            c.itemconfig(self.updated_box, state="normal")
        else:
            c.itemconfig(self.updated_box, state="hidden")
        if snap["cur_state"] is not None and snap["state"] in ("running", "paused"):
            x, y = self.cell_xy(snap["cur_state"], 0)
            c.coords(self.now_arrow, x - ARW / 2, y + RH / 2)
            c.itemconfig(self.now_arrow, state="normal")
        else:
            c.itemconfig(self.now_arrow, state="hidden")
        b = snap["bellman"]
        if b is None:
            self.bellman_label.config(text=(
                "Running the learned policy: the table is not changed.\n"
                if not snap["train"] and snap["state"] in ("running", "paused")
                else "Start training to see each step's update here.\n"))
        else:
            s, a, s2, old, r, nxt, new = b
            self.bellman_label.config(text=(
                f"Last update:  {STATE_NAMES[s]} → {ACTION_NAMES[a]} → {STATE_NAMES[s2]}\n"
                f"Q ← {old:.2f} + {ALPHA}·[ {r:+.0f} + {GAMMA}·{nxt:.2f} − {old:.2f} ]"
                f" = {new:.2f}"))

    # -- current numbers ------------------------------------------------------
    def build_now(self, parent):
        f = card(parent, "Now")
        f.pack(fill="x")
        grid = tk.Frame(f, bg=CARD)
        grid.pack(fill="x", pady=(4, 0))
        self.now_vals = {}
        names = ["Episode", "Step", "Yaw", "State", "Action", "Reward"]
        for i, name in enumerate(names):
            col = tk.Frame(grid, bg=CARD)
            col.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 26), pady=1)
            tk.Label(col, text=name, bg=CARD, fg=INK2, font=(FONT, 11)).pack(anchor="w")
            v = tk.Label(col, text="–", bg=CARD, fg=INK, font=(FONT, 17, "bold"))
            v.pack(anchor="w")
            self.now_vals[name] = v

    def update_now(self, snap):
        active = snap["state"] in ("running", "paused")
        dash = "–"
        if active and snap["train"]:
            episode = f"{snap['ep']} / {NUM_EPISODES}"
            step = f"{snap['step']} / {NUM_STEPS}"
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

    # -- yaw gauge (redrawn every tick) ---------------------------------------
    def build_gauge(self, parent):
        f = card(parent, "Yaw since the episode started")
        f.pack(fill="x", pady=(12, 0))
        w = 440
        self.gc = tk.Canvas(f, width=w, height=96, bg=CARD, highlightthickness=0)
        self.gc.pack(fill="x", pady=(4, 0))
        tk.Label(f, text="gray tick = live reading    blue ▼ = reading used for the state",
                 bg=CARD, fg=INK2, font=(FONT, 11), anchor="w").pack(fill="x")
        c = self.gc
        # positive yaw = drifted LEFT, so it's drawn on the left
        self.gx = lambda d: w / 2 - np.clip(d, -GAUGE_RANGE, GAUGE_RANGE) / GAUGE_RANGE * (w / 2 - 16)
        c.create_rectangle(self.gx(YAW_LEFT), 30, self.gx(YAW_RIGHT), 56,
                           fill=STRAIGHT_ZONE, outline="")
        c.create_line(self.gx(GAUGE_RANGE), 43, self.gx(-GAUGE_RANGE), 43, fill=BORDER, width=2)
        for d in (GAUGE_RANGE, YAW_LEFT, 0, YAW_RIGHT, -GAUGE_RANGE):
            c.create_line(self.gx(d), 34, self.gx(d), 52, fill=MUTED)
            c.create_text(self.gx(d), 66, text=f"{d:+d}°" if d else "0°", fill=INK2,
                          font=(FONT, 11))
        mid_l = (self.gx(GAUGE_RANGE) + self.gx(YAW_LEFT)) / 2
        mid_r = (self.gx(YAW_RIGHT) + self.gx(-GAUGE_RANGE)) / 2
        c.create_text(mid_l, 84, text="Drifted Left", fill=INK2, font=(FONT, 12))
        c.create_text(self.gx(0), 84, text="Straight", fill=INK2, font=(FONT, 12))
        c.create_text(mid_r, 84, text="Drifted Right", fill=INK2, font=(FONT, 12))

    def draw_gauge(self, snap):
        c = self.gc
        c.delete("dyn")
        if snap["state"] not in ("running", "paused"):
            return
        if snap["live"] is not None:
            x = self.gx(snap["live"])
            c.create_line(x, 30, x, 56, fill=MUTED, width=2, tags="dyn")
        if snap["yaw"] is not None:
            x = self.gx(snap["yaw"])
            c.create_polygon(x - 8, 8, x + 8, 8, x, 24, fill=GOOD, tags="dyn")
            c.create_line(x, 24, x, 56, fill=GOOD, width=2, tags="dyn")
            right_side = x < 300
            c.create_text(x + 12 if right_side else x - 12, 15, text=f"{snap['yaw']:+.1f}°",
                          anchor="w" if right_side else "e", fill=INK,
                          font=(FONT, 13, "bold"), tags="dyn")

    # -- reward chart ---------------------------------------------------------
    def build_chart(self, parent):
        f = card(parent, "")
        f.pack(fill="x", pady=(12, 0))
        self.chart_title = f.winfo_children()[0]
        self.cc = tk.Canvas(f, width=440, height=170, bg=CARD, highlightthickness=0)
        self.cc.pack(fill="x", pady=(4, 0))
        self.chart_hover = tk.Label(f, text="", bg=CARD, fg=INK, font=(FONT, 12),
                                    anchor="w")
        self.chart_hover.pack(fill="x")
        self.cc.bind("<Motion>", self.hover_chart)
        self.cc.bind("<Leave>", lambda e: self.chart_hover.config(text=""))
        self.chart_data, self.chart_what = [], ""

    X0, X1, Y0, Y1 = 44, 430, 24, 146

    def draw_chart(self, snap):
        c = self.cc
        c.delete("all")
        if snap["train"]:
            data = snap["history"][-200:]
            first = len(snap["history"]) - len(data)
            lo, hi = min(REWARDS) * NUM_STEPS, max(REWARDS) * NUM_STEPS
            self.chart_title.config(text="Total reward per training episode")
            self.chart_what, unit = "Episode", "episode"
        else:
            data = list(np.cumsum(snap["run_rewards"]))
            first = 1
            lo = min([0.0] + data)
            hi = max([max(REWARDS) * 10.0] + data)
            self.chart_title.config(text="Total reward so far (Run)")
            self.chart_what, unit = "Step", "step"
        self.chart_data, self.chart_first = data, first
        x0, x1, y0, y1 = self.X0, self.X1, self.Y0, self.Y1
        Y = lambda v: y1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (y1 - y0)
        for v in sorted({lo, 0, hi}):
            c.create_line(x0, Y(v), x1, Y(v), fill=INK2 if v == 0 else BORDER)
            if v == 0 or abs(Y(v) - Y(0)) > 14:      # skip labels that would overlap
                c.create_text(x0 - 6, Y(v), text=f"{v:+g}" if v else "0", anchor="e",
                              fill=INK2, font=(FONT, 11))
        if not data:
            c.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=f"No {unit}s yet",
                          fill=MUTED, font=(FONT, 12))
            return
        n = len(data)
        X = lambda i: x0 + (i / max(n - 1, 1)) * (x1 - x0)
        if n > 1:
            c.create_line(*[v for i, d in enumerate(data) for v in (X(i), Y(d))],
                          fill=GOOD, width=2)
        for i, d in enumerate(data):
            if n <= 40:
                c.create_oval(X(i) - 4, Y(d) - 4, X(i) + 4, Y(d) + 4, fill=GOOD,
                              outline=CARD, width=2)
        c.create_text(x0, y1 + 14, text=f"{unit} {first}", anchor="w", fill=INK2,
                      font=(FONT, 11))
        c.create_text(x1, y1 + 14, text=f"{first + n - 1}", anchor="e", fill=INK2,
                      font=(FONT, 11))

    def hover_chart(self, e):
        if not self.chart_data:
            return
        n = len(self.chart_data)
        i = int(round((min(max(e.x, self.X0), self.X1) - self.X0)
                      / (self.X1 - self.X0) * (n - 1)))
        what = "reward" if self.chart_what == "Episode" else "total reward so far"
        self.chart_hover.config(text=f"{self.chart_what} {self.chart_first + i}: "
                                     f"{what} {self.chart_data[i]:+g}")

    # -- recent steps ---------------------------------------------------------
    def build_log(self, parent):
        f = card(parent, "Recent steps")
        f.pack(fill="both", expand=True, pady=(12, 0))
        self.logbox = tk.Text(f, height=8, width=70, bg=CARD, fg=INK, bd=0, wrap="none",
                              font=(MONO, 11), highlightthickness=0)
        self.logbox.pack(fill="both", expand=True, pady=(4, 0))
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
        if active:
            eps_text = f"ε (exploring) = {snap['eps']:.2f}"
        else:
            eps_text = f"ε (exploring) = {EPS_START:g} → {EPS_END:g}"
        self.eps_label.config(text=eps_text)
        if snap["version"] != self.seen_version:
            if snap["version"] > 0:
                self.status.config(text=snap["message"])
            self.seen_version = snap["version"]
            self.update_table(snap)
            self.update_now(snap)
            self.draw_chart(snap)
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
