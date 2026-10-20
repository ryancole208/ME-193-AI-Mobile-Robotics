"""
The sumo behavior tree (matches the diagram in the README / assignment PNG).

    ? root Selector (priority, re-checked from the left every tick)
    |- 0  -> Robot out of ring?            -> Stop, publish lost, wait for song, exit
    |- 0b -> Color sensor data stale?      -> Stop (safety: we can't see the edge)
    |- 1  -> Blue while moving forward?    -> Drive backward 1 s (RUNNING until done)
    |- 2  -> Camera sees opponent (fresh)? -> Steer toward it
    |- 3  -> Lever past deadband?          -> Tank drive
    |- 4  Stop and wait (fallback)

Branch 0b is the "color sensor stale/disconnected => stop" rule from the spec.
It sits right under "out of ring" so it overrides everything else, including
the reverse reflex.

All nodes read the Blackboard `bb` (one snapshot of inputs per tick) and write
motor commands through bb.drive. They never touch Bluetooth or MQTT directly.
"""

from dataclasses import dataclass, field

import config
from behavior_tree import Action, Condition, Selector, Sequence, Status
from vision import chase_command, clamp_speed


@dataclass
class Blackboard:
    now: float = 0.0
    tracker: object = None          # LineTracker
    color_fresh: bool = True
    direction: object = None        # motion.Direction
    opponent: object = None         # vision.Opponent if fresh, else None
    levers: tuple | None = None     # (left %, right %) if the Controller is fresh
    teleop_unlocked: bool = False
    drive: object = None            # has set(left, right) and stop()
    link: object = None             # has announce_lost(reason), song_done, expects_song
    exit_requested: bool = False
    lost_reason: str | None = None
    notes: dict = field(default_factory=dict)


# Name of each root branch, in priority order, and the value it maps to in the
# MQTT robot/state "branch" field (the contract only knows reverse/chase/teleop/idle/null).
BRANCHES = ["out_of_ring", "color_safety", "reverse", "chase", "teleop", "idle"]
CONTRACT_BRANCH = {"out_of_ring": None, "color_safety": "idle", "reverse": "reverse",
                   "chase": "chase", "teleop": "teleop", "idle": "idle", None: None}


# ---------------------------------------------------------------------------
# Actions with state
# ---------------------------------------------------------------------------

class LostAction(Action):
    """Branch 0: stop, tell the UNO Q we lost, wait for the sad song, then ask to exit.

    RUNNING while waiting, so the tree keeps ticking (and keeps the motors
    stopped) instead of blocking the loop for up to 8 s.
    """

    def __init__(self, timeout_s=config.SONG_DONE_TIMEOUT_S):
        super().__init__("Stop, publish lost, wait for song_done, exit")
        self.timeout_s = timeout_s
        self._t0 = None

    def tick(self, bb):
        bb.drive.stop()
        if self._t0 is None:
            self._t0 = bb.now
            bb.lost_reason = bb.tracker.out_reason or "out of ring"
            bb.link.announce_lost(bb.lost_reason)
            if bb.link.expects_song:
                print(f"[lost] waiting up to {self.timeout_s:.0f} s for event/song_done ...")
        done = bb.link.song_done or not bb.link.expects_song
        timed_out = bb.now - self._t0 >= self.timeout_s
        if done or timed_out:
            if timed_out and not done:
                print("[lost] no song_done received (timeout); exiting anyway")
            bb.exit_requested = True
            return Status.SUCCESS
        return Status.RUNNING

    def halt(self):
        pass  # being out is final: nothing can interrupt this branch


class ReverseAction(Action):
    """Branch 1: back away from the edge for REVERSE_TIME_S, even if blue disappears."""

    def __init__(self, time_s=config.REVERSE_TIME_S, speed=config.REVERSE_SPEED):
        super().__init__(f"Drive backward for {time_s:g} s")
        self.time_s = time_s
        self.speed = speed
        self._t0 = None

    def tick(self, bb):
        if self._t0 is None:
            self._t0 = bb.now
        if bb.now - self._t0 >= self.time_s:
            self._t0 = None
            bb.drive.stop()
            return Status.SUCCESS
        bb.drive.set(-self.speed, -self.speed)
        return Status.RUNNING

    def halt(self):
        self._t0 = None


# ---------------------------------------------------------------------------
# Conditions and one-tick actions
# ---------------------------------------------------------------------------

def is_out(bb):
    return bb.tracker.is_out


def color_stale(bb):
    return not bb.color_fresh


def blue_while_forward(bb):
    # The tracker is ON_EDGE only if blue appeared while moving forward
    # (or uncertain, with LEAN_WHEN_UNCERTAIN = "in").
    return bb.tracker.zone.value == "on_edge"


def sees_opponent(bb):
    return bb.opponent is not None


def lever_active(bb):
    if bb.levers is None or not bb.teleop_unlocked:
        return False
    return max(abs(bb.levers[0]), abs(bb.levers[1])) > config.TELEOP_DEADBAND


def stop(bb):
    bb.drive.stop()
    return Status.SUCCESS


def chase(bb):
    bb.drive.set(*chase_command(bb.opponent))
    return Status.SUCCESS


def lever_to_speed(p, deadband=config.TELEOP_DEADBAND, max_speed=config.TELEOP_MAX_SPEED):
    """Lever % -> wheel %, proportional, with the deadband removed so that speed
    starts at 0 just past the deadband and reaches max_speed at full lever."""
    if abs(p) <= deadband:
        return 0.0
    mag = (abs(p) - deadband) / (100 - deadband) * max_speed
    return clamp_speed(mag if p > 0 else -mag)


def teleop(bb):
    l, r = bb.levers
    bb.drive.set(lever_to_speed(l), lever_to_speed(r))
    return Status.SUCCESS


def build_tree():
    return Selector("Root: priority selector", [
        Sequence("out_of_ring", [Condition("Robot out of ring?", is_out), LostAction()]),
        Sequence("color_safety", [Condition("Color sensor data stale?", color_stale),
                                  Action("Stop (can't see the edge)", stop)]),
        Sequence("reverse", [Condition("Color sensor sees blue while moving forward?", blue_while_forward),
                             ReverseAction()]),
        Sequence("chase", [Condition("Camera sees opponent (fresh)?", sees_opponent),
                           Action("Steer toward opponent", chase)]),
        Sequence("teleop", [Condition("Controller lever past deadband?", lever_active),
                            Action("Tank drive from levers", teleop)]),
        Action("idle", stop),
    ])


def active_branch(root):
    """Name of the root child that handled the last tick (see BRANCHES)."""
    child = root.active_child
    return None if child is None else child.name
