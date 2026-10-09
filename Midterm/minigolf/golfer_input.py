"""
Where the golfer aims: motor twist, keyboard and webcam.

  * aim        -- heading of the putt in degrees (0 = north/+y, + = clockwise/east)
                  = base aim + twist.
                  base aim: the suggested line, turned with ←/→ or A/D (Shift = fine).
                  twist:    twisting the Double Motor turns the putter face
                            (AIM_TWIST_GAIN, at most AIM_TWIST_MAX_DEG either way). It is
                            re-zeroed whenever a new putt is set up, or with recenter().
                  (Optional, AIM_POSE_JOYSTICK: hold your wrist left/right of centre to
                  turn the base aim like a joystick.)
  * push/pull  -- sideways wrist motion around impact adds a few degrees.
  * power      -- keyboard only: hold Space to charge the meter, release to putt.

The game reads the aim when the swing fires and the push around impact, so the aim
and PoseTracker snapshots are recorded every frame into a short history.
"""

from bisect import bisect_right
from collections import deque

import config
from pose_tracker import Smoother, map_to_table


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class GolferInput:
    def __init__(self, pose=None, twist=None):
        """twist: object with .twist_deg (degrees or None) and .recenter_twist(), e.g. a GolfMotor."""
        self.pose = pose
        self.twist = twist
        self.base_aim_deg = 0.0
        self.twist_offset_deg = 0.0
        self.aim_deg = 0.0
        self.stick = 0.0              # wrist joystick -1..1 (0 inside the deadzone)
        self.charge_t = None          # keyboard: time Space went down (None = not charging)
        self._smoother = Smoother()
        self._t = deque()
        self._wrist = deque()         # mirrored, normalised wrist x
        self._aim_t = deque()
        self._aim = deque()
        self._last_snap_t = None

    @property
    def tracking(self):
        return self.pose is not None and self.pose.tracking

    @property
    def source(self):
        return "pose" if self.tracking else "keyboard"

    # -- per-frame update ---------------------------------------------------------
    @property
    def twisting(self):
        return self.twist is not None and self.twist.twist_deg is not None

    def recenter(self):
        """Whatever way the motor is twisted now becomes 'straight at the base aim'."""
        if self.twist is not None:
            self.twist.recenter_twist()
        self.twist_offset_deg = 0.0
        self.aim_deg = wrap_deg(self.base_aim_deg)

    def set_aim(self, deg, t=None):
        self.base_aim_deg = wrap_deg(deg)
        self.recenter()
        if t is not None:
            self._aim_t.clear()
            self._aim.clear()
            self._record_aim(t)

    def update(self, t, dt, *, left=False, right=False, fine=False, frozen=False):
        """frozen=True (ball rolling, menu, ...) records the aim but doesn't turn it."""
        if self.pose is not None:
            self.add_snapshot(self.pose.snapshot())
        if not frozen:
            rate = config.AIM_KEY_RATE_DEG_S * (config.AIM_KEY_FINE_FACTOR if fine else 1.0)
            turn = ((1 if right else 0) - (1 if left else 0)) * rate
            if self.tracking and config.AIM_POSE_JOYSTICK:
                turn += self.stick * config.AIM_POSE_RATE_DEG_S
            self.base_aim_deg = wrap_deg(self.base_aim_deg + turn * dt)
            tw = self.twist.twist_deg if self.twist is not None else None
            if tw is not None:
                m = config.AIM_TWIST_MAX_DEG
                self.twist_offset_deg = max(-m, min(m, config.AIM_TWIST_GAIN * tw))
            self.aim_deg = wrap_deg(self.base_aim_deg + self.twist_offset_deg)
        self._record_aim(t)

    def _record_aim(self, t):
        self._aim_t.append(t)
        self._aim.append(self.aim_deg)
        while self._aim_t and t - self._aim_t[0] > config.POSE_HISTORY_S * 2:
            self._aim_t.popleft(); self._aim.popleft()

    def add_snapshot(self, snap):
        """Record one PoseSnapshot (duplicates of the same camera frame are skipped)."""
        if snap is None or snap.t == self._last_snap_t:
            return
        self._last_snap_t = snap.t
        if not snap.tracking or snap.wrist_x is None:
            self._smoother.reset()
            self.stick = 0.0
            return
        u = self._smoother.update(map_to_table(snap.wrist_x, config.POSE_X_RANGE, 2.0))   # -1..1
        dz = config.AIM_POSE_DEADZONE
        self.stick = 0.0 if abs(u) <= dz else (abs(u) - dz) / (1 - dz) * (1 if u > 0 else -1)
        self._t.append(snap.t)
        self._wrist.append(snap.wrist_x)
        while self._t and snap.t - self._t[0] > config.POSE_HISTORY_S * 2:
            self._t.popleft(); self._wrist.popleft()

    # -- keyboard power meter -------------------------------------------------------
    def begin_charge(self, t):
        if self.charge_t is None:
            self.charge_t = t

    def charge_fraction(self, t):
        """0..1, sweeping up over KEYBOARD_CHARGE_S and back down while Space is held."""
        if self.charge_t is None:
            return 0.0
        k = ((t - self.charge_t) / config.KEYBOARD_CHARGE_S) % 2.0
        return k if k <= 1.0 else 2.0 - k

    def release_charge(self, t):
        """Strength of the keyboard putt (None if Space wasn't being held)."""
        if self.charge_t is None:
            return None
        f = self.charge_fraction(t)
        self.charge_t = None
        lo, hi = config.KEYBOARD_STRENGTH_RANGE
        return lo + f * (hi - lo)

    # -- queries ------------------------------------------------------------------
    @staticmethod
    def _index_at(times, t, max_age_s=0.5):
        if not times:
            return None
        i = max(0, bisect_right(times, t) - 1)
        return i if abs(times[i] - t) <= max_age_s else None

    def aim_at(self, t):
        i = self._index_at(self._aim_t, t, max_age_s=1.0)
        return self._aim[i] if i is not None else self.aim_deg

    def push_deg(self, t_impact):
        """Push/pull from sideways wrist motion around impact (0 without webcam tracking)."""
        a = self._index_at(self._t, t_impact - config.PUSH_WINDOW_S)
        b = self._index_at(self._t, t_impact + config.PUSH_AFTER_S)
        if a is None or b is None or b <= a:
            return 0.0
        dx = self._wrist[b] - self._wrist[a]
        if abs(dx) < config.PUSH_DEADZONE:
            return 0.0
        m = config.PUSH_MAX_DEG
        return max(-m, min(m, dx * config.PUSH_DEG_PER_UNIT))

    def putt_heading_deg(self, t_onset, t_impact):
        return wrap_deg(self.aim_at(t_onset) + self.push_deg(t_impact))
