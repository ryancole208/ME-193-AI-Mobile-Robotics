"""
Where the bowler stands and where they aim, from the webcam (preferred) or keyboard.

  * position   -- lateral release point on the foul line (m). Webcam: the wrist's
                  sideways position (same POSE_X_RANGE calibration as ping pong).
                  Keyboard: ←/→.
  * preset aim -- A/D turn the aim, like Wii's A + D-pad.
  * swing aim  -- sideways wrist motion during the forward swing adds a small angle.

The game reads the position at throw onset and the swing aim around the release, so
PoseTracker snapshots are recorded every frame into a short history.
"""

from bisect import bisect_right
from collections import deque

import config
from pose_tracker import Smoother, map_to_table

STAND_RANGE_M = config.LANE_WIDTH_M - 2 * config.BALL_RADIUS_M


class BowlerInput:
    def __init__(self, pose=None):
        self.pose = pose
        self.keyboard_x = 0.0
        self.preset_aim_deg = 0.0
        self.kb_strength = config.KEYBOARD_THROW_STRENGTH
        self._smoother = Smoother()
        self._t = deque()
        self._wrist = deque()     # mirrored, normalised wrist x
        self._pos = deque()       # smoothed stand position (m)
        self._last_snap_t = None

    @property
    def tracking(self):
        return self.pose is not None and self.pose.tracking

    @property
    def source(self):
        return "pose" if self.tracking else "keyboard"

    # -- per-frame update ---------------------------------------------------------
    def update(self, t, dt, *, left=False, right=False, aim_left=False, aim_right=False):
        half = STAND_RANGE_M / 2
        step = ((1 if right else 0) - (1 if left else 0)) * config.KEYBOARD_BOWLER_SPEED_M_S * dt
        self.keyboard_x = max(-half, min(half, self.keyboard_x + step))
        turn = ((1 if aim_right else 0) - (1 if aim_left else 0)) * config.PRESET_AIM_RATE_DEG_S * dt
        m = config.PRESET_AIM_MAX_DEG
        self.preset_aim_deg = max(-m, min(m, self.preset_aim_deg + turn))
        if self.pose is not None:
            self.add_snapshot(self.pose.snapshot())

    def change_strength(self, steps):
        lo, hi = config.KEYBOARD_STRENGTH_RANGE
        self.kb_strength = max(lo, min(hi, self.kb_strength + steps * config.KEYBOARD_STRENGTH_STEP))

    def add_snapshot(self, snap):
        """Record one PoseSnapshot (duplicates of the same camera frame are skipped)."""
        if snap is None or snap.t == self._last_snap_t:
            return
        self._last_snap_t = snap.t
        if not snap.tracking or snap.wrist_x is None:
            self._smoother.reset()
            return
        pos = self._smoother.update(map_to_table(snap.wrist_x, config.POSE_X_RANGE, STAND_RANGE_M))
        self._t.append(snap.t)
        self._wrist.append(snap.wrist_x)
        self._pos.append(pos)
        while self._t and snap.t - self._t[0] > config.POSE_HISTORY_S * 2:
            self._t.popleft(); self._wrist.popleft(); self._pos.popleft()

    # -- queries ------------------------------------------------------------------
    def _index_at(self, t, max_age_s=0.5):
        if not self._t:
            return None
        i = max(0, bisect_right(self._t, t) - 1)
        return i if abs(self._t[i] - t) <= max_age_s else None

    def position_at(self, t):
        i = self._index_at(t)
        return self._pos[i] if i is not None else self.keyboard_x

    def current_position(self):
        if self.tracking and self._pos:
            return self._pos[-1]
        return self.keyboard_x

    def swing_aim_deg(self, t_release):
        """Angle from sideways wrist motion around the release (0 without webcam tracking)."""
        a = self._index_at(t_release - config.SWING_AIM_WINDOW_S)
        b = self._index_at(t_release + config.SWING_AIM_AFTER_S)
        if a is None or b is None or b <= a:
            return 0.0
        dx = self._wrist[b] - self._wrist[a]
        if abs(dx) < config.SWING_AIM_DEADZONE:
            return 0.0
        m = config.SWING_AIM_MAX_DEG
        return max(-m, min(m, dx * config.SWING_AIM_DEG_PER_UNIT))

    def launch_angle_deg(self, t_release):
        m = config.LAUNCH_ANGLE_MAX_DEG
        return max(-m, min(m, self.preset_aim_deg + self.swing_aim_deg(t_release)))
