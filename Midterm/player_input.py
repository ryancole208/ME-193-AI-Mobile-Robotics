"""
Combines the paddle-position sources the game can use:
  * PoseTracker (webcam wrist) -- preferred whenever it is tracking
  * KeyboardPaddle (arrow keys) -- fallback when the camera is off / no person
"""

from collections import deque

import config


class KeyboardPaddle:
    """Arrow-key paddle. update() is called every frame with which arrows are held."""

    def __init__(self, speed=config.KEYBOARD_PADDLE_SPEED_M_S, table_width=config.TABLE_WIDTH_M):
        self.speed = speed
        self.half = table_width / 2
        self.x = 0.0
        self.held = "center"
        self._history = deque(maxlen=240)  # (t, x)

    def update(self, t, dt, left, right):
        direction = (1 if right else 0) - (1 if left else 0)
        self.x = max(-self.half, min(self.half, self.x + direction * self.speed * dt))
        self.held = {1: "right", -1: "left"}.get(direction, "center")
        self._history.append((t, self.x))

    def x_at(self, t):
        for ts, x in reversed(self._history):
            if ts <= t:
                return x
        return self.x

    def direction_at(self, t):
        return self.held


class PlayerInput:
    def __init__(self, pose=None, keyboard=None):
        self.pose = pose
        self.keyboard = keyboard or KeyboardPaddle()

    def _pose_ok(self):
        return self.pose is not None and self.pose.tracking

    @property
    def source(self):
        return "pose" if self._pose_ok() else "keyboard"

    def paddle_x_at(self, t):
        if self._pose_ok():
            x = self.pose.paddle_x_at(t)
            if x is not None:
                return x
        return self.keyboard.x_at(t)

    def direction_at(self, t):
        if self._pose_ok():
            return self.pose.direction_at(t)
        return self.keyboard.direction_at(t)

    def current_paddle_x(self):
        if self._pose_ok():
            x = self.pose.snapshot().paddle_x
            if x is not None:
                return x
        return self.keyboard.x
