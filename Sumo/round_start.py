"""
Round start: "armed" -> (start push) -> [countdown] -> "running".

  * Armed: motors stopped, nothing else happens. The round starts when either
    lever passes START_THRESHOLD. So a lever that is already pushed at power-on
    can't start the round by accident, both levers must first be seen centered.
  * Countdown (COUNTDOWN_S > 0 only; default 0 = none, your choice).
  * Running: the behavior tree is ticked.

Teleop gate: the start push itself must not drive the robot (it would lurch
forward). With REQUIRE_RECENTER_AFTER_START, teleop stays locked until both
levers have returned to center once after the start.
"""

import config

ARMED, COUNTDOWN, RUNNING = "armed", "countdown", "running"


def centered(l, r, limit=config.LEVER_CENTER):
    return abs(l) <= limit and abs(r) <= limit


class RoundStart:
    def __init__(self, *, threshold=config.START_THRESHOLD, countdown_s=config.COUNTDOWN_S,
                 require_recenter=config.REQUIRE_RECENTER_AFTER_START, log=print):
        self.threshold = threshold
        self.countdown_s = countdown_s
        self.phase = ARMED
        self.teleop_unlocked = not require_recenter
        self._seen_centered = False
        self._countdown_end = None
        self._last_count = None
        self.log = log

    def update(self, now, levers):
        """levers: (left %, right %) or None if the Controller has no fresh data.
        Returns the phase."""
        if self.phase == ARMED:
            if levers is None:
                return self.phase
            l, r = levers
            if centered(l, r):
                self._seen_centered = True
            elif self._seen_centered and max(abs(l), abs(r)) >= self.threshold:
                self.log(f"[start] start push detected (levers {l:+.0f}/{r:+.0f})")
                if self.countdown_s > 0:
                    self.phase = COUNTDOWN
                    self._countdown_end = now + self.countdown_s
                else:
                    self._begin()
        elif self.phase == COUNTDOWN:
            remaining = self._countdown_end - now
            count = int(remaining) + 1
            if remaining <= 0:
                self._begin()
            elif count != self._last_count:
                self._last_count = count
                self.log(f"[start] {count}...")
        if self.phase == RUNNING and not self.teleop_unlocked and levers is not None \
                and centered(*levers):
            self.teleop_unlocked = True
            self.log("[start] levers back at center -> teleop enabled")
        return self.phase

    def _begin(self):
        self.phase = RUNNING
        self.log("[start] ROUND STARTED" + ("" if self.teleop_unlocked else
                                             " (teleop locked until both levers return to center)"))
