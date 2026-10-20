"""
Line-crossing state machine: are we inside the ring or out?

The floor is the same color inside and outside, so one reading can't tell us.
Instead we track the history of crossings of the blue tape. The color sensor is
at the FRONT of the robot, ahead of the wheels:

    back [wheels]------[sensor] front

    Event (debounced)            real direction   meaning                          new state
    ---------------------------  ---------------  -------------------------------  ---------
    INSIDE  + blue appears       forward          front reached the edge           ON_EDGE   (branch 1 reverses)
    INSIDE  + blue appears       backward         sensor is the LAST part to reach OUT
                                                  the line => whole robot is past it
              ... within RETAPE_WINDOW_S of backing off the tape: it is the SAME tape again
                  (reading flickering at its edge while we reverse) -> ON_EDGE, keep reversing
    ON_EDGE + floor again        forward          front crossed the line            OUT (if CROSS_FORWARD_IS_OUT)
              ... within EDGE_OVERSHOOT_GRACE_S of reaching the tape: momentum overshoot while
                  braking -> stay ON_EDGE (keep reversing); we come back over the tape and
                  "floor while backward" makes us INSIDE. Not back within OVERSHOOT_RECOVER_S -> OUT
    ON_EDGE + floor again        backward         backed off the line               INSIDE
    any     + uncertain direction                 lean per LEAN_WHEN_UNCERTAIN      ("in" = treat as staying inside)

The robot starts INSIDE. OUT is final (latched) for the rest of the round.
"""

from enum import Enum

import config
from motion import Direction


class Zone(Enum):
    INSIDE = "inside"        # sensor over the floor, robot inside
    ON_EDGE = "on_edge"      # sensor over the tape, robot still inside
    OUT = "out"


class LineTracker:
    def __init__(self, *, cross_forward_is_out=config.CROSS_FORWARD_IS_OUT,
                 lean=config.LEAN_WHEN_UNCERTAIN, grace_s=config.EDGE_OVERSHOOT_GRACE_S,
                 recover_s=config.OVERSHOOT_RECOVER_S, retape_s=config.RETAPE_WINDOW_S, log=print):
        if lean not in ("in", "out"):
            raise ValueError("LEAN_WHEN_UNCERTAIN must be 'in' or 'out'")
        self.cross_forward_is_out = cross_forward_is_out
        self.lean = lean
        self.grace_s = grace_s
        self.recover_s = recover_s
        self.retape_s = retape_s
        self.log = log
        self.zone = Zone.INSIDE
        self.out_reason = None
        self.entered_edge_forward = False   # True while ON_EDGE that began moving forward
        self.uncertain_events = 0
        self.edge_t = None                  # when we reached the tape (ON_EDGE), if `now` was given
        self.overshoot_t = None             # when momentum carried the sensor past the tape (else None)
        self.backed_off_t = None            # when we last backed off the tape (ON_EDGE -> INSIDE)

    @property
    def is_out(self):
        return self.zone is Zone.OUT

    def _resolve(self, direction, what):
        """Turn UNCERTAIN into FORWARD/BACKWARD according to the configured lean.

        For "blue appears", leaning IN means assuming FORWARD (front reached the edge).
        For "floor again", leaning IN means assuming BACKWARD (backed off the line).
        """
        if direction is not Direction.UNCERTAIN:
            return direction
        self.uncertain_events += 1
        if what == "blue":
            guess = Direction.FORWARD if self.lean == "in" else Direction.BACKWARD
        else:
            guess = Direction.BACKWARD if self.lean == "in" else Direction.FORWARD
        self.log(f"[line] direction UNCERTAIN when {what} appeared -> leaning {self.lean.upper()} "
                 f"(treated as {guess.value})")
        return guess

    def on_blue(self, direction, detail="", now=None):
        """Debounced change floor -> blue. `now` enables the overshoot grace period."""
        if self.zone is Zone.ON_EDGE and self.overshoot_t is not None:
            self.overshoot_t = None
            self.log(f"[line] back over the tape after the overshoot {detail}")
            return self.zone
        if self.zone is not Zone.INSIDE:
            return self.zone
        d = self._resolve(direction, "blue")
        if d is Direction.FORWARD:
            self.zone = Zone.ON_EDGE
            self.entered_edge_forward = True
            self.edge_t = now
            self.log(f"[line] blue while moving FORWARD -> front at the edge, still inside {detail}")
        elif self._recently_backed_off(now):
            self.zone = Zone.ON_EDGE
            self.entered_edge_forward = False
            self.edge_t = now
            self.log(f"[line] blue while moving BACKWARD {now - self.backed_off_t:.2f} s after backing off "
                     f"the tape (< {self.retape_s:g} s) -> same tape again, NOT out; keep reversing {detail}")
        else:
            self._go_out("blue appeared while moving backward (pushed out)", detail)
        return self.zone

    def on_floor(self, direction, detail="", now=None):
        """Debounced change blue -> floor."""
        if self.zone is not Zone.ON_EDGE:
            return self.zone
        d = self._resolve(direction, "floor")
        if d is Direction.BACKWARD:
            self.zone = Zone.INSIDE
            self.entered_edge_forward = False
            self.edge_t = self.overshoot_t = None
            self.backed_off_t = now
            self.log(f"[line] floor while moving BACKWARD -> backed off the line, inside {detail}")
        elif self.cross_forward_is_out and self._in_grace(now):
            self.overshoot_t = now
            self.log(f"[line] crossed the tape {now - self.edge_t:.2f} s after reaching it "
                     f"(< {self.grace_s:g} s grace) -> momentum overshoot, NOT out; keep reversing "
                     f"(out if not back over the tape within {self.recover_s:g} s) {detail}")
        elif self.cross_forward_is_out:
            self._go_out("crossed the line moving forward", detail)
        else:
            self.zone = Zone.INSIDE
            self.entered_edge_forward = False
            self.log(f"[line] floor while moving FORWARD (CROSS_FORWARD_IS_OUT off) -> assume inside {detail}")
        return self.zone

    def _in_grace(self, now):
        return (now is not None and self.edge_t is not None and self.grace_s > 0
                and now - self.edge_t <= self.grace_s)

    def _recently_backed_off(self, now):
        return (now is not None and self.backed_off_t is not None and self.retape_s > 0
                and now - self.backed_off_t <= self.retape_s)

    def update(self, now, detail=""):
        """Call every tick: an overshoot that never came back over the tape means out."""
        if (self.zone is Zone.ON_EDGE and self.overshoot_t is not None
                and now - self.overshoot_t > self.recover_s):
            self._go_out(f"crossed the line moving forward (overshoot: did not come back within "
                         f"{self.recover_s:g} s)", detail)

    def _go_out(self, reason, detail):
        self.zone = Zone.OUT
        self.out_reason = reason
        self.log(f"[line] OUT OF RING: {reason} {detail}")
