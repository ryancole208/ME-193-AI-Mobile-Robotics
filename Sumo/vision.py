"""
Opponent data from the UNO Q camera, and the chase steering math.

Freshness: clocks on different devices are never compared. A vision message is
"fresh" only if
  * the LAPTOP received it at most VISION_FRESH_S ago (laptop clock), and
  * the UNO Q says the camera frame was at most VISION_MAX_FRAME_AGE_MS old
    when it published (frame_age_ms, measured on the UNO Q's own clock).
Plus: visible, confidence >= VISION_MIN_CONFIDENCE, valid x, and the UNO Q not
reporting itself offline / camera missing.
"""

import json
import threading
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Opponent:
    seq: int
    visible: bool
    x: float | None             # -1 (left edge of image) .. +1 (right edge)
    distance_cm: float | None
    confidence: float
    frame_age_ms: int
    received_at: float          # laptop clock (time.monotonic or sim clock)


def parse_opponent(payload, received_at):
    """JSON string -> Opponent, or raise ValueError with a reason."""
    d = json.loads(payload)
    if not isinstance(d, dict):
        raise ValueError("not a JSON object")
    try:
        x = d.get("x")
        dist = d.get("distance_cm")
        return Opponent(seq=int(d["seq"]), visible=bool(d["visible"]),
                        x=None if x is None else float(x),
                        distance_cm=None if dist is None else float(dist),
                        confidence=float(d["confidence"]), frame_age_ms=int(d["frame_age_ms"]),
                        received_at=received_at)
    except (KeyError, TypeError) as e:
        raise ValueError(f"bad field: {e}") from None


class VisionStore:
    """Latest messages from the UNO Q. Written by the MQTT thread, read by the main loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._opponent = None
        self._unoq_status = None    # dict or None (never heard)
        self.bad_messages = 0

    def set_opponent(self, opp):
        with self._lock:
            # Drop out-of-order messages, but accept a big jump back (UNO Q restarted).
            if self._opponent is not None and 0 < self._opponent.seq - opp.seq < 1000:
                return
            self._opponent = opp

    def set_unoq_status(self, status):
        with self._lock:
            self._unoq_status = status

    @property
    def opponent(self):
        with self._lock:
            return self._opponent

    @property
    def unoq_status(self):
        with self._lock:
            return self._unoq_status

    def unoq_ok(self):
        """False only if the UNO Q says it's offline or its camera is missing.
        Never having heard a status is OK (fresh vision proves the UNO Q is alive)."""
        st = self.unoq_status
        if st is None:
            return True
        return st.get("state") == "online" and st.get("camera", "ok") == "ok"

    def fresh_opponent(self, now, link_connected=True):
        """The current target, or None if the chase branch must not run."""
        opp = self.opponent
        if opp is None or not link_connected or not self.unoq_ok():
            return None
        if now - opp.received_at > config.VISION_FRESH_S:
            return None
        if opp.frame_age_ms > config.VISION_MAX_FRAME_AGE_MS:
            return None
        if not opp.visible or opp.x is None or opp.confidence < config.VISION_MIN_CONFIDENCE:
            return None
        if not -1.5 <= opp.x <= 1.5:
            return None
        return opp


def chase_command(opp):
    """(left %, right %) to steer toward the opponent.

    Proportional steering on x: turn = GAIN * x. Base speed grows from MIN to
    MAX as the opponent gets closer AND more centered, so we charge when lined
    up and turn gently while searching.
    """
    x = max(-1.0, min(1.0, config.CHASE_X_SIGN * opp.x))
    if opp.distance_cm is None:
        closeness = 0.0
    else:
        span = max(1e-6, config.CHASE_FAR_CM - config.CHASE_NEAR_CM)
        closeness = min(1.0, max(0.0, (config.CHASE_FAR_CM - opp.distance_cm) / span))
    centered = 1.0 - abs(x)
    base = config.CHASE_MIN_SPEED + (config.CHASE_MAX_SPEED - config.CHASE_MIN_SPEED) * closeness * centered
    turn = config.CHASE_STEER_GAIN * x
    left, right = base + turn, base - turn
    # Keep the turn difference when a side saturates: shift both down together.
    over = max(left, right) - 100
    if over > 0:
        left, right = left - over, right - over
    return clamp_speed(left), clamp_speed(right)


def clamp_speed(v):
    return max(-100.0, min(100.0, float(v)))
