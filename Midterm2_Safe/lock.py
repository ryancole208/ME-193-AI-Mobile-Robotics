"""Lock demo state machine (no camera, no model, no Flask: easy to test and to
reuse on the UNO Q).

    asleep --press--> awake (LEDs on) --3 digits--> result (UNLOCKED / STAYS LOCKED) --> asleep

While awake, a digit registers when the same count (1-5) stays on top with
confidence >= CONF_THRESHOLD for HOLD_SECONDS. `none` (0) never registers, and
each digit must differ from the one before it.
"""
import config
from gestures import CODE_COUNTS


def validate_code(code):
    code = tuple(int(d) for d in code)
    if len(code) != 3:
        raise ValueError(f"secret code must have 3 digits, got {code}")
    if any(d not in CODE_COUNTS for d in code):
        raise ValueError(f"secret code digits must be 1-5, got {code}")
    if len(set(code)) != len(code):
        raise ValueError(f"secret code must use three different numbers, got {code}")
    return code


class LockDemo:
    def __init__(self, code=None, hold_seconds=None, conf_threshold=None,
                 result_seconds=None, awake_timeout=None):
        self.code = validate_code(config.SECRET_CODE if code is None else code)
        self.hold_seconds = config.HOLD_SECONDS if hold_seconds is None else hold_seconds
        self.conf_threshold = config.CONF_THRESHOLD if conf_threshold is None else conf_threshold
        self.result_seconds = config.RESULT_SECONDS if result_seconds is None else result_seconds
        self.awake_timeout = config.AWAKE_TIMEOUT_SECONDS if awake_timeout is None else awake_timeout
        self._sleep()

    def _sleep(self):
        self.state = "asleep"
        self.entered = []
        self.hold_count = None
        self.hold_start = None
        self.progress = 0.0
        self.message = "Push the button to wake the safe"
        self.result = None
        self._until = None
        self._last_activity = None

    def press(self, now):
        """The push button: wake up (or restart) and light the LEDs."""
        self._sleep()
        self.state = "awake"
        self.message = "Show your first number"
        self._last_activity = now

    def reset(self):
        self._sleep()

    def update(self, count, confidence, now):
        """Feed the latest smoothed prediction. Returns the new digit if one registered."""
        if self.state == "result":
            if now >= self._until:
                self._sleep()
            return None
        if self.state != "awake":
            return None
        if now - self._last_activity > self.awake_timeout:
            self._sleep()
            self.message = "Timed out. Push the button to try again"
            return None

        ok = count in CODE_COUNTS and confidence >= self.conf_threshold
        if not ok:
            self._hold(None, now)
            self.message = ("Show a number (1-5)" if count not in CODE_COUNTS
                            else f"Hold {count} steady (confidence {confidence:.0%})")
            return None
        if self.entered and count == self.entered[-1]:
            self._hold(None, now)
            self.message = "Change your number"
            return None

        self._hold(count, now)
        self._last_activity = now
        self.progress = min(1.0, (now - self.hold_start) / self.hold_seconds)
        self.message = f"Holding {count}..."
        if self.progress < 1.0:
            return None

        self.entered.append(count)
        self._hold(None, now)
        if len(self.entered) < len(self.code):
            self.message = f"Got {count}. Next number"
            return count
        self.state = "result"
        self.result = "unlocked" if tuple(self.entered) == self.code else "locked"
        self.message = "UNLOCKED" if self.result == "unlocked" else "STAYS LOCKED"
        self._until = now + self.result_seconds
        return count

    def _hold(self, count, now):
        if count is None:
            self.hold_count, self.hold_start, self.progress = None, None, 0.0
        elif count != self.hold_count:
            self.hold_count, self.hold_start, self.progress = count, now, 0.0

    def snapshot(self):
        return {
            "state": self.state,
            "leds_on": self.state != "asleep",
            "entered": list(self.entered),
            "n_digits": len(self.code),
            "hold_count": self.hold_count,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result": self.result,
        }
