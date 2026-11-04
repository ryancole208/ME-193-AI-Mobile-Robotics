"""Matrix commands (Python says WHAT, sketch/sketch.ino knows HOW).

Copied/adapted from unoq_app/python/gait/display.py. Calls go through a worker
thread doing Bridge.call(), so a slow or restarting MCU never stalls counting.
Base screens are only re-sent when they change (plus every 5 s, in case the
MCU restarted), and the count/distance at most twice per second.

Codes must match the #defines in sketch/sketch.ino.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

log = logging.getLogger("step.display")

DISPLAY_PROTOCOL_VERSION = 3

ST_SEARCHING, ST_LINK_LOST, ST_UNSURE, ST_IDLE = 1, 2, 3, 4
ST_CFG_ERROR, ST_LEG_MISMATCH, ST_REC, ST_NN_FAIL, ST_HIP_FAIL = 10, 11, 12, 13, 14
ST_ANKLE_LOST, ST_FLASH1, ST_FLASH3, ST_CHECK = 15, 16, 17, 18
BASE_STATUS = (ST_SEARCHING, ST_LINK_LOST, ST_UNSURE, ST_IDLE)

DM_STEPS, DM_DIST = 0, 1

STEP_STAND = 0
STEP_FACE_BASE = 1          # + face index 0..5 (+X -X +Y -Y +Z -Z)
STEP_GESTURE_CHECK = 20
STEP_BUILD = 21
STEP_SET_CONFIRM = 22
STEP_RESET_CONFIRM = 23


class BridgeSender:
    def __init__(self, timeout_s: int = 3):
        from arduino.app_utils import Bridge  # only available on the board

        self.Bridge = Bridge
        self.timeout_s = timeout_s
        self.q: queue.Queue = queue.Queue(maxsize=200)
        self.errors = 0
        self._last_error = ""
        threading.Thread(target=self._worker, name="display", daemon=True).start()

    def handshake(self):
        try:
            return int(self.Bridge.call("display_version", timeout=self.timeout_s))
        except Exception as e:
            log.error(f"While asking the sketch for display_version: {type(e).__name__}: {e}. Likely fix: the "
                      "sketch is not running or is an older build - restart the app so App Lab re-flashes it")
            return None

    def send(self, name: str, *args) -> None:
        try:
            self.q.put_nowait((name, args))
        except queue.Full:
            log.error("display queue full (the sketch stopped answering?) - dropping a command")

    def _worker(self) -> None:
        while True:
            name, args = self.q.get()
            try:
                self.Bridge.call(name, *args, timeout=self.timeout_s)
            except Exception as e:
                self.errors += 1
                msg = f"{type(e).__name__}: {e}"
                if msg != self._last_error or self.errors % 50 == 1:
                    log.error(f"While sending {name}{args} to the sketch: {msg}. Likely fix: check that "
                              "sketch/sketch.ino provides this function (names must match exactly)")
                    self._last_error = msg


class Display:
    REFRESH_S = 5.0
    VALUE_MIN_S = 0.5           # refresh at most twice a second

    def __init__(self, sender, clock=time.monotonic):
        self.sender = sender
        self.clock = clock
        self._base = None
        self._base_t = 0.0
        self._value_t = -1e9
        self.mode = DM_STEPS

    def _set_base(self, key: tuple, min_interval: float = 0.0) -> None:
        now = self.clock()
        if key == self._base and now - self._base_t < self.REFRESH_S:
            return
        if key != self._base and key[0] == "show_value" and now - self._value_t < min_interval \
                and self._base and self._base[0] == "show_value" and self._base[2] == key[2]:
            return                       # same screen, new number: wait for the 2 Hz slot
        self._base, self._base_t = key, now
        if key[0] == "show_value":
            self._value_t = now
        self.sender.send(*key)

    def value(self, text: str, mode: int) -> None:
        self.mode = mode
        self._set_base(("show_value", str(text), int(mode)), self.VALUE_MIN_S)

    def status(self, code: int) -> None:
        if code in BASE_STATUS:
            self._set_base(("show_status", int(code)))
        else:
            self.sender.send("show_status", int(code))    # transients are never deduplicated

    def setup(self, step: int, progress: int) -> None:
        self._set_base(("show_setup", int(step), int(progress)))

    def banner(self, text: str) -> None:
        self.sender.send("show_banner", str(text)[:15])

    def invalidate(self) -> None:
        self._base = None

    def handshake(self):
        return self.sender.handshake()
