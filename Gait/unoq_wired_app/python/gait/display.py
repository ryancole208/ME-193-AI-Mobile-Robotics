"""The display commands Python sends to the sketch.

Python never draws pixels: it says WHAT to show, the sketch knows HOW. Calls go
through a sender:
  * BridgeSender - real board: a worker thread does Bridge.call() so a slow MCU
    can never stall the classifier.
  * RecordingSender - offline replay: remembers (and optionally prints) every call.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from .constants import (
    DISPLAY_PROTOCOL_VERSION, MODEL_A, MODEL_B, ST_IDLE, ST_NO_DATA, ST_UNSURE,
)

log = logging.getLogger("gait.display")

BASE_STATUS = (ST_NO_DATA, ST_UNSURE, ST_IDLE)


class RecordingSender:
    def __init__(self, clock=time.monotonic, echo=None):
        self.calls: list[tuple] = []
        self.clock = clock
        self.echo = echo

    def send(self, name: str, *args) -> None:
        self.calls.append((self.clock(), name, args))
        if self.echo:
            self.echo(f"display: {name}{args}")

    def handshake(self) -> int:
        return DISPLAY_PROTOCOL_VERSION

    def names(self) -> list[tuple]:
        return [(n, a) for _, n, a in self.calls]


class BridgeSender:
    """Sends commands with Bridge.call from one worker thread, in order."""

    def __init__(self, timeout_s: int = 3):
        from arduino.app_utils import Bridge  # only available on the board

        self.Bridge = Bridge
        self.timeout_s = timeout_s
        self.q: queue.Queue = queue.Queue(maxsize=200)
        self.errors = 0
        self._last_error = ""
        threading.Thread(target=self._worker, name="display", daemon=True).start()

    def handshake(self) -> int | None:
        try:
            return int(self.Bridge.call("display_version", timeout=self.timeout_s))
        except Exception as e:
            log.error(f"While asking the sketch for display_version: {type(e).__name__}: {e}. Likely fix: the sketch "
                      "is not running or is an older version - restart the app so App Lab re-flashes sketch/sketch.ino")
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
                              "sketch/sketch.ino provides this function (names must match exactly) and that the "
                              "app was restarted after editing the sketch")
                    self._last_error = msg


class Display:
    """Remembers the current base screen so it is only re-sent when it changes
    (plus a refresh every few seconds in case the MCU restarted)."""

    REFRESH_S = 5.0

    def __init__(self, sender, clock=time.monotonic):
        self.sender = sender
        self.clock = clock
        self._base = None
        self._base_t = 0.0
        self._model = None

    # base screens ------------------------------------------------------
    def _set_base(self, key: tuple) -> None:
        now = self.clock()
        if key != self._base or now - self._base_t > self.REFRESH_S:
            self._base, self._base_t = key, now
            self.sender.send(*key)

    def activity(self, class_id: int) -> None:
        self._set_base(("show_activity", int(class_id)))

    def status(self, code: int) -> None:
        if code in BASE_STATUS:
            self._set_base(("show_status", int(code)))
        else:
            self.sender.send("show_status", int(code))  # transients and flags are never deduplicated

    def setup(self, step: int, progress: int) -> None:
        self._set_base(("show_setup", int(step), int(progress)))

    def invalidate(self) -> None:
        """Force the next base command to be sent (e.g. after a transient)."""
        self._base = None

    # other -------------------------------------------------------------
    def model(self, model: str, banner: bool) -> None:
        mid = MODEL_A if model == "A" else MODEL_B
        if banner or mid != self._model:
            self.sender.send("show_model", mid, 1 if banner else 0)
        self._model = mid

    def value(self, kind: int, value: int) -> None:
        self.sender.send("show_value", int(kind), int(value))

    def handshake(self):
        return self.sender.handshake()
