"""Replay: feed a recorded CSV through the LIVE pipeline (gestures, smoothing,
display commands), so the system can be tested without wearing the pod.

  * In the app (RUN_MODE "replay <csv>"): real time, drives the real matrix.
  * Offline (tools/replay.py): as fast as possible, prints the display commands.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .live import LiveController
from .protocol import SampleChunk
from .recording import Recording

log = logging.getLogger("gait.replay")
CHUNK = 10  # samples per fake packet, like the real node


def recording_chunks(rec: Recording, chunk: int = CHUNK):
    """Split a recording into SampleChunks of consecutive indices."""
    idx, raw = rec.index, rec.raw
    if len(idx) == 0:
        return
    breaks = np.where(np.diff(idx) != 1)[0] + 1
    for seg_i, seg_raw in zip(np.split(idx, breaks), np.split(raw, breaks)):
        for k in range(0, len(seg_i), chunk):
            yield SampleChunk(int(seg_i[k]), seg_raw[k: k + chunk])


class SampleClock:
    """Time = stream sample index / 100 (so timers in live mode follow the data)."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def replay_offline(rec: Recording, cfg: dict, settings: dict, paths, display, write_logs: bool = False) -> LiveController:
    clock = SampleClock()
    ctrl = LiveController(cfg, settings, paths, display, link=None, clock=clock, write_logs=write_logs, log_raw=False,
                          run_name="replay")
    ctrl.start()
    for ch in recording_chunks(rec):
        clock.t = (ch.start_index + len(ch.raw)) / 100.0
        ctrl.on_chunk(ch)
        ctrl.tick()
    return ctrl


class ReplayController:
    """In-app replay in real time (speed 1.0) on the real display."""

    def __init__(self, rec: Recording, cfg: dict, settings: dict, paths, display, speed: float = 1.0):
        self.clock = SampleClock()
        self.live = LiveController(cfg, settings, paths, display, link=None, clock=self.clock, log_raw=False,
                                   run_name="replay-" + time.strftime("%Y%m%d-%H%M%S"))
        self.chunks = list(recording_chunks(rec))
        self.speed = speed
        self.i = 0
        self.wall0 = None
        self.t0 = self.chunks[0].start_index / 100.0 if self.chunks else 0.0
        self.next_mode = None
        self.done = False
        self.rec = rec

    def start(self) -> None:
        log.info(f"replaying {self.rec.path} ({self.rec.duration_s:.0f} s) at {self.speed}x through the live pipeline")
        self.live.start()

    def on_link_event(self, ev) -> None:
        pass

    def on_chunk(self, chunk) -> None:
        pass  # the real node is ignored during replay

    def tick(self) -> None:
        if self.done:
            return
        if self.wall0 is None:
            self.wall0 = time.monotonic()
        due = self.t0 + (time.monotonic() - self.wall0) * self.speed
        while self.i < len(self.chunks) and self.chunks[self.i].start_index / 100.0 <= due:
            ch = self.chunks[self.i]
            self.clock.t = (ch.start_index + len(ch.raw)) / 100.0
            self.live.on_chunk(ch)
            self.i += 1
        self.live.tick()
        if self.i >= len(self.chunks):
            self.done = True
            log.info(f"replay finished: {self.live.summary()}")
