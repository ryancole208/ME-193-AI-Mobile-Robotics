"""Picks the mode at startup and switches between modes.

Startup logic:
  1. RUN_MODE file (one-shot; deleted when read): diag | setup | holdout | replay <csv>
  2. config_wired.json exists and is valid -> live mode (with the model last shown)
  3. otherwise -> setup mode
Nothing else ever triggers setup. To redo setup on purpose: the redo gesture,
`RUN_MODE` = setup, or delete/rename config_wired.json.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from .config import backup_existing, load_config, load_shown_model
from .constants import DISPLAY_PROTOCOL_VERSION, ST_CFG_SETUP, STEP_DIAG_DONE
from .diag import DiagController
from .errors import ConfigError, GaitError
from .imu_link import ImuLink
from .live import LiveController
from .paths import Paths
from .recording import read_recording
from .replay import ReplayController
from .settings import load_settings
from .setup_mode import SetupController, archive_setup_data, setup_is_complete_and_built

log = logging.getLogger("gait.runner")

ONE_SHOT = ("diag", "setup", "holdout", "replay")


def read_run_mode(paths: Paths) -> tuple[str, str]:
    """('normal', '') or a one-shot mode from the RUN_MODE file (which is then deleted)."""
    p = paths.run_mode
    if not p.exists():
        return "normal", ""
    text = p.read_text().strip()
    mode, _, arg = text.partition(" ")
    mode = mode.lower()
    if mode in ONE_SHOT:
        p.unlink()
        log.info(f"RUN_MODE '{text}' read and removed (one-shot)")
        return mode, arg.strip()
    if mode not in ("", "normal", "live"):
        log.error(f"While reading RUN_MODE: '{text}' is not a known mode. Likely fix: write one of "
                  f"{', '.join(ONE_SHOT)} (replay needs a CSV path) - starting normally")
    return "normal", ""


class Runner:
    def __init__(self, paths: Paths, display, clock=time.monotonic,
                 build_in_thread: bool = True):
        self.paths = paths
        self.display = display
        self.clock = clock
        self.build_in_thread = build_in_thread
        self.link = ImuLink()
        self.ctrl = None
        self.settings = None
        self.mode = ""

    # ------------------------------------------------------------ startup
    def boot(self) -> None:
        self.paths.ensure()
        v = self.display.handshake()
        if v != DISPLAY_PROTOCOL_VERSION:
            log.error(f"display handshake returned {v!r}, expected {DISPLAY_PROTOCOL_VERSION}: the matrix may not work "
                      "(see the error above)")
        try:
            self.settings = load_settings(self.paths.setup_yaml)
        except ConfigError as e:
            log.error(str(e))
            self.display.status(ST_CFG_SETUP)
            self.display.setup(STEP_DIAG_DONE, 0)
            self.mode = "broken_settings"
            return
        mode, arg = read_run_mode(self.paths)
        if mode == "diag":
            self._start(DiagController(self.settings, self.paths, self.display, self.link, self.clock), "diag")
        elif mode == "holdout":
            self._start(SetupController(self.settings, self.paths, self.display, self.link, self.clock, mode="holdout"), "holdout")
        elif mode == "replay":
            self._replay(arg)
        elif mode == "setup":
            self.redo_setup("RUN_MODE setup")
        else:
            self.normal_start()

    def normal_start(self) -> None:
        try:
            cfg = load_config(self.paths.config)
        except ConfigError as e:
            if self.paths.config.exists():
                broken = self.paths.config.with_name(f"config_wired.broken-{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json")
                os.replace(self.paths.config, broken)
                log.error(f"{e}\nThe file was renamed to {broken.name}; starting setup.")
            else:
                log.info("no config_wired.json: starting the one-time setup")
            self.display.status(ST_CFG_SETUP)   # "CFG" then "?": missing, invalid or not from a wired setup
            if setup_is_complete_and_built(self.paths):
                # A complete set of recordings that was already built into a config: the
                # config was deleted on purpose, so record afresh instead of rebuilding.
                archive_setup_data(self.paths)
            self._start_setup()
            return
        shown = load_shown_model(self.paths.state, cfg["display"]["default_model"])
        self._start(LiveController(cfg, self.settings, self.paths, self.display, self.link, self.clock, shown_model=shown), "live")

    def redo_setup(self, why: str) -> None:
        backup = backup_existing(self.paths.config)
        log.warning(f"redo setup ({why}): config_wired.json backed up as {backup.name if backup else '(none)'}; "
                    "previous recordings archived")
        archive_setup_data(self.paths)
        self._start_setup()

    def _start_setup(self) -> None:
        self._start(SetupController(self.settings, self.paths, self.display, self.link, self.clock,
                                    build_in_thread=self.build_in_thread), "setup")

    def _replay(self, arg: str) -> None:
        try:
            cfg = load_config(self.paths.config)
            path = (self.paths.root / arg) if arg and not os.path.isabs(arg) else arg
            rec = read_recording(path)
        except GaitError as e:
            log.error(f"replay impossible: {e}")
            self.display.setup(STEP_DIAG_DONE, 0)
            self.mode = "replay_failed"
            return
        self._start(ReplayController(rec, cfg, self.settings, self.paths, self.display), "replay")

    def _start(self, ctrl, mode: str) -> None:
        old = self.ctrl
        if old is not None and hasattr(old, "close"):
            old.close()
        self.ctrl, self.mode = ctrl, mode
        log.info(f"===== mode: {mode} =====")
        self.display.invalidate()
        ctrl.start()

    # ------------------------------------------------------------ running
    def on_event(self, ev) -> None:
        chunks = self.link.handle(ev)
        if self.ctrl is None:
            return
        self.ctrl.on_link_event(ev)
        for ch in chunks:
            self.ctrl.on_chunk(ch)

    def tick(self) -> None:
        if self.ctrl is None:
            return
        self.ctrl.tick()
        nxt = getattr(self.ctrl, "next_mode", None)
        if nxt == "live":
            self.ctrl.next_mode = None
            self.normal_start()
        elif nxt == "redo_setup":
            self.ctrl.next_mode = None
            self.redo_setup("confirmed by the redo-setup gesture")
