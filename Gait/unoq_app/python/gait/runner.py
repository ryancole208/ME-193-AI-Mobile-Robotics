"""Picks the mode at startup and switches between modes.

Startup logic:
  1. RUN_MODE file (one-shot; deleted when read): diag | setup | holdout | replay <csv> | redo <parts>
  2. config.json exists and is valid -> live mode (with the model last shown)
  3. otherwise -> setup mode
Nothing else ever triggers setup. To redo EVERYTHING on purpose: the redo
gesture, `RUN_MODE` = setup, or delete/rename config.json.

To redo only SOME of it, write `redo <parts>` to RUN_MODE, parts being a
comma-separated list from gait.setup_mode.REDO_PARTS ("stand", "gestures",
"still", "walking", "running", "biking"), e.g. `redo gestures`. Only the named
recordings are archived; config.json is left alone until the (much shorter)
redo finishes, then it is rebuilt whole - the kept recordings feed back in
unchanged, so only the redone part's section actually changes. `redo all` (or
an empty/unknown part list) falls back to a full redo.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from .config import backup_existing, load_config, load_shown_model
from .constants import DISPLAY_PROTOCOL_VERSION, ST_CFG_ERROR, STEP_DIAG_DONE
from .diag import DiagController
from .errors import ConfigError, GaitError
from .link_state import LinkMonitor
from .live import LiveController
from .paths import Paths
from .recording import read_recording
from .replay import ReplayController
from .settings import load_settings
from .setup_mode import REDO_PARTS, SetupController, archive_setup_data, archive_setup_parts, setup_is_complete_and_built

log = logging.getLogger("gait.runner")

ONE_SHOT = ("diag", "setup", "holdout", "replay", "redo")


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
    def __init__(self, paths: Paths, display, link_write=lambda text: False, clock=time.monotonic,
                 build_in_thread: bool = True):
        self.paths = paths
        self.display = display
        self.link_write = link_write
        self.clock = clock
        self.build_in_thread = build_in_thread
        self.link = LinkMonitor()
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
            self.display.status(ST_CFG_ERROR)
            self.display.setup(STEP_DIAG_DONE, 0)
            self.mode = "broken_settings"
            return
        mode, arg = read_run_mode(self.paths)
        if mode == "diag":
            self._start(DiagController(self.settings, self.paths, self.display, self.link, self.clock, self.link_write), "diag")
        elif mode == "holdout":
            self._start(SetupController(self.settings, self.paths, self.display, self.link, self.clock, mode="holdout"), "holdout")
        elif mode == "replay":
            self._replay(arg)
        elif mode == "setup":
            self.redo_setup("RUN_MODE setup")
        elif mode == "redo":
            self.redo_setup_parts(arg, "RUN_MODE redo")
        else:
            self.normal_start()

    def normal_start(self) -> None:
        try:
            cfg = load_config(self.paths.config)
        except ConfigError as e:
            if self.paths.config.exists():
                broken = self.paths.config.with_name(f"config.broken-{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json")
                os.replace(self.paths.config, broken)
                log.error(f"{e}\nThe broken file was renamed to {broken.name}; starting setup.")
                self.display.status(ST_CFG_ERROR)
            else:
                log.info("no config.json: starting the one-time setup")
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
        log.warning(f"redo setup ({why}): config.json backed up as {backup.name if backup else '(none)'}; "
                    "previous recordings archived")
        archive_setup_data(self.paths)
        self._start_setup()

    def redo_setup_parts(self, arg: str, why: str) -> None:
        """Redo only some recordings (see REDO_PARTS): config.json is left in
        place - still valid - until setup finishes and rebuilds it whole, so
        an interruption just leaves live mode running on the old config."""
        names = [n.strip() for n in arg.split(",") if n.strip()]
        if not names or "all" in names:
            self.redo_setup(f"{why} {arg}".strip() if names else f"{why}: no part named, redoing everything")
            return
        bad = [n for n in names if n not in REDO_PARTS]
        if bad:
            log.error(f"{why} {arg}: unknown part(s) {bad}; valid names are {', '.join(REDO_PARTS)} or 'all'. "
                      "Likely fix: fix the RUN_MODE file and restart. Redoing everything instead.")
            self.redo_setup(why)
            return
        moved = archive_setup_parts(self.paths, names)
        log.warning(f"{why} {arg}: recalibrating {names}; previous recordings for them archived as "
                    f"{[p.name for p in moved]}. Everything else is kept as-is.")
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
