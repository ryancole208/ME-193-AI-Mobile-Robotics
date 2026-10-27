"""Setup mode (the one-time calibration) and holdout mode (test recordings).

Driven entirely by the matrix and tap gestures - no laptop, no button:

  1. stand   Stand still 10 s (countdown). Gives gravity direction + noise.
  2. for Still, Walk, Run, Bike (skipping any already saved in data/setup/):
       countdown  letter + 10 s countdown. A 4-tap gesture here SKIPS the
                  activity for now (e.g. leave biking for another trip).
       record     letter + progress bar for record_s (120 s). Then the whole
                  matrix flashes 3 times and the CSV is saved immediately.
     After Still: gesture check - do the switch gesture 5 times; each one is
     confirmed with a flash. Used to tune the tap thresholds.
  3. build   When all four exist: windows, features, CV, config.json,
             reports/setup_report.md. Shows a check mark and both accuracies,
             then live mode starts.
     If something was skipped: shows the missing letter + "?" and stops;
     the next start resumes from the first missing activity.

Time is counted in RECEIVED samples (100 per second), so if the link drops the
countdown/recording simply pauses until data flows again.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from .config import save_config
from .constants import (
    CLASSES, CLASS_LETTERS, ST_CHECK, ST_FLASH1, ST_FLASH3, ST_LINK_LOST, ST_SEARCHING, STEP_BUILD,
    STEP_COUNTDOWN_BASE, STEP_DIAG_DONE, STEP_GESTURE_CHECK, STEP_HOLDOUT_DONE, STEP_PARTIAL,
    STEP_RECORD_BASE, STEP_STAND, VAL_ACC_A, VAL_ACC_B,
)
from .errors import GaitError
from .gesture import GestureDetector, GestureParams
from .paths import Paths
from .recording import Recorder, read_recording, write_recording
from .report import setup_report
from .training import base_signal, build_config, converter_for, stand_stats

log = logging.getLogger("gait.setup")

FS = 100
LINK_LOST_S = 2.0
SKIP_GRACE_S = 3.0      # a skip gesture judged in the first 3 s of recording still counts
RESULT_SHOW_S = 9.0     # check mark + two scrolling accuracies, then live mode
BUILT_MARKER = "built.json"


def setup_files(paths: Paths) -> dict:
    d = paths.setup_data
    return {"stand": d / "stand.csv", "gestures": d / "gestures.csv",
            **{c: d / f"{CLASSES[c]}.csv" for c in range(len(CLASSES))}}


def archive_setup_data(paths: Paths) -> Path | None:
    """Move data/setup to data/setup-archive-<date> so a redo starts from scratch."""
    d = paths.setup_data
    if not d.exists() or not any(d.iterdir()):
        return None
    dest = d.with_name(f"setup-archive-{datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
    shutil.move(str(d), str(dest))
    d.mkdir(parents=True, exist_ok=True)
    log.info(f"previous setup recordings moved to {dest}")
    return dest


REDO_PARTS = ("stand", "gestures", *CLASSES)  # names accepted by a selective redo (see Runner.redo_setup_parts)


def archive_setup_parts(paths: Paths, parts: list[str]) -> list[Path]:
    """Move only the named recordings out of data/setup - each name is "stand",
    "gestures", or one of CLASSES - so the next setup re-records just those and
    resumes the rest from what is already saved (SetupController.start()
    already skips any activity whose CSV still exists). Also drops the stale
    build marker so a build runs again even though config.json is untouched;
    building re-reads every recording from disk, so the kept ones flow through
    unchanged and only the redone part's section of config.json changes.
    Names not in REDO_PARTS are a bug in the caller, not handled here."""
    f = setup_files(paths)
    by_name = {"stand": f["stand"], "gestures": f["gestures"], **{CLASSES[c]: f[c] for c in range(len(CLASSES))}}
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    moved = []
    for name in parts:
        src = by_name[name]
        if src.exists():
            dest = src.with_name(f"{src.stem}-archive-{stamp}{src.suffix}")
            shutil.move(str(src), str(dest))
            moved.append(dest)
    marker = paths.setup_data / BUILT_MARKER
    if marker.exists():
        marker.unlink()
    return moved


def setup_is_complete_and_built(paths: Paths) -> bool:
    f = setup_files(paths)
    return all(p.exists() for p in f.values() if p.name != "gestures.csv") and (paths.setup_data / BUILT_MARKER).exists()


class SetupController:
    def __init__(self, settings: dict, paths: Paths, display, link=None, clock=time.monotonic,
                 mode: str = "setup", build_in_thread: bool = True):
        self.s = settings
        self.paths = paths
        self.display = display
        self.link = link
        self.clock = clock
        self.mode = mode                   # "setup" | "holdout"
        self.build_in_thread = build_in_thread
        self.next_mode: str | None = None
        self.files = setup_files(paths)
        self.conv = converter_for(base_signal(settings))
        self.steps: list[tuple] = []
        self.step: tuple | None = None
        self.step_samples = 0
        self.recorder: Recorder | None = None
        self.detector: GestureDetector | None = None
        self.gestures_seen = 0
        self.skipped: list[int] = []
        self.saved: list[Path] = []
        self.last_data_t: float | None = None
        self.noise = {"amag_std_g": 0.02, "gmag_std_dps": 2.0}
        self._build_thread: threading.Thread | None = None
        self._build_result = None
        self._build_error: str | None = None
        self._result_t: float | None = None
        self.failed: str | None = None
        self.stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ------------------------------------------------------------ plan
    def start(self) -> None:
        self.paths.ensure()
        if self.mode == "holdout":
            self.steps = [s for c in range(len(CLASSES)) for s in (("countdown", c), ("record", c))]
            log.info("holdout mode: record each activity once more for tools/evaluate")
        else:
            f = self.files
            have = {k: p.exists() for k, p in f.items()}
            if have["stand"]:
                self._load_noise()
            else:
                self.steps.append(("stand",))
            for c in range(len(CLASSES)):
                if not have[c]:
                    self.steps += [("countdown", c), ("record", c)]
                if c == 0 and not have["gestures"]:
                    self.steps.append(("gesture_check",))
            missing = [CLASSES[c] for c in range(len(CLASSES)) if not have[c]]
            log.info(f"setup: already recorded: {[k if isinstance(k, str) else CLASSES[k] for k, v in have.items() if v]}; "
                     f"to record now: {missing or 'nothing'}")
        self.steps.append(("finish",))
        self.display.status(ST_SEARCHING)
        log.info("setup waiting for data from the ankle node")

    def _load_noise(self) -> None:
        try:
            conv = converter_for(base_signal(self.s))
            _, self.noise = stand_stats(read_recording(self.files["stand"]), conv)
        except GaitError as e:
            self._fail(str(e))

    def _detector(self) -> GestureDetector:
        # permissive tap threshold: the real one is tuned from the gesture check
        return GestureDetector(GestureParams(tap_g=1.5, amag_noise_g=self.noise["amag_std_g"],
                                             gmag_noise_dps=self.noise["gmag_std_dps"]))

    def _advance(self) -> None:
        self.step = self.steps.pop(0)
        self.step_samples = 0
        kind = self.step[0]
        if kind == "stand":
            self.recorder = Recorder({"kind": "stand", "leg": self.s["leg"]})
            log.info(f"STEP stand still for {self.s['stand_s']} s")
        elif kind == "countdown":
            self.detector = self._detector()
            c = self.step[1]
            log.info(f"STEP get ready for {CLASSES[c]} ({CLASS_LETTERS[c]}): {self.s['countdown_s']} s countdown "
                     "(tap 4 times now to skip it)")
        elif kind == "record":
            c = self.step[1]
            self.recorder = Recorder({"kind": self.mode, "class": CLASSES[c], "leg": self.s["leg"]})
            log.info(f"STEP recording {CLASSES[c]} for {self._record_s()} s - vary your pace!")
        elif kind == "gesture_check":
            self.detector = self._detector()
            self.recorder = Recorder({"kind": "gestures", "leg": self.s["leg"]})
            self.gestures_seen = 0
            log.info(f"STEP gesture check: stand still and do the 4-tap switch gesture "
                     f"{self.s['gesture_check_count']} times (wait for the flash after each)")
        elif kind == "finish":
            self._finish()
        self._show()

    def _record_s(self) -> float:
        return self.s["holdout_record_s"] if self.mode == "holdout" else self.s["record_s"]

    # ------------------------------------------------------------ display
    def _show(self) -> None:
        if self.step is None:
            return
        kind = self.step[0]
        sec = self.step_samples / FS
        if kind == "stand":
            self.display.setup(STEP_STAND, max(0, int(self.s["stand_s"] - sec + 0.999)))
        elif kind == "countdown":
            self.display.setup(STEP_COUNTDOWN_BASE + self.step[1], max(0, int(self.s["countdown_s"] - sec + 0.999)))
        elif kind == "record":
            self.display.setup(STEP_RECORD_BASE + self.step[1], min(100, int(100 * sec / self._record_s())))
        elif kind == "gesture_check":
            self.display.setup(STEP_GESTURE_CHECK, self.gestures_seen)

    # ------------------------------------------------------------ data
    def on_link_event(self, ev) -> None:
        pass

    def on_chunk(self, chunk) -> None:
        self.last_data_t = self.clock()
        if self.failed:
            return
        if self.step is None:
            self._advance()
        kind = self.step[0]
        if kind in ("build", "finish", "done", "partial"):
            return
        n = len(chunk.raw)
        if self.recorder is not None and kind in ("stand", "record", "gesture_check"):
            self.recorder.add(chunk.start_index, chunk.raw)
        if self.detector is not None and kind in ("countdown", "record", "gesture_check"):
            step_before = self.step
            for ev in self.detector.push(chunk.start_index, self.conv.convert(chunk.raw), chunk.reset):
                self._gesture(ev)
                if self.step is not step_before:  # a skip moved us on; this chunk is done
                    return
        self.step_samples += n
        sec = self.step_samples / FS
        if kind == "stand" and sec >= self.s["stand_s"]:
            rec = self.recorder.recording()
            write_recording(self.files["stand"], rec)
            self.saved.append(self.files["stand"])
            try:
                _, self.noise = stand_stats(rec, self.conv)
            except GaitError as e:
                self._fail(str(e))
                return
            log.info(f"stand-still saved: noise {self.noise}")
            self._advance()
        elif kind == "countdown" and sec >= self.s["countdown_s"]:
            self._advance()
        elif kind == "record" and sec >= self._record_s():
            self._save_recording()
            self._advance()
        elif kind == "gesture_check" and (self.gestures_seen >= self.s["gesture_check_count"]
                                          or sec >= self.s["gesture_check_timeout_s"]):
            if self.gestures_seen < self.s["gesture_check_count"]:
                log.warning(f"gesture check timed out with {self.gestures_seen} gestures; thresholds are tuned "
                            "from what was seen (defaults if fewer than 3)")
            write_recording(self.files["gestures"], self.recorder.recording())
            self.display.status(ST_FLASH3)
            self.display.invalidate()
            self._advance()
        else:
            self._show()

    def _save_recording(self) -> None:
        c = self.step[1]
        rec = self.recorder.recording()
        if self.mode == "holdout":
            path = self.paths.holdout_data / f"{CLASSES[c]}-{self.stamp}.csv"
        else:
            path = self.files[c]
        write_recording(path, rec)
        self.saved.append(path)
        log.info(f"saved {path} ({rec.n} samples, {rec.duration_s:.0f} s)")
        self.display.status(ST_FLASH3)
        self.display.invalidate()

    def _gesture(self, ev) -> None:
        kind = self.step[0]
        log.info(f"setup gesture during {kind}: {ev.describe()}")
        if ev.kind != "switch":
            return
        if kind == "gesture_check":
            self.gestures_seen += 1
            self.display.status(ST_FLASH1)
            self.display.invalidate()
            log.info(f"gesture {self.gestures_seen}/{self.s['gesture_check_count']} confirmed")
        elif kind == "countdown" or (kind == "record" and self.step_samples < SKIP_GRACE_S * FS):
            c = self.step[1]
            log.info(f"{CLASSES[c]} SKIPPED by gesture - it will be asked for again on the next setup start")
            self.skipped.append(c)
            self.display.status(ST_FLASH1)
            self.display.invalidate()
            if kind == "countdown" and self.steps and self.steps[0] == ("record", c):
                self.steps.pop(0)
            self.recorder = None
            self._advance()

    # ------------------------------------------------------------ finish / build
    def _finish(self) -> None:
        if self.mode == "holdout":
            self.step = ("done",)
            self.display.setup(STEP_HOLDOUT_DONE, 1)
            log.info(f"holdout recordings saved: {[str(p) for p in self.saved]}. Run tools/evaluate.")
            return
        missing = [c for c in range(len(CLASSES)) if not self.files[c].exists()]
        if missing:
            self.step = ("partial",)
            self.display.setup(STEP_PARTIAL, missing[0])
            log.warning(f"setup paused: still missing {[CLASSES[c] for c in missing]}. Restart the app when you are "
                        f"ready to record it; setup resumes from {CLASSES[missing[0]]}.")
            return
        self.step = ("build",)
        self.display.setup(STEP_BUILD, 0)
        log.info("all activities recorded: building the models")
        if self.build_in_thread:
            self._build_thread = threading.Thread(target=self._build, name="setup-build", daemon=True)
            self._build_thread.start()
        else:
            self._build()
            self._after_build()

    def _build(self) -> None:
        try:
            f = self.files
            stand = read_recording(f["stand"])
            activity = {c: [read_recording(f[c])] for c in range(len(CLASSES))}
            gestures = read_recording(f["gestures"]) if f["gestures"].exists() else None
            res = build_config(self.s, stand, activity, gestures, progress=lambda m: log.info(f"build: {m}"))
            backup = save_config(res.config, self.paths.config)
            if backup:
                log.info(f"previous config.json backed up as {backup.name}")
            setup_report(res.config, res.report, self.paths.reports / "setup_report.md")
            (self.paths.setup_data / BUILT_MARKER).write_text(json.dumps({"built": res.config["created"]}))
            self._build_result = res
        except GaitError as e:
            self._build_error = str(e)
        except Exception as e:  # loud and complete: this is the one long computation
            self._build_error = (f"While building the models: unexpected {type(e).__name__}: {e}. Likely fix: send "
                                 f"this traceback with the recordings in data/setup.\n{traceback.format_exc()}")

    def _after_build(self) -> None:
        if self._build_error:
            self._fail(self._build_error)
            return
        cfg = self._build_result.config
        a, b = cfg["model_a"]["cv_balanced_accuracy"], cfg["model_b"]["cv_balanced_accuracy"]
        log.info(f"setup complete: config.json written. CV balanced accuracy A {a:.1%}, B {b:.1%}. "
                 "Full report: reports/setup_report.md")
        self.display.status(ST_CHECK)
        self.display.value(VAL_ACC_A, int(round(a * 100)))
        self.display.value(VAL_ACC_B, int(round(b * 100)))
        self.step = ("done",)
        self._result_t = self.clock()

    def _fail(self, msg: str) -> None:
        self.failed = msg
        self.step = ("failed",)
        log.error(f"SETUP FAILED - {msg}")
        self.display.setup(STEP_DIAG_DONE, 0)   # big X

    # ------------------------------------------------------------ timers
    def tick(self) -> None:
        now = self.clock()
        if self._build_thread is not None and not self._build_thread.is_alive():
            self._build_thread = None
            self._after_build()
        if self._result_t is not None and now - self._result_t > RESULT_SHOW_S:
            self._result_t = None
            self.next_mode = "live"
        if self.step is None or self.step[0] in ("countdown", "record", "stand", "gesture_check"):
            if self.last_data_t is not None and now - self.last_data_t > LINK_LOST_S:
                self.display.status(ST_LINK_LOST)
            elif self.last_data_t is None:
                self.display.status(ST_SEARCHING)
            else:
                self._show()
