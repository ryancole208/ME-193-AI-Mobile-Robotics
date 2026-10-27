"""Live mode: classify every window with BOTH models, show one of them.

The gesture only changes which model is SHOWN; both are always computed and
logged, so switching is instant and the comparison stays fair.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime

from .config import save_shown_model
from .constants import (
    MODEL_A, ST_LEG_MISMATCH, ST_NO_DATA, ST_REDO_WINDOW_OFF, ST_REDO_WINDOW_ON, ST_UNSURE,
    STEP_SET_CONFIRM, UNSURE,
)
from .paths import Paths
from .pipeline import Classifier, Smoother, label_name
from .recording import RecordingWriter, rotate_logs

log = logging.getLogger("gait.live")

NO_DATA_S = 2.0
REDO_WINDOW_S = 30.0
CONFIRM_S = 5.0
CONFIRM_GRACE_S = 2.5        # a confirm train that started in time may be judged a little later
INVALID_STREAK_FOR_Q = 3     # "?" after this many gap-ridden windows in a row
AUTO_ALTERNATE_S = 4.0
FS = 100


class LiveController:
    def __init__(self, cfg: dict, settings: dict, paths: Paths, display, link=None, clock=time.monotonic,
                 shown_model: str | None = None, write_logs: bool = True, log_raw: bool | None = None,
                 run_name: str | None = None):
        self.cfg = cfg
        self.settings = settings
        self.paths = paths
        self.display = display
        self.link = link                 # ImuLink or None (replay)
        self.clock = clock
        self.next_mode: str | None = None
        self.leg = settings["leg"]
        self.leg_mismatch = cfg["signal"]["leg"] != self.leg
        self.clf = Classifier(cfg, leg=self.leg)
        self.smooth = {"A": Smoother(), "B": Smoother()}
        self.model = shown_model or cfg["display"]["default_model"]
        self.switch_mode = cfg["display"]["switch_mode"]
        self.invalid_streak = 0
        self.first_data_t: float | None = None
        self.last_data_t: float | None = None
        self.last_index = 0
        self.redo_open = False
        self.redo_start = 0.0
        self.confirm_t: float | None = None
        self.confirm_index = 0
        self.last_alt_t: float | None = None
        self.counts = {"windows": 0, "invalid": 0, "gesture_windows": 0, "switches": 0, "long": 0, "near_miss": 0}
        self.events: list = []           # every gesture event (replay/tests read this)
        self.results: list = []

        stamp = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self._log_f = self._gest_f = self._raw = None
        if write_logs:
            paths.logs.mkdir(parents=True, exist_ok=True)
            rotate_logs(paths.logs, keep=20 - 3)  # room for this run's 3 files
            self._log_f = (paths.logs / f"run-{stamp}.csv").open("w", newline="")
            self._log = csv.writer(self._log_f)
            self._log.writerow(["time", "t_s", "window_start", "valid", "missing_pct", "raw_a", "raw_b",
                                "smooth_a", "smooth_b", "shown_model", "displayed", "gesture_window",
                                "dist_a", "dist_b", "packet_loss_pct", "i2c_errors"])
            self._gest_f = (paths.logs / f"gestures-{stamp}.csv").open("w", newline="")
            self._gest = csv.writer(self._gest_f)
            self._gest.writerow(["time", "kind", "taps", "gaps_ms", "peaks_g", "start_index", "reason"])
            if log_raw if log_raw is not None else settings.get("log_raw", True):
                self._raw = RecordingWriter(paths.logs / f"raw-{stamp}.csv",
                                            {"kind": "live", "class": "unknown", "leg": self.leg})
            log.info(f"logging this run to {paths.logs}/run-{stamp}.csv (+ gestures-, raw-)")

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        log.info(f"live mode: showing model {self.model} ({'1D' if self.model == 'A' else '6D'}), "
                 f"switch mode {self.switch_mode}, leg {self.leg}")
        self.display.model(self.model, banner=False)
        self.display.status(ST_NO_DATA if self.last_data_t is None else ST_UNSURE)
        if self.leg_mismatch:
            log.warning(f"While starting live mode: setup was done on the {self.cfg['signal']['leg']} leg but setup.yaml "
                        f"says leg: {self.leg}. Likely fix: move the IMU back, or fix setup.yaml, or redo setup on "
                        "this leg. Classifying anyway with the setup.yaml leg.")
            self.display.status(ST_LEG_MISMATCH)

    def close(self) -> None:
        for f in (self._log_f, self._gest_f):
            if f:
                f.close()
        if self._raw:
            self._raw.close()

    # ------------------------------------------------------------ input
    def on_link_event(self, ev) -> None:
        pass  # link facts are read from self.link in tick()

    def on_chunk(self, chunk) -> None:
        now = self.clock()
        if self.first_data_t is None:
            self.first_data_t = now
            # The window counts from when IMU data FIRST arrived (not from power-on, and
            # not from the start of live mode right after setup). Replay has no link: use now.
            first = self.link.first_data_t if self.link and self.link.first_data_t is not None else now
            self.redo_start = first
            if now - first <= REDO_WINDOW_S:
                self.redo_open = True
                self.display.status(ST_REDO_WINDOW_ON)
                log.info(f"redo-setup window open for {REDO_WINDOW_S - (now - first):.0f} s "
                         "(8 taps, then 4 more within 5 s, redoes setup)")
        self.last_data_t = now
        self.last_index = chunk.start_index + len(chunk.raw)
        if self._raw:
            self._raw.add(chunk.start_index, chunk.raw)
        results, events = self.clf.push(chunk.start_index, chunk.raw, chunk.reset)
        for ev in events:
            self._gesture(ev, now)
        for r in results:
            self._window(r, now)

    # ------------------------------------------------------------ windows
    def _window(self, r, now: float) -> None:
        self.results.append(r)
        self.counts["windows"] += 1
        if not r.valid:
            self.counts["invalid"] += 1
            self.invalid_streak += 1
        else:
            self.invalid_streak = 0
            if r.gesture:
                self.counts["gesture_windows"] += 1
            else:
                self.smooth["A"].push(r.pred_a)
                self.smooth["B"].push(r.pred_b)
        self._refresh(now)
        if self._log_f:
            st = self.link.stats if self.link else None
            self._log.writerow([
                datetime.now().isoformat(timespec="milliseconds"), f"{r.start / FS:.2f}", r.start, int(r.valid),
                f"{r.missing_frac * 100:.1f}", label_name(r.pred_a), label_name(r.pred_b),
                label_name(self.smooth["A"].smoothed), label_name(self.smooth["B"].smoothed), self.model,
                self._displayed_name(), int(r.gesture),
                "" if r.dist_a is None else f"{r.dist_a:.3f}", "" if r.dist_b is None else f"{r.dist_b:.3f}",
                f"{st.loss_pct:.2f}" if st else "", "" if not self.link else self.link.i2c_errors,
            ])
            self._log_f.flush()

    def _displayed_name(self) -> str:
        if self.invalid_streak >= INVALID_STREAK_FOR_Q:
            return "?"
        return label_name(self.smooth[self.model].shown) or "?"

    def _refresh(self, now: float) -> None:
        """Choose the base screen from the link state and the shown model's smoother."""
        if self.confirm_t is not None:
            return  # "SET?" owns the screen
        if self.last_data_t is None or now - self.last_data_t > NO_DATA_S:
            self.display.status(ST_NO_DATA)
        elif self.invalid_streak >= INVALID_STREAK_FOR_Q:
            self.display.status(ST_UNSURE)
        else:
            shown = self.smooth[self.model].shown
            if shown is None or shown == UNSURE:
                self.display.status(ST_UNSURE)
            else:
                self.display.activity(shown)

    # ------------------------------------------------------------ gestures
    def _gesture(self, ev, now: float) -> None:
        self.events.append(ev)
        msg = ev.describe()
        if ev.kind == "near_miss":
            self.counts["near_miss"] += 1
            log.info(f"gesture near-miss: {msg}")
        else:
            log.info(f"gesture: {msg}")
        if self._gest_f:
            self._gest.writerow([datetime.now().isoformat(timespec="milliseconds"), ev.kind, ev.n_taps,
                                 " ".join(f"{g:.0f}" for g in ev.gaps_ms), " ".join(f"{t.peak_g:.2f}" for t in ev.taps),
                                 ev.start_index, ev.reason])
            self._gest_f.flush()

        if self.confirm_t is not None:            # waiting for the redo-setup confirmation
            if ev.kind == "switch" and ev.start_index - self.confirm_index <= CONFIRM_S * FS:
                log.warning("redo setup CONFIRMED by gesture: the old config_wired.json will be backed up and setup starts")
                self.confirm_t = None
                self.next_mode = "redo_setup"
            else:
                log.info("redo setup cancelled (a different gesture was made)")
                self._cancel_confirm()
            return

        if ev.locked_out:
            self.counts["near_miss"] += 1
            log.info(f"gesture ignored: within {self.cfg['gesture']['lockout_s']} s of the previous one")
            return
        if ev.kind == "long":
            self.counts["long"] += 1
            if self.redo_open:
                log.info("8 taps inside the redo-setup window: showing SET? - tap 4 times within 5 s to confirm")
                self.confirm_t = now
                self.confirm_index = self.last_index
                self.display.setup(STEP_SET_CONFIRM, 0)
            else:
                log.info("long gesture (8 taps) outside the redo-setup window: no action in the wired app")
        elif ev.kind == "switch":
            if self.switch_mode != "gesture":
                log.info("switch gesture ignored: switch_mode is auto_alternate")
                return
            self.counts["switches"] += 1
            self.model = "B" if self.model == "A" else "A"
            log.info(f"switched display to model {self.model} ({'1D' if self.model == 'A' else '6D'})")
            self.display.model(self.model, banner=True)
            self.display.invalidate()
            self._refresh(now)
            try:
                save_shown_model(self.paths.state, self.model)
            except OSError as e:
                log.error(f"While saving the shown model to state.json: {e}. Likely fix: check the app folder is writable")

    def _cancel_confirm(self) -> None:
        self.confirm_t = None
        self.display.invalidate()
        self._refresh(self.clock())

    # ------------------------------------------------------------ timers
    def tick(self) -> None:
        now = self.clock()
        if self.redo_open and now - self.redo_start > REDO_WINDOW_S:
            self.redo_open = False
            self.display.status(ST_REDO_WINDOW_OFF)
            log.info("redo-setup window closed")
        if self.confirm_t is not None and now - self.confirm_t > CONFIRM_S + CONFIRM_GRACE_S \
                and self.clf.gestures.in_progress == 0:
            log.info("redo setup cancelled (no confirmation within 5 s)")
            self._cancel_confirm()
        if self.switch_mode == "auto_alternate" and self.last_data_t is not None:
            if self.last_alt_t is None:
                self.last_alt_t = now
            elif now - self.last_alt_t >= AUTO_ALTERNATE_S:
                self.last_alt_t = now
                self.model = "B" if self.model == "A" else "A"
                self.display.model(self.model, banner=False)
                self.display.invalidate()
        self._refresh(now)

    def summary(self) -> dict:
        return dict(self.counts, model=self.model, model_id=MODEL_A if self.model == "A" else 1)
