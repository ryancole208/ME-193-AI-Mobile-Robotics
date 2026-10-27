"""diag: first-power-on check of the wired IMU. Lay the MPU-6050 flat and still on a table.

Five checks, in order, then a PASS/FAIL table in the app log and reports/diag.md:

  1 I2C          bus up, something answers at 0x68, WHO_AM_I = 0x68, bus speed
  2 Sampling     30 s: effective rate (expect 100 Hz), gaps from seq and t0_us, I2C errors
  3 Still        |a| about 1.00 g and low gyro noise (catches range / byte-order bugs)
  4 Matrix       display handshake, then every animation and status screen once
  5 Config       config_wired.json valid, or "will start setup"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .config import load_config
from .constants import (
    CLASSES, DISPLAY_PROTOCOL_VERSION, ST_CFG_SETUP, ST_CHECK, ST_FLASH3, ST_IDLE, ST_LEG_MISMATCH, ST_NO_DATA,
    ST_REDO_WINDOW_OFF, ST_REDO_WINDOW_ON, ST_UNSURE, STEP_BUILD, STEP_COUNTDOWN_BASE, STEP_DIAG, STEP_DIAG_DONE,
    STEP_GESTURE_CHECK, STEP_PARTIAL, STEP_RECORD_BASE, STEP_SET_CONFIRM, STEP_STAND, VAL_ACC_A, VAL_ACC_B,
)
from .errors import ConfigError
from .imu_link import STATE_BUS_ERROR, STATE_NO_DEVICE, STATE_OK, STATE_TEXT
from .paths import Paths
from .training import base_signal, converter_for

log = logging.getLogger("gait.diag")

STATUS_TIMEOUT_S = 15.0
STREAM_S = 30.0
BUS = "Wire = I2C2: SDA = D20/PB11, SCL = D21/PB10 (header pins next to AREF)"


@dataclass
class Check:
    num: int
    name: str
    result: str = "PENDING"   # PASS | FAIL | WARN | INFO
    details: str = ""
    fix: str = ""


def matrix_script():
    S = [(f"{name} animation", [("model", ("A", False)), ("activity", (c,))], 3.0) for c, name in enumerate(CLASSES)]
    S += [
        ("model B indicator + 6D banner", [("model", ("B", True))], 2.0),
        ("model A indicator + 1D banner", [("model", ("A", True))], 2.0),
        ("no IMU data (IMU, X)", [("status", (ST_NO_DATA,))], 3.0),
        ("unsure (?)", [("status", (ST_UNSURE,))], 1.5),
        ("config missing (CFG, ?)", [("status", (ST_IDLE,)), ("status", (ST_CFG_SETUP,))], 2.5),
        ("leg mismatch (LEG, ?)", [("status", (ST_LEG_MISMATCH,))], 2.5),
        ("redo-setup marker", [("status", (ST_REDO_WINDOW_ON,))], 2.0),
        ("stand-still countdown", [("status", (ST_REDO_WINDOW_OFF,)), ("setup", (STEP_STAND, 7))], 1.5),
        ("countdown W", [("setup", (STEP_COUNTDOWN_BASE + 1, 9))], 1.5),
        ("recording R 40 %", [("setup", (STEP_RECORD_BASE + 2, 40))], 1.5),
        ("gesture check 3/5", [("setup", (STEP_GESTURE_CHECK, 3))], 1.5),
        ("building (spinner)", [("setup", (STEP_BUILD, 0))], 1.5),
        ("partial: B missing", [("setup", (STEP_PARTIAL, 3))], 1.5),
        ("SET? confirm", [("setup", (STEP_SET_CONFIRM, 0))], 2.5),
        ("flash x3 + check mark", [("status", (ST_FLASH3,)), ("status", (ST_CHECK,))], 2.5),
        ("accuracy A 87 % / B 91 %", [("value", (VAL_ACC_A, 87)), ("value", (VAL_ACC_B, 91))], 4.5),
    ]
    return S


class DiagController:
    def __init__(self, settings: dict, paths: Paths, display, link, clock=time.monotonic):
        self.s = settings
        self.paths = paths
        self.display = display
        self.link = link
        self.clock = clock
        self.next_mode = None
        self.checks = [Check(1, "I2C + MPU-6050"), Check(2, f"Sampling {STREAM_S:.0f} s"), Check(3, "IMU still on table"),
                       Check(4, "LED matrix"), Check(5, "config_wired.json")]
        self.current = 0
        self.t0 = 0.0
        self.samples: list = []
        self.collecting = False
        self.stats0 = None
        self.err0 = 0
        self.overflow0 = 0
        self.idx_first = self.idx_last = None
        self.received = 0
        self.script = matrix_script()
        self.script_i = -1
        self.script_t = None
        self.finished = False
        self.conv = converter_for(base_signal(settings))

    def start(self) -> None:
        log.info("diag: running 5 checks - lay the MPU-6050 flat and still on a table")
        self._begin(0)

    def _begin(self, i: int) -> None:
        self.current = i
        self.t0 = self.clock()
        if i < len(self.checks):
            log.info(f"diag check {i + 1}/5: {self.checks[i].name}")
            self.display.setup(STEP_DIAG, i + 1)

    def _done(self, result: str, details: str, fix: str = "") -> None:
        c = self.checks[self.current]
        c.result, c.details, c.fix = result, details, fix
        log.log(logging.ERROR if result == "FAIL" else logging.INFO,
                f"diag {c.num} {c.name}: {result} - {details}" + (f" | fix: {fix}" if fix else ""))
        self._begin(self.current + 1)

    def on_link_event(self, ev) -> None:
        pass

    def on_chunk(self, chunk) -> None:
        if not self.collecting:
            return
        if self.idx_first is None:
            self.idx_first = chunk.start_index
        self.idx_last = chunk.start_index + len(chunk.raw)
        self.received += len(chunk.raw)
        if len(self.samples) < 400:
            self.samples.append(self.conv.convert(chunk.raw))

    def tick(self) -> None:
        if self.finished:
            return
        now = self.clock()
        el = now - self.t0
        L = self.link
        i = self.current
        if i == 0:
            if L.state == STATE_OK and L.last_data_t is not None:
                self._done("PASS", f"{BUS}; 0x68 answers; WHO_AM_I 0x{L.who_am_i:02X}; bus {L.bus_hz // 1000} kHz")
                self._start_sampling()
            elif el > STATUS_TIMEOUT_S:
                if L.status_t is None:
                    self._done("FAIL", "no imu_status from the sketch in 15 s",
                               "the sketch is not running or is out of date: restart the app so App Lab re-flashes "
                               "sketch/sketch.ino; `arduino-app-cli monitor` should print 'gait wired ready'")
                elif L.state == STATE_OK:
                    self._done("FAIL", f"{BUS}: sensor OK (WHO_AM_I 0x{L.who_am_i:02X}) but no imu_packet reached Python",
                               "the Bridge names in sketch.ino and main.py must match exactly ('imu_packet'); "
                               "check the app log for packet errors")
                elif L.state == STATE_BUS_ERROR:
                    self._done("FAIL", f"{BUS}: bus not ready", "restart the app; if it persists, re-flash with "
                               "`arduino-app-cli app clean-cache user:unoq_wired_app --force`")
                elif L.state == STATE_NO_DEVICE:
                    self._done("FAIL", f"{BUS}: nothing answers at 0x68 (I2C errors {L.i2c_errors})",
                               "check the 4 wires: VCC to 3.3V, GND to GND, SDA to SDA (D20), SCL to SCL (D21); "
                               "AD0 must be low or open (0x69 if high); try a shorter cable")
                else:
                    self._done("FAIL", f"WHO_AM_I is 0x{(L.who_am_i or 0):02X}, expected 0x68 "
                               f"(state: {STATE_TEXT.get(L.state, L.state)})",
                               "the chip is not an MPU-6050 (0x70 = MPU-6500, 0x71 = MPU-9250): registers may differ")
                for _ in range(2):
                    self._done("FAIL", "skipped", "fix check 1 first")
        elif i == 1:
            if el >= STREAM_S:
                self._finish_sampling(el)
        elif i == 2:
            self._still_check()
        elif i == 3:
            self._matrix_step(now)
        elif i == 4:
            self._config_check()
            self._report()

    def _start_sampling(self) -> None:
        st = self.link.stats
        self.stats0 = (st.packets, st.samples, st.packets_lost_seq, st.samples_missing_t0, st.fifo_gaps,
                       st.seq_gaps, st.resyncs)
        self.err0, self.overflow0 = self.link.i2c_errors, self.link.fifo_overflows
        self.collecting = True

    def _finish_sampling(self, el: float) -> None:
        self.collecting = False
        st = self.link.stats
        now_vals = (st.packets, st.samples, st.packets_lost_seq, st.samples_missing_t0, st.fifo_gaps,
                    st.seq_gaps, st.resyncs)
        packets, samples, lost, missing, fifo, seq_gaps, resyncs = (b - a for a, b in zip(self.stats0, now_vals))
        errors = self.link.i2c_errors - self.err0
        overflows = self.link.fifo_overflows - self.overflow0
        rate_rx = self.received / el
        span = (self.idx_last - self.idx_first) if self.idx_first is not None else 0
        rate_sensor = span / el
        det = (f"{samples} samples in {el:.1f} s: received {rate_rx:.1f} Hz, sensor clock {rate_sensor:.1f} Hz; "
               f"packets {packets}, seq gaps {seq_gaps} ({lost} packets), t0_us gaps {fifo} "
               f"({missing} samples missing), FIFO overflows {overflows}, resyncs {resyncs}, I2C errors {errors}")
        if samples == 0:
            self._done("FAIL", det, "no samples arrived: check the app log for imu_packet errors")
        elif not 95 <= rate_sensor <= 105:
            self._done("FAIL", det, "the MPU-6050 is not sampling at 100 Hz: check SMPLRT_DIV (9) and CONFIG "
                       "(DLPF_CFG 3) in sketch.ino")
        elif errors > 0:
            self._done("FAIL" if errors > 5 else "WARN", det,
                       "I2C errors: shorten the cable, twist SDA and SCL each with a GND wire, keep 100 kHz")
        elif lost or fifo or overflows or resyncs:
            self._done("WARN", det, "gaps mean the sketch fell behind reading the FIFO (a long Bridge stall?)")
        elif rate_rx < 95:
            self._done("FAIL", det, "packets are dropped between the sketch and Python")
        else:
            self._done("PASS", det)

    def _still_check(self) -> None:
        if not self.samples:
            self._done("FAIL", "no samples to check", "fix checks 1-2 first")
            return
        x = np.vstack(self.samples)
        amag = np.linalg.norm(x[:, 0:3], axis=1)
        gstd = x[:, 3:6].std(axis=0)
        gmean = x[:, 3:6].mean(axis=0)
        det = (f"|a| mean {amag.mean():.3f} g (std {amag.std():.4f}); gyro std {np.round(gstd, 3).tolist()} dps, "
               f"gyro offset {np.round(gmean, 2).tolist()} dps; accel mean {np.round(x[:, 0:3].mean(0), 3).tolist()} g")
        m = amag.mean()
        if abs(m - 1.0) > 0.06:
            if m > 7.0:
                fix = "about 8 g: the MPU-6050 is at ±2 g - check ACCEL_CONFIG = 0x18 is written in sketch.ino"
            elif m < 0.3 or m > 3.0:
                fix = "wildly off: byte order - the FIFO is big-endian and the sketch must swap to little-endian"
            else:
                fix = "is the IMU lying still? Otherwise check the accel range (±16 g = 2048 LSB/g)"
            self._done("FAIL", det, fix)
        elif gstd.max() > 1.0:
            self._done("FAIL", det, "gyro noise high: is the IMU still on the table? Otherwise check GYRO_CONFIG = 0x18")
        elif np.abs(gmean).max() > 30:
            self._done("WARN", det, "large gyro offset - harmless for the models (no gyro means are used)")
        else:
            self._done("PASS", det)

    def _matrix_step(self, now: float) -> None:
        if self.script_i == -1:
            v = self.display.handshake()
            if v != DISPLAY_PROTOCOL_VERSION:
                self._done("FAIL", f"display_version returned {v!r}, expected {DISPLAY_PROTOCOL_VERSION}",
                           "the sketch is not running or is the Bluetooth app's sketch: restart this app")
                return
            self.script_i = 0
        if self.script_t is not None and now - self.script_t < self.script[self.script_i - 1][2]:
            return
        if self.script_i >= len(self.script):
            self.display.status(ST_IDLE)
            self._done("PASS", f"handshake OK; played {len(self.script)} screens: " + ", ".join(s[0] for s in self.script),
                       "PASS means the sketch answered - confirm by eye that each screen looked right")
            return
        desc, cmds, _ = self.script[self.script_i]
        log.info(f"matrix: {desc}")
        for method, args in cmds:
            getattr(self.display, method)(*args)
        self.script_i += 1
        self.script_t = now

    def _config_check(self) -> None:
        try:
            cfg = load_config(self.paths.config)
        except ConfigError as e:
            if not self.paths.config.exists():
                self._done("INFO", "config_wired.json not found: setup starts on the next normal start")
            else:
                self._done("FAIL", str(e), "it is renamed to config_wired.broken-<date>.json and setup starts next time")
            return
        det = (f"valid; created {cfg['created']}; CV balanced accuracy A {cfg['model_a']['cv_balanced_accuracy']:.2f}, "
               f"B {cfg['model_b']['cv_balanced_accuracy']:.2f}; leg {cfg['signal']['leg']}")
        if cfg["signal"]["leg"] != self.s["leg"]:
            self._done("WARN", det, f"setup.yaml says leg: {self.s['leg']} - the matrix will show LEG ?")
        else:
            self._done("PASS", det + " -> live mode on the next normal start")

    def _report(self) -> None:
        self.finished = True
        ok = all(c.result in ("PASS", "INFO", "WARN") for c in self.checks)
        lines = ["# diag report (wired)", "", f"{datetime.now().isoformat(timespec='seconds')} - overall: "
                 f"**{'PASS' if ok else 'FAIL'}**", "",
                 "| # | check | result | details | likely cause / fix |", "|---|---|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c.num} | {c.name} | **{c.result}** | {c.details.replace('|', '/')} | {c.fix.replace('|', '/')} |")
        L = self.link
        lines += ["", f"- sensor state: {STATE_TEXT.get(L.state, L.state)}, WHO_AM_I {L.who_am_i}, bus {L.bus_hz} Hz",
                  f"- I2C errors {L.i2c_errors}, re-inits {L.reinits}, FIFO overflows {L.fifo_overflows}",
                  f"- samples/packet seen {sorted(L.n_values)}; stream events {L.stats.events[-10:]}", ""]
        self.paths.reports.mkdir(parents=True, exist_ok=True)
        path = self.paths.reports / "diag.md"
        path.write_text("\n".join(lines))
        for line in lines:
            log.info(line)
        log.info(f"diag finished: {'PASS' if ok else 'FAIL'} - saved to {path}. Restart the app for normal use.")
        self.display.setup(STEP_DIAG_DONE, 1 if ok else 0)
