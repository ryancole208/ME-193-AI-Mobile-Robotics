"""diag: the first-power-on hardware check. Put the pod flat on a table and run it.

Runs seven checks in order, then prints a PASS/FAIL table to the app log and
saves it to reports/diag.md (paste that file back for help):

  1 Bluetooth access     relay container up, BlueZ reachable, adapter powered
  2 GaitNode link        found, connected, subscribed; MTU and samples/packet
  3 Streaming (30 s)     effective sample rate, packet loss, gaps (seq and t0_us)
  4 Pod still on table   |a| about 1.00 g and low gyro noise (catches byte-order / scaling bugs)
  5 Battery              a battery notification arrived (0 = reading unavailable)
  6 Matrix               display handshake, then every animation and status screen once
  7 config.json          valid -> live mode next start; missing -> setup next start
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .config import load_config
from .constants import (
    CLASSES, DISPLAY_PROTOCOL_VERSION, MIN_MTU_FOR_FULL_PACKET, ST_CFG_ERROR, ST_CHECK, ST_FLASH3,
    ST_IDLE, ST_LEG_MISMATCH, ST_LINK_LOST, ST_LOW_BATT_OFF, ST_LOW_BATT_ON, ST_REDO_WINDOW_OFF,
    ST_REDO_WINDOW_ON, ST_SEARCHING, ST_UNSURE, STEP_BUILD, STEP_COUNTDOWN_BASE, STEP_DIAG, STEP_DIAG_DONE,
    STEP_GESTURE_CHECK, STEP_PARTIAL, STEP_RECORD_BASE, STEP_SET_CONFIRM, STEP_STAND, VAL_ACC_A, VAL_ACC_B,
    VAL_BATTERY,
)
from .errors import ConfigError
from .paths import Paths
from .training import base_signal, converter_for

log = logging.getLogger("gait.diag")

BT_TIMEOUT_S = 45.0       # first run: the relay waits until App Lab has installed bleak
CONNECT_TIMEOUT_S = 60.0
STREAM_S = 30.0
BATTERY_GRACE_S = 5.0


@dataclass
class Check:
    num: int
    name: str
    result: str = "PENDING"   # PASS | FAIL | WARN | INFO
    details: str = ""
    fix: str = ""


# Screens shown by check 6: (description, list of (method, args), seconds)
def matrix_script():
    S = []
    for c, name in enumerate(CLASSES):
        S.append((f"{name} animation", [("model", ("A", False)), ("activity", (c,))], 3.0))
    S += [
        ("model B indicator + 6D banner", [("model", ("B", True))], 2.0),
        ("model A indicator + 1D banner", [("model", ("A", True))], 2.0),
        ("searching (scanning dot)", [("status", (ST_SEARCHING,))], 2.0),
        ("link lost (X)", [("status", (ST_LINK_LOST,))], 1.5),
        ("unsure (?)", [("status", (ST_UNSURE,))], 1.5),
        ("config error (CFG, X)", [("status", (ST_IDLE,)), ("status", (ST_CFG_ERROR,))], 2.5),
        ("leg mismatch (LEG, ?)", [("status", (ST_LEG_MISMATCH,))], 2.5),
        ("low battery icon", [("status", (ST_LOW_BATT_ON,))], 1.5),
        ("battery bar 60 %", [("status", (ST_LOW_BATT_OFF,)), ("value", (VAL_BATTERY, 60))], 2.5),
        ("battery n/a", [("value", (VAL_BATTERY, -1))], 2.5),
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
    def __init__(self, settings: dict, paths: Paths, display, link, clock=time.monotonic, link_write=None,
                 timeouts: dict | None = None):
        self.s = settings
        self.paths = paths
        self.display = display
        self.link = link
        self.clock = clock
        self.link_write = link_write or (lambda text: False)
        t = timeouts or {}
        self.bt_timeout = t.get("bt", BT_TIMEOUT_S)
        self.connect_timeout = t.get("connect", CONNECT_TIMEOUT_S)
        self.stream_s = t.get("stream", STREAM_S)
        self.next_mode = None
        self.checks = [Check(1, "Bluetooth access"), Check(2, "GaitNode link"), Check(3, f"Streaming {self.stream_s:.0f} s"),
                       Check(4, "Pod still on table"), Check(5, "Battery value"), Check(6, "LED matrix"),
                       Check(7, "config.json")]
        self.current = 0
        self.t0 = None
        self.samples: list = []
        self.stream_start = None
        self.stats0 = None
        self.idx_first = self.idx_last = None
        self.received = 0
        self.battery_before = None
        self.script = matrix_script()
        self.script_i = -1
        self.script_t = 0.0
        self.finished = False
        self.conv = converter_for(base_signal(settings))

    def start(self) -> None:
        log.info("diag: running 7 checks - put the ankle pod flat and still on a table")
        self._begin(0)

    def _begin(self, i: int) -> None:
        self.current = i
        self.t0 = self.clock()
        if i < len(self.checks):
            log.info(f"diag check {i + 1}/7: {self.checks[i].name}")
            self.display.setup(STEP_DIAG, i + 1)

    def _done(self, result: str, details: str, fix: str = "") -> None:
        c = self.checks[self.current]
        c.result, c.details, c.fix = result, details, fix
        log.log(logging.ERROR if result == "FAIL" else logging.INFO, f"diag {c.num} {c.name}: {result} - {details}"
                + (f" | fix: {fix}" if fix else ""))
        self._begin(self.current + 1)

    def on_link_event(self, ev) -> None:
        pass

    def on_chunk(self, chunk) -> None:
        if self.stream_start is None:
            return
        if self.idx_first is None:
            self.idx_first = chunk.start_index
        self.idx_last = chunk.start_index + len(chunk.raw)
        self.received += len(chunk.raw)
        if len(self.samples) < 4000:
            self.samples.append(self.conv.convert(chunk.raw))

    # ------------------------------------------------------------ the checks
    def tick(self) -> None:
        if self.finished:
            return
        now = self.clock()
        el = now - self.t0
        L = self.link
        i = self.current
        if i == 0:
            if L.relay_up and L.bluez.get("ok"):
                self._done("PASS", f"relay up; {L.bluez.get('detail')} {L.bluez.get('adapters', '')}")
            elif el > self.bt_timeout:
                if not L.relay_up:
                    self._done("FAIL", f"cannot reach the BLE relay ({L.relay_error})",
                               "the ble-relay container is not running. Check `docker ps` shows "
                               "<app>-ble-relay-1; if not, App Lab did not apply bricks/ble_link/brick_compose.yaml "
                               "(app.yaml must list `- ble_link: {}`). On the very first run it waits for bleak to "
                               "install (needs internet) - check `docker logs` of the relay")
                else:
                    self._done("FAIL", f"relay up but BlueZ not usable: {L.bluez.get('detail')}",
                               "if 'powered off': run `bluetoothctl power on` on the board; if a D-Bus "
                               "permission error: the relay must run as user 1000 with /run/dbus mounted")
        elif i == 1:
            if L.connected and L.n_this_connection is not None:
                mtu_txt = f"MTU {L.mtu} ({L.mtu_source})"
                det = (f"found, connected and subscribed; RSSI {L.rssi} dBm, {mtu_txt}, "
                       f"n = {L.n_this_connection} samples/packet")
                if L.info_problems:
                    self._done("FAIL", det + "; Info: " + "; ".join(L.info_problems),
                               "the Pico W firmware does not match the shared protocol (version 1, ±16 g, ±2000 dps, 100 Hz)")
                elif L.mtu is not None and L.mtu < MIN_MTU_FOR_FULL_PACKET and L.mtu_source != "default":
                    self._done("WARN", det, f"MTU below {MIN_MTU_FOR_FULL_PACKET}: works with n < 10 but uses more packets")
                else:
                    self._done("PASS", det)
                self._start_stream()
            elif el > self.connect_timeout:
                self._done("FAIL", f"no data from GaitNode within {self.connect_timeout:.0f} s (BLE state: {L.state})",
                           "power the Pico W, keep it within a few metres, check it advertises as 'GaitNode' "
                           "(e.g. with a phone BLE scanner app). If it connects but sends nothing, it is not seeing "
                           "the subscription (CCCD write) - check the Pico W log")
                self._skip_stream_checks("no link")
        elif i == 2:
            if el >= self.stream_s:
                self._finish_stream(el)
        elif i == 3:
            self._still_check()
        elif i == 4:
            if L.battery_mV is not None and (L.battery_t or 0) >= (self.stream_start or 0):
                self._battery_done()
            elif el > BATTERY_GRACE_S:
                if L.battery_mV is not None:
                    self._battery_done(" (value from before the test)")
                else:
                    self._done("FAIL", "no battery notification or read in the whole test",
                               "the Pico W must notify the battery characteristic every 10 s and answer PING; "
                               "check that it sends notifications only after the CCCD write")
        elif i == 5:
            self._matrix_step(now)
        elif i == 6:
            self._config_check()
            self._report()

    def _start_stream(self) -> None:
        st = self.link.stats
        self.stats0 = (st.packets, st.samples, st.packets_lost_seq, st.samples_missing_t0, st.fifo_gaps, st.seq_gaps,
                       st.t0_misaligned, st.resyncs)
        self.stream_start = self.clock()
        self.link_write("PING")

    def _skip_stream_checks(self, why: str) -> None:
        for _ in range(3):
            self._done("FAIL", f"skipped ({why})", "fix check 2 first")

    def _finish_stream(self, el: float) -> None:
        st = self.link.stats
        now_vals = (st.packets, st.samples, st.packets_lost_seq, st.samples_missing_t0, st.fifo_gaps, st.seq_gaps,
                    st.t0_misaligned, st.resyncs)
        d = [b - a for a, b in zip(self.stats0, now_vals)]
        packets, samples, lost, missing, fifo, seq_gaps, misaligned, resyncs = d
        rate_rx = self.received / el
        span = (self.idx_last - self.idx_first) if self.idx_first is not None else 0
        rate_sensor = span / el
        loss = 100 * lost / (packets + lost) if packets + lost else 100.0
        det = (f"{samples} samples in {el:.1f} s: received {rate_rx:.1f} Hz, sensor clock {rate_sensor:.1f} Hz; "
               f"packets {packets}, lost {lost} ({loss:.2f} %), seq gaps {seq_gaps}, FIFO (t0_us) gaps {fifo}, "
               f"missing samples by t0_us {missing}, misaligned t0 {misaligned}, resyncs {resyncs}")
        if samples == 0:
            self._done("FAIL", det, "no IMU data: see check 2")
        elif not 95 <= rate_sensor <= 105:
            self._done("FAIL", det, "the sensor clock is not 100 Hz: check the MPU-6050 sample-rate divider "
                       "(SMPLRT_DIV) and DLPF on the Pico W, and that t0_us advances 10000 per sample")
        elif loss > 2.0 or rate_rx < 95:
            self._done("FAIL", det, "too much loss: move the pod closer, check the MTU (n = 10 needs >= 130), "
                       "reduce other 2.4 GHz traffic; RESEND <seq> exists if loss stays above 2 %")
        elif fifo > 0 or resyncs > 0 or misaligned > 0:
            self._done("WARN", det, "FIFO overflows / resyncs mean the Pico W fell behind reading the MPU-6050")
        else:
            self._done("PASS", det)

    def _still_check(self) -> None:
        if not self.samples:
            self._done("FAIL", "no samples to check", "fix checks 2-3 first")
            return
        x = np.vstack(self.samples)
        amag = np.linalg.norm(x[:, 0:3], axis=1)
        gstd = x[:, 3:6].std(axis=0)
        gmean = x[:, 3:6].mean(axis=0)
        det = (f"|a| mean {amag.mean():.3f} g (std {amag.std():.4f}); gyro std {np.round(gstd, 3).tolist()} dps, "
               f"gyro mean (offset) {np.round(gmean, 2).tolist()} dps; accel mean {np.round(x[:, 0:3].mean(0), 3).tolist()} g")
        m = amag.mean()
        if abs(m - 1.0) > 0.06:
            if m > 7.0:
                fix = "about 8 g: the MPU-6050 is at ±2 g but the UNO Q assumes ±16 g - set ACCEL_CONFIG to ±16 g"
            elif m < 0.3 or m > 3.0:
                fix = ("wildly off: almost certainly byte order - the FIFO is big-endian and every int16 must be "
                       "swapped to little-endian before sending")
            else:
                fix = "check the accel range (±16 g, 2048 LSB/g) and that the pod is not moving"
            self._done("FAIL", det, fix)
        elif gstd.max() > 1.0:
            self._done("FAIL", det, "gyro noise high: is the pod really still on a table? Otherwise suspect byte order "
                       "or a wrong gyro range (±2000 dps = 16.4 LSB/(deg/s))")
        elif np.abs(gmean).max() > 30:
            self._done("WARN", det, "large gyro offset - harmless for the models (no gyro means are used), "
                       "but check the gyro range")
        else:
            self._done("PASS", det)

    def _battery_done(self, note: str = "") -> None:
        mv = self.link.battery_mV
        if mv == 0:
            self._done("PASS", f"0 mV received = reading unavailable (battery reading disabled on the Pico W){note}")
        else:
            low = mv < self.s["low_battery_mV"]
            self._done("WARN" if low else "PASS", f"{mv} mV{note}" + (" - below low_battery_mV" if low else ""),
                       "charge/replace the ankle battery" if low else "")

    def _matrix_step(self, now: float) -> None:
        if self.script_i == -1:
            v = self.display.handshake()
            if v != DISPLAY_PROTOCOL_VERSION:
                self._done("FAIL", f"display_version returned {v!r}, expected {DISPLAY_PROTOCOL_VERSION}",
                           "the sketch is not running or is out of date: restart the app so App Lab re-flashes "
                           "sketch/sketch.ino, and check `arduino-app-cli monitor` for 'gait display ready'")
                return
            self.script_i, self.script_t = 0, None
        if self.script_t is not None and now - self.script_t < self.script[self.script_i - 1][2]:
            return  # the previous screen is still showing
        if self.script_i >= len(self.script):
            self.display.status(ST_IDLE)
            self._done("PASS", f"handshake OK; played {len(self.script)} screens: "
                       + ", ".join(s[0] for s in self.script),
                       "PASS means the sketch answered - please confirm by eye that each screen looked right")
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
                self._done("INFO", "config.json not found: the app will start setup on its next normal start")
            else:
                self._done("FAIL", str(e), "it will be renamed to config.broken-<date>.json and setup starts on the next normal start")
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
        lines = [f"# diag report", "", f"{datetime.now().isoformat(timespec='seconds')} - overall: "
                 f"**{'PASS' if ok else 'FAIL'}**", "",
                 "| # | check | result | details | likely cause / fix |", "|---|---|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c.num} | {c.name} | **{c.result}** | {c.details.replace('|', '/')} | {c.fix.replace('|', '/')} |")
        L = self.link
        lines += ["", "## Link details", "", f"- relay: {'up' if L.relay_up else 'down'} {L.relay_error}",
                  f"- BlueZ: {L.bluez}", f"- Info characteristic: {L.info}", f"- MTU {L.mtu} ({L.mtu_source}), "
                  f"n values seen {sorted(L.n_values)}", f"- connections: {L.connect_count}",
                  f"- stream events: {L.stats.events[-10:]}", ""]
        text = "\n".join(lines)
        self.paths.reports.mkdir(parents=True, exist_ok=True)
        path = self.paths.reports / "diag.md"
        path.write_text(text)
        for line in lines:
            log.info(line)
        log.info(f"diag finished: {'PASS' if ok else 'FAIL'} - table saved to {path}. Restart the app (without "
                 "RUN_MODE) for normal use.")
        self.display.setup(STEP_DIAG_DONE, 1 if ok else 0)
        self.ok = ok
