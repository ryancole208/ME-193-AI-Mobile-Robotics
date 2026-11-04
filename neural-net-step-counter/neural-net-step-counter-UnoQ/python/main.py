"""Neural-Net Step Counter - Python side (UNO Q Linux MPU). ME 193.

Counts every footfall (right + left heel strikes) live, using a small dilated
temporal convolutional network (StepTCN) trained in PyTorch on the laptop and
run here in plain numpy, from
  * the ankle: Pico W + MPU-6050 over BLE (100 Hz, 6-axis, firmware unchanged)
  * the hip:   GY-61 (ADXL335) on A0-A2, sampled at 100 Hz by sketch/sketch.ino
and estimates the distance walked, both with a 95 % interval. The matrix shows
steps or distance (4 heel taps switch); an HTTP API serves the laptop.

Startup:
  1. setup.yaml is read (a broken one shows "CFG X" and the defaults are used).
  2. RUN_MODE (one-shot file, deleted when read): "diag" = also print a live
     diag block every 2 s; "setup" = redo the one-time setup.
  3. config.json missing/invalid -> the one-time setup (setup_mode.py) runs first.
  4. model.npz is checked (stepnet_np.check_model); if it is missing or fails,
     the baseline detector counts and the matrix shows "NN X" once.
  5. SELF-CHECK, printed as PASS/FAIL per item (status screen on a failure):
       model   loads, prep_hash matches, numpy == PyTorch reference (< 1e-4)
       hip     each axis within 5-95 % of the ADC range, |a| = 1 +- 0.25 g when still
       ble     ankle packets arriving with no gap for 3 s
       http    the API port is bound
Modes (setup.yaml `mode`, or POST /api/mode): live = count and display;
record = the same + sessions/ for training.

Everything runs in App.run()'s loop thread; the BLE reader, the Bridge
callbacks and the HTTP server only put work into queues.
"""

import atexit
import logging
import queue
import sys
import threading
import time
import traceback

import numpy as np
from arduino.app_utils import App, Bridge

import display as D
from api import ApiServer, Shared
from ble_client import RelayClient
from counter import EventLog
from fusion import prep_hash
from hip_stream import wiring_problem
from pipeline import Engine
from settings import (
    MODEL_UPLOAD, RUN_MODE, StepError, app_yaml_ports, load_settings, DEFAULTS,
)
from stepnet_np import check_model

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("step.main")

# ---------------------------------------------------------------- settings + one-shot mode
display = D.Display(D.BridgeSender())
try:
    settings = load_settings()
except StepError as e:
    import copy
    log.error(f"{e}. Running with the built-in defaults instead.")
    display.status(D.ST_CFG_ERROR)
    settings = copy.deepcopy(DEFAULTS)

run_mode = ""
if RUN_MODE.exists():
    run_mode = RUN_MODE.read_text().strip().lower()
    RUN_MODE.unlink()
    log.info(f"RUN_MODE = {run_mode!r} (one-shot)")

ver = display.handshake()
if ver != D.DISPLAY_PROTOCOL_VERSION:
    log.error(f"sketch display protocol is {ver}, expected {D.DISPLAY_PROTOCOL_VERSION}. Likely fix: restart the "
              "app so App Lab re-flashes sketch/sketch.ino (or clean-cache if it stays stale)")
log.info(f"preprocessing hash (contract 9.3): {prep_hash()}")
if settings["net"]["token"] == "change-me":
    log.warning('net.token is still "change-me": set your own token in setup.yaml')

# ---------------------------------------------------------------- engine + inputs
work: queue.Queue = queue.Queue(maxsize=5000)


def _put(item) -> None:
    try:
        work.put_nowait(item)
    except queue.Full:
        log.error("work queue full: the counting loop is not keeping up - dropping sensor data")


shared = Shared()
evlog = EventLog()
engine = Engine(settings, display, shared, evlog, diag_mode=run_mode == "diag", force_setup=run_mode == "setup")

ble = RelayClient(on_event=lambda ev: _put(("ble", ev)))


def on_hip_packet(data):
    # The sketch sends MsgPack bin (std::vector<uint8_t>) -> bytes here.
    _put(("hip", bytes(data), time.monotonic()))


def on_hip_status(late, skipped, adc_errors, adc_bits, samples):
    _put(("hip_status", (late, skipped, adc_errors, adc_bits, samples), time.monotonic()))


Bridge.provide("hip_packet", on_hip_packet)
Bridge.provide("hip_status", on_hip_status)

# ---------------------------------------------------------------- HTTP API + model upload
_upload_lock = threading.Lock()


def model_upload(body: bytes) -> dict:
    """API thread: check the uploaded file BEFORE it can replace anything.
    Only a model that passes every check is handed to the main loop, which
    keeps the old one as model.prev.npz. A failing upload is deleted and the
    active model stays."""
    with _upload_lock:
        MODEL_UPLOAD.write_bytes(body)
        model, checks = check_model(MODEL_UPLOAD)
        for name, c in checks.items():
            log.info(f"uploaded model check {name}: {'PASS' if c['ok'] else 'FAIL'} - {c['detail']}")
        if model is None:
            MODEL_UPLOAD.unlink(missing_ok=True)
            active = engine.model.version if engine.model else None
            log.error(f"uploaded model REJECTED; still active: {active or 'baseline'}")
            return {"ok": False, "checks": checks, "active_version": active}
        r = shared.command("activate_model", (model, MODEL_UPLOAD))
        return {"ok": bool(r.get("ok")), "checks": checks, "active_version": r.get("active_version")}


api = ApiServer(settings["net"]["port"], settings["net"]["token"], shared, evlog, model_upload)
api.start()
if settings["net"]["mode"] == "push":
    log.warning("net.mode: push is not implemented: App Lab publishes the API port (app.yaml ports:), "
                "so the laptop pulls. Using pull.")
ble.start()
atexit.register(engine.shutdown)       # App Lab stop -> flush and close an open recording


# ---------------------------------------------------------------- self-check
class SelfCheck:
    HIP_WAIT_S = 15        # no hip packet by then -> FAIL
    HIP_STILL_WAIT_S = 20  # look this long for a still second to test |a|
    BLE_WAIT_S = 60
    BLE_WINDOW_S = 3.0
    BLE_TRIES = 5

    def __init__(self):
        self.t0 = time.monotonic()
        self.results: dict = {}
        self._ble_mark = None
        self._ble_tries = 0
        self._hip_t0 = None
        m = engine.model_checks
        self.put("model", engine.model is not None,
                 "; ".join(f"{k}: {'ok' if v['ok'] else 'FAIL'} ({v['detail']})" for k, v in m.items()))
        port = settings["net"]["port"]
        published = port in app_yaml_ports()
        self.put("http", api.bound and published,
                 (f"listening on {port}" if api.bound else f"bind failed: {api.error}")
                 + ("" if published else f"; port {port} is NOT listed under ports: in app.yaml"))

    def put(self, name: str, ok: bool, detail: str) -> None:
        if name in self.results:
            return
        self.results[name] = ok
        log.log(logging.INFO if ok else logging.ERROR, f"SELF-CHECK {name:5s} {'PASS' if ok else 'FAIL'}: {detail}")
        if not ok:
            if name == "hip":
                display.status(D.ST_HIP_FAIL)
            elif name == "ble":
                display.status(D.ST_ANKLE_LOST)
            elif name == "http":
                display.banner("API X")
        if len(self.results) == 4:
            bad = [k for k, v in self.results.items() if not v]
            log.info("SELF-CHECK: ALL PASS" if not bad else f"SELF-CHECK: FAILED {bad} (counting continues "
                     "with what works: baseline if no model, the other sensor if one is missing)")

    def tick(self, now: float) -> None:
        if len(self.results) == 4:
            return
        if "hip" not in self.results:
            self._hip(now)
        if "ble" not in self.results:
            self._ble(now)

    def _hip(self, now: float) -> None:
        h = engine.hip
        if h.first_data_t is None:
            if now - self.t0 > self.HIP_WAIT_S:
                self.put("hip", False, "no hip packets from the sketch (is sketch.ino running? see "
                         "`arduino-app-cli monitor`)")
            return
        g = engine.fusion.hip.val
        if len(g) < 200:
            return
        raw = g * h.scale + h.offset
        problem = wiring_problem(raw[-100:].mean(axis=0))
        if problem:                            # needs no calibration, so it is judged even during setup
            self.put("hip", False, problem)
            return
        if engine.in_setup:
            # Setup's 6-face step IS the hip calibration: judging |a| = 1 g before
            # it with nominal values would fail a GY-61 whose real supply is 3.0 V.
            self._hip_t0 = None
            return
        if self._hip_t0 is None:
            self._hip_t0 = now                 # the "never still" timeout starts after setup
        frac = raw / 16383.0
        # the stillest 1 s window in the buffer
        best = None
        for s in range(0, len(g) - 100 + 1, 25):
            sd = float(np.linalg.norm(g[s:s + 100], axis=1).std())
            if best is None or sd < best[0]:
                best = (sd, s)
        sd, s = best
        win_frac = frac[s:s + 100].mean(axis=0)
        in_range = bool(np.all((win_frac > 0.05) & (win_frac < 0.95)))
        amag = float(np.linalg.norm(g[s:s + 100], axis=1).mean())
        detail = f"axes at {np.round(win_frac * 100, 1).tolist()} % of full scale, |a| {amag:.3f} g (cal {h.cal['method']})"
        if sd < 0.03:
            self.put("hip", in_range and abs(amag - 1.0) <= 0.25, detail + f", still (|a| std {sd:.3f} g)")
        elif now - max(h.first_data_t, self._hip_t0) > self.HIP_STILL_WAIT_S:
            self.put("hip", in_range, detail + " - never still, so |a| = 1 g was not tested; axis ranges only")

    def _ble(self, now: float) -> None:
        a = engine.ankle
        if a.last_data_t is None or now - a.last_data_t > 1.0:
            self._ble_mark = None
            if now - self.t0 > self.BLE_WAIT_S:
                self.put("ble", False, f"no ankle packets within {self.BLE_WAIT_S} s (relay {a.state}; Pico W "
                         "on and advertising as GaitNode? unoq_app not running?)")
            return
        st = a.stats
        if self._ble_mark is None:
            self._ble_mark = (now, st.packets, st.gaps)
            return
        t, p, g = self._ble_mark
        if now - t < self.BLE_WINDOW_S:
            return
        if st.gaps == g and st.packets - p >= 20:
            self.put("ble", True, f"{st.packets - p} packets in {now - t:.1f} s, no gaps, MTU {a.mtu}, "
                     f"n {sorted(a.n_values)}")
            return
        self._ble_tries += 1
        if self._ble_tries >= self.BLE_TRIES:
            self.put("ble", False, f"{st.gaps - g} gap(s) / {st.packets - p} packets in the last 3 s window "
                     f"(loss {st.loss_pct:.1f} %); move the UNO Q and the ankle closer, same side of the body")
        self._ble_mark = None


selfcheck = SelfCheck()
_errors = 0


def loop():
    """Called repeatedly by App.run(): drain the queues, then the 0.5 s step."""
    global _errors
    try:
        try:
            item = work.get(timeout=0.05)
            for _ in range(500):
                kind = item[0]
                if kind == "ble":
                    engine.on_ble(item[1])
                elif kind == "hip":
                    engine.on_hip_packet(item[1], item[2])
                elif kind == "hip_status":
                    engine.on_hip_status(item[1], item[2])
                item = work.get_nowait()
        except queue.Empty:
            pass
        now = time.monotonic()
        engine.tick(now)
        selfcheck.tick(now)
    except Exception as e:
        _errors += 1
        log.error(f"unexpected {type(e).__name__}: {e} (error #{_errors}). Likely fix: this is a bug - send "
                  f"this log.\n{traceback.format_exc()}")


App.run(user_loop=loop)
