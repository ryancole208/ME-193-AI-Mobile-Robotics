"""Gait Activity Classifier - WIRED test version (no Pico W, no Bluetooth).

The MPU-6050 is wired to the UNO Q's header I2C pins. The sketch reads its
FIFO, builds packets in the same format as the Bluetooth app's ankle node and
sends them here over Bridge; from there everything is the same as the
Bluetooth app: gap detection, units, leg transform, windows, features, two
kNN models, smoothing, gestures, setup, live mode, logging and the matrix.

  sketch.ino --Bridge.notify("imu_packet", bytes)--> on_packet --> queue --> Runner
             --Bridge.notify("imu_status", ...)  --> on_status (once per second)

Modes: normal start = live if config_wired.json is valid, else the one-time
setup. One-shot modes: write "diag", "setup", "holdout" or "replay <csv>" into
the file RUN_MODE in the app folder and restart (tools/wired.py does that).
"""

import logging
import queue
import sys
import time
import traceback

from arduino.app_utils import App, Bridge

from gait.display import BridgeSender, Display
from gait.imu_link import ImuEvent
from gait.paths import Paths
from gait.runner import Runner

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s.%(msecs)03d %(levelname)s - %(name)s:  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gait.main")

events: queue.Queue = queue.Queue(maxsize=2000)


def _put(ev: ImuEvent) -> None:
    try:
        events.put_nowait(ev)
    except queue.Full:
        log.error("event queue full: the classifier is not keeping up - dropping IMU data")


def on_packet(data):
    # The sketch sends a std::vector<uint8_t>, which msgpack delivers as bytes;
    # accept a list of ints too, in case a library version sends an array.
    _put(ImuEvent("packet", data=bytes(data), t=time.monotonic()))


def on_status(state, who_am_i, i2c_errors, reinits, fifo_overflows, bus_hz):
    _put(ImuEvent("status", status=(state, who_am_i, i2c_errors, reinits, fifo_overflows, bus_hz), t=time.monotonic()))


Bridge.provide("imu_packet", on_packet)
Bridge.provide("imu_status", on_status)

display = Display(BridgeSender())
runner = Runner(Paths.default(), display)
runner.boot()

_errors = 0


def loop():
    """Called repeatedly by App.run(): handle queued IMU events, then timers."""
    global _errors
    try:
        try:
            runner.on_event(events.get(timeout=0.1))
            for _ in range(200):  # drain whatever else has arrived
                runner.on_event(events.get_nowait())
        except queue.Empty:
            pass
        runner.tick()
    except Exception as e:
        _errors += 1
        log.error(f"While running mode '{runner.mode}': unexpected {type(e).__name__}: {e} (error #{_errors}). "
                  f"Likely fix: this is a bug - send this log.\n{traceback.format_exc()}")


App.run(user_loop=loop)
