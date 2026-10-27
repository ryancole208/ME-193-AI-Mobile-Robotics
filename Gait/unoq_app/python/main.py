"""Gait Activity Classifier - Python side (runs on the UNO Q's Linux MPU).

Worn on the waist. Receives IMU data from the Pico W ankle node ("GaitNode")
over Bluetooth LE, classifies still / walking / running / biking with two kNN
models (A = "1D": peak shin-axis acceleration only, B = "Full IMU": all six
channels), and shows the result on the LED matrix as an animated stick figure.
A 4-tap foot gesture switches which model is shown.

How the pieces fit:
  ble_link brick   relay container (bleak + BlueZ) -> Unix socket -> LinkEvents
  gait.runner      picks setup / live / diag / holdout / replay at startup
  gait.*           packet parsing, features, kNN, gestures, setup, live, diag
  sketch.ino       draws everything on the matrix (Python sends small commands)

Modes: normal start = live if config.json is valid, else the one-time setup.
One-shot modes: write "diag", "setup", "holdout" or "replay <csv>" into the file
RUN_MODE in the app folder, then restart the app. See README.md.
"""

import logging
import queue
import sys
import traceback

from arduino.app_utils import App

from ble_link import BleLink
from gait.display import BridgeSender, Display
from gait.paths import Paths
from gait.runner import Runner

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s.%(msecs)03d %(levelname)s - %(name)s:  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gait.main")

events: queue.Queue = queue.Queue(maxsize=5000)


def on_link_event(ev):
    try:
        events.put_nowait(ev)
    except queue.Full:
        log.error("event queue full: the classifier is not keeping up - dropping BLE data")


link = BleLink(on_event=on_link_event)
display = Display(BridgeSender())
runner = Runner(Paths.default(), display, link_write=link.write_control)
runner.boot()

_errors = 0


def loop():
    """Called repeatedly by App.run(): handle queued BLE events, then timers."""
    global _errors
    try:
        try:
            ev = events.get(timeout=0.1)
            runner.on_event(ev)
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
