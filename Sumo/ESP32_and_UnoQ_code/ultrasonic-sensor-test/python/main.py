"""Ultrasonic Sensor Test - no MQTT, no laptop, no behavior tree.

The sketch does all the real work (ping, filter, detect, draw the LED
matrix). This file just logs what it's seeing to the console, once a
second, so you don't have to open App Lab's Serial Monitor to follow
along - but the LED matrix alone is enough to test with if you'd rather
just watch the board.
"""

import logging
import threading
import time

from arduino.app_utils import App, Bridge, Logger

logger = Logger("UltrasonicTest", level=logging.INFO)

_lock = threading.Lock()
_latest = {"visible": False, "distance_cm": None, "confidence": 0.0, "sensor_ok": True}


def _on_vision_update(visible: bool, distance_cm: float, confidence: float, sensor_ok: bool):
    """Called by the sketch (Bridge.notify) on every filtered reading.
    Only updates shared state here - no Bridge.call is ever made in
    response, so there's no risk of the Bridge-thread deadlock documented
    in the "RQ-D2 Vision" app's README (that only bites when a notify
    callback tries to call back into the sketch)."""
    with _lock:
        _latest["visible"] = bool(visible)
        _latest["distance_cm"] = None if distance_cm < 0 else float(distance_cm)
        _latest["confidence"] = float(confidence)
        _latest["sensor_ok"] = bool(sensor_ok)


Bridge.provide("vision_update", _on_vision_update)


def loop():
    with _lock:
        visible = _latest["visible"]
        distance_cm = _latest["distance_cm"]
        confidence = _latest["confidence"]
        sensor_ok = _latest["sensor_ok"]

    dist_str = "?" if distance_cm is None else f"{distance_cm:.1f}cm"
    logger.info(f"distance={dist_str} visible={visible} confidence={confidence:.2f} sensor_ok={sensor_ok}")
    time.sleep(1.0)


App.run(user_loop=loop)
