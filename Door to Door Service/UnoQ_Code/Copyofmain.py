"""Subscribes to this car's minifig color on the laptop's MQTT feed
(door-to-door-service-MQTT.py) and forwards each update to the sketch over the
Bridge: a scaled (col, row) position for the LED matrix dot, plus the pixel
offset (dx) that the sketch's PID loop drives to zero to center the car.
"""

import json
import logging

import paho.mqtt.client as mqtt
from arduino.app_utils import *

# ===================== THIS CAR'S MINIFIG =====================
# Set to "green" or "blue" to match the minifig mounted on THIS car.
# The other car runs this same file with the opposite value.
MINIFIG_COLOR = "green"
# =================================================================

if MINIFIG_COLOR not in ("green", "blue"):
    raise ValueError(f'MINIFIG_COLOR must be "green" or "blue", got {MINIFIG_COLOR!r}')

# ===================== MQTT SOURCE =====================
# Must match mqttlib.py / door-to-door-service-MQTT.py on the laptop.
BROKER = "test.mosquitto.org"
PORT = 1883
MQTT_TOPIC = f"ME193/RQ-D2/{MINIFIG_COLOR}"
# =========================================================

MATRIX_COLS = 13
MATRIX_ROWS = 8


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to MQTT broker {BROKER}:{PORT} (reason={reason_code})")
    client.subscribe(MQTT_TOPIC)


def _on_connect_fail(client, userdata):
    print(f"FAILED to connect to MQTT broker {BROKER}:{PORT}")


def _on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"Disconnected from MQTT broker {BROKER}:{PORT} (reason={reason_code})")


def _on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    print(f"{msg.topic} -> {data}")

    found = bool(data.get("found", False))
    col = 0
    row = 0
    dx = 0

    if found:
        w = data.get("w") or 1
        h = data.get("h") or 1
        x = data.get("x", w / 2)
        y = data.get("y", h / 2)
        dx = int(data.get("dx", 0))
        col = _clamp(round(x / w * (MATRIX_COLS - 1)), 0, MATRIX_COLS - 1)
        row = _clamp(round(y / h * (MATRIX_ROWS - 1)), 0, MATRIX_ROWS - 1)

    # Sketch signature: update_target(bool found, int col, int row, int dx)
    Bridge.call("update_target", found, col, row, dx)


logging.basicConfig(level=logging.INFO)

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.enable_logger()  # TEMP: surfaces paho's internal connect/handshake errors
mqtt_client.on_connect = _on_connect
mqtt_client.on_connect_fail = _on_connect_fail
mqtt_client.on_disconnect = _on_disconnect
mqtt_client.on_message = _on_message

try:
    mqtt_client.connect(BROKER, PORT)
    print("connect() returned; waiting for broker handshake...")
except Exception:
    import traceback
    traceback.print_exc()

mqtt_client.loop_start()  # background thread handles MQTT I/O; Bridge calls happen from it

App.run()
