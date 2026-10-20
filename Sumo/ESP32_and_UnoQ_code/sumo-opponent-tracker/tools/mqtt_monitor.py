"""Standalone debug tool: subscribes to every topic in the Sumo Opponent
Tracker's MQTT contract and prints every message that arrives, with a
timestamp.

Run this from your own laptop (or anywhere with internet access) while
testing the robot - it doesn't need App Lab, the board, or the behavior
tree running:

    python tools/mqtt_monitor.py

It uses the exact same mqttlib.py as the app itself (imported from
../python), so if this prints nothing for vision/opponent, the robot
either isn't running or isn't reaching the broker - check the robot's own
console log first.

NOTE: the topic strings below are duplicated from main.py's TUNABLE: MQTT
contract section (this script can't import main.py directly - main.py
imports arduino.app_utils, which only exists on the board, not a plain
laptop Python install). Keep these in sync with main.py if you ever
change a topic name there.
"""

import json
import os
import sys
import time

# mqttlib.py lives in ../python relative to this file; this lets the tool
# import it without needing App Lab or the board.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from mqttlib import MQTTClient  # noqa: E402

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC_PREFIX = "ME193/Rogers/RQ-D2/"

TOPICS = [
    TOPIC_PREFIX + "vision/opponent",
    TOPIC_PREFIX + "unoq/status",
    TOPIC_PREFIX + "laptop/status",
    TOPIC_PREFIX + "robot/state",
    TOPIC_PREFIX + "event/lost",
    TOPIC_PREFIX + "event/song_done",
]


def _on_any_message(topic, payload):
    timestamp = time.strftime("%H:%M:%S")
    try:
        pretty = json.dumps(json.loads(payload))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pretty = payload
    print(f"[{timestamp}] {topic} -> {pretty}")


def main():
    print(f"Connecting to {BROKER}:{PORT} ...")
    with MQTTClient(BROKER, PORT) as client:
        for topic in TOPICS:
            client.subscribe(topic, _on_any_message)
        print("Subscribed to:")
        for topic in TOPICS:
            print(f"  {topic}")
        print("Listening - press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopping.")


if __name__ == "__main__":
    main()
