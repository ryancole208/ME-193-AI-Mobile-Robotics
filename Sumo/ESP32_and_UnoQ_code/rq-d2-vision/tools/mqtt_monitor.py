"""Standalone debug tool: subscribes to every topic in the RQ-D2 MQTT
contract and prints every message that arrives, with a timestamp.

Run this from your own laptop (or anywhere with internet access) while
testing the robot - it doesn't need App Lab, the board, or the behavior
tree running:

    python tools/mqtt_monitor.py

It uses the exact same mqttlib.py and config.py as the app itself, so if
this prints nothing for vision/opponent, the robot either isn't running
or isn't reaching the broker - check the robot's own console log first.
"""

import json
import os
import sys
import time

# mqttlib.py and config.py live in ../python relative to this file; this
# lets the tool import them without duplicating the broker/topic settings.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import config  # noqa: E402
from mqttlib import MQTTClient  # noqa: E402

TOPICS = [
    config.TOPIC_VISION,
    config.TOPIC_UNOQ_STATUS,
    config.TOPIC_LAPTOP_STATUS,
    config.TOPIC_ROBOT_STATE,
    config.TOPIC_EVENT_LOST,
    config.TOPIC_EVENT_SONG_DONE,
]


def _on_any_message(topic, payload):
    timestamp = time.strftime("%H:%M:%S")
    try:
        pretty = json.dumps(json.loads(payload))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pretty = payload
    print(f"[{timestamp}] {topic} -> {pretty}")


def main():
    print(f"Connecting to {config.MQTT_BROKER}:{config.MQTT_PORT} ...")
    with MQTTClient(config.MQTT_BROKER, config.MQTT_PORT) as client:
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
