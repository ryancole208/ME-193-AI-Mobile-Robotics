"""
Prints the hit-streak record that the Godot rhythm game publishes over MQTT.

Godot sends {"type":"score","streak":..,"max_streak":..} to bridge.py over UDP after every judged
ball; whenever max_streak beats the record since the bridge started, bridge.py publishes it as a
float ("7.0") to bridge_config.MQTT_TOPIC on the broker in ../../mqttlib.py. The bridge publishes
0.0 when it starts.

    python mqtt_receiver.py            # Ctrl+C to quit
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # ../../mqttlib.py

import bridge_config as bc
from mqttlib import BROKER, MQTTClient


def on_message(topic, payload):
    stamp = datetime.now().strftime("%H:%M:%S")
    try:
        hits = int(float(payload))
    except ValueError:
        print(f"[{stamp}] {topic}: {payload!r} (not a number)")
        return
    if hits == 0:
        print(f"[{stamp}] {topic}: 0 -- bridge started, record reset")
    else:
        print(f"[{stamp}] {topic}: new record -- {hits} consecutive hits")


def main():
    print(f"Listening on {BROKER}, topic {bc.MQTT_TOPIC} (Ctrl+C to quit)")
    try:
        with MQTTClient() as client:
            client.subscribe(bc.MQTT_TOPIC, on_message)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
