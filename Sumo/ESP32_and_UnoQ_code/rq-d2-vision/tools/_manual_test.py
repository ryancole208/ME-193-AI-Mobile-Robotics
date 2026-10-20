"""One-off manual verification script (not part of the app) - publishes a
scripted sequence of robot/state, laptop/status and event/lost messages so
the running app's LED icon logic and lost->song->song_done flow can be
checked end-to-end without the real laptop program. Safe to delete."""
import json
import sys
import time

sys.path.insert(0, "../python")
import config
from mqttlib import MQTTClient

results = {"song_done": False}


def on_song_done(topic, payload):
    results["song_done"] = True
    print("GOT song_done:", payload)


with MQTTClient(config.MQTT_BROKER, config.MQTT_PORT) as client:
    client.subscribe(config.TOPIC_EVENT_SONG_DONE, on_song_done)
    time.sleep(1)

    sequence = [
        ("armed", None),
        ("running", "reverse"),
        ("running", "chase"),
        ("running", "teleop"),
        ("running", "idle"),
    ]
    for phase, branch in sequence:
        payload = json.dumps({"phase": phase, "branch": branch})
        client.publish(config.TOPIC_ROBOT_STATE, payload)
        print("sent robot/state:", payload)
        time.sleep(2.5)

    client.publish(config.TOPIC_LAPTOP_STATUS, json.dumps({"state": "online"}))
    print("sent laptop/status online")
    time.sleep(2)

    client.publish(config.TOPIC_EVENT_LOST, json.dumps({"reason": "test"}))
    print("sent event/lost")

    waited = 0
    while not results["song_done"] and waited < 8:
        time.sleep(0.5)
        waited += 0.5
    print("song_done received:", results["song_done"])
