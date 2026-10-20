"""Standalone debug tool: publishes fake robot/state and event/lost
messages so you can test the UNO Q's LED icons and sad song WITHOUT the
laptop's behavior tree running at all.

Run this from your own laptop (or anywhere with internet access):

    python tools/fake_robot.py

Then pick an option from the menu. Watch the LED matrix on the real robot
update as you do.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import config  # noqa: E402
from mqttlib import MQTTClient  # noqa: E402

# (menu label, phase, branch) - branch is only meaningful when phase == "running"
STATE_OPTIONS = [
    ("Armed (waiting for start lever)", "armed", None),
    ("Running / reverse", "running", "reverse"),
    ("Running / chase", "running", "chase"),
    ("Running / teleop", "running", "teleop"),
    ("Running / idle", "running", "idle"),
    ("Lost (phase only, no song)", "lost", None),
]


def _publish_robot_state(client, phase, branch):
    payload = {"phase": phase, "branch": branch}
    client.publish(config.TOPIC_ROBOT_STATE, json.dumps(payload))
    print(f"-> published robot/state: {payload}")


def _publish_event_lost(client):
    payload = {"reason": "fake_robot.py manual trigger"}
    client.publish(config.TOPIC_EVENT_LOST, json.dumps(payload))
    print(f"-> published event/lost: {payload}")
    print("   (watching for event/song_done ...)")


def _publish_laptop_status(client, online):
    payload = {"state": "online" if online else "offline"}
    client.publish(config.TOPIC_LAPTOP_STATUS, json.dumps(payload))
    print(f"-> published laptop/status: {payload}")


def _on_song_done(topic, payload):
    print(f"<- {topic}: {payload}  (sad song finished playing on the robot)")


def _print_menu():
    print("\nWhat do you want to send?")
    for i, (label, _, _) in enumerate(STATE_OPTIONS, start=1):
        print(f"  {i}. robot/state -> {label}")
    n = len(STATE_OPTIONS)
    print(f"  {n + 1}. event/lost (triggers the sad song)")
    print(f"  {n + 2}. laptop/status -> online")
    print(f"  {n + 3}. laptop/status -> offline")
    print("  q. quit")


def main():
    print(f"Connecting to {config.MQTT_BROKER}:{config.MQTT_PORT} ...")
    with MQTTClient(config.MQTT_BROKER, config.MQTT_PORT) as client:
        client.subscribe(config.TOPIC_EVENT_SONG_DONE, _on_song_done)
        print("Connected.")
        while True:
            _print_menu()
            choice = input("> ").strip().lower()
            if choice == "q":
                break
            if not choice.isdigit():
                print("Please enter a number (or q).")
                continue
            choice = int(choice)
            n = len(STATE_OPTIONS)
            if 1 <= choice <= n:
                _, phase, branch = STATE_OPTIONS[choice - 1]
                _publish_robot_state(client, phase, branch)
            elif choice == n + 1:
                _publish_event_lost(client)
            elif choice == n + 2:
                _publish_laptop_status(client, True)
            elif choice == n + 3:
                _publish_laptop_status(client, False)
            else:
                print("Not a valid option.")
            time.sleep(0.2)  # give paho a moment to actually send before the next prompt


if __name__ == "__main__":
    main()
