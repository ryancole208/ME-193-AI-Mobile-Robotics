"""Check that MQTT works from this computer through test.mosquitto.org.

Run the receiver in one terminal, then the sender in another:

    python Test.py receive              # wait for messages on TOPIC and print them
    python Test.py send                 # send one test message to TOPIC
    python Test.py send "hello there"   # send your own message

Both sides wait for the broker to actually accept the connection, and say so if it doesn't.
Press Ctrl+C to stop the receiver.
"""

import sys
import threading
import time

import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "ME193/RQ-D2/test"
CONNECT_TIMEOUT = 15   # seconds to wait for the broker to accept us


def connect():
    """Connect to the broker and return the client, or exit if the broker never accepts."""
    connected = threading.Event()

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"Broker refused the connection: {reason_code}")
        else:
            connected.set()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    print(f"Connecting to {BROKER}:{PORT}...")
    try:
        client.connect(BROKER, PORT)
    except OSError as e:
        sys.exit(f"Could not reach {BROKER}:{PORT}: {e}")
    client.loop_start()

    if not connected.wait(CONNECT_TIMEOUT):
        client.loop_stop()
        sys.exit(f"FAILED: {BROKER} did not accept the connection within {CONNECT_TIMEOUT}s.")
    print(f"Connected to {BROKER}.")
    return client


def send(text):
    client = connect()
    info = client.publish(TOPIC, text, qos=1)   # qos=1: the broker confirms it got the message
    info.wait_for_publish(timeout=10)
    if info.is_published():
        print(f"SENT to {TOPIC}: {text}")
    else:
        print(f"FAILED: the broker never confirmed the message on {TOPIC}.")
    client.loop_stop()
    client.disconnect()


def receive():
    client = connect()
    client.on_message = lambda c, u, msg: print(
        f"[{time.strftime('%H:%M:%S')}] {msg.topic}: {msg.payload.decode('utf-8', errors='replace')}")
    client.subscribe(TOPIC, qos=1)
    print(f"Listening on {TOPIC}. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("send", "receive"):
        sys.exit(__doc__)
    if sys.argv[1] == "send":
        send(" ".join(sys.argv[2:]) or f"test message from this computer at {time.strftime('%H:%M:%S')}")
    else:
        receive()
