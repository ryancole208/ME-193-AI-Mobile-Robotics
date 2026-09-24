import time

from mqttlib import MQTTClient

TOPIC = "ME193"

def on_message(topic, payload):
    print(f"Got it back: [{topic}] {payload}")

try:
    with MQTTClient() as client:
        while True:
            client.subscribe(TOPIC, on_message)
            time.sleep(1)  # give the subscription time to reach the broker

            #client.publish(TOPIC, '2')
            #print(f"Published '2' to '{TOPIC}' on test.mosquitto.org")

            #sleep(1)  # give the message time to come back before disconnecting
except KeyboardInterrupt:
    print("Exiting...")