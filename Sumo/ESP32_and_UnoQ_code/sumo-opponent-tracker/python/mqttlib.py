import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883


class MQTTClient:
    def __init__(self, broker=BROKER, port=PORT):
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_message = self._on_message
        self._callbacks = {}
        self._broker = broker
        self._port = port

    def __enter__(self):
        self._client.connect(self._broker, self._port)
        self._client.loop_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.loop_stop()
        self._client.disconnect()

    def subscribe(self, topic, callback):
        self._callbacks[topic] = callback
        self._client.subscribe(topic)

    def publish(self, topic, payload):
        self._client.publish(topic, payload)

    def _on_message(self, client, userdata, msg):
        callback = self._callbacks.get(msg.topic)
        if callback:
            callback(msg.topic, msg.payload.decode("utf-8", errors="replace"))
