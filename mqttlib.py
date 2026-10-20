import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883


class MQTTClient:
    """Small wrapper around paho-mqtt v2.

    Everything after `port` is optional and keyword-only; with the defaults this
    behaves exactly like the original version (connect on __enter__, raise if the
    broker is unreachable, QoS 0, no retain, no Last Will).

    Optional extras:
      will_topic / will_payload / will_qos / will_retain
          Last Will: the broker publishes this if we vanish without a clean disconnect.
      background_connect=True
          __enter__ never raises: paho keeps retrying in its network thread until the
          broker answers (and after every later drop, as it always does).
      on_connect=callable()
          Called (on paho's network thread) after every successful (re)connect,
          after subscriptions have been restored.
    Subscriptions are re-sent after every reconnect, because the broker forgets
    them when the connection drops.
    """

    def __init__(self, broker=BROKER, port=PORT, *, will_topic=None, will_payload=None,
                 will_qos=0, will_retain=False, background_connect=False, on_connect=None):
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_message = self._on_message
        self._client.on_connect = self._on_connect
        self._callbacks = {}
        self._qos = {}
        self._broker = broker
        self._port = port
        self._background_connect = background_connect
        self._user_on_connect = on_connect
        if will_topic is not None:
            # Must be set before connecting; the broker stores it with the session.
            self._client.will_set(will_topic, will_payload, qos=will_qos, retain=will_retain)

    def __enter__(self):
        if self._background_connect:
            self._client.reconnect_delay_set(min_delay=1, max_delay=8)
            self._client.connect_async(self._broker, self._port)
        else:
            self._client.connect(self._broker, self._port)
        self._client.loop_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.loop_stop()
        self._client.disconnect()

    @property
    def is_connected(self):
        return self._client.is_connected()

    def subscribe(self, topic, callback, qos=0):
        self._callbacks[topic] = callback
        self._qos[topic] = qos
        self._client.subscribe(topic, qos)

    def publish(self, topic, payload, qos=0, retain=False):
        """Returns paho's MQTTMessageInfo (call .wait_for_publish(timeout) to block)."""
        return self._client.publish(topic, payload, qos=qos, retain=retain)

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if getattr(reason_code, "is_failure", False):
            return
        for topic in self._callbacks:
            client.subscribe(topic, self._qos.get(topic, 0))
        if self._user_on_connect is not None:
            self._user_on_connect()

    def _on_message(self, client, userdata, msg):
        callback = self._callbacks.get(msg.topic)
        if callback:
            callback(msg.topic, msg.payload.decode("utf-8", errors="replace"))
