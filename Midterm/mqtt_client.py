"""
Thin wrapper around ../mqttlib.py (unchanged) adding what the game needs:

  * non-blocking start: mqttlib's connect raises if the broker is unreachable, so
    connection attempts run in a background thread and are retried
  * connection status for the UI: paho on_connect / on_disconnect callbacks are
    attached to the paho client that mqttlib.MQTTClient creates (its `_client`)
  * publish_score(): FLOAT payload ("7.0") to f"{MQTT_TOPIC_BASE}/{PLAYER_NAME}";
    the latest score is re-sent after a reconnect so the broker never stays stale

Once connected, paho's own network loop (started by mqttlib) handles reconnects.
"""

import sys
import threading
from collections import deque
from pathlib import Path

import config

_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from mqttlib import MQTTClient  # noqa: E402  (../mqttlib.py)


def score_topic(base=config.MQTT_TOPIC_BASE, name=config.PLAYER_NAME):
    return f"{base}/{name}"


def score_payload(score):
    return str(float(score))


class ScorePublisher:
    def __init__(self, topic=None, client_factory=MQTTClient, retry_s=config.MQTT_RETRY_INTERVAL_S):
        self.topic = topic or score_topic()
        self._factory = client_factory
        self._retry_s = retry_s
        self._client = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._last_score = None
        self.connected = False
        self.status = "idle"
        self.published = deque(maxlen=50)  # recent (topic, payload), for tests/debug overlay

    # -- lifecycle --------------------------------------------------------------
    def start(self):
        self.status = "connecting"
        self._thread = threading.Thread(target=self._connect_loop, name="MQTT", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._client is not None:
            try:
                self._client.__exit__(None, None, None)
            except Exception as e:
                print(f"[mqtt] disconnect error: {e}")
        self.connected = False
        self.status = "stopped"

    def _connect_loop(self):
        while not self._stop.is_set():
            client = self._factory()
            paho = client._client
            paho.on_connect = self._on_connect
            paho.on_disconnect = self._on_disconnect
            self._client = client  # set first: on_connect may fire before __enter__ returns
            try:
                client.__enter__()  # mqttlib: connect + loop_start
                return
            except Exception as e:
                self._client = None
                self.status = f"offline (retrying): {e}"
                self._stop.wait(self._retry_s)

    # -- paho callbacks (network thread) ----------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        failed = getattr(reason_code, "is_failure", False)
        if failed:
            self.connected = False
            self.status = f"refused: {reason_code}"
            return
        print(f"[mqtt] connected; score topic = {self.topic}")
        # Same lock as publish_score: no score can slip in between the resend and the flag flip.
        with self._lock:
            if self._last_score is not None:
                self._send(self._last_score)
            self.connected = True
            self.status = "connected"

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = False
        if not self._stop.is_set():
            self.status = "disconnected (reconnecting)"

    # -- publishing -------------------------------------------------------------
    def publish_score(self, score):
        with self._lock:
            self._last_score = float(score)
            if self.connected:
                self._send(float(score))

    def _send(self, score):
        payload = score_payload(score)
        try:
            self._client.publish(self.topic, payload)
            self.published.append((self.topic, payload))
        except Exception as e:
            print(f"[mqtt] publish failed: {e}")
