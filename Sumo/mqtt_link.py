"""
The laptop's half of the MQTT contract with the UNO Q (see README for the table).

    in : vision/opponent (QoS 0), unoq/status (QoS 1, retained), event/song_done (QoS 1)
    out: laptop/status (QoS 1, retained, Last Will {"state":"offline"}),
         robot/state (QoS 1, only on change), event/lost (QoS 1)

paho calls our subscribe callbacks on ITS network thread. They only parse and
store data (VisionStore uses a lock; song_done is a threading.Event), and the
main loop reads it on its next tick.

The broker is public and can be slow or away: MQTTClient(background_connect=True)
never raises on startup and keeps reconnecting, and on every (re)connect we
re-announce laptop/status and the latest robot/state. Nothing here is needed for
the color-sensor reflex, which keeps working with MQTT down.
"""

import json
import sys
import threading
import time
from pathlib import Path

import config
from vision import VisionStore, parse_opponent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # ../mqttlib.py
from mqttlib import MQTTClient  # noqa: E402

ONLINE = json.dumps({"state": "online"})
OFFLINE = json.dumps({"state": "offline"})


class LinkBase:
    """Shared logic; subclasses implement _send() and connected."""

    expects_song = True

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.store = VisionStore()
        self._song_done = threading.Event()
        self._lost_announced = False
        self._last_state = None
        self.sent = []          # (topic, payload) log, handy for tests

    @property
    def connected(self):
        raise NotImplementedError

    def _send(self, topic, payload, qos=0, retain=False):
        raise NotImplementedError

    def _record(self, topic, payload):
        self.sent.append((topic, payload))
        if len(self.sent) > 1000:
            del self.sent[:500]

    # -- outgoing ---------------------------------------------------------------
    def publish_state(self, phase, branch):
        """robot/state, only when (phase, branch) changes."""
        state = (phase, branch)
        if state == self._last_state:
            return
        self._last_state = state
        self._send(config.T_ROBOT_STATE, json.dumps({"phase": phase, "branch": branch}), qos=1)

    def announce_lost(self, reason):
        self._song_done.clear()
        self._lost_announced = True
        self.publish_state("lost", None)
        self._send(config.T_EVENT_LOST, json.dumps({"reason": reason}), qos=1)

    @property
    def song_done(self):
        return self._song_done.is_set()

    # -- incoming (called on the network thread) ------------------------------------
    def handle_message(self, topic, payload):
        try:
            if topic == config.T_VISION:
                self.store.set_opponent(parse_opponent(payload, self.clock()))
            elif topic == config.T_UNOQ_STATUS:
                st = json.loads(payload)
                if not isinstance(st, dict):
                    raise ValueError("not an object")
                old = self.store.unoq_status
                self.store.set_unoq_status(st)
                if old is None or old.get("state") != st.get("state") or old.get("camera") != st.get("camera"):
                    print(f"[mqtt] UNO Q status: {st}")
            elif topic == config.T_SONG_DONE:
                if self._lost_announced:
                    print("[mqtt] song_done received")
                    self._song_done.set()
        except (ValueError, TypeError) as e:
            self.store.bad_messages += 1
            if self.store.bad_messages <= 5 or self.store.bad_messages % 100 == 0:
                print(f"[mqtt] ignored bad message on {topic}: {e} ({payload[:80]!r})")

    def start(self):
        pass

    def close(self):
        pass


class MqttLink(LinkBase):
    def __init__(self, clock=time.monotonic, client_factory=MQTTClient):
        super().__init__(clock)
        self._factory = client_factory
        self._client = None
        self._was_connected = False

    def start(self):
        self._client = self._factory(will_topic=config.T_LAPTOP_STATUS, will_payload=OFFLINE,
                                     will_qos=1, will_retain=True, background_connect=True,
                                     on_connect=self._on_connect)
        self._client.subscribe(config.T_VISION, self.handle_message, qos=0)
        self._client.subscribe(config.T_UNOQ_STATUS, self.handle_message, qos=1)
        self._client.subscribe(config.T_SONG_DONE, self.handle_message, qos=1)
        self._client.__enter__()
        print("[mqtt] connecting to the broker in the background ...")

    def _on_connect(self):
        print(f"[mqtt] connected{' again' if self._was_connected else ''}; laptop/status = online")
        self._was_connected = True
        self._log_send(config.T_LAPTOP_STATUS, ONLINE)
        self._client.publish(config.T_LAPTOP_STATUS, ONLINE, qos=1, retain=True)
        if self._last_state is not None:   # the UNO Q may have missed changes while we were away
            phase, branch = self._last_state
            payload = json.dumps({"phase": phase, "branch": branch})
            self._log_send(config.T_ROBOT_STATE, payload)
            self._client.publish(config.T_ROBOT_STATE, payload, qos=1)

    @staticmethod
    def _log_send(topic, payload):
        print(f"[mqtt] -> {topic.removeprefix(config.TOPIC_PREFIX)} {payload}")

    @property
    def connected(self):
        return self._client is not None and self._client.is_connected

    def _send(self, topic, payload, qos=0, retain=False):
        self._record(topic, payload)
        self._log_send(topic, payload)
        if self._client is None:
            return
        # While disconnected paho queues QoS 1 messages and delivers them after reconnecting.
        try:
            self._client.publish(topic, payload, qos=qos, retain=retain)
        except Exception as e:
            print(f"[mqtt] publish to {topic} failed: {e}")

    def close(self):
        if self._client is None:
            return
        try:
            if self.connected:
                # A clean disconnect does NOT fire the Last Will, so say "offline" ourselves.
                self._log_send(config.T_LAPTOP_STATUS, OFFLINE)
                info = self._client.publish(config.T_LAPTOP_STATUS, OFFLINE, qos=1, retain=True)
                info.wait_for_publish(timeout=1.5)
        except Exception as e:
            print(f"[mqtt] could not publish offline status: {e}")
        try:
            self._client.__exit__(None, None, None)
        except Exception as e:
            print(f"[mqtt] disconnect error: {e}")
        print("[mqtt] disconnected")


class NullLink(LinkBase):
    """--no-unoq: no MQTT at all. Vision never arrives, so the chase branch never runs."""

    expects_song = False

    @property
    def connected(self):
        return False

    def _send(self, topic, payload, qos=0, retain=False):
        self._record(topic, payload)
        print(f"[no-unoq] would publish {topic.removeprefix(config.TOPIC_PREFIX)} {payload}")
