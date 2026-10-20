"""
mqttlib.py (backward compatibility + new options) and the laptop's MQTT contract.

The last tests talk to the real public broker (test.mosquitto.org) and are skipped
automatically when it can't be reached.
"""

import json
import socket
import threading
import time
import uuid

import pytest

import config
import mqtt_link
import mqttlib
from mqtt_link import MqttLink, NullLink


class FakePaho:
    """Records what mqttlib asks paho to do."""

    instances = []

    def __init__(self, api_version):
        self.calls = []
        self.on_message = self.on_connect = None
        self.connected = False
        FakePaho.instances.append(self)

    def will_set(self, *a, **kw): self.calls.append(("will_set", a, kw))
    def connect(self, host, port): self.calls.append(("connect", host, port)); self.connected = True
    def connect_async(self, host, port): self.calls.append(("connect_async", host, port))
    def reconnect_delay_set(self, **kw): self.calls.append(("reconnect_delay_set", kw))
    def loop_start(self): self.calls.append(("loop_start",))
    def loop_stop(self): self.calls.append(("loop_stop",))
    def disconnect(self): self.calls.append(("disconnect",))
    def subscribe(self, topic, qos=0): self.calls.append(("subscribe", topic, qos))
    def publish(self, topic, payload, qos=0, retain=False):
        self.calls.append(("publish", topic, payload, qos, retain))
        return None
    def is_connected(self): return self.connected


@pytest.fixture
def fake_paho(monkeypatch):
    FakePaho.instances.clear()
    monkeypatch.setattr(mqttlib.mqtt, "Client", FakePaho)
    return FakePaho


def test_old_usage_unchanged(fake_paho):
    got = []
    with mqttlib.MQTTClient() as c:
        c.subscribe("t", lambda topic, payload: got.append((topic, payload)))
        c.publish("t", "2")
    calls = fake_paho.instances[0].calls
    assert calls[0] == ("connect", "test.mosquitto.org", 1883)       # still blocking connect
    assert ("subscribe", "t", 0) in calls
    assert ("publish", "t", "2", 0, False) in calls                  # QoS 0, no retain
    assert not any(c[0] == "will_set" for c in calls)


def test_old_usage_still_raises_when_broker_unreachable(monkeypatch):
    class Failing(FakePaho):
        def connect(self, host, port):
            raise OSError("unreachable")
    monkeypatch.setattr(mqttlib.mqtt, "Client", Failing)
    with pytest.raises(OSError):
        with mqttlib.MQTTClient():
            pass


def test_new_options(fake_paho):
    hooks = []
    c = mqttlib.MQTTClient(will_topic="w", will_payload="bye", will_qos=1, will_retain=True,
                           background_connect=True, on_connect=lambda: hooks.append(1))
    c.subscribe("a", lambda *_: None, qos=1)
    c.__enter__()
    p = fake_paho.instances[0]
    assert ("will_set", ("w", "bye"), {"qos": 1, "retain": True}) in p.calls
    assert any(x[0] == "connect_async" for x in p.calls)
    c.publish("x", "y", qos=1, retain=True)
    assert ("publish", "x", "y", 1, True) in p.calls
    # a (re)connect re-subscribes and calls the hook
    p.calls.clear()
    p.on_connect(p, None, None, type("RC", (), {"is_failure": False})())
    assert ("subscribe", "a", 1) in p.calls and hooks == [1]
    # a refused connection does neither
    p.on_connect(p, None, None, type("RC", (), {"is_failure": True})())
    assert hooks == [1]


def test_link_uses_contract_topics_qos_retain_and_will(fake_paho):
    link = MqttLink(clock=lambda: 0.0)
    link.start()
    p = fake_paho.instances[0]
    assert ("will_set", (config.T_LAPTOP_STATUS, '{"state": "offline"}'), {"qos": 1, "retain": True}) in p.calls
    subs = {c[1]: c[2] for c in p.calls if c[0] == "subscribe"}
    assert subs == {config.T_VISION: 0, config.T_UNOQ_STATUS: 1, config.T_SONG_DONE: 1}
    p.connected = True
    p.on_connect(p, None, None, type("RC", (), {"is_failure": False})())
    assert ("publish", config.T_LAPTOP_STATUS, '{"state": "online"}', 1, True) in p.calls
    link.publish_state("armed", None)
    link.publish_state("armed", None)                  # unchanged: not re-sent
    link.publish_state("running", "teleop")
    states = [c for c in p.calls if c[0] == "publish" and c[1] == config.T_ROBOT_STATE]
    assert [json.loads(c[2]) for c in states] == [{"phase": "armed", "branch": None},
                                                   {"phase": "running", "branch": "teleop"}]
    assert all(c[3] == 1 and c[4] is False for c in states)
    link.announce_lost("pushed")
    assert ("publish", config.T_EVENT_LOST, '{"reason": "pushed"}', 1, False) in p.calls


def test_song_done_only_counts_after_lost():
    link = NullLink()
    link.handle_message(config.T_SONG_DONE, "{}")       # stale song_done from before: ignored
    assert not link.song_done
    link.announce_lost("x")
    link.handle_message(config.T_SONG_DONE, "{}")
    assert link.song_done


def test_bad_messages_are_counted_not_raised():
    link = NullLink()
    link.handle_message(config.T_VISION, "garbage")
    link.handle_message(config.T_UNOQ_STATUS, "[1]")
    assert link.store.bad_messages == 2


# ---------------------------------------------------------------- live broker
def _broker_reachable():
    try:
        socket.create_connection((mqttlib.BROKER, mqttlib.PORT), timeout=3).close()
        return True
    except OSError:
        return False


live = pytest.mark.skipif(not _broker_reachable(), reason="test.mosquitto.org not reachable")


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


@live
def test_live_roundtrip_retained_and_last_will(monkeypatch):
    """Real broker: retained 'online', then the Last Will 'offline' after an unclean drop."""
    prefix = f"ME193/Rogers/RQ-D2-test-{uuid.uuid4().hex[:8]}/"
    status_topic = prefix + "laptop/status"
    monkeypatch.setattr(config, "T_LAPTOP_STATUS", status_topic)

    link = MqttLink(clock=time.monotonic)
    link.start()
    assert _wait(lambda: link.connected), "laptop link did not connect"

    seen = []
    lock = threading.Lock()

    def on_status(topic, payload):
        with lock:
            seen.append(json.loads(payload)["state"])

    with mqttlib.MQTTClient(background_connect=True) as watcher:
        watcher.subscribe(status_topic, on_status, qos=1)
        assert _wait(lambda: "online" in seen), f"no retained online status, saw {seen}"
        # Simulate a crash: kill the socket without a DISCONNECT packet.
        link._client._client.socket().close()
        assert _wait(lambda: seen[-1:] == ["offline"], timeout=20), f"no Last Will, saw {seen}"
    link._client._client.loop_stop()
    # clean up the retained message on the broker
    with mqttlib.MQTTClient() as c:
        c.publish(status_topic, "", qos=1, retain=True).wait_for_publish(timeout=3)
