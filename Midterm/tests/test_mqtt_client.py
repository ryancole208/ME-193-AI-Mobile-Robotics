"""MQTT wrapper around ../mqttlib.py (fake client unless --hardware)."""

import time
from types import SimpleNamespace

import pytest

import mqtt_client
from mqtt_client import ScorePublisher, score_payload, score_topic


class FakePaho:
    on_connect = None
    on_disconnect = None


class FakeMQTTClient:
    """Same surface as mqttlib.MQTTClient: __enter__/__exit__/publish, plus _client."""
    fail_times = 0
    instances = []

    def __init__(self):
        self._client = FakePaho()
        self.sent = []
        self.exited = False
        FakeMQTTClient.instances.append(self)

    def __enter__(self):
        if FakeMQTTClient.fail_times > 0:
            FakeMQTTClient.fail_times -= 1
            raise OSError("broker unreachable")
        self._client.on_connect(self._client, None, None, SimpleNamespace(is_failure=False), None)
        return self

    def __exit__(self, *exc):
        self.exited = True

    def publish(self, topic, payload):
        self.sent.append((topic, payload))


@pytest.fixture(autouse=True)
def reset_fake():
    FakeMQTTClient.fail_times = 0
    FakeMQTTClient.instances = []


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_uses_mqttlib_from_parent_folder():
    import mqttlib
    assert mqtt_client.MQTTClient is mqttlib.MQTTClient
    assert mqttlib.__file__.endswith("ME 193 AI Mobile Robotics\\mqttlib.py") or \
        mqttlib.__file__.replace("\\", "/").endswith("ME 193 AI Mobile Robotics/mqttlib.py")


def test_payload_is_float_string():
    assert score_payload(7) == "7.0"
    assert score_payload(0) == "0.0"
    assert float(score_payload(12)) == 12.0


def test_topic():
    assert score_topic("ME193/RQ-D2", "Name") == "ME193/RQ-D2/Name"


def test_publishes_every_change_after_connect():
    pub = ScorePublisher(client_factory=FakeMQTTClient)
    pub.start()
    assert wait_for(lambda: pub.connected)
    assert pub.status == "connected"
    for s in (1, 2, 3, 0):
        pub.publish_score(s)
    client = FakeMQTTClient.instances[-1]
    assert client.sent == [("ME193/RQ-D2/Name", p) for p in ("1.0", "2.0", "3.0", "0.0")]
    pub.stop()
    assert client.exited and not pub.connected


def test_retries_until_broker_reachable_and_sends_latest_score():
    FakeMQTTClient.fail_times = 2
    pub = ScorePublisher(client_factory=FakeMQTTClient, retry_s=0.01)
    pub.publish_score(5)            # before connection: remembered, not sent
    pub.start()
    assert wait_for(lambda: pub.connected)
    assert len(FakeMQTTClient.instances) == 3
    assert FakeMQTTClient.instances[-1].sent == [("ME193/RQ-D2/Name", "5.0")]
    pub.stop()


def test_status_tracks_disconnect_and_reconnect():
    pub = ScorePublisher(client_factory=FakeMQTTClient)
    pub.start()
    assert wait_for(lambda: pub.connected)
    paho = FakeMQTTClient.instances[-1]._client
    pub.publish_score(4)
    paho.on_disconnect(paho, None, None, 7, None)
    assert not pub.connected and "reconnecting" in pub.status
    pub.publish_score(6)            # while offline
    paho.on_connect(paho, None, None, SimpleNamespace(is_failure=False), None)
    assert pub.connected
    assert FakeMQTTClient.instances[-1].sent[-1] == ("ME193/RQ-D2/Name", "6.0")
    pub.stop()


@pytest.mark.hardware
def test_real_broker_round_trip():
    """Publishes via the wrapper and receives it back through mqttlib on the real broker."""
    from mqttlib import MQTTClient
    got = []
    topic = score_topic(name="Name_pytest")
    with MQTTClient() as listener:
        listener.subscribe(topic, lambda t, p: got.append(p))
        time.sleep(1.0)
        pub = ScorePublisher(topic=topic)
        pub.start()
        assert wait_for(lambda: pub.connected, timeout=10), pub.status
        pub.publish_score(7)
        assert wait_for(lambda: "7.0" in got, timeout=10)
        pub.stop()
