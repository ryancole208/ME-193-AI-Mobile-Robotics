"""bridge.py packets, with fake motor / pose / UDP (no hardware, no network)."""

import json
import socket

import bridge
import bridge_config as bc
import config
from bridge import Bridge, face_for, parse_command, x_for
from rhythm_motor import RhythmSwing
from udp_link import UdpReceiver, UdpSender


class FakeLink:
    def __init__(self):
        self.msgs = []

    def send(self, msg):
        self.msgs.append(msg)
        return json.dumps(msg).encode("utf-8")

    def close(self):
        pass


class FakeRoll:
    def __init__(self, angle=None):
        self.angle = angle


class FakeMotor:
    def __init__(self):
        self.connected = False
        self.detector = None
        self.roll = None
        self.roll_deg = None
        self.quiet = False
        self.status = "scanning"
        self.zeroed = 0
        self.pulses = []

    def calibrate_neutral(self):
        self.zeroed += 1

    def pulse(self, ms):
        self.pulses.append(ms)


def calibrated_motor():
    m = FakeMotor()
    m.connected, m.detector, m.roll = True, object(), FakeRoll(angle=2.0)
    return m


class FakeSnap:
    def __init__(self, paddle_x):
        self.paddle_x = paddle_x


class FakePose:
    def __init__(self):
        self.tracking = False
        self.paddle_x = None

    def snapshot(self):
        return FakeSnap(self.paddle_x)


def make(motor=None, pose=None, published=None):
    return Bridge(motor, pose, godot=FakeLink(), swing_link=FakeLink(),
                  publish_score=None if published is None else published.append)


def cmd(obj):
    msg, why = parse_command(json.dumps(obj).encode("utf-8"))
    assert why is None, why
    return msg


def test_face_follows_the_raw_roll():
    assert face_for(None) == "black"
    assert face_for(0.0) == "black"
    assert face_for(bc.FACE_FLIP_DEG) == "black"
    assert face_for(-bc.FACE_FLIP_DEG - 1) == "yellow"
    assert face_for(170.0) == "yellow"


def test_x_maps_the_table_to_0_1_and_clamps():
    w = config.TABLE_WIDTH_M
    assert x_for(0.0, w) == 0.5
    assert x_for(-w / 2, w) == 0.0 and x_for(w / 2, w) == 1.0
    assert x_for(5.0, w) == 1.0 and x_for(-5.0, w) == 0.0


def test_disabled_hardware_sends_neutral_state_and_false_status():
    b = make()
    assert b.state(0.0) == {"type": "state", "x": 0.5, "roll": 0.0, "face": "black"}
    assert b.status() == {"type": "status", "motor": False, "pose": False}


def test_pose_x_is_held_while_the_wrist_is_lost():
    pose = FakePose()
    b = make(pose=pose)
    pose.tracking, pose.paddle_x = True, config.TABLE_WIDTH_M / 4
    assert b.state(0.0)["x"] == 0.75 and b.status()["pose"] is True
    pose.tracking, pose.paddle_x = False, None
    assert b.state(0.1)["x"] == 0.75 and b.status()["pose"] is False


def test_motor_status_needs_connection_and_calibration():
    m = FakeMotor()
    b = make(motor=m)
    m.connected = True
    assert b.status()["motor"] is False          # connected but still calibrating
    m.detector = object()
    assert b.status()["motor"] is True


def test_roll_zeroed_once_per_roll_filter_after_its_first_sample():
    m = FakeMotor()
    b = make(motor=m)
    assert not b.zero_after_calibration()        # not calibrated yet
    m.roll = FakeRoll(angle=None)
    assert not b.zero_after_calibration()        # calibrated, no sample yet: would be a no-op
    m.roll.angle = 3.0
    assert b.zero_after_calibration() and m.zeroed == 1
    assert not b.zero_after_calibration() and m.zeroed == 1
    m.roll = FakeRoll(angle=1.0)                 # reconnect -> new filter -> zero again
    assert b.zero_after_calibration() and m.zeroed == 2


def test_state_face_from_raw_roll_but_roll_is_calmed():
    m = FakeMotor()
    b = make(motor=m)
    m.roll_deg = 150.0
    st = b.state(1.0)
    assert st["face"] == "yellow"
    assert abs(st["roll"]) <= config.ROLL_DISPLAY_MAX_DEG


def test_swing_is_sent_at_once_with_age_since_the_peak(monkeypatch):
    b = make(motor=FakeMotor())
    monkeypatch.setattr(bridge, "clock", lambda: 10.040)
    b.on_swing(RhythmSwing(t=10.0, onset=9.97, strength=3.21459))
    assert b.swing_link.msgs == [{"type": "swing", "strength": 3.215, "age_ms": 40.0}]
    assert b.last_swing_t == 10.0


def test_udp_sender_delivers_one_json_packet():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    tx = UdpSender("127.0.0.1", rx.getsockname()[1])
    try:
        tx.send({"type": "status", "motor": False, "pose": False})
        assert json.loads(rx.recv(2048).decode("utf-8")) == {"type": "status", "motor": False, "pose": False}
    finally:
        tx.close()
        rx.close()


# -- commands from Godot (4243) ------------------------------------------------

def test_zero_rezeroes_like_the_c_key():
    m = calibrated_motor()
    b = make(motor=m)
    b.roll_display.shown, b.roll_display.bias = 25.0, 4.0
    assert b.handle(cmd({"type": "zero"})) == "roll zeroed"
    assert m.zeroed == 1
    assert b.roll_display.shown == 0.0 and b.roll_display.bias == 0.0
    assert not b.zero_after_calibration()        # already zeroed this filter: no second auto-zero


def test_zero_is_ignored_until_the_motor_is_calibrated():
    m = FakeMotor()
    b = make(motor=m)
    assert "ignored" in b.handle(cmd({"type": "zero"}))
    m.roll = FakeRoll(angle=None)                # calibrated, but no sample yet
    assert "ignored" in b.handle(cmd({"type": "zero"}))
    assert "ignored" in make().handle(cmd({"type": "zero"}))     # --no-motor
    assert m.zeroed == 0


def test_haptic_pulses_the_fixed_length_when_the_motor_is_ready():
    m = calibrated_motor()
    b = make(motor=m)
    assert b.handle(cmd({"type": "haptic"})) == f"pulse {bc.HAPTIC_PULSE_MS} ms"
    assert m.pulses == [bc.HAPTIC_PULSE_MS]


def test_haptic_is_ignored_when_disabled_or_not_ready(monkeypatch):
    m = calibrated_motor()
    m.detector = None                            # connected, still calibrating
    assert "ignored" in make(motor=m).handle(cmd({"type": "haptic"}))
    assert "ignored" in make().handle(cmd({"type": "haptic"}))   # --no-motor
    m = calibrated_motor()
    monkeypatch.setattr(bc, "HAPTIC_ENABLED", False)
    assert "ignored" in make(motor=m).handle(cmd({"type": "haptic"}))
    assert m.pulses == []


def test_score_publishes_the_record_only_when_it_goes_up():
    published = []
    b = make(published=published)
    for streak, best in [(0, 0), (3, 3), (5, 5), (0, 5), (2, 5), (1, 1), (6, 6), (6, 6)]:
        b.handle(cmd({"type": "score", "streak": streak, "max_streak": best}))
    assert published == [3.0, 5.0, 6.0]          # 0 is not above the start; ties and drops don't publish
    assert all(isinstance(v, float) for v in published)
    assert b.record == 6


def test_score_accepts_whole_floats_from_godot():
    assert cmd({"type": "score", "streak": 2.0, "max_streak": 7.0})["max_streak"] == 7


def test_malformed_packets_are_ignored_with_a_reason():
    bad = [b"\xff\xfe", b"not json", b"[1, 2]", b'"zero"', b"{}", b'{"type": "explode"}',
           b'{"type": "score"}', b'{"type": "score", "streak": 1}',
           b'{"type": "score", "streak": 1, "max_streak": -2}',
           b'{"type": "score", "streak": 1, "max_streak": 2.5}',
           b'{"type": "score", "streak": 1, "max_streak": "9"}',
           b'{"type": "score", "streak": true, "max_streak": 3}',
           b"[" * 100000]
    for data in bad:
        msg, why = parse_command(data)
        assert msg is None and why, data[:40]


def test_receiver_drains_every_waiting_packet_without_blocking():
    rx = UdpReceiver("127.0.0.1", 0)
    port = rx.sock.getsockname()[1]
    tx = UdpSender("127.0.0.1", port)
    try:
        assert rx.drain() == []                  # nothing waiting: returns at once
        assert rx.wait(0.01) is False
        for kind in ("zero", "haptic"):
            tx.send({"type": kind})
        assert rx.wait(2.0) is True
        got = []
        for _ in range(20):                      # both datagrams may not be queued at the same instant
            got += rx.drain()
            if len(got) == 2:
                break
            rx.wait(0.1)
        assert [parse_command(d)[0]["type"] for d in got] == ["zero", "haptic"]
    finally:
        tx.close()
        rx.close()
