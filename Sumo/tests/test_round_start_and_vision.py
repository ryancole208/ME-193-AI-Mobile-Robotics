"""Round start (armed -> running, teleop lock) and vision freshness / chase math."""

import json

import pytest

import config
from round_start import ARMED, COUNTDOWN, RUNNING, RoundStart
from vision import VisionStore, chase_command, parse_opponent

Q = dict(log=lambda *_: None)


# ---------------------------------------------------------------- round start
def test_needs_center_before_push():
    rs = RoundStart(countdown_s=0, **Q)
    assert rs.update(0.0, (90, 0)) == ARMED       # lever already pushed at power-on: ignored
    assert rs.update(0.1, (0, 0)) == ARMED
    assert rs.update(0.2, (0, 90)) == RUNNING


def test_small_push_does_not_start():
    rs = RoundStart(countdown_s=0, **Q)
    rs.update(0.0, (0, 0))
    assert rs.update(0.1, (config.START_THRESHOLD - 1, 0)) == ARMED
    assert rs.update(0.2, None) == ARMED          # no controller data


def test_teleop_locked_until_recenter():
    rs = RoundStart(countdown_s=0, require_recenter=True, **Q)
    rs.update(0.0, (0, 0))
    rs.update(0.1, (90, 90))
    assert rs.phase == RUNNING and not rs.teleop_unlocked
    rs.update(0.2, (50, 50))
    assert not rs.teleop_unlocked
    rs.update(0.3, (0, 0))
    assert rs.teleop_unlocked


def test_recenter_option_off():
    rs = RoundStart(countdown_s=0, require_recenter=False, **Q)
    rs.update(0.0, (0, 0))
    rs.update(0.1, (90, 90))
    assert rs.teleop_unlocked


def test_optional_countdown():
    rs = RoundStart(countdown_s=5, **Q)
    rs.update(0.0, (0, 0))
    assert rs.update(1.0, (90, 0)) == COUNTDOWN
    assert rs.update(5.9, (0, 0)) == COUNTDOWN
    assert rs.update(6.0, (0, 0)) == RUNNING


# ---------------------------------------------------------------- vision
def msg(**kw):
    d = dict(seq=1, visible=True, x=0.0, distance_cm=40.0, confidence=0.9, frame_age_ms=50)
    d.update(kw)
    return json.dumps(d)


def store_with(payload, received_at=10.0):
    st = VisionStore()
    st.set_opponent(parse_opponent(payload, received_at))
    return st


def test_fresh_opponent_accepted():
    assert store_with(msg()).fresh_opponent(10.2) is not None


@pytest.mark.parametrize("payload,now,why", [
    (msg(), 10.0 + config.VISION_FRESH_S + 0.01, "too old by the laptop's clock"),
    (msg(frame_age_ms=config.VISION_MAX_FRAME_AGE_MS + 1), 10.1, "frame too old on the UNO Q"),
    (msg(visible=False, x=None, distance_cm=None), 10.1, "not visible"),
    (msg(confidence=config.VISION_MIN_CONFIDENCE - 0.01), 10.1, "low confidence"),
    (msg(x=None), 10.1, "no x"),
])
def test_opponent_rejected(payload, now, why):
    assert store_with(payload).fresh_opponent(now) is None, why


def test_rejected_when_broker_down_or_unoq_offline_or_camera_missing():
    st = store_with(msg())
    assert st.fresh_opponent(10.1, link_connected=False) is None
    st.set_unoq_status({"state": "offline"})
    assert st.fresh_opponent(10.1) is None
    st.set_unoq_status({"state": "online", "camera": "missing", "fps": 0})
    assert st.fresh_opponent(10.1) is None
    st.set_unoq_status({"state": "online", "camera": "ok", "fps": 15})
    assert st.fresh_opponent(10.1) is not None


def test_out_of_order_dropped_but_restart_accepted():
    st = store_with(msg(seq=10))
    st.set_opponent(parse_opponent(msg(seq=9, x=0.9), 10.0))
    assert st.opponent.seq == 10
    st.set_opponent(parse_opponent(msg(seq=5000, x=0.9), 10.0))
    st.set_opponent(parse_opponent(msg(seq=1, x=0.1), 10.0))      # UNO Q restarted
    assert st.opponent.seq == 1


@pytest.mark.parametrize("bad", ["not json", "[1,2]", json.dumps({"seq": 1}),
                                 json.dumps({"seq": "a", "visible": True, "x": 0, "distance_cm": 1,
                                             "confidence": 1, "frame_age_ms": 1})])
def test_bad_payloads_raise_valueerror(bad):
    with pytest.raises(ValueError):
        parse_opponent(bad, 0.0)


def test_chase_steering_and_speed():
    centered_close = chase_command(parse_opponent(msg(x=0.0, distance_cm=config.CHASE_NEAR_CM), 0))
    assert centered_close == (config.CHASE_MAX_SPEED, config.CHASE_MAX_SPEED)
    far = chase_command(parse_opponent(msg(x=0.0, distance_cm=None), 0))
    assert far == (config.CHASE_MIN_SPEED, config.CHASE_MIN_SPEED)
    right = chase_command(parse_opponent(msg(x=0.8, distance_cm=30), 0))
    left = chase_command(parse_opponent(msg(x=-0.8, distance_cm=30), 0))
    assert right[0] > right[1] and left[0] < left[1]
    assert max(right) <= 100 and min(right) >= -100
