"""RQ-D2 Vision - Linux-side app.

This is the "glue" between the sketch (which does the actual ultrasonic
measuring) and MQTT (which talks to the laptop's behavior tree). It:

  1. Receives filtered distance/detection updates pushed from the sketch
     over the Bridge, and publishes one to MQTT for every single one
     (ME193/Rogers/RQ-D2/vision/opponent).
  2. Subscribes to the laptop's robot/state, event/lost and laptop/status
     topics, and tells the sketch which LED icon to show.
  3. Publishes this app's own status (ME193/Rogers/RQ-D2/unoq/status)
     once a second.
  4. Logs distance/detection/MQTT status to the console once a second.

Everything that talks to MQTT is wrapped in try/except: a flaky public
broker must never crash this app, because the sensor, LED matrix and
buzzer all need to keep working even with no network at all.
"""

import json
import logging
import threading
import time

from arduino.app_utils import App, Bridge, Logger

import config
from mqttlib import MQTTClient

logger = Logger("RQD2Vision", level=logging.INFO)

# ===================================================================
# ===== Shared state =====
# ===================================================================
# Written from three different places - the sketch's Bridge pushes, MQTT's
# background thread (paho runs callbacks on its own thread, not ours), and
# this file's own main loop - so every read/write of these goes through
# _state_lock to avoid torn reads (e.g. reading a brand new distance_cm
# together with a stale confidence from a half-finished update).
_state_lock = threading.Lock()

_vision = {
    "visible": False,
    "distance_cm": None,     # None = sensor currently has no valid reading
    "confidence": 0.0,
    "sensor_ok": True,
    "received_at": None,     # time.monotonic() this app received the update (NOT the sketch's clock)
}
_seq = 0                     # increments once per vision/opponent message published
_update_count = 0            # counts vision updates between status lines, to compute an actual measurement rate (fps)

_robot_state = {"phase": None, "branch": None}
_laptop_online = False
_song_in_progress = False    # True from event/lost until the sketch reports the song finished

_mqtt_client = None           # set once the first MQTT connection succeeds
_mqtt_connected = False

_last_icon_sent = None        # (icon_name, size_step) - only redraw the matrix when this changes, as required


# ===================================================================
# ===== Bridge: receiving from the sketch =====
# ===================================================================

def _on_vision_update(visible: bool, distance_cm: float, confidence: float, sensor_ok: bool):
    """Called by the sketch (Bridge.notify) on every filtered ultrasonic
    reading. distance_cm arrives as -1.0 when the sketch currently has no
    valid reading - we turn that into Python's None here so the rest of
    the app only has to deal with one "unknown" value instead of two.

    IMPORTANT: this callback runs on the Bridge library's own dispatch
    thread (seen in logs as "Bridge.read_loop"). Testing showed that
    calling Bridge.call(...) from THIS thread - e.g. to redraw an LED icon
    - reliably self-deadlocks: that thread is the same one needed to
    receive the call's own response, so it can only ever time out (10s),
    never succeed. That's why this function only updates shared state and
    publishes to MQTT (no Bridge calls) - _update_led()'s Bridge calls are
    made only from loop()'s own thread instead (see below).
    """
    global _seq, _update_count
    with _state_lock:
        _vision["visible"] = bool(visible)
        _vision["distance_cm"] = None if distance_cm < 0 else float(distance_cm)
        _vision["confidence"] = float(confidence)
        _vision["sensor_ok"] = bool(sensor_ok)
        _vision["received_at"] = time.monotonic()
        _seq += 1
        _update_count += 1

    _publish_vision()


def _on_song_finished():
    """Called by the sketch (Bridge.notify) once the sad song's 5 notes
    have all finished playing. Same Bridge-thread caveat as
    _on_vision_update above - no Bridge.call here, just state + MQTT."""
    global _song_in_progress
    _song_in_progress = False
    _safe_publish(config.TOPIC_EVENT_SONG_DONE, "{}", qos=1)


Bridge.provide("vision_update", _on_vision_update)
Bridge.provide("song_finished", _on_song_finished)


_BRIDGE_CALL_FAILED = object()  # sentinel: lets callers tell "failed" apart from a real None/void return


def _safe_bridge_call(name, *args):
    """Bridge.call wrapped so a slow/misbehaving sketch can never crash
    this app - same defensive spirit as _safe_publish below."""
    try:
        return Bridge.call(name, *args)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: must never raise
        logger.warning(f"Bridge.call({name!r}) failed: {exc!r}")
        return _BRIDGE_CALL_FAILED


# ===================================================================
# ===== MQTT: connecting, with retry, and never crashing =====
# ===================================================================

def _safe_publish(topic, payload, qos=0):
    """Publishes if we're currently connected; silently drops the message
    otherwise. vision/opponent messages are fire-and-forget by design
    (QoS 0, a new one is on its way in ~60ms anyway), so dropping one while
    disconnected is harmless - the alternative (crashing, or blocking the
    whole app on a dead broker) would be far worse.

    NOTE: mqttlib.py's publish() doesn't accept a qos argument (see its
    docstring/module - we were asked to keep it unmodified), so `qos` here
    is currently just documentation of INTENT for a future mqttlib.py that
    supports it; every publish actually goes out at QoS 0 today.
    """
    if _mqtt_client is None or not _mqtt_connected:
        return
    try:
        _mqtt_client.publish(topic, payload)
    except Exception as exc:  # noqa: BLE001 - a flaky public broker must never crash this app
        logger.warning(f"MQTT publish to {topic} failed: {exc!r}")


def _on_robot_state(topic, payload):
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(f"robot/state: couldn't parse payload {payload!r}")
        return
    with _state_lock:
        _robot_state["phase"] = data.get("phase")
        _robot_state["branch"] = data.get("branch")
    # LED icon updates happen from loop() (see its comment) - this just
    # records the new state; loop() picks it up within ~0.1s.


def _on_laptop_status(topic, payload):
    global _laptop_online
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(f"laptop/status: couldn't parse payload {payload!r}")
        return
    _laptop_online = data.get("state") == "online"
    # LED icon updates happen from loop() (see its comment) - this just
    # records the new state; loop() picks it up within ~0.1s.


def _on_event_lost(topic, payload):
    """The laptop decided OUR robot got pushed out of the ring. Show the
    sad face right away and start the song; event/song_done is published
    once the sketch confirms the song actually finished (see
    _on_song_finished above).

    This runs on paho's own MQTT thread, not the Bridge's dispatch thread,
    so (unlike _on_vision_update above) it's safe to call Bridge.call
    directly here - that's what lets the sad face appear immediately
    instead of waiting for loop()'s next ~0.1s tick."""
    global _song_in_progress
    _song_in_progress = True
    _set_icon("lost")
    _safe_bridge_call("play_sad_song")


def _wire_subscriptions(client):
    client.subscribe(config.TOPIC_ROBOT_STATE, _on_robot_state)
    client.subscribe(config.TOPIC_LAPTOP_STATUS, _on_laptop_status)
    client.subscribe(config.TOPIC_EVENT_LOST, _on_event_lost)


def _mqtt_worker():
    """Connects to the broker, retrying with exponential backoff if the
    broker is unreachable, and keeps retrying forever in the background.
    Runs in its own thread so a dead/slow broker at startup never blocks
    the sensor/LED/buzzer side of this app.

    Once the FIRST connection succeeds, paho-mqtt's own background thread
    (started by MQTTClient.__enter__ via loop_start()) automatically
    reconnects on its own if the connection later drops - so this loop
    only has to handle getting connected in the first place.
    """
    global _mqtt_client, _mqtt_connected
    delay = config.MQTT_RECONNECT_INITIAL_DELAY_S
    while True:
        try:
            client = MQTTClient(config.MQTT_BROKER, config.MQTT_PORT)
            client.__enter__()  # same connect()+loop_start() a "with" block would do
        except Exception as exc:  # noqa: BLE001 - must keep retrying no matter what goes wrong
            logger.warning(f"MQTT connect to {config.MQTT_BROKER}:{config.MQTT_PORT} failed: {exc!r}; retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, config.MQTT_RECONNECT_MAX_DELAY_S)
            continue

        # mqttlib.py doesn't expose connect/disconnect notifications (and
        # we were asked not to modify it), so we attach paho's own
        # on_connect/on_disconnect handlers directly to the underlying
        # client it keeps at `_client` - this only READS state, nothing
        # about mqttlib.py's behavior changes.
        def _on_disconnect(c, userdata, flags, reason_code, properties):
            global _mqtt_connected
            _mqtt_connected = False
            logger.warning(f"MQTT disconnected (reason={reason_code}); paho will keep retrying in the background")
            # LED icon updates happen from loop() (see its comment) - this
            # just records the new state; loop() picks it up within ~0.1s.

        client._client.on_disconnect = _on_disconnect

        _mqtt_client = client
        _mqtt_connected = True
        _wire_subscriptions(client)
        logger.info(f"Connected to MQTT broker {config.MQTT_BROKER}:{config.MQTT_PORT}")
        return  # paho's background thread now owns the connection; nothing left to do here


# ===================================================================
# ===== Publishing vision/opponent =====
# ===================================================================

def _publish_vision():
    with _state_lock:
        visible = _vision["visible"]
        distance_cm = _vision["distance_cm"]
        confidence = _vision["confidence"]
        received_at = _vision["received_at"]
        seq = _seq

    age_ms = 0 if received_at is None else int((time.monotonic() - received_at) * 1000)
    payload = {
        "seq": seq,
        "visible": visible,
        "x": 0.0 if visible else None,  # a single ultrasonic sensor can't tell left from right
        "distance_cm": distance_cm,
        "confidence": confidence,
        "frame_age_ms": age_ms,
        "source": "ultrasonic",
    }
    _safe_publish(config.TOPIC_VISION, json.dumps(payload), qos=0)


# ===================================================================
# ===== LED icon selection =====
# ===================================================================
# Picks ONE icon from everything we currently know, in priority order
# (most urgent problem first), then tells the sketch to draw it - but only
# if it's actually different from what's already on screen.

def _size_step_for_distance(distance_cm):
    if distance_cm is None:
        return config.LED_CHASE_SIZE_STEPS // 2  # unknown distance: a reasonable middle size
    steps = config.LED_CHASE_SIZE_STEPS
    # Closer (smaller distance_cm) -> bigger box, so invert the fraction.
    frac = 1.0 - max(0.0, min(1.0, distance_cm / config.RING_DIAMETER_CM))
    return max(0, min(steps - 1, round(frac * (steps - 1))))


def _set_icon(name, size_step=0):
    global _last_icon_sent
    state = (name, size_step)
    if state == _last_icon_sent:
        return  # only update on change
    _last_icon_sent = state

    if name == "chasing":
        _safe_bridge_call("show_icon_chasing", size_step)
    else:
        _safe_bridge_call(f"show_icon_{name}")


def _update_led():
    with _state_lock:
        sensor_ok = _vision["sensor_ok"]
        distance_cm = _vision["distance_cm"]
        received_at = _vision["received_at"]
        phase = _robot_state["phase"]
        branch = _robot_state["branch"]

    # The sketch itself hasn't said anything is wrong, but if we haven't
    # heard from it AT ALL in a while, something's wrong at a level the
    # sketch can't report on its own (e.g. the Bridge link itself is down).
    sensor_silent = (
        received_at is None
        or (time.monotonic() - received_at) > config.SENSOR_SILENCE_TIMEOUT_S
    )

    if not sensor_ok or sensor_silent:
        _set_icon("sensor_error")
    elif not _mqtt_connected:
        _set_icon("no_network")
    elif not _laptop_online:
        _set_icon("waiting_laptop")
    elif phase == "lost":
        _set_icon("lost")
    elif phase == "armed":
        _set_icon("armed")
    elif phase == "running":
        if branch == "reverse":
            _set_icon("reversing")
        elif branch == "chase":
            _set_icon("chasing", _size_step_for_distance(distance_cm))
        elif branch == "teleop":
            _set_icon("teleop")
        else:  # "idle", None, or anything unrecognized
            _set_icon("idle")
    else:
        # No robot/state received yet, or an unrecognized phase: a safe,
        # quiet default rather than guessing.
        _set_icon("idle")


# ===================================================================
# ===== Status publishing + logging (once a second) =====
# ===================================================================

_last_status_time = 0.0


def _publish_and_log_status(now):
    global _update_count

    with _state_lock:
        distance_cm = _vision["distance_cm"]
        visible = _vision["visible"]
        confidence = _vision["confidence"]
        sensor_ok = _vision["sensor_ok"]
        count = _update_count
        _update_count = 0

    fps = count / config.STATUS_LOG_INTERVAL_S

    status_payload = {
        "state": "online",
        "camera": "ok" if sensor_ok else "missing",  # "camera" per the contract; this app reports the ultrasonic sensor's health here
        "fps": round(fps, 1),
    }
    _safe_publish(config.TOPIC_UNOQ_STATUS, json.dumps(status_payload), qos=1)

    dist_str = "?" if distance_cm is None else f"{distance_cm:.1f}cm"
    logger.info(
        f"distance={dist_str} visible={visible} confidence={confidence:.2f} "
        f"sensor_ok={sensor_ok} rate={fps:.1f}Hz mqtt_connected={_mqtt_connected} "
        f"laptop_online={_laptop_online} robot_phase={_robot_state['phase']} robot_branch={_robot_state['branch']}"
    )


# ===================================================================
# ===== Startup =====
# ===================================================================

_configured = False


def _push_config_once():
    """Pushes config.py's detection numbers into the sketch. Retried from
    the main loop (not blocking) in case the sketch's Bridge isn't ready
    on the very first attempt right after boot."""
    global _configured
    result = _safe_bridge_call(
        "configure",
        config.MIN_VALID_DISTANCE_CM,
        config.RING_DIAMETER_CM,
        config.LOSE_RANGE_CM,
        config.DEBOUNCE_COUNT,
    )
    if result is not _BRIDGE_CALL_FAILED:
        _configured = True
    # else: leave _configured False so loop() retries on the next iteration


def loop():
    global _last_status_time, _configured
    now = time.monotonic()

    if not _configured:
        _push_config_once()

    # All LED icon Bridge.call()s happen here, on loop()'s own thread -
    # see _on_vision_update's docstring for why that matters. _set_icon()
    # only actually calls the sketch when the icon changes, so calling
    # this every ~0.1s is cheap in the steady state.
    if not _song_in_progress:
        _update_led()

    if now - _last_status_time >= config.STATUS_LOG_INTERVAL_S:
        _last_status_time = now
        _publish_and_log_status(now)

    time.sleep(0.1)


threading.Thread(target=_mqtt_worker, daemon=True).start()

App.run(user_loop=loop)
