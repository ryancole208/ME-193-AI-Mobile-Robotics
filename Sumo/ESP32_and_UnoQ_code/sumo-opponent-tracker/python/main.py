"""Sumo Opponent Tracker - full pipeline.

Camera/detection (reused/built earlier):
  - camera_discovery.py finds the ESP32-CAM; camera_stream.py reads its
    MJPEG feed in the background.
  - color_detector.py / distance_estimator.py / config_store.py: a
    classical HSV color-based detector finds the opponent, rejects false
    positives (ring tape, walls, lights), and estimates its distance. All
    tunables live in config.json and can be adjusted live from the debug
    web page (served by the arduino:web_ui brick) without restarting -
    see that page for the click-to-sample color tuning, distance
    calibration, and recording/replay tools.

MQTT + robot status (this section): talks to the laptop's behavior tree
over the public broker test.mosquitto.org, using mqttlib.py exactly as
given (see its own file - it is NOT modified here). This is the "glue"
between the camera detector and the rest of the robot:

  1. Publishes vision/opponent for every processed camera frame.
  2. Publishes unoq/status (camera health + measurement rate) once a
     second.
  3. Subscribes to robot/state (shown on the LED matrix), laptop/status
     (are we even talking to a laptop right now?), and event/lost (play
     the sad song, show the sad face, then publish event/song_done once
     the sketch confirms the song actually finished).
  4. Also runs a small self-echo "round trip time" probe (see
     _rtt_worker) so the console shows a real, measured number for how
     long a message takes to go out to the broker and back - this is
     NOT part of the official MQTT contract, just a diagnostic.

Threading model (important - see the comments where each piece is
defined for why): camera detection runs entirely on loop()'s own thread.
MQTT subscription callbacks run on paho's own background thread. The
sketch's "song finished" notification runs on the Bridge library's own
dispatch thread. All shared state three of those touch is behind
_state_lock. Bridge.call() (telling the sketch which icon to show, or to
play the song) is ONLY ever made from loop()'s own thread - see
_on_song_finished's docstring for why calling it from a Bridge-provided
callback instead reliably deadlocks. A flaky public broker, or a
slow/unresponsive sketch, must never crash this app - every MQTT and
Bridge call is wrapped so a failure there is logged and ignored, not
fatal, since the camera/LED/buzzer all need to keep working regardless
of what the network is doing.

Startup: this app is meant to be set to "Run at startup" in App Lab, and
must work correctly with no laptop connected at all (the LED matrix
should settle on the "waiting for laptop" icon and the camera/vision
publishing keeps working regardless - only the robot/state-driven icons
need a laptop).
"""

import base64
import copy
import json
import os
import threading
import time

import cv2
import numpy as np
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

import config_store
from camera_discovery import discover_camera
from camera_stream import MjpegStreamReader
from color_detector import ColorDetector
from mqttlib import MQTTClient
import opponent_locator  # turns the mask into the opponent's x/distance (see that file)

# ===================== TUNABLE: camera/detection/debug page =====================
STREAM_FPS = 12  # how many frames per second to push to the debug page and run detection on
STATUS_INTERVAL = 1.0  # seconds between printed console status lines / unoq-status publishes
STALE_FRAME_AGE = 2.0  # a frame older than this (seconds) counts as "stale"
REDISCOVER_AFTER = 5.0  # seconds with no camera/stale stream before retrying discovery
LED_UPDATE_MAX_HZ = 15  # LED matrix updates are capped to this rate even if the detector runs faster
CALIBRATION_SAMPLE_BOX_FRAC = 0.2  # size (as a fraction of frame width/height) of the box sampled by "Set <color> here"
LED_SIZE_STEPS = 4  # number of distinct box sizes the "chasing" LED icon can show (0=farthest, LED_SIZE_STEPS-1=closest)
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "recordings")

# ===================== TUNABLE: opponent position from the mask =====================
# Used by opponent_locator.py (see its docstring for the step-by-step method).
# These are deliberately here, NOT in config.json (which holds the mask calibration).
GROUP_RADIUS = 18  # px (frames are 320x240). Clusters closer than ~2x this merge into one group.
#                    Robot split into several groups? Raise it. Unrelated specks joining the robot? Lower it.
MIN_PIXELS = 150  # a group needs at least this many white mask pixels to count (fewer = noise)
X_SMOOTHING = 0.5  # 0..1: weight of the newest frame in x. 1 = no smoothing (twitchy), lower = smoother but slower
DEBOUNCE_FRAMES = 3  # frames in a row a group must be seen before visible=True (same count as RQ-D2 Vision)
NEAR_CM = 5.0  # distance reported when the group's lowest pixel is on the own-robot line (bottom of detection area)
FAR_CM = 100.0  # distance reported when it's on the horizon line (top of detection area); 100 = ring diameter
FULL_CONFIDENCE_PIXELS = 1500  # a group this big (or bigger) gets full marks for size in `confidence`
STABILITY_FRAMES = 5  # how many recent frames are checked to see if the center is steady
STABILITY_SPREAD = 0.3  # if x jumped around by this much (or more) over those frames, steadiness score is 0

# ===================== TUNABLE: MQTT contract =====================
# All topic names live here, in one place, matching the laptop side exactly.
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
TOPIC_PREFIX = "ME193/Rogers/RQ-D2/"

TOPIC_VISION = TOPIC_PREFIX + "vision/opponent"
TOPIC_UNOQ_STATUS = TOPIC_PREFIX + "unoq/status"
TOPIC_LAPTOP_STATUS = TOPIC_PREFIX + "laptop/status"
TOPIC_ROBOT_STATE = TOPIC_PREFIX + "robot/state"
TOPIC_EVENT_LOST = TOPIC_PREFIX + "event/lost"
TOPIC_EVENT_SONG_DONE = TOPIC_PREFIX + "event/song_done"
# Not part of the official contract - a private topic this app uses to
# measure its own round-trip time through the broker (see _rtt_worker).
TOPIC_RTT_PROBE = TOPIC_PREFIX + "debug/rtt_probe"

# How long to wait before retrying a dropped/failed initial MQTT connection,
# doubling up to the cap each time it fails again (a brief broker hiccup
# retries fast; a genuinely down broker doesn't get hammered).
MQTT_RECONNECT_INITIAL_DELAY_S = 1.0
MQTT_RECONNECT_MAX_DELAY_S = 30.0

RTT_PROBE_INTERVAL_S = 5.0  # how often to measure/log the self-echo round-trip time
# =====================================================

config = config_store.load_config()
detector = ColorDetector()  # still builds the calibrated mask; its own target pick is no longer used
ui = WebUI()


def _make_tracker(debounce_frames=DEBOUNCE_FRAMES):
    """One OpponentTracker built from the TUNABLE values above."""
    return opponent_locator.OpponentTracker(
        group_radius=GROUP_RADIUS, min_pixels=MIN_PIXELS, x_smoothing=X_SMOOTHING,
        debounce_frames=debounce_frames, near_cm=NEAR_CM, far_cm=FAR_CM,
        full_confidence_pixels=FULL_CONFIDENCE_PIXELS,
        stability_frames=STABILITY_FRAMES, stability_spread=STABILITY_SPREAD,
    )


tracker = _make_tracker()  # used only by loop() (the live camera)


def _roi_rows(frame_shape):
    """Pixel rows of the horizon and own-robot lines, computed the same way
    color_detector.py does - the top and bottom of the detection area."""
    height = frame_shape[0]
    return int(height * config["horizon_row_frac"]), int(height * config["own_robot_row_frac"])

_reader = None
_last_status_print = 0.0
_last_discovery_attempt = 0.0
_last_processed_frame_time = None  # the frame timestamp last run through the detector, so we don't reprocess the same frame twice
_last_result = None
_last_debug = None  # the "debug" dict from the last detector.detect() call - holds per_color_masks + diagnostics for status logging
_last_led_sent = None  # (icon_name, col, size_step) - only re-sent to the sketch when this changes
_last_led_update_time = 0.0
_sample_point = (0.5, 0.5)  # (x_frac, y_frac) of the last spot clicked on the live image - what "Set <color> here" samples from
_mask_channel = "combined"  # which mask the debug page's mask image shows: "combined", or one color name to isolate it

# ----- Shared state written from more than one thread - see module docstring -----
# MQTT callbacks run on paho's own thread; the sketch's "song finished"
# notification runs on the Bridge dispatch thread; loop() reads all of this
# from its own thread. Every read/write of the fields below goes through
# _state_lock to avoid torn reads (e.g. a fresh phase paired with a stale
# branch from a half-finished update).
_state_lock = threading.Lock()
_robot_state = {"phase": None, "branch": None}
_laptop_last_state = None  # the most recent laptop/status "state" field seen, or None if never received
_song_in_progress = False  # True from event/lost until the sketch reports the song finished

_mqtt_client = None  # an mqttlib.MQTTClient, set once the first connection succeeds
_mqtt_connected = False
_vision_seq = 0  # increments once per vision/opponent message published
_rtt_pending = {}  # {probe_id: send_time (time.monotonic())} - at most one outstanding probe in practice


# ===================================================================
# ===== Camera connection (reused pattern from earlier steps) =====
# ===================================================================

def _connect_to(base_url):
    global _reader
    if _reader is not None:
        _reader.stop()
    stream_url = base_url + "/stream"
    print(f"[main] connecting to {stream_url}")
    print(f"[main] watch it in your browser at {base_url}/")
    _reader = MjpegStreamReader(stream_url)
    _reader.start()


def _try_discover_and_connect():
    base_url = discover_camera()
    if base_url is None:
        print("[main] couldn't find the ESP32-CAM on the network; will keep retrying")
        return False
    _connect_to(base_url)
    return True


# ===================================================================
# ===== Image helpers =====
# ===================================================================

def _decode(jpeg_bytes):
    return cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def _encode_jpeg(bgr_image):
    ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _sample_box(frame_shape):
    """The box that gets sampled by "Set <color> here" - centered on
    _sample_point, which defaults to the middle of the frame but moves
    wherever you last clicked on the live image."""
    height, width = frame_shape[:2]
    box_w, box_h = int(width * CALIBRATION_SAMPLE_BOX_FRAC), int(height * CALIBRATION_SAMPLE_BOX_FRAC)
    cx, cy = int(_sample_point[0] * width), int(_sample_point[1] * height)
    x0 = max(0, min(width - box_w, cx - box_w // 2))
    y0 = max(0, min(height - box_h, cy - box_h // 2))
    return x0, y0, box_w, box_h


# Added on top of the sampled spread, per channel, as a small cushion
# against frame-to-frame noise - NOT a guess at the color's real range
# (that part comes straight from the sampled pixels, see below).
_SAMPLE_SAFETY_MARGIN = {"h": 5, "s": 15, "v": 15}
_SAMPLE_PERCENTILE_LOW = 5
_SAMPLE_PERCENTILE_HIGH = 95


def _sample_hsv_range(bgr_image):
    """Samples the sample box (see _sample_box) and returns an HSV range
    computed from the *actual pixels* seen there - the 5th-95th percentile
    of each channel, plus a small safety cushion - rather than a guessed
    margin. This is the fast, systematic way to tune a color: click on it
    in the live image, then press its "Set <color> here" button. See
    _on_learn_color for how repeated clicks on the same target widen the
    range instead of replacing it, so a couple of clicks across a piece's
    lit and shadowed sides converges on a solid range."""
    x0, y0, box_w, box_h = _sample_box(bgr_image.shape)
    region = bgr_image[y0:y0 + box_h, x0:x0 + box_w]
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    result = {}
    for i, channel in enumerate(("h", "s", "v")):
        lo = float(np.percentile(hsv_region[:, i], _SAMPLE_PERCENTILE_LOW))
        hi = float(np.percentile(hsv_region[:, i], _SAMPLE_PERCENTILE_HIGH))
        margin = _SAMPLE_SAFETY_MARGIN[channel]
        max_value = 179 if channel == "h" else 255
        result[channel] = [int(max(0, lo - margin)), int(min(max_value, hi + margin))]
    return result


def _union_hsv_range(a, b):
    """Combines two HSV ranges into the smallest one covering both - used
    so clicking the same color again and pressing its button *widens* the
    learned range to also include the new spot, instead of discarding
    whatever was learned from previous clicks."""
    return {channel: [min(a[channel][0], b[channel][0]), max(a[channel][1], b[channel][1])] for channel in ("h", "s", "v")}


# ===================================================================
# ===== Debug page: config editing (unchanged from earlier steps) =====
# ===================================================================

def _on_update_config(client_id, data):
    """A slider moved: live-applies one setting without saving to disk.
    data = {"path": ["colors", "azure", "h"], "value": [90, 120]} - path is
    the list of keys to walk into the nested config dict."""
    node = config
    for key in data["path"][:-1]:
        node = node[key]
    node[data["path"][-1]] = data["value"]


def _on_save_config(client_id, data):
    config_store.save_config(config)
    ui.send_message("config", config)


def _on_set_sample_point(client_id, data):
    """The live image was clicked: remember where, in fractions of the
    frame (0-1 each way) so it still makes sense if the browser resizes
    the image. The sample box drawn on the live view moves there too."""
    global _sample_point
    x_frac = max(0.0, min(1.0, float(data["x_frac"])))
    y_frac = max(0.0, min(1.0, float(data["y_frac"])))
    _sample_point = (x_frac, y_frac)


# Where a given "Set <color> here" button's result gets written in config:
# the 4 opponent colors live under config["colors"][name]; floor/tape are
# their own top-level keys.
_LEARNABLE_TARGETS = {"azure", "pink", "white", "wheel", "floor", "tape"}


def _on_learn_color(client_id, data):
    """Fast, systematic color tuning: samples the box at _sample_point and
    merges a range computed from the actual pixels there into whichever
    target was requested (see _sample_hsv_range/_union_hsv_range) -
    azure/pink/white/wheel update that opponent color, floor/tape update
    the learned floor/tape colors. Clicking the same target again widens
    its range rather than replacing it, so a few clicks across a piece's
    lit and shadowed sides converges on a solid range."""
    target = data["target"]
    if target not in _LEARNABLE_TARGETS:
        ui.send_message("calibration_result", {"success": False, "message": f"unknown target {target!r}", "kind": "color"})
        return

    frame_bytes, _ = _reader.get_latest_frame() if _reader is not None else (None, None)
    if frame_bytes is None:
        ui.send_message("calibration_result", {"success": False, "message": "no frame available", "kind": "color"})
        return

    new_range = _sample_hsv_range(_decode(frame_bytes))
    if target in ("floor", "tape"):
        existing = config[f"{target}_color"]
        config[f"{target}_color"] = _union_hsv_range(existing, new_range) if existing is not None else new_range
    else:
        existing = config["colors"][target]
        config["colors"][target] = _union_hsv_range(existing, new_range) if existing is not None else new_range

    ui.send_message("calibration_result", {"success": True, "message": f"set {target} from the clicked spot", "kind": "color"})
    ui.send_message("config", config)


def _on_reset_colors(client_id, data):
    """Clears all learned/tuned colors back to the plain defaults - use
    this if a stray click widened a range too far and it's easier to start
    over than to keep clicking good spots to try to out-widen the bad one."""
    defaults = config_store.DEFAULT_CONFIG
    config["colors"] = copy.deepcopy(defaults["colors"])
    config["floor_color"] = defaults["floor_color"]
    config["tape_color"] = defaults["tape_color"]
    ui.send_message("calibration_result", {"success": True, "message": "reset all colors to defaults", "kind": "color"})
    ui.send_message("config", config)


def _on_set_line(client_id, data):
    """Click-to-set for the horizon/own-robot lines: uses the y position
    of the last clicked spot (see _on_set_sample_point) instead of a
    fraction slider."""
    line = data["line"]
    if line not in ("horizon", "own_robot"):
        ui.send_message("calibration_result", {"success": False, "message": f"unknown line {line!r}", "kind": "color"})
        return
    config[f"{line}_row_frac"] = _sample_point[1]
    ui.send_message("calibration_result", {"success": True, "message": f"set {line} line from the clicked spot", "kind": "color"})
    ui.send_message("config", config)


# ===================================================================
# ===== Debug page: distance calibration (unchanged from earlier steps) =====
# ===================================================================

def _on_capture_distance_point(client_id, data):
    """Runs the detector on the current frame and, if a candidate blob is
    found, records its position/size alongside the distance you measured
    by hand. Repeat at several distances to build the calibration curve."""
    frame_bytes, _ = _reader.get_latest_frame() if _reader is not None else (None, None)
    if frame_bytes is None:
        ui.send_message("calibration_result", {"success": False, "message": "no frame available", "kind": "distance"})
        return

    bgr = _decode(frame_bytes)
    target = detector.find_raw_target(bgr, config)
    if target is None:
        ui.send_message("calibration_result", {
            "success": False,
            "message": "no candidate blob found - check the color sliders and that something's in view",
            "kind": "distance",
        })
        return

    distance_cm = float(data["distance_cm"])
    x, y, w, h = target["bbox"]
    height = bgr.shape[0]
    config["distance_calibration_points"].append({
        "row_frac": (y + h) / height,
        "bbox_height_frac": h / height,
        "distance_cm": distance_cm,
    })
    ui.send_message("calibration_result", {
        "success": True,
        "message": f"captured point at {distance_cm:.0f}cm ({len(config['distance_calibration_points'])} total)",
        "kind": "distance",
    })
    ui.send_message("config", config)


def _on_clear_distance_calibration(client_id, data):
    config["distance_calibration_points"] = []
    ui.send_message("calibration_result", {"success": True, "message": "cleared distance calibration", "kind": "distance"})
    ui.send_message("config", config)


# ===================================================================
# ===== Debug page: recording / replay (unchanged from earlier steps) =====
# ===================================================================

def _on_record_frame(client_id, data):
    frame_bytes, _ = _reader.get_latest_frame() if _reader is not None else (None, None)
    if frame_bytes is None:
        ui.send_message("record_result", {"success": False, "error": "no frame available yet"})
        return
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    filename = f"frame_{time.time():.3f}.jpg"
    with open(os.path.join(RECORDINGS_DIR, filename), "wb") as f:
        f.write(frame_bytes)
    print(f"[main] recorded frame to recordings/{filename}")
    ui.send_message("record_result", {"success": True, "filename": filename})
    _on_list_recordings(client_id, {})


def _on_list_recordings(client_id, data):
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    ui.send_message("recordings_list", {"files": sorted(os.listdir(RECORDINGS_DIR))})


def _on_replay_frame(client_id, data):
    """Runs the full detector on a previously recorded frame and pushes
    the result through the same "frame" message the live view uses."""
    filename = data["filename"]
    try:
        with open(os.path.join(RECORDINGS_DIR, filename), "rb") as f:
            frame_bytes = f.read()
    except OSError as exc:
        ui.send_message("calibration_result", {"success": False, "message": f"couldn't read {filename}: {exc!r}", "kind": "color"})
        return
    bgr = _decode(frame_bytes)
    _, debug = detector.detect(bgr, config, time.monotonic())
    # A fresh tracker with no debounce, so a single replayed frame can show
    # visible=True - and so replaying never disturbs the live tracker's memory.
    horizon_row, own_robot_row = _roi_rows(bgr.shape)
    result = _make_tracker(debounce_frames=1).update(debug["mask"], horizon_row, own_robot_row)
    overlay = opponent_locator.draw_overlay(bgr, result, horizon_row, own_robot_row)
    _push_frame_message(overlay, debug, result, source=filename, fps=0.0, frame_age=0.0, state="replay")


def _on_set_mask_channel(client_id, data):
    """Which mask to show on the debug page: "combined" (what the detector
    actually uses) or one single color's own raw mask, picked from the
    dropdown - isolating one color makes it obvious which one is too broad
    (e.g. lighting up the whole floor) instead of guessing from the
    combined picture."""
    global _mask_channel
    channel = data.get("channel")
    if not channel:
        print(f"[main] set_mask_channel: ignoring malformed message {data!r}")
        return
    _mask_channel = channel


ui.on_message("update_config", _on_update_config)
ui.on_message("save_config", _on_save_config)
ui.on_message("set_sample_point", _on_set_sample_point)
ui.on_message("set_mask_channel", _on_set_mask_channel)
ui.on_message("learn_color", _on_learn_color)
ui.on_message("reset_colors", _on_reset_colors)
ui.on_message("set_line", _on_set_line)
ui.on_message("capture_distance_point", _on_capture_distance_point)
ui.on_message("clear_distance_calibration", _on_clear_distance_calibration)
ui.on_message("record_frame", _on_record_frame)
ui.on_message("list_recordings", _on_list_recordings)
ui.on_message("replay_frame", _on_replay_frame)
ui.on_connect(lambda client_id: (ui.send_message("config", config), _on_list_recordings(client_id, {})))


def _draw_calibration_box(overlay_bgr):
    """Draws the box that "Set <color> here" samples from, wherever it
    currently is (see _sample_point) - so it's visible on the live image
    instead of being an invisible target you have to guess at."""
    x0, y0, box_w, box_h = _sample_box(overlay_bgr.shape)
    cv2.rectangle(overlay_bgr, (x0, y0), (x0 + box_w, y0 + box_h), (0, 165, 255), 2)


def _select_mask(debug):
    """Picks which mask to show, per _mask_channel: the combined mask the
    detector actually uses, or one isolated color's raw mask (before
    cleanup) so you can tell exactly which color is too broad."""
    if _mask_channel == "combined":
        return debug["mask"]
    return debug["per_color_masks"].get(_mask_channel, debug["mask"])


def _push_frame_message(overlay_bgr, debug, result, source, fps, frame_age, state):
    _draw_calibration_box(overlay_bgr)
    try:
        mask_bgr = cv2.cvtColor(_select_mask(debug), cv2.COLOR_GRAY2BGR)
    except Exception as exc:  # noqa: BLE001 - a bad mask_channel must never kill the main loop/app
        print(f"[main] _select_mask({_mask_channel!r}) failed, falling back to combined mask: {exc!r}")
        mask_bgr = cv2.cvtColor(debug["mask"], cv2.COLOR_GRAY2BGR)
    ui.send_message("frame", {
        "source": source,
        "image": _encode_jpeg(overlay_bgr),
        "mask": _encode_jpeg(mask_bgr),
        "mask_channel": _mask_channel,
        "state": state,
        "fps": fps,
        "frame_age": frame_age,
        "warnings": debug.get("diagnostics", {}).get("warnings", []),
        "detection": {
            "visible": result.visible,
            "x": result.x,
            "distance_cm": result.distance_cm,
            "confidence": result.confidence,
            "group_count": result.group_count,
            "group_pixels": result.group_pixels,
            "processing_ms": result.processing_ms,
        },
    })


# ===================================================================
# ===== Bridge: receiving from the sketch =====
# ===================================================================

_BRIDGE_CALL_FAILED = object()  # sentinel: lets callers tell "failed" apart from a real None/void return


def _safe_bridge_call(name, *args):
    """Bridge.call wrapped so a slow/unresponsive sketch can never crash
    this app - same defensive spirit as _safe_publish below."""
    try:
        return Bridge.call(name, *args)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: must never raise
        print(f"[main] Bridge.call({name!r}) failed: {exc!r}")
        return _BRIDGE_CALL_FAILED


def _on_song_finished():
    """Called by the sketch (Bridge.notify) once the sad song's 5 notes
    have all finished playing.

    IMPORTANT: this runs on the Bridge library's own dispatch thread (the
    same one that delivers Bridge.call's own responses) - calling
    Bridge.call(...) from inside a function the sketch calls INTO Python
    reliably self-deadlocks, because that thread would then be waiting on
    itself. This is a real bug hit and fixed while building the sibling
    "RQ-D2 Vision" app (ultrasonic version of this same contract) - see
    that app's README for the full story. The fix is the same here: this
    callback only updates shared state and publishes to MQTT; it never
    calls Bridge.call. All Bridge.call()s that talk to the LED matrix
    happen only from loop()'s own thread (see _update_led)."""
    global _song_in_progress
    _song_in_progress = False
    _safe_publish(TOPIC_EVENT_SONG_DONE, "{}")


Bridge.provide("song_finished", _on_song_finished)


# ===================================================================
# ===== MQTT: connecting, with retry, and never crashing =====
# ===================================================================

def _safe_publish(topic, payload):
    """Publishes if we're currently connected; silently drops the message
    otherwise. vision/opponent messages are fire-and-forget by design - a
    new one is on its way in ~1/STREAM_FPS seconds anyway - so dropping one
    while disconnected is harmless; the alternative (crashing, or blocking
    the whole app on a dead broker) would be far worse."""
    if _mqtt_client is None or not _mqtt_connected:
        return
    try:
        _mqtt_client.publish(topic, payload)
    except Exception as exc:  # noqa: BLE001 - a flaky public broker must never crash this app
        print(f"[main] MQTT publish to {topic} failed: {exc!r}")


def _on_robot_state(topic, payload):
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"[main] robot/state: couldn't parse payload {payload!r}")
        return
    with _state_lock:
        _robot_state["phase"] = data.get("phase")
        _robot_state["branch"] = data.get("branch")
    # The new icon is picked up by loop()'s own _update_led() call within
    # ~1/STREAM_FPS seconds - see that function for why Bridge calls only
    # ever happen from loop()'s thread, never from here.


def _on_laptop_status(topic, payload):
    global _laptop_last_state
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"[main] laptop/status: couldn't parse payload {payload!r}")
        return
    with _state_lock:
        _laptop_last_state = data.get("state")


def _on_event_lost(topic, payload):
    """The laptop decided our robot got pushed out of the ring. Show the
    sad face right away and start the song; event/song_done is published
    once the sketch confirms the song actually finished (see
    _on_song_finished above).

    This runs on paho's own MQTT thread, not the Bridge dispatch thread, so
    (unlike _on_song_finished above) it's safe to call Bridge.call directly
    here - that's what lets the sad face appear instantly instead of
    waiting for loop()'s next ~1/STREAM_FPS tick."""
    global _song_in_progress
    _song_in_progress = True
    _set_icon("lost")
    _safe_bridge_call("play_sad_song")


def _on_rtt_probe_echo(topic, payload):
    """Our own round-trip probe (see _rtt_worker) coming back to us via our
    own subscription. Logs how long it took to go UNO Q -> broker -> UNO Q
    - not laptop-to-laptop, since comparing clocks across devices isn't
    reliable (and isn't needed here: this measures what matters, which is
    how much of vision/opponent's "freshness" is being spent on the network
    hop through the broker, on this device's own clock throughout)."""
    try:
        data = json.loads(payload)
        probe_id = data["id"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
        return
    send_time = _rtt_pending.pop(probe_id, None)
    if send_time is not None:
        rtt_ms = (time.monotonic() - send_time) * 1000
        print(f"[main] MQTT round-trip time through {MQTT_BROKER}: {rtt_ms:.0f}ms")


def _wire_subscriptions(client):
    client.subscribe(TOPIC_ROBOT_STATE, _on_robot_state)
    client.subscribe(TOPIC_LAPTOP_STATUS, _on_laptop_status)
    client.subscribe(TOPIC_EVENT_LOST, _on_event_lost)
    client.subscribe(TOPIC_RTT_PROBE, _on_rtt_probe_echo)


def _mqtt_worker():
    """Connects to the broker, retrying with exponential backoff if it's
    unreachable, forever, in the background. Runs in its own thread so a
    dead/slow broker at startup never blocks the camera/LED/buzzer side of
    this app.

    Once the FIRST connection succeeds, paho-mqtt's own background thread
    (started by MQTTClient.__enter__ via loop_start()) automatically
    reconnects on its own if the connection later drops - so this loop only
    has to handle getting connected in the first place."""
    global _mqtt_client, _mqtt_connected
    delay = MQTT_RECONNECT_INITIAL_DELAY_S
    while True:
        try:
            client = MQTTClient(MQTT_BROKER, MQTT_PORT)
            client.__enter__()  # same connect()+loop_start() a "with" block would do
        except Exception as exc:  # noqa: BLE001 - must keep retrying no matter what goes wrong
            print(f"[main] MQTT connect to {MQTT_BROKER}:{MQTT_PORT} failed: {exc!r}; retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, MQTT_RECONNECT_MAX_DELAY_S)
            continue

        # mqttlib.py doesn't expose connect/disconnect notifications (and we
        # were asked not to modify it), so we attach paho's own on_disconnect
        # handler directly to the underlying client it keeps at `_client` -
        # this only READS state, nothing about mqttlib.py's behavior changes.
        def _on_disconnect(c, userdata, flags, reason_code, properties):
            global _mqtt_connected
            _mqtt_connected = False
            print(f"[main] MQTT disconnected (reason={reason_code}); paho will keep retrying in the background")

        client._client.on_disconnect = _on_disconnect

        _mqtt_client = client
        _mqtt_connected = True
        _wire_subscriptions(client)
        print(f"[main] connected to MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
        return  # paho's background thread now owns the connection; nothing left to do here


def _rtt_worker():
    """Publishes a small probe message to TOPIC_RTT_PROBE every
    RTT_PROBE_INTERVAL_S and relies on our own subscription to the same
    topic (see _wire_subscriptions/_on_rtt_probe_echo) to measure how long
    it took to come back. Runs in its own thread so a slow/disconnected
    broker never affects the camera/LED loop's timing."""
    probe_id = 0
    while True:
        time.sleep(RTT_PROBE_INTERVAL_S)
        probe_id += 1
        _rtt_pending[probe_id] = time.monotonic()
        _safe_publish(TOPIC_RTT_PROBE, json.dumps({"id": probe_id}))


# ===================================================================
# ===== Publishing vision/opponent =====
# ===================================================================

def _publish_vision(result, age):
    global _vision_seq
    _vision_seq += 1
    payload = {
        "seq": _vision_seq,
        "visible": result.visible,
        # Same keys/types as RQ-D2 Vision; only x carries real left/right info
        # here (RQ-D2's ultrasonic sensor always sends 0.0). float() turns
        # numpy numbers into plain Python ones so json.dumps never chokes.
        "x": float(result.x) if result.visible else None,
        "distance_cm": float(result.distance_cm) if result.visible else None,
        "confidence": float(result.confidence),
        "frame_age_ms": int(age * 1000),
        "source": "camera",  # RQ-D2 Vision sends "ultrasonic" here
    }
    _safe_publish(TOPIC_VISION, json.dumps(payload))


# ===================================================================
# ===== LED icon selection =====
# ===================================================================
# Picks ONE icon from everything we currently know, in priority order
# (most urgent problem first), then tells the sketch to draw it - but only
# if it's actually different from what's already on screen (and not faster
# than LED_UPDATE_MAX_HZ).

def _size_step_for_distance(distance_cm):
    """Maps a distance in cm to a LED box size step (0=smallest/farthest,
    LED_SIZE_STEPS-1=largest/closest), over the NEAR_CM..FAR_CM range.
    Falls back to a fixed middle size with no distance estimate - still
    shows *that* something was found, just without size-by-distance."""
    if distance_cm is None:
        return LED_SIZE_STEPS // 2

    near, far = NEAR_CM, FAR_CM
    if far == near:
        return LED_SIZE_STEPS // 2

    # Closer (smaller distance_cm) -> bigger box, so this fraction is inverted.
    frac = 1.0 - (max(near, min(far, distance_cm)) - near) / (far - near)
    return max(0, min(LED_SIZE_STEPS - 1, round(frac * (LED_SIZE_STEPS - 1))))


def _set_icon(name, col=0, size_step=0):
    """Sends one icon to the sketch, but only when it has actually changed
    and not more often than LED_UPDATE_MAX_HZ."""
    global _last_led_sent, _last_led_update_time
    now = time.monotonic()
    if now - _last_led_update_time < 1.0 / LED_UPDATE_MAX_HZ:
        return
    state = (name, col, size_step)
    if state == _last_led_sent:
        return
    _last_led_sent = state
    _last_led_update_time = now

    if name == "no_camera":
        _safe_bridge_call("show_no_camera")
    elif name == "searching":
        _safe_bridge_call("show_searching")
    elif name == "target":
        _safe_bridge_call("show_target", col, size_step)
    else:
        _safe_bridge_call(f"show_icon_{name}")


def _is_laptop_online():
    """Same rule as RQ-D2 Vision: online whenever the most recent
    laptop/status message said "online" - no timeout."""
    with _state_lock:
        return _laptop_last_state == "online"


def _update_led(is_stale):
    """Checked in priority order, top to bottom - e.g. a dead camera always
    shows the X, even if everything else about the robot's state is fine."""
    if is_stale:
        _set_icon("no_camera")
        return
    if not _mqtt_connected:
        _set_icon("no_network")
        return
    if not _is_laptop_online():
        _set_icon("waiting_laptop")
        return

    with _state_lock:
        phase = _robot_state["phase"]
        branch = _robot_state["branch"]

    if phase == "lost":
        _set_icon("lost")
    elif phase == "armed":
        _set_icon("armed")
    elif phase == "running":
        if branch == "reverse":
            _set_icon("reversing")
        elif branch == "chase":
            if _last_result is not None and _last_result.visible:
                col = max(0, min(12, round(((_last_result.x + 1) / 2) * 12)))
                _set_icon("target", col, _size_step_for_distance(_last_result.distance_cm))
            else:
                _set_icon("searching")
        elif branch == "teleop":
            _set_icon("teleop")
        else:  # "idle", None, or anything unrecognized
            _set_icon("idle")
    else:
        # No robot/state received yet, or an unrecognized phase: a safe,
        # quiet default rather than guessing. (In practice this is mostly
        # unreachable - a laptop that's online but has never sent
        # robot/state is rare - but it's a safe fallback regardless.)
        _set_icon("idle")


# ===================================================================
# ===== Diagnostic logging (mask coverage, HSV bounds/percentiles) =====
# ===================================================================
# TEMPORARY, added to track down the "combined mask is a solid rectangle"
# bug: prints, once a second, exactly what the user's diagnostic request
# asked for - no detection behavior changes here, only logging.

def _print_mask_diagnostics():
    if _last_debug is None or "diagnostics" not in _last_debug:
        return
    diagnostics = _last_debug["diagnostics"]
    coverage = diagnostics["coverage_pct"]
    roi_hsv = diagnostics["roi_hsv"]
    excluded_masks = diagnostics.get("excluded_masks", [])

    parts = []
    for name, pct in sorted(coverage.items(), key=lambda kv: -kv[1]):
        flag = " <-- EXCLUDED (saturated)" if name in excluded_masks else ""
        parts.append(f"{name}={pct:.1f}%{flag}")
    print(f"[diag] ROI coverage: {'  '.join(parts)}")

    for warning in diagnostics.get("warnings", []):
        print(f"[diag] WARNING: {warning}")

    if roi_hsv is not None:
        h, s, v = roi_hsv["h"], roi_hsv["s"], roi_hsv["v"]
        print(
            f"[diag] ROI HSV (p10/median/p90): "
            f"H={h['p10']:.0f}/{h['median']:.0f}/{h['p90']:.0f}  "
            f"S={s['p10']:.0f}/{s['median']:.0f}/{s['p90']:.0f}  "
            f"V={v['p10']:.0f}/{v['median']:.0f}/{v['p90']:.0f}"
        )

    for name, color_range in config["colors"].items():
        print(f"[diag] colors.{name} bounds: h={color_range['h']} s={color_range['s']} v={color_range['v']}")
    print(
        f"[diag] floor_fallback_enabled={config['floor_fallback_enabled']} "
        f"floor_color={'set' if config['floor_color'] is not None else 'NOT CALIBRATED'} "
        f"tape_color={'set' if config['tape_color'] is not None else 'NOT CALIBRATED'}"
    )


# ===================================================================
# ===== Main loop =====
# ===================================================================

def loop():
    global _last_status_print, _last_discovery_attempt, _last_processed_frame_time, _last_result, _last_debug

    now = time.monotonic()
    frame_bytes, frame_time = _reader.get_latest_frame() if _reader is not None else (None, None)
    age = None if frame_time is None else now - frame_time
    is_stale = age is None or age > STALE_FRAME_AGE

    if is_stale and now - _last_discovery_attempt >= REDISCOVER_AFTER:
        _last_discovery_attempt = now
        _try_discover_and_connect()

    if not is_stale and frame_time is not None and frame_time != _last_processed_frame_time:
        _last_processed_frame_time = frame_time
        bgr = _decode(frame_bytes)
        # 1. The (unchanged) detector builds the calibrated mask: debug["mask"].
        _, debug = detector.detect(bgr, config, now)
        _last_debug = debug
        # 2. Group the mask's clusters and work out visible/x/distance from them.
        horizon_row, own_robot_row = _roi_rows(bgr.shape)
        _last_result = tracker.update(debug["mask"], horizon_row, own_robot_row)
        # 3. Debug page picture: chosen group's outline + center dot.
        overlay = opponent_locator.draw_overlay(bgr, _last_result, horizon_row, own_robot_row)
        _push_frame_message(
            overlay, debug, _last_result,
            source="live", fps=_reader.get_fps(), frame_age=age, state="ok",
        )
        _publish_vision(_last_result, age)

    # The LED icon depends on more than just the camera (MQTT/laptop/robot
    # state can change on their own), so this runs every tick, not just when
    # a new camera frame arrived - _set_icon()'s own throttling keeps this
    # cheap. Skipped entirely while the sad song is playing: the sad face
    # was already shown immediately by _on_event_lost, and we deliberately
    # leave it alone for the ~5s the song takes rather than let some other
    # status flicker interrupt it (same choice made in the sibling
    # "RQ-D2 Vision" app, for the same reason).
    if not _song_in_progress:
        _update_led(is_stale)

    if now - _last_status_print >= STATUS_INTERVAL:
        _last_status_print = now
        fps = _reader.get_fps() if _reader is not None else 0.0
        _safe_publish(TOPIC_UNOQ_STATUS, json.dumps({
            "state": "online",
            "camera": "missing" if is_stale else "ok",
            "fps": round(fps, 1),
        }))

        with _state_lock:
            phase, branch = _robot_state["phase"], _robot_state["branch"]
        if age is None:
            print(f"[main] state=no_frames_yet mqtt_connected={_mqtt_connected} laptop_online={_is_laptop_online()}")
        elif is_stale:
            print(f"[main] state=stale fps={fps:.1f} frame_age={age:.2f}s mqtt_connected={_mqtt_connected} laptop_online={_is_laptop_online()}")
        elif _last_result is not None:
            r = _last_result
            x_str = f"{r.x:+.2f}" if r.x is not None else "null"
            dist_str = f"{r.distance_cm:.1f}" if r.distance_cm is not None else "null"
            print(
                f"[main] visible={r.visible} x={x_str} distance_cm={dist_str} "
                f"groups={r.group_count} group_pixels={r.group_pixels} confidence={r.confidence:.2f} "
                f"fps={fps:.1f} frame_age={age:.2f}s proc={r.processing_ms:.1f}ms "
                f"mqtt_connected={_mqtt_connected} laptop_online={_is_laptop_online()} "
                f"robot_phase={phase} robot_branch={branch}"
            )
            _print_mask_diagnostics()

    time.sleep(1.0 / STREAM_FPS)


print("[main] looking for the ESP32-CAM...")
_try_discover_and_connect()

threading.Thread(target=_mqtt_worker, daemon=True).start()
threading.Thread(target=_rtt_worker, daemon=True).start()

App.run(user_loop=loop)
