"""All tunable values for the RQ-D2 vision/status app, in one place.

This is the file to edit when you want to retune the robot - change the
ring size, how quickly it reacts, which MQTT broker/topics it talks to,
and so on - without touching main.py, mqttlib.py, or the sketch.

Two different kinds of settings live here:

  - LIVE settings (the detection numbers below) are read by main.py at
    startup and pushed into the sketch over the Bridge with a single
    Bridge.call("configure", ...). Change one of these, restart the app,
    and the new value takes effect - no need to touch sketch/sketch.ino.

  - REFERENCE settings (pin numbers, ping rate) are compiled into
    sketch/sketch.ino itself, because they're tied to the physical
    wiring/hardware timing and the Bridge can't rewire a pin. They're
    still listed here, matching the .ino exactly, so every tunable the
    assignment asks for is visible in one place - but changing them means
    editing sketch/sketch.ino and re-uploading, not just editing this file.
"""

# ===================== MQTT broker =====================
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883

# All topics for this robot live under this prefix.
TOPIC_PREFIX = "ME193/Rogers/RQ-D2/"

TOPIC_VISION = TOPIC_PREFIX + "vision/opponent"
TOPIC_UNOQ_STATUS = TOPIC_PREFIX + "unoq/status"
TOPIC_LAPTOP_STATUS = TOPIC_PREFIX + "laptop/status"
TOPIC_ROBOT_STATE = TOPIC_PREFIX + "robot/state"
TOPIC_EVENT_LOST = TOPIC_PREFIX + "event/lost"
TOPIC_EVENT_SONG_DONE = TOPIC_PREFIX + "event/song_done"

# ===================== LIVE: detection tuning =====================
# Pushed to the sketch at startup via Bridge.call("configure", ...).
# See the README's "Tuning the maximum range" section before changing these.

# Readings closer than this are treated as sensor noise (HC-SR04's own
# blind zone / an echo bouncing off the robot's own chassis), not a real
# object. 5cm matches this robot's sensor mounting (confirmed with Ryan).
MIN_VALID_DISTANCE_CM = 5.0

# "Detect" threshold: the ring's diameter. Anything farther than this is
# outside the ring (a wall, a person, another team's robot) and should be
# ignored. 100cm matches this competition's ring (confirmed with Ryan).
RING_DIAMETER_CM = 100.0

# "Lose" threshold (hysteresis): once something is detected, it has to
# retreat past RING_DIAMETER_CM + this margin before we call it lost.
# Without this gap, an opponent sitting exactly at the ring's edge would
# flicker between detected/not-detected on every small sensor wobble.
HYSTERESIS_MARGIN_CM = 15.0
LOSE_RANGE_CM = RING_DIAMETER_CM + HYSTERESIS_MARGIN_CM

# How many consecutive ultrasonic readings (at ~16Hz, see PING_INTERVAL_MS
# below) must agree before the detected/not-detected state is allowed to
# flip. Higher = steadier but slower to react; lower = snappier but more
# prone to flickering on noisy readings.
DEBOUNCE_COUNT = 3

# Number of distinct box sizes the "chasing" LED icon can show (0=farthest/
# smallest, LED_CHASE_SIZE_STEPS-1=closest/largest). Must match the length
# of CHASE_HALF_SIZE in sketch/sketch.ino.
LED_CHASE_SIZE_STEPS = 4

# ===================== REFERENCE: hardware/timing =====================
# Compiled into sketch/sketch.ino - listed here for visibility, but you
# must edit the .ino (and re-upload) to actually change these.
PIN_TRIG = 9     # HC-SR04 Trig
PIN_ECHO = 8     # HC-SR04 Echo (see README for the voltage-protection note)
PIN_BUZZER = 5   # passive buzzer signal
PING_INTERVAL_MS = 60    # ~16.7 Hz; HC-SR04 needs >=60ms between pings
MEDIAN_WINDOW = 5         # "median of the last 5 readings" per the assignment

# ===================== Status / logging =====================
STATUS_LOG_INTERVAL_S = 1.0   # how often to print a status line and publish unoq/status

# How long without ANY update from the sketch (total Bridge/sketch silence,
# not just an ultrasonic timeout - that's handled on the sketch side)
# before main.py itself treats the sensor as unresponsive.
SENSOR_SILENCE_TIMEOUT_S = 2.0

# How long to wait before retrying a dropped/failed MQTT connection,
# doubling up to the cap each time it fails again (so a brief broker hiccup
# retries fast, but we don't hammer a broker that's genuinely down).
MQTT_RECONNECT_INITIAL_DELAY_S = 1.0
MQTT_RECONNECT_MAX_DELAY_S = 30.0
