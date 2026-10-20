"""
Every tunable value for the sumo robot, in one place.

Values marked MEASURE could not be checked without spinning the wheels; run
`python hardware_check.py` (see README) and copy the numbers it prints here.
Speeds are LEGO percentages (-100..100) unless stated otherwise.
"""

import legoeducation as le

# --------------------------------------------------------------------------
# LEGO hardware (all three devices share one Connection Card)
# --------------------------------------------------------------------------
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = "1131"
NOTIFICATION_MS = 20          # requested sensor update period (library allows 15..1000);
                              # measured ~30 ms actual on this hardware
BLE_RECONNECT_INTERVAL_S = 2.0

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
TICK_HZ = 50                  # behavior-tree ticks per second (spec: 20-50 Hz)
STATUS_PRINT_S = 1.0          # print a [status] line this often on real hardware (None = off)
MOTOR_RESEND_S = 0.2         # re-send an unchanged motor command this often, so nothing
                              # else (e.g. a Controller paired directly) can take over

# --------------------------------------------------------------------------
# Round start
# --------------------------------------------------------------------------
START_THRESHOLD = 70          # |lever %| that starts the round from the "armed" phase
LEVER_CENTER = 10             # |lever %| at or below this counts as "centered"
REQUIRE_RECENTER_AFTER_START = True   # ignore teleop until both levers return to center once
COUNTDOWN_S = 0               # seconds between the start push and the round (0 = none)

# --------------------------------------------------------------------------
# Branch 0: out of ring
# --------------------------------------------------------------------------
SONG_DONE_TIMEOUT_S = 8.0     # max wait for event/song_done before exiting
CROSS_FORWARD_IS_OUT = True   # blue then floor while moving forward => front crossed => OUT
LEAN_WHEN_UNCERTAIN = "in"    # "in": treat an uncertain-direction crossing as staying inside
                              # (your choice); "out": end the round instead
EDGE_OVERSHOOT_GRACE_S = 0.5  # a forward crossing this soon after first touching the tape is
                              # momentum carrying us over while braking, NOT "out": keep reversing
OVERSHOOT_RECOVER_S = 2.0     # ...but if we haven't come back over the tape within this long
                              # after the overshoot, we really are out
RETAPE_WINDOW_S = 1.5         # blue while moving BACKWARD this soon after backing off the tape is
                              # the SAME tape again (flicker at its edge), not "pushed out"

# --------------------------------------------------------------------------
# Branch 1: edge reflex
# --------------------------------------------------------------------------
REVERSE_TIME_S = 1.0
REVERSE_SPEED = 70

# --------------------------------------------------------------------------
# Branch 2: chase
# --------------------------------------------------------------------------
VISION_FRESH_S = 0.5          # max age of the last vision message, by the LAPTOP's clock
VISION_MAX_FRAME_AGE_MS = 500 # max frame_age_ms reported by the UNO Q
VISION_MIN_CONFIDENCE = 0.5
CHASE_X_SIGN = +1             # +1: x > 0 means the opponent is to our RIGHT (flip if it turns away)
CHASE_STEER_GAIN = 40         # % wheel-speed difference per unit of x (proportional control)
CHASE_MIN_SPEED = 35          # forward speed when the opponent is far / off-center
CHASE_MAX_SPEED = 100         # forward speed when it is close and centered
CHASE_NEAR_CM = 15            # at or closer than this => "close"
CHASE_FAR_CM = 80             # at or farther than this => "far" (null distance counts as far)

# --------------------------------------------------------------------------
# Branch 3: teleop
# --------------------------------------------------------------------------
TELEOP_DEADBAND = 12          # |lever %| below this is ignored
TELEOP_MAX_SPEED = 100        # wheel speed at full lever

# --------------------------------------------------------------------------
# Staleness (any of these => that input is ignored)
# --------------------------------------------------------------------------
COLOR_STALE_S = 0.25          # no color sample for this long => stop (can't see the edge)
CONTROLLER_STALE_S = 0.5
MOTOR_STALE_S = 0.5           # wheel-speed / IMU data

# --------------------------------------------------------------------------
# Color classification (KNN, trained by calibrate_color.py)
# --------------------------------------------------------------------------
COLOR_SAMPLES_FILE = "color_samples.json"
KNN_K = 5
KNN_MAX_DISTANCE = 4.0        # nearest-sample distance (in feature std-devs) above which a
                              # reading is "unknown" (None disables)
DEBOUNCE_SAMPLES = 2          # consecutive identical readings needed to change floor<->blue
                              # (2 x ~30 ms; also adds to the overshoot past the tape)
TAPE_WIDTH_M = 0.03           # MEASURE: width of the blue tape
COLOR_SAMPLE_PERIOD_S = 0.03  # measured time between Color Sensor notifications
# Speed-aware debounce: if we move so fast that the tape can't produce DEBOUNCE_SAMPLES
# readings (tape_width / |v| / sample_period < DEBOUNCE_SAMPLES + 1), one reading is
# accepted, so a hard shove can't carry us across the tape unnoticed.
# Above TAPE_WIDTH_M / COLOR_SAMPLE_PERIOD_S (1 m/s for 3 cm) the tape can be missed entirely.

# --------------------------------------------------------------------------
# HOW to decide "floor" vs "tape". Pick one:
#   "knn"  : KNN on raw RGB + reflection, trained on YOUR floor and tape with
#            calibrate_color.py (needs color_samples.json; the program refuses to
#            start without it). Learns your exact tape/lighting/sensor height and
#            says "unknown" for readings unlike anything recorded.
#   "lego" : the sensor's own color name: any of TAPE_LEGO_COLORS = tape,
#            anything else = floor. No calibration needed.
# Both go through the same debouncer and line tracker.
# --------------------------------------------------------------------------
COLOR_METHOD = "knn"
TAPE_LEGO_COLORS = (le.LEGO_COLOR_BLUE, le.LEGO_COLOR_AZURE)   # "lego" mode: colors that mean tape

# --------------------------------------------------------------------------
# Motion direction estimate (wheels + IMU complementary filter, see motion.py)
# --------------------------------------------------------------------------
# How the Double Motor sits in the robot, relative to the FRONT (= the color sensor side).
# "Robot left/right" means as seen from behind the robot, looking toward the sensor.
DRIVE_INVERTED = False        # MEASURE (--spin): True if movement_move_tank(+, +) drives AWAY
                              # from the color sensor (Double Motor mounted backward)
DRIVE_SWAP_SIDES = False      # MEASURE (--spin): True if the motor's "left" side (motor[0])
                              # is the ROBOT's right wheel
LEFT_WHEEL_SIGN = -1          # MEASURE: sign of the robot-left wheel's .speed when moving forward
RIGHT_WHEEL_SIGN = +1         # MEASURE: sign of the robot-right wheel's .speed when moving forward
WHEEL_SPEED_TO_MPS = 0.0028   # MEASURE: m/s of ground speed per unit of motor `speed`
IMU_FORWARD_AXIS = "x"        # MEASURE: accelerometer axis that points along "forward"
IMU_FORWARD_SIGN = +1         # MEASURE: +1 if that axis increases when pushed forward
IMU_RAW_PER_G = 1000.0        # accelerometer counts per g (measured ~992-1000 at rest)
FUSION_TAU_S = 0.3            # normal mode: how quickly the estimate settles back on the wheels
SLIP_MPS = 0.15               # wheels vs IMU disagreement that means "wheels are slipping"
SLIP_MAX_S = 3.0              # slip mode (pure IMU integration) is forced to end after this long
IMU_TILT_FIELD = None         # MEASURE: "pitch" or "roll" (the tilt angle that changes when the
                              # nose is lifted), or None = no tilt compensation
IMU_TILT_GAIN = 0.0           # MEASURE: forward-axis accel counts per decidegree of that tilt
DIRECTION_MIN_MPS = 0.03      # |fused speed| below this => direction UNCERTAIN
IMU_REST_BIAS_ALPHA = 0.02    # how fast the accel bias re-zeroes while sitting still

# --------------------------------------------------------------------------
# Controller direct-drive watchdog (see lego_io.py)
# --------------------------------------------------------------------------
DIRECT_DRIVE_SPEED = 5        # wheel |speed| that counts as "moving" while we command 0
DIRECT_DRIVE_TIME_S = 0.3

# --------------------------------------------------------------------------
# MQTT (broker host/port live in ../mqttlib.py)
# --------------------------------------------------------------------------
TOPIC_PREFIX = "ME193/Rogers/RQ-D2/"
T_VISION = TOPIC_PREFIX + "vision/opponent"
T_UNOQ_STATUS = TOPIC_PREFIX + "unoq/status"
T_LAPTOP_STATUS = TOPIC_PREFIX + "laptop/status"
T_ROBOT_STATE = TOPIC_PREFIX + "robot/state"
T_EVENT_LOST = TOPIC_PREFIX + "event/lost"
T_SONG_DONE = TOPIC_PREFIX + "event/song_done"
