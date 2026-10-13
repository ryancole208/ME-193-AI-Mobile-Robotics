"""
Every tunable value for the hardware bridge (bridge.py) lives here.

The bridge connects the paddle hardware to the Godot rebuild (PingPongRhythm/): Godot owns the
music, the clock, ball scheduling and judging; Python only reads the Double Motor IMU and the
webcam pose and sends them over UDP on localhost (one JSON object per packet, UTF-8).

    Python -> Godot  (GODOT_PORT):  state ~60 Hz, swing (at each swing), status (1 Hz)
    Python -> Godot  (GODOT_PORT):  tag (once per stable tag / once when lost)
    Godot -> Python  (BRIDGE_PORT): haptic, zero, score

This file imports nothing, so `bridge.py --no-hardware` starts instantly. The motor, IMU,
webcam and pose settings are NOT duplicated here -- they stay in ../config.py
(read through rhythm/config.py) and are only loaded when the bridge needs them.
"""

# =============================================================================
# UDP (localhost)
# =============================================================================
GODOT_HOST = "127.0.0.1"
GODOT_PORT = 4242                      # Godot listens here (state / swing / status)
BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 4243                     # the bridge listens here (haptic / zero / score)

# =============================================================================
# Send rates
# =============================================================================
STATUS_INTERVAL_S = 1.0                # {"type":"status"} once per second
STATE_RATE_HZ = 60                     # {"type":"state"} rate (hardware mode)
PRINT_STATE_INTERVAL_S = 1.0           # print the latest state packet this often (sending stays at STATE_RATE_HZ)

# =============================================================================
# Feature flags
# =============================================================================
MOTOR_ENABLED = True                   # False = never connect to the Double Motor (same as --no-motor)
POSE_ENABLED = True                    # False = never open the webcam (same as --no-pose)
POSE_PREVIEW = False                   # True = the pose thread also builds its preview image (unused by the bridge)
HAPTIC_ENABLED = True                  # False = ignore {"type":"haptic"} from Godot
TAGS_ENABLED = True                    # False = no AprilTag song selection (same as --no-tags)
DISPLAY_ENABLED = True                 # False = no camera view window (same as --no-display)
MQTT_ENABLED = True                    # False = never touch the network (same as --no-mqtt)
MQTT_TOPIC = "ME193/Rogers/Ryan"       # the bridge's score topic (consecutive-hit record); broker is in ../../mqttlib.py
PRINT_SENDS = True                     # print every status and swing packet (state: see PRINT_STATE_INTERVAL_S)

# =============================================================================
# Paddle
# =============================================================================
HAPTIC_PULSE_MS = 80                  # fixed pulse length for every {"type":"haptic"} (pygame's longest
                                       # pulse); the IMU ignores swings for this + RHYTHM_HAPTIC_SETTLE_MS

# =============================================================================
# Paddle tilt and face (paddle_tracker.py) -> state.roll and state.face
# =============================================================================
# Neutral grip: paddle upright, BLACK face toward the screen.
#   TILT = rotating it in the screen plane (windshield wiper)  -> state.roll, + = top to your right
#   FLIP = twisting the handle so the other face points at the screen -> state.face
#   SWING = the forward hit (swing detection is ../config.py's, unchanged)
# Axis mapping: run  python bridge.py --imu-debug  and copy the "grav" readings ("+x", "-z", ...):
TRACK_NEUTRAL_GRAVITY = "+x"           # PLACEHOLDER gravity axis in the neutral grip (= along the handle)
TRACK_TILT_RIGHT_GRAVITY = "+y"        # PLACEHOLDER gravity axis after tilting the top 90 deg to your RIGHT
TRACK_GYRO_DEG_PER_RAW = 0.1           # PLACEHOLDER gyro raw -> deg/s (--imu-debug prints a "scale?" check)
TRACK_GYRO_SIGN = 1.0                  # -1 if --imu-debug shows gyro tilt and accel tilt with opposite signs

# Tilt filter and output
TRACK_ACCEL_TAU_S = 0.20               # complementary filter: the tilt reaches the accelerometer's in ~this
TRACK_ACCEL_TOLERANCE_G = 0.20         # accel trusted only while | |a| - 1 g | is below this
TRACK_MIN_PLANE_GRAVITY = 0.5          # ...and at least this much gravity lies in the screen plane (x 1 g)
TRACK_DEADZONE_DEG = 4.0               # tilts smaller than this show as straight
TRACK_GAIN = 0.6                       # shown tilt = (real tilt - deadzone) x this (reduced, as before)
TRACK_TILT_MAX_DEG = 90.0              # clamp (Godot's Paddle max_roll_degrees clamps again)
TRACK_SMOOTH_S = 0.05                  # light output smoothing (time constant; 0 = none)

# Flip (face)
TRACK_FLIP_TO_YELLOW_DEG = 110.0       # black -> yellow once the twist passes this...
TRACK_FLIP_TO_BLACK_DEG = 70.0         # ...and back to black below this (hysteresis: no flicker edge-on)
TRACK_FLIP_DEBOUNCE_S = 0.06           # the new face must hold this long before it is sent
TRACK_FLIP_SETTLE_S = 2.0              # while still, the twist settles to the face shown (gyro drift)
TRACK_GYRO_DEADZONE_DPS = 2.0          # twist rates below this are ignored (noise)

# Swings, gyro bias, stillness
TRACK_SWING_HOLD_S = 0.25              # tilt and face hold during a swing / haptic pulse and this long after
TRACK_SWING_REWIND_S = 0.12            # at swing onset, go back to the tilt/face from this long before
TRACK_STILL_GYRO_DPS = 20.0            # slower than this (and |a| ~ 1 g) = still: face settling allowed
TRACK_BIAS_STILL_DPS = 6.0             # slower than this = gyro bias is learnt...
TRACK_BIAS_TAU_S = 4.0                 # ...with this time constant

# --imu-debug
IMU_DEBUG_RATE_HZ = 10                 # debug lines per second

# =============================================================================
# AprilTag song selection (tag16h5, read from the webcam frames PoseTracker already grabs;
# with --no-pose the tag scanner opens the webcam itself)
# =============================================================================
ALLOWED_TAG_IDS = (0, 4, 6, 7, 8, 9, 10, 11)   # the printed song tags; any other ID is ignored
TAG_RATE_HZ = 10                       # detection passes per second (at most)
TAG_STABLE_FRAMES = 3                  # the same ID in this many passes in a row -> {"type":"tag","id":N}
TAG_MIN_SIDE_PX = 30                   # ignore tags whose shortest side is smaller (far / false positives)
TAG_LOST_S = 1.0                       # no stable tag seen this long -> {"type":"tag","id":-1}
TAG_ERROR_CORRECTION = 0.0             # cv2.aruco errorCorrectionRate: 0 = exact codes only (OpenCV's
                                       # default 0.6 "repairs" bits -- the main source of 16h5 false positives)

# =============================================================================
# Camera view (a window with what the webcam sees: pose overlay + tag outlines and IDs)
# =============================================================================
DISPLAY_FPS = 15                       # window refreshes per second (its own thread)
DISPLAY_WIDTH = 640                    # px; the camera image is scaled to this width
DISPLAY_TITLE = "Bridge camera (Esc closes this view)"
