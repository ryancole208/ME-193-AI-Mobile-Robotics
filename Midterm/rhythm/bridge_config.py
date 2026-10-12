"""
Every tunable value for the hardware bridge (bridge.py) lives here.

The bridge connects the paddle hardware to the Godot rebuild (PingPongRhythm/): Godot owns the
music, the clock, ball scheduling and judging; Python only reads the Double Motor IMU and the
webcam pose and sends them over UDP on localhost (one JSON object per packet, UTF-8).

    Python -> Godot  (GODOT_PORT):  state ~60 Hz, swing (at each swing), status (1 Hz)
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
MQTT_ENABLED = True                    # False = never touch the network (same as --no-mqtt)
MQTT_TOPIC = "ME193/Rogers/Ryan"       # the bridge's score topic (consecutive-hit record); broker is in ../../mqttlib.py
PRINT_SENDS = True                     # print every status and swing packet (state: see PRINT_STATE_INTERVAL_S)

# =============================================================================
# Paddle
# =============================================================================
# "face" = the side physically toward the screen, from the RAW roll (the calmed roll is capped
# at ROLL_DISPLAY_MAX_DEG and fades to 0, so it can never flip). Neutral grip = black toward the
# screen; past this twist the yellow side faces the screen.
FACE_FLIP_DEG = 90.0
HAPTIC_PULSE_MS = 80                   # fixed pulse length for every {"type":"haptic"} (pygame's longest
                                       # pulse); the IMU ignores swings for this + RHYTHM_HAPTIC_SETTLE_MS
