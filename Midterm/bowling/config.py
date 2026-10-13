"""
Every tunable value for the bowling game lives here.

This file first loads ../config.py (the ping-pong config) and copies all of its
UPPER_CASE values, so the motor, IMU calibration, webcam, pose, MQTT and haptic
settings you already tuned for ping pong apply here too. Anything below overrides
or adds to them. ../config.py itself is never modified.

Values marked  # PLACEHOLDER  are guesses to tune on real hardware (see README.md).
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIDTERM = HERE.parent

# The bowling folder must stay first on sys.path (so `import config` is this file);
# Midterm goes after it so motor_imu / pose_tracker / mqtt_client can be reused.
if str(MIDTERM) not in sys.path:
    sys.path.append(str(MIDTERM))

_spec = importlib.util.spec_from_file_location("midterm_config", MIDTERM / "config.py")
_midterm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_midterm)
globals().update({k: v for k, v in vars(_midterm).items() if k.isupper() and k not in ("HERE", "MIDTERM")})
del _spec, _midterm

# =============================================================================
# Difficulty (the only menu choice, like Wii Sports)
# =============================================================================
DIFFICULTIES = {
    "Bumpers":    {"bumpers": True},
    "No Bumpers": {"bumpers": False},
}
DEFAULT_DIFFICULTY = "No Bumpers"

# =============================================================================
# Throw detection (IMU). A throw is one whole pendulum motion: backswing, forward
#   swing, follow-through. It starts when the thresholds trip, ends once the motor
#   has been quiet for THROW_END_QUIET_S, and the RELEASE is the moment of peak
#   strength (the bottom of the forward swing). Start thresholds default to the
#   ping-pong swing thresholds; run `python throw_probe.py` to tune.
# =============================================================================
THROW_MODE = SWING_MODE                               # "accel", "gyro", "either" or "both"
THROW_ACCEL_THRESHOLD_G = SWING_ACCEL_THRESHOLD_G     # dynamic |a| (g) that starts a throw
THROW_GYRO_THRESHOLD_RAW = SWING_GYRO_THRESHOLD_RAW   # |omega| (raw) that starts a throw
THROW_END_FRACTION = 0.5                 # "still moving" while above this fraction of the start thresholds
THROW_END_QUIET_S = 0.20                 # quiet this long = throw finished
THROW_MIN_S = 0.30                       # shorter motions are twitches, not throws
THROW_MAX_S = 2.0                        # a throw is forced to end after this long
THROW_COOLDOWN_S = 1.0                   # ignore motion this long after a throw ends

# Hook / spin from twisting the motor at release (Wii-style wrist twist)
SPIN_GYRO_AXIS = "y"                     # PLACEHOLDER "x", "y", "z" (the forearm axis) or None = no IMU spin
SPIN_SIGN = 1.0                          # PLACEHOLDER flip to -1.0 if twisting hooks the wrong way
SPIN_FULL_RAW = 4000.0                   # PLACEHOLDER mean twist rate (raw) that gives full spin
SPIN_DEADZONE = 0.12                     # |spin| below this counts as a straight ball
SPIN_WINDOW_S = 0.12                     # twist is averaged over +/- this long around release

# Throw strength -> ball speed
BALL_SPEED_AT_THRESHOLD = 6.0            # m/s for a throw that just tripped the thresholds (strength 1.0)
BALL_SPEED_PER_STRENGTH = 1.6            # m/s added per unit of strength above 1.0
BALL_SPEED_MIN = 4.5                     # m/s  (about 16 km/h)
BALL_SPEED_MAX = 10.0                    # m/s  (36 km/h, a very fast ball)

# =============================================================================
# Aim (webcam). Before the throw your wrist's sideways position is where you stand
# on the approach (Wii: D-pad). Sideways wrist motion during the forward swing angles
# the ball. A/D on the keyboard turns the aim, like Wii's A + D-pad.
# =============================================================================
SWING_AIM_WINDOW_S = 0.25                # wrist motion measured over this long before release...
SWING_AIM_AFTER_S = 0.08                 # ...until this long after it
SWING_AIM_DEG_PER_UNIT = 15.0            # degrees per unit of wrist dx (fraction of image width)
SWING_AIM_DEADZONE = 0.02                # wrist dx below this adds no angle
SWING_AIM_MAX_DEG = 2.5
PRESET_AIM_MAX_DEG = 4.0                 # A/D aim limit
PRESET_AIM_RATE_DEG_S = 1.5              # A/D aim turn rate
LAUNCH_ANGLE_MAX_DEG = 6.0               # hard limit on preset + swing angle
AIM_ARM_DELAY_S = 0.6                    # throws starting sooner than this after the bowler is ready are ignored

# =============================================================================
# Lane and pins (metres, regulation). x = lateral (left -, right +), y = along the lane
# (0 = foul line), z = height.
# =============================================================================
LANE_WIDTH_M = 1.054                     # 41.5 in, 39 boards
GUTTER_WIDTH_M = 0.235
HEAD_PIN_Y_M = 18.288                    # foul line to head pin, 60 ft
PIN_SPACING_M = 0.3048                   # 12 in between neighbouring pins
PIN_ROW_SPACING_M = 0.2639               # 12 in * cos(30 deg)
PIT_Y_M = HEAD_PIN_Y_M + 0.868           # end of the pin deck
APPROACH_LENGTH_M = 4.6
BALL_RADIUS_M = 0.109
BALL_MASS_KG = 6.8
PIN_RADIUS_M = 0.0605                    # at the belly
PIN_FALLEN_RADIUS_M = 0.11               # a lying pin sweeps a wider area
PIN_HEIGHT_M = 0.381
PIN_MASS_KG = 1.53

# Physics
SIM_DT_S = 1 / 400
OIL_LENGTH_M = 12.0                      # the ball barely hooks on the oiled front of the lane...
OIL_HOOK_FRACTION = 0.10                 # ...(this fraction of the full hook)...
HOOK_ACCEL_M_S2 = 0.8                    # ...and hooks this hard on the dry back end at full spin
LANE_DECEL_M_S2 = 0.12                   # rolling resistance
BALL_PIN_RESTITUTION = 0.75
PIN_PIN_RESTITUTION = 0.4
BUMPER_RESTITUTION = 0.6
PIN_TOPPLE_SPEED = 0.60                  # a hit giving a pin this much speed always knocks it over
PIN_WOBBLE_SPEED = 0.25                  # below this a hit never does; in between it's a chance
PIN_FRICTION_M_S2 = 12.0                 # deceleration of a sliding, fallen pin
PIN_FALL_TIME_S = 0.35                   # toppling animation
MAX_ROLL_S = 9.0                         # safety limit on one roll
SETTLE_SPEED = 0.05                      # pins slower than this count as settled

# =============================================================================
# Game flow
# =============================================================================
RESULT_DISPLAY_S = 2.0                   # how long a roll's result is shown
STRIKE_DISPLAY_S = 2.6
FRAMES_PER_GAME = 10

# =============================================================================
# View (pinhole camera behind the bowler that follows the ball down the lane)
# =============================================================================
VIEW_CAMERA_HEIGHT_M = 1.6 
VIEW_CAMERA_AIM_Y_M = -4.0               # camera position while aiming (behind the foul line)
VIEW_FOLLOW_BACK_M = 3.4                 # camera trails the ball by this much
VIEW_PIN_STOP_BACK_M = 4.6               # ...and stops this far in front of the head pin
VIEW_FOLLOW_SMOOTHING = 6.0              # 1/s, how fast the camera catches up
VIEW_FOCAL_PX = 950
VIEW_HORIZON_Y_PX = 250
VIEW_NEAR_M = 0.9

# =============================================================================
# Keyboard fallback
# =============================================================================
KEYBOARD_BOWLER_SPEED_M_S = 0.6          # ←/→ move the bowler
KEYBOARD_THROW_STRENGTH = 2.0            # Space throw strength (↑/↓ change it)
KEYBOARD_STRENGTH_RANGE = (1.0, 3.5)
KEYBOARD_STRENGTH_STEP = 0.25
KEYBOARD_SPIN = 0.7                      # Z / C held at release = hook left / right

# =============================================================================
# AprilTag selector (stub, inherited from ping pong)
# =============================================================================
APRILTAG_COMMANDS = {
    0: ("start", None),
    1: ("difficulty", "Bumpers"),
    2: ("difficulty", "No Bumpers"),
}
