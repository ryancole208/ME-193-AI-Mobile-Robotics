"""
Every tunable value for the minigolf game lives here.

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

# The minigolf folder must stay first on sys.path (so `import config` is this file);
# Midterm goes after it so motor_imu / pose_tracker / mqtt_client can be reused.
if str(MIDTERM) not in sys.path:
    sys.path.append(str(MIDTERM))

_spec = importlib.util.spec_from_file_location("midterm_config", MIDTERM / "config.py")
_midterm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_midterm)
globals().update({k: v for k, v in vars(_midterm).items() if k.isupper() and k not in ("HERE", "MIDTERM")})
del _spec, _midterm

# =============================================================================
# Difficulty (the only menu choice)
# =============================================================================
DIFFICULTIES = {
    "Easy": {"cup_radius_m": 0.085, "capture_speed": 1.6, "guide_m": 2.6, "obstacle_speed": 0.8},
    "Hard": {"cup_radius_m": 0.054, "capture_speed": 1.3, "guide_m": 0.9, "obstacle_speed": 1.25},
}
DEFAULT_DIFFICULTY = "Easy"

# =============================================================================
# Putt detection (IMU). The putt fires the moment you SWING QUICKLY: when the
#   swing-plane rotation rate trips the threshold (held for PUTT_CONFIRM_SAMPLES
#   samples, so a single bump doesn't count). POWER is the peak strength over the
#   next PUTT_PEAK_WINDOW_S. A slow backswing stays under the threshold. Twisting
#   the motor (see TWIST_*) is excluded from the swing rate, so aiming never putts.
#   A putt is gentler than a ping-pong swing, so the thresholds are 40% of the
#   ping-pong ones. Run `python putt_probe.py` to tune.
# =============================================================================
PUTT_MODE = "gyro"                                     # "accel", "gyro", "either" or "both"
PUTT_ACCEL_THRESHOLD_G = 0.4 * SWING_ACCEL_THRESHOLD_G  # PLACEHOLDER dynamic |a| (g) that fires a putt
PUTT_GYRO_THRESHOLD_RAW = 0.4 * SWING_GYRO_THRESHOLD_RAW  # PLACEHOLDER swing-plane |omega| (raw) that fires a putt
PUTT_CONFIRM_SAMPLES = 2                 # consecutive samples over the threshold needed to fire
PUTT_PEAK_WINDOW_S = 0.12                # power = peak strength over this long after firing
PUTT_COOLDOWN_S = 1.0                    # ignore motion this long after a putt (follow-through)

# =============================================================================
# Twist = putter angle (IMU). Twisting the motor turns the putter face. The gyro's
#   rotation rate around the twist axis is integrated into an angle; it is re-zeroed
#   whenever the game sets up a new putt (and with C), so however you're holding the
#   motor then is "straight at the suggested line".
#   TWIST_AXIS = "vertical" measures rotation around gravity (turning a putter face
#   with the motor hanging down, however it sits in your hand), so the pendulum
#   swing itself barely affects it. Or name a motor axis ("x", "y", "z").
# =============================================================================
TWIST_AXIS = "vertical"                  # PLACEHOLDER "vertical", "x", "y" or "z"
TWIST_SIGN = 1.0                         # PLACEHOLDER flip to -1.0 if twisting right turns the aim left
TWIST_DEG_PER_RAW_S = 0.1                # PLACEHOLDER gyro raw units -> deg/s (guess: raw = 0.1 deg/s)
TWIST_DEADZONE_RAW = 80.0                # rates below this are sensor noise (stops the angle drifting)
TWIST_FREEZE_FRACTION = 0.5              # twist isn't counted while the swing rate is above this x threshold
AIM_TWIST_GAIN = 1.0                     # aim degrees per degree of twist
AIM_TWIST_MAX_DEG = 60.0                 # twist can turn the aim this far either way from the suggested line

# Putt strength -> ball speed. Rolling distance is roughly v^2 / (2 * decel).
PUTT_SPEED_AT_THRESHOLD = 1.2            # PLACEHOLDER m/s for a putt that just tripped the thresholds
PUTT_SPEED_PER_STRENGTH = 1.0            # PLACEHOLDER m/s added per unit of strength above 1.0
PUTT_SPEED_MIN = 0.4                     # m/s  (a tap, about 10 cm)
PUTT_SPEED_MAX = 4.6                     # m/s  (a full-length smash)

# =============================================================================
# Aim (keyboard + webcam). ←/→ or A/D turn the aim (on top of the motor twist).
#   Sideways wrist motion around impact pushes/pulls the putt a few degrees.
#   The old wrist-joystick aim (hold your wrist left/right of centre to turn) is off
#   now that the twist sets the angle; set AIM_POSE_JOYSTICK = True to bring it back.
# =============================================================================
AIM_KEY_RATE_DEG_S = 45.0                # ←/→ or A/D turn rate
AIM_KEY_FINE_FACTOR = 0.25               # hold Shift for fine aim
AIM_POSE_JOYSTICK = False
AIM_POSE_DEADZONE = 0.30                 # wrist within this fraction of centre (-1..1) = no turning
AIM_POSE_RATE_DEG_S = 50.0               # turn rate with the wrist at the edge of POSE_X_RANGE
PUSH_WINDOW_S = 0.25                     # wrist motion measured over this long before impact...
PUSH_AFTER_S = 0.08                      # ...until this long after it
PUSH_DEG_PER_UNIT = 30.0                 # degrees per unit of wrist dx (fraction of image width)
PUSH_DEADZONE = 0.02                     # wrist dx below this adds no angle
PUSH_MAX_DEG = 6.0
AIM_ARM_DELAY_S = 0.6                    # putts starting sooner than this after the golfer is ready are ignored

# =============================================================================
# Course generation (metres). Each hole is a corridor of square cells from the tee
#   to the cup; walls run along every cell edge that borders the outside.
#   x = east, y = north, z = height. The tee always faces north (+y).
# =============================================================================
HOLES_PER_GAME = 5
CELL_M = 1.25                            # corridor width
WALL_HALF_THICK_M = 0.03                 # walls are capsules this thick around each cell edge
WALL_HEIGHT_M = 0.10
TEE_BACK_M = 0.30                        # tee sits this far behind the centre of the first cell
# Per hole: (min cells, max cells, min turns, max turns, obstacle budget)
HOLE_PLANS = [
    (5, 7, 0, 0, 0),
    (6, 9, 1, 1, 0),
    (8, 11, 1, 2, 3),
    (9, 12, 2, 2, 4),
    (10, 13, 2, 3, 5),
]
RUN_LENGTH = (2, 5)                      # straight run length between turns (cells)
WIDEN_CHANCE = 0.5                       # chance a long straight run gets a second lane (holes 3+)
MAX_STROKES = 8                          # pick up after this many strokes

# Halloween obstacles (holes 3+). Each is placed on a straight cell, off to one side,
# always leaving a gap the ball fits through.
PUMPKIN_RADIUS_M = 0.17
TOMBSTONE_LENGTH_M = 0.55                # juts in from a wall, blocking about half the lane
TOMBSTONE_HALF_THICK_M = 0.07
TOMBSTONE_HEIGHT_M = 0.45
GHOST_RADIUS_M = 0.17                    # floats back and forth across the lane
GHOST_PERIOD_S = (2.4, 3.6)
SPINNER_LENGTH_M = 0.50                  # bone spinner: half-length of the rotating bone
SPINNER_HALF_THICK_M = 0.035
SPINNER_OMEGA_RAD_S = (1.2, 2.0)
SLIME_RADIUS_M = 0.45                    # sticky green slime: much more rolling friction
SLIME_DECEL_FACTOR = 6.0
PIT_RADIUS_M = 0.27                      # witch's cauldron: +1 stroke, ball returns to where it was
PIT_PENALTY_STROKES = 1
OBSTACLE_RESTITUTION = 0.65
# Which obstacles may appear on each hole (index 0 = hole 1). The first one listed is the
# hole's signature obstacle and is always placed first; the rest are shuffled.
HOLE_OBSTACLES = [
    (),
    (),
    ("tombstone", "pumpkin", "slime"),
    ("ghost", "pumpkin", "tombstone", "pit"),
    ("spinner", "ghost", "tombstone", "pit", "slime"),
]

# =============================================================================
# Ball physics
# =============================================================================
GOLF_BALL_RADIUS_M = 0.0214
SIM_DT_S = 1 / 500
ROLL_DECEL_M_S2 = 0.55                   # constant rolling resistance...
ROLL_DRAG_PER_S = 0.12                   # ...plus this much per m/s of speed
WALL_RESTITUTION = 0.72
STOP_SPEED = 0.03                        # slower than this = the ball has stopped
CUP_DROP_S = 0.45                        # drop animation once holed
CUP_LIP_DEFLECT_DEG = 25.0               # a ball too fast for the cup is deflected by up to this much...
CUP_LIP_SPEED_KEEP = 0.75                # ...and keeps this fraction of its speed
MAX_PUTT_S = 25.0                        # safety limit on one putt

# =============================================================================
# Game flow
# =============================================================================
INTRO_S = 2.6                            # hole flyover before the first putt
RESULT_DISPLAY_S = 1.4                   # how long "Splash!" etc. are shown
HOLED_DISPLAY_S = 2.6                    # how long the hole's result is shown

# =============================================================================
# View (pinhole camera behind the ball, like Wii Sports putting)
# =============================================================================
VIEW_FOCAL_PX = 780
VIEW_CENTER_Y_PX = 400                   # screen row of the optical axis
VIEW_NEAR_M = 0.15
VIEW_AIM_BACK_M = 2.2                    # camera distance behind the ball while aiming
VIEW_AIM_HEIGHT_M = 1.05
VIEW_AIM_PITCH_DEG = 22.0
VIEW_ROLL_HEIGHT_M = 1.25                # while the ball rolls the camera turns to watch it...
VIEW_ROLL_PITCH_DEG = 28.0
VIEW_FOLLOW_MAX_M = 3.2                  # ...and follows once it gets farther than this
VIEW_SMOOTHING = 5.0                     # 1/s
VIEW_OVERVIEW_PITCH_DEG = 62.0           # hold V (and the hole intro) for an overhead view

# =============================================================================
# Keyboard fallback: hold Space to charge, release to putt
# =============================================================================
KEYBOARD_CHARGE_S = 1.4                  # meter fills in this long, then sweeps back down
KEYBOARD_STRENGTH_RANGE = (0.2, 4.4)     # strength at an empty / full meter

# =============================================================================
# AprilTag selector (stub, inherited from ping pong)
# =============================================================================
APRILTAG_COMMANDS = {
    0: ("start", None),
    1: ("difficulty", "Easy"),
    2: ("difficulty", "Hard"),
}
