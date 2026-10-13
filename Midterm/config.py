"""
Every tunable value for the human side of the ping-pong game lives here.

Values marked  # PLACEHOLDER  are sensible guesses that you should tune on real
hardware (see README.md -> "Tuning"). Units are given in the comments.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

# =============================================================================
# MQTT  (broker host/port stay in ../mqttlib.py -- do not duplicate them here)
# =============================================================================
MQTT_TOPIC_BASE = "ME193/RQ-D2"          # score topic = f"{MQTT_TOPIC_BASE}/{PLAYER_NAME}"
PLAYER_NAME = "Name"                     # the <Name> segment of the score topic
MQTT_RETRY_INTERVAL_S = 3.0              # wait between initial connect attempts
MQTT_ENABLED = True                      # False = never touch the network

# =============================================================================
# BLE / Double Motor
# =============================================================================
MOTOR_ENABLED = True                     # False = skip the motor entirely (keyboard swings only)
BLE_SCAN_TIMEOUT_S = 4.0                 # length of each scan; the nearest (strongest RSSI) motor wins
BLE_DEVICE_NAME_FILTER = None            # e.g. "Double Motor"; None = any Double Motor (product-ID filtered)
BLE_CARD_COLOR = None                    # later, with a Connection Card: e.g. legoeducation.LEGO_COLOR_RED
BLE_CARD_SERIAL = None                   # later, with a Connection Card: e.g. "0049" (string keeps leading zeros)
BLE_RECONNECT_INTERVAL_S = 2.0           # wait between scan/connect attempts after a failure or drop
BLE_NOTIFICATION_MS = 20                 # IMU notification period; library allows 15..1000 ms

# =============================================================================
# IMU / swing detection
#   Units: the library does not document IMU scaling. Acceleration is normalised
#   to g by measuring gravity at rest right after connecting (hold the paddle
#   still for ~0.5 s). Gyro thresholds are in the device's RAW units -- run
#   `python imu_probe.py` and swing to see typical values.
# =============================================================================
IMU_CALIBRATION_SAMPLES = 25             # samples averaged at rest to measure 1 g in raw units
IMU_CALIBRATION_MAX_STD = 0.05           # max relative std-dev of |a| during calibration (else fallback)
IMU_ACCEL_RAW_PER_G_FALLBACK = 1000.0    # PLACEHOLDER raw units per g if calibration fails (milli-g guess)
IMU_GRAVITY_ALPHA = 0.95                 # low-pass factor for gravity estimate (0..1, closer to 1 = slower)
SWING_MODE = "either"                    # "accel", "gyro", "either" or "both" thresholds must trip
SWING_ACCEL_THRESHOLD_G = 1.2            # PLACEHOLDER dynamic (gravity-removed) |a| in g
SWING_GYRO_THRESHOLD_RAW = 3000.0        # PLACEHOLDER |omega| in raw gyro units
SWING_COOLDOWN_S = 0.40                  # ignore new swings for this long after one starts
SWING_PEAK_WINDOW_S = 0.12               # how long after onset the peak strength is measured
SWING_EXCLUDE_ROLL_GYRO = True           # twisting the handle (roll axis) never counts as a swing

# =============================================================================
# Handle roll (twist) -- the hub IS the paddle handle
#   Run `python imu_axis_test.py` and copy the suggested values here.
#   Roll = rotation around the handle's long axis. Positive roll = the face
#   toward the SCREEN (black, in the neutral grip) turns toward your RIGHT
#   (clockwise seen from above). Zero = neutral grip (face vertical, BLACK side
#   toward the screen, YELLOW toward you), captured at game start and with C.
# =============================================================================
IMU_ROLL_AXIS = "x"                      # PLACEHOLDER hub axis along the handle: "x", "y" or "z"
IMU_ROLL_SIGN = 1.0                      # +1 / -1 so the screen-side face turning right is positive (verified)
IMU_ACCEL_ROLL_SIGN = 1.0                # PLACEHOLDER +1 / -1 so the accelerometer roll agrees with the gyro
IMU_GYRO_DEG_PER_RAW = 0.1               # PLACEHOLDER gyro raw units -> deg/s
ROLL_FILTER_ALPHA = 0.98                 # complementary filter: gyro weight per 20 ms (accel gets the rest)
ROLL_STILL_TOLERANCE_G = 0.15            # accel correction only while | |a| - 1 g | is below this
ROLL_MIN_GRAVITY_FRACTION = 0.35         # ...and at least this much of gravity lies across the handle
                                         #    (a vertical handle cannot measure roll from gravity)
ROLL_GYRO_DEADZONE_DPS = 4.0             # roll rates below this are noise (limits drift)
ROLL_HISTORY_S = 1.0                     # roll history kept to look up the angle at swing onset

# =============================================================================
# Spin
#   Spin is two numbers in [-SPIN_MAX, SPIN_MAX]:
#     side  > 0 curves the ball to the right (+x), < 0 to the left
#     top   > 0 topspin (dips, kicks forward), < 0 backspin (floats, bounces short)
# =============================================================================
SPIN_INPUT = "rate"                      # "rate" (twist speed in the swing), "angle" (roll at swing), "both"
SPIN_RATE_DEADZONE_DPS = 120.0           # PLACEHOLDER twist speed below this = no sidespin
SPIN_RATE_FULL_DPS = 700.0               # PLACEHOLDER twist speed that gives full sidespin (1.0)
SPIN_ANGLE_DEADZONE_DEG = 12.0           # roll angle below this = no sidespin
SPIN_ANGLE_FULL_DEG = 50.0               # roll angle that gives full sidespin
SPIN_SIDE_SIGN = 1.0                     # flip if twisting right should curve the ball left
SPIN_TOP_SOURCE = "imu"                  # "imu" (vertical swing motion) or "off"
SPIN_TOP_DEADZONE_G = 0.5                # PLACEHOLDER mean up/down accel in the swing below this = no top/back
SPIN_TOP_FULL_G = 2.0                    # PLACEHOLDER up/down accel that gives full top/backspin
SPIN_SENSITIVITY = 1.0                   # multiplies all IMU spin
SPIN_MAX = 1.0                           # clamp on each spin component (keeps the ball playable)

MAGNUS_STRENGTH = 0.12                   # sideways accel (m/s^2) per unit sidespin per m/s of ball speed
TOPSPIN_ARC_MULT = 0.35                  # topspin lowers the arc by this fraction at full spin
BACKSPIN_ARC_MULT = 0.30                 # backspin raises (floats) the arc by this fraction
TOPSPIN_BOUNCE_SHIFT = 0.10              # topspin bounces this much earlier (fraction of the flight)
BACKSPIN_BOUNCE_SHIFT = 0.06             # backspin bounces this much later
TOPSPIN_KICK = 0.30                      # speed after the bounce x (1 + kick * spin)
BACKSPIN_CHECK = 0.30                    # speed after the bounce x (1 - check * spin)
TOPSPIN_SECOND_ARC_MULT = 0.6            # height of the bounce arc x (1 - mult * spin): low, skidding
BACKSPIN_SECOND_ARC_MULT = 0.3           # height of the bounce arc x (1 + mult * spin): sits up

SPIN_LABEL_THRESHOLD = 0.55              # show "TOPSPIN!" / "CURVE!" above this spin
SPIN_LABEL_S = 1.0

KEYBOARD_SPIN_AMOUNT = 0.8               # spin from Q/E (side) and W/S (top/back) held during Space
KEYBOARD_ROLL_DISPLAY_DEG = 40.0         # on-screen paddle twist while Q/E are held

# =============================================================================
# Haptic feedback (pulse the motor on a successful hit)
# =============================================================================
HAPTIC_ENABLED = True
HAPTIC_PULSE_MS = 80                     # pulse duration
HAPTIC_SPEED = 60                        # % speed, -100..100
HAPTIC_MOTOR = "both"                    # "left", "right" or "both"
HAPTIC_SETTLE_S = 0.15                   # after a pulse ends, ignore the IMU this much longer
                                         # (no swing detection, no accel roll correction)

# =============================================================================
# Pose tracking (MediaPipe Tasks PoseLandmarker)
# =============================================================================
CAMERA_ENABLED = True                    # False = no webcam (keyboard paddle only)
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DOMINANT_HAND = "right"                  # "right" or "left"
MIRROR_CAMERA = True                     # mirror so moving right moves the paddle right
POSE_MODEL_VARIANT = "lite"              # "lite", "full" or "heavy" (heavier = slower, more accurate)
POSE_MODEL_PATH = HERE / "models" / f"pose_landmarker_{POSE_MODEL_VARIANT}.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  f"pose_landmarker_{POSE_MODEL_VARIANT}/float16/latest/"
                  f"pose_landmarker_{POSE_MODEL_VARIANT}.task")
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_MIN_WRIST_VISIBILITY = 0.4          # below this the wrist counts as "not tracked"
POSE_SMOOTHING = 0.5                     # EMA factor for paddle position (0 = none, 0.9 = heavy)
POSE_X_RANGE = (0.15, 0.85)              # mirrored-image wrist x mapped to the table's left..right edge
POSE_DIRECTION_FRAMES = 6                # frames of wrist history used for swing direction
POSE_DIRECTION_THRESHOLD = 0.04          # min wrist dx (fraction of image width) to count as left/right
POSE_HISTORY_S = 1.5                     # seconds of wrist history kept

# =============================================================================
# Game
# =============================================================================
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 720
FPS = 60

# Table (metres, regulation size). x = lateral (left -/right +), y = along the table
# (0 = my end, TABLE_LENGTH = opponent end), z = height above the table.
TABLE_WIDTH_M = 1.525
TABLE_LENGTH_M = 2.74
PLAYER_HIT_Y_M = -0.15                   # my paddle's plane (just behind my end)
OPPONENT_HIT_Y_M = TABLE_LENGTH_M + 0.15
BALL_RADIUS_M = 0.02
BALL_ARC_HEIGHT_M = 0.30                 # cosmetic arc height of each flight

# 3D view camera (VIEW_* = the on-screen camera; CAMERA_* above is the webcam).
# Pinhole camera behind and above my end of the table, looking down the table.
VIEW_CAMERA_X_M = 0.0                    # lateral camera position
VIEW_CAMERA_BACK_M = 2.2                 # distance behind my end of the table
VIEW_CAMERA_HEIGHT_M = 0.92              # height above the table surface
VIEW_CAMERA_PITCH_DEG = 9.0              # look-down angle (0 = level)
VIEW_FOV_DEG = 62.0                      # horizontal field of view
VIEW_CENTER_Y_PX = 320                   # screen row the camera's optical axis passes through
VIEW_NEAR_M = 0.05                       # near clipping plane

TABLE_HEIGHT_M = 0.76                    # table surface above the floor (regulation)
TABLE_THICKNESS_M = 0.04
NET_HEIGHT_M = 0.1525
NET_OVERHANG_M = 0.1525                  # how far the net posts stick out past the table sides

# Difficulty = ball speed along the table (m/s). "Dynamic" = the RL opponent (see the
# Dynamic section at the end) picks every shot's speed; ball_speed is only the base for MY returns.
DIFFICULTIES = {
    "Easy":    {"ball_speed": 2.2},
    "Medium":  {"ball_speed": 3.2},
    "Hard":    {"ball_speed": 4.5},
    "Dynamic": {"ball_speed": 3.2},
}
DYNAMIC_DIFFICULTY = "Dynamic"           # the difficulty name that switches to the RL opponent
DEFAULT_DIFFICULTY = "Easy"

HIT_WINDOW_BEFORE_S = 0.25               # swing may start this long before ball reaches my paddle plane
HIT_WINDOW_AFTER_S = 0.20                # ...or this long after
LATERAL_HIT_TOLERANCE_M = 0.25           # |paddle x - ball x| allowed for a hit

RETURN_ANGLE_LEFT_RIGHT_DEG = 12.0       # return angle for a left/right swing
RETURN_ANGLE_CENTER_JITTER_DEG = 3.0     # random spread for a centre swing
RETURN_ANGLE_MAX_DEG = 20.0              # hard limit on any return angle
RETURN_SPEED_MIN_MULT = 0.8              # return speed = difficulty speed * mult, clamped to [min, max]
RETURN_SPEED_MAX_MULT = 1.5
RETURN_STRENGTH_GAIN = 0.35              # mult = 1 + gain * (normalised strength - 1)

SERVE_DELAY_S = 1.2                      # pause before each serve
MISS_DISPLAY_S = 1.0                     # how long "MISS" is shown before the next serve

# Simulated opponent
OPPONENT_PLACEMENT_SPREAD_M = 0.45       # random lateral spread of opponent shots (+/- around centre)
OPPONENT_REACTION_S = 0.05               # delay between ball arrival and its return
OPPONENT_MISS_PROBABILITY = 0.0          # chance the simulated opponent misses
OPPONENT_TIMEOUT_S = 2.0                 # no answer this long after arrival = opponent missed (robot/MQTT later)
# Spin penalty: a ball with spin magnitude m (0..SPIN_MAX) adds miss chance and placement scatter.
# Bigger factor = the opponent struggles more. Easy hurts the opponent most, so Easy stays easiest for you.
OPPONENT_SPIN_PENALTY = {"Easy": 1.0, "Medium": 0.6, "Hard": 0.3, "Dynamic": 0.6}
OPPONENT_SPIN_MISS_MAX = 0.35            # added miss chance at full spin with factor 1.0
OPPONENT_SPIN_SCATTER_M = 0.25           # added random placement (+/- m) at full spin with factor 1.0

# =============================================================================
# Look: paddles, ball, table (RGB colours)
# =============================================================================
PADDLE_BLADE_W_M = 0.16                  # real blade size; PADDLE_DRAW_SCALE enlarges it on screen
PADDLE_BLADE_H_M = 0.165
PADDLE_THICKNESS_M = 0.012
PADDLE_HANDLE_L_M = 0.10
PADDLE_DRAW_SCALE = 1.35
PADDLE_HEIGHT_M = 0.14                   # my blade centre above the table
PLAYER_PADDLE_FRONT = (250, 214, 40)     # the YELLOW face of your real paddle
PLAYER_PADDLE_BACK = (22, 22, 26)        # the BLACK face of your real paddle
# Mirrored faces: the camera looks over your shoulder, so it shows the face pointing at YOU.
# True  = neutral grip has BLACK toward the screen -> the game shows YELLOW toward you.
# False = old mapping (the game shows BLACK toward you in the neutral grip).
PADDLE_FACE_INVERT = True
PADDLE_WOOD = (196, 146, 92)             # handle and edge
PADDLE_RIM_GLOW = True                   # glowing rim so the paddle stands out against the orange table
PADDLE_RIM_COLOR = (140, 50, 220)        # purple
PLAYER_PADDLE_ALPHA = 235                # 255 = opaque
OPPONENT_PADDLE_FRONT = (150, 70, 220)   # skeleton's paddle: purple / lime
OPPONENT_PADDLE_BACK = (120, 220, 60)
SWING_ANIM_S = 0.28                      # length of the on-screen swing animation

BALL_COLOR = (250, 250, 246)             # white: an orange ball disappears on the orange table
BALL_SHADE_COLOR = (120, 112, 150)       # shaded side of the ball
BALL_OUTLINE_COLOR = (40, 20, 50)        # thin dark outline (keeps it visible over the white lines)
BALL_SEAM_COLOR = (70, 40, 110)          # the rotating spin marking
BALL_VISUAL_SCALE = 1.5                  # draw the ball bigger than 40 mm so it reads at the far end
BALL_SPIN_VIS_RPS = 6.0                  # on-screen rotation (rev/s) at full spin
TRAIL_ENABLED = True
TRAIL_LENGTH_S = 0.16
TRAIL_DOTS = 9
TRAIL_COLORS = {"none": (255, 255, 255), "top": (255, 70, 200), "back": (80, 200, 255),
                "side": (150, 255, 80)}  # none / topspin (magenta) / backspin (cyan) / sidespin (lime)
TRAIL_HALO_ALPHA = 70                    # dark halo under each trail dot (contrast on the orange table)
SHADOW_COLOR = (40, 8, 20)               # ball shadow (dark plum reads well on orange)

TABLE_TOP_COLOR = (240, 118, 20)         # Halloween-orange playing surface
TABLE_LINE_COLOR = (250, 250, 250)       # white edge and centre lines
TABLE_EDGE_COLOR = (140, 58, 14)         # darker burnt-orange edge / thickness (underside is darker still)
TABLE_LEG_COLOR = (40, 40, 48)
NET_COLOR = (235, 235, 240)
NET_POST_COLOR = (50, 50, 60)

# =============================================================================
# Theme (theme.py). Colours per theme; swap THEME to change the whole look.
# =============================================================================
THEME = "halloween"                      # "halloween" or "classic"
AMBIENT_ANIMATION = True                 # floating ghosts, flickering pumpkins
AMBIENT_BATS = True
AMBIENT_LEAVES = True
BACKGROUND_SOFTEN = 0.45                 # 0 = crisp scenery, 1 = fully faded into the sky colour
BACKGROUND_BLUR = True                   # soft-focus scenery (pre-rendered, no per-frame cost)
GHOST_COUNT = 4
PUMPKIN_COUNT = 6
BAT_COUNT = 4
LEAF_COUNT = 14
THEME_SEED = 7                           # where the scenery goes (any integer; same seed = same scene)
# Decoration placement. None = seeded spots that never overlap each other, the table, the
# ball path, the paddles or the skeleton. Or place them by hand, in metres:
#   PUMPKIN_SPOTS = [(-2.0, 2.0), (2.4, 4.5)]      # (x lateral, y along the table) on the floor
#   GHOST_SPOTS = [(-2.5, 8.0, 1.0)]               # (x, y, height above the table)
# Hand-placed spots are checked too: one that overlaps something is skipped with a warning.
PUMPKIN_SPOTS = None
GHOST_SPOTS = None
DECOR_MIN_GAP_PX = 10                    # min screen gap between decorations
DECOR_AVOID_HUD = True                   # keep decorations out from under the HUD panels
# Playful system fonts; the first one installed wins (no font files are shipped).
FONT_TITLE = "kristenitc,segoeprint,inkfree,trebuchetms,segoeui"
FONT_BODY = "segoeprint,inkfree,trebuchetms,segoeui"
HALLOWEEN_COLORS = {
    "sky_top": (16, 10, 40), "sky_bottom": (64, 30, 92), "horizon_glow": (120, 60, 120),
    "moon": (255, 244, 200), "moon_glow": (255, 220, 140), "star": (255, 250, 220),
    "hill_far": (40, 26, 66), "hill_near": (30, 20, 50),
    "floor_a": (52, 40, 62), "floor_b": (46, 35, 56), "floor_line": (34, 26, 44), "fog": (70, 42, 96),
    "ghost": (244, 244, 255), "pumpkin": (250, 130, 30), "pumpkin_dark": (200, 90, 20),
    "pumpkin_glow": (255, 230, 90), "stem": (80, 140, 40), "bat": (20, 14, 30),
    "leaf": [(230, 120, 30), (200, 70, 30), (240, 180, 50)],
    "panel": (30, 14, 50), "panel_border": (255, 140, 30), "text": (255, 246, 230),
    "accent": (255, 140, 30), "accent2": (150, 255, 80), "dim": (210, 190, 230),
    "button": (70, 30, 110), "button_selected": (255, 140, 30), "start": (110, 200, 50),
    "good": (150, 255, 80), "warn": (255, 200, 60), "bad": (255, 90, 90), "off": (130, 120, 150),
}
CLASSIC_COLORS = {
    "sky_top": (22, 24, 34), "sky_bottom": (46, 50, 64), "horizon_glow": (60, 64, 80),
    "floor_a": (60, 60, 66), "floor_b": (56, 56, 62), "floor_line": (44, 44, 50), "fog": (50, 54, 66),
    "panel": (0, 0, 0), "panel_border": (230, 230, 235), "text": (255, 255, 255),
    "accent": (70, 140, 230), "accent2": (70, 200, 110), "dim": (190, 190, 200),
    "button": (60, 64, 80), "button_selected": (70, 140, 230), "start": (60, 170, 100),
    "good": (70, 200, 110), "warn": (235, 190, 60), "bad": (225, 80, 70), "off": (130, 130, 140),
}

# =============================================================================
# Skeleton opponent (purely visual -- driven by whatever Opponent is in use)
# =============================================================================
SKELETON_BEHIND_HIT_PLANE_M = 0.12       # how far behind the opponent's hit plane it stands
SKELETON_TRACK_SPEED_M_S = 1.6           # how fast it shuffles sideways toward the ball
SKELETON_BONE = (240, 234, 216)
SKELETON_OUTLINE = (52, 38, 70)
SKELETON_EYES = (60, 24, 90)
SKELETON_EYE_GLOW = (150, 255, 80)
SKELETON_BOWTIE = (255, 140, 30)
SKELETON_GRIP_ALONG_HANDLE = 0.6         # where the hand holds the handle (0 = at the blade, 1 = at the butt)
SKELETON_WRIST_DEG = 35.0                # fixed wrist bend: the paddle points this far above the forearm line
SKELETON_BACKHAND_SIDE_M = 0.0           # ball arriving right of (hips + this) on screen -> backhand
SKELETON_AIM_EXAGGERATION = 2.5          # paddle-face yaw toward the shot target, x the true angle
SKELETON_AIM_MAX_DEG = 40.0
SKELETON_SPIN_TILT_DEG = 35.0            # face tilt at full top/backspin (closed / open face)
SKELETON_SIDESPIN_DEG = 30.0             # extra face angle + handle lean at full sidespin
SKELETON_RECOVER_S = 0.30                # back to the ready pose after the follow-through

# =============================================================================
# AprilTag selector (stub for later -- see apriltag_selector.py)
# =============================================================================
APRILTAG_ENABLED = False
APRILTAG_COMMANDS = {                    # tag id -> (command kind, value)
    0: ("start", None),
    1: ("difficulty", "Easy"),
    2: ("difficulty", "Medium"),
    3: ("difficulty", "Hard"),
    4: ("difficulty", "Dynamic"),
}
APRILTAG_HOLD_S = 0.5                    # tag must stay visible this long to trigger

# =============================================================================
# Feature flags / dev aids
# =============================================================================
SHOW_CAMERA_PREVIEW = True
PREVIEW_WIDTH = 240                      # px, height follows the camera aspect ratio
DEBUG_OVERLAY_DEFAULT = False            # toggle at runtime with F1
KEYBOARD_FALLBACK = True                 # space = swing, arrows = paddle (always available as backup)
KEYBOARD_PADDLE_SPEED_M_S = 1.5          # paddle speed when steering with arrow keys
KEYBOARD_SWING_STRENGTH = 1.5            # normalised strength of a spacebar swing
CONTROLS_BAR_DEFAULT = True              # bottom controls bar visible at start (toggle with H)

# =============================================================================
# Dynamic difficulty: reinforcement-learning opponent (rl_agent.py, rl_opponent.py)
#   GOAL: not to beat you, but to find the hardest shots you can still return, so the
#   game gets harder as you improve while rallies keep going.
# =============================================================================
# ---- Reward: edit this block to change what the agent wants -------------------------------
# reward = base_return       * [you returned it]
#        + hard_return_bonus * [you returned it] * D ** difficulty_exponent
#        - miss_penalty      * [you missed it]
#        + rally_weight      * [you returned it] * min(rally, rally_cap) / rally_cap
# D (0..1) = the shot's difficulty: weighted mix of its speed, how far it lands from your
# paddle (full credit at distance_full_m) and its spin strength.
# The four weights can be changed at any time without losing what the agent learned (it
# learns each part separately). Changing difficulty_exponent, difficulty_weights,
# distance_full_m or rally_cap changes WHAT is learned, so a saved model is reset.
REWARD = {
    "base_return": 0.1,                  # small reward for any returned shot
    "hard_return_bonus": 1.0,            # main reward; band-adjusted while you play
    "difficulty_exponent": 2.0,          # >1 keeps easy shots nearly worthless (D=0.3 -> 0.09, D=0.9 -> 0.81)
    "miss_penalty": 0.2,                 # the shot was too hard; band-adjusted while you play
    "rally_weight": 0.05,                # small bonus for keeping long rallies alive (0 = off)
    "rally_cap": 10,
    "difficulty_weights": {"speed": 0.4, "distance": 0.4, "spin": 0.2},
    "distance_full_m": 1.0,
    # Target zone: keep your rolling return rate inside this band.
    "target_band": (0.70, 0.85),
    "window": 20,                        # rolling return rate over your last N shots
    "min_samples": 8,                    # no band adjustment before this many shots
    "adjust_every": 5,                   # adjust the weights once every N shots
    "adjust_rate": 0.03,                 # above band: bonus x (1 + rate); below: penalty x (1 + rate)
    "relax_rate": 0.015,                 # inside the band: both move this fraction back toward their defaults
    "weight_limits": (0.5, 6.0),         # bonus / penalty stay within these multiples of their defaults
}
# ---- Learning --------------------------------------------------------------------------------
RL_GAMMA = 0.7                           # discount: how much the agent values set-ups over the next shots
RL_ALPHA = 0.1                           # learning-rate floor (each entry starts at 1/visits, then this)
RL_BACKOFF_K = 8.0                       # joint-vs-separate blend: beta = visits / (visits + K)
# ---- Exploration -> exploitation -----------------------------------------------------------
RL_EPSILON_START = 1                   # mostly exploring at first
RL_EPSILON_DECAY = 0.85                  # epsilon x this ...
RL_HITS_PER_STEP = 10                     # ... every N of your hits
RL_EPSILON_MIN = 0.05
RL_HIT_COUNT_MODE = "session"            # "session" = all your hits (keeps progressing after a miss); "rally"
RL_PROGRESSIVE_UNLOCK = True             # start with easier shots, unlock harder ones as epsilon falls
RL_UNLOCK_FAST_EPSILON = 0.5             # fast shots allowed once epsilon < this
RL_UNLOCK_TOPBACK_EPSILON = 0.3          # topspin / backspin allowed once epsilon < this
# ---- Shot commands (also what a physical robot would receive) ------------------------------
RL_PLACEMENT_X_M = {"left": -0.45, "center": 0.0, "right": 0.45}   # where the ball lands (your x)
RL_SPEEDS_M_S = {"slow": 2.2, "medium": 3.2, "fast": 4.5}
RL_SPIN_AMOUNT = 0.7                     # spin of a spin shot (0..SPIN_MAX)
# ---- State buckets --------------------------------------------------------------------------
RL_PADDLE_ZONE_M = 0.25                  # your paddle x < -this = left, > +this = right, else centre
RL_RALLY_LONG = 3                        # rally length >= this counts as a long rally
# ---- Persistence / display ------------------------------------------------------------------
RL_PERSIST = True                        # load at start, save on Esc-to-menu and on quit
RL_SAVE_PATH = HERE / "models" / "dynamic_agent.json"
RL_PANEL_DEBUG_ONLY = False              # True = show the Dynamic panel only with the F1 overlay
