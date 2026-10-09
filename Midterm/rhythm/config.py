"""
Every tunable value for Concert Rally (the rhythm ping-pong game) lives here.

This file first loads ../config.py (the ping-pong config) and copies all of its
UPPER_CASE values, so the motor, IMU calibration, webcam, pose and roll settings you
already tuned for ping pong apply here too (hardware settings stay in ONE place).
Anything below overrides or adds to them. ../config.py itself is never modified.

Your latency calibration is saved to calibration.json (next to this file) and loaded
at the bottom of this file, overriding the INPUT_OFFSET_MS / VISUAL_OFFSET_MS defaults.

Values marked  # DEFAULT  are sensible starting points picked without hardware
testing -- tune them if they feel off. Units are given in the comments.
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIDTERM = HERE.parent

# The rhythm folder must stay first on sys.path (so `import config` is this file);
# Midterm goes after it so motor_imu / pose_tracker / paddle / scene can be reused.
if str(MIDTERM) not in sys.path:
    sys.path.append(str(MIDTERM))

_spec = importlib.util.spec_from_file_location("midterm_config", MIDTERM / "config.py")
_midterm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_midterm)
globals().update({k: v for k, v in vars(_midterm).items() if k.isupper() and k not in ("HERE", "MIDTERM")})
del _spec, _midterm

# =============================================================================
# Window
# =============================================================================
WINDOW_TITLE = "Concert Rally"
WINDOW_RESIZABLE = True                  # drag the window edges; everything re-lays out
# WINDOW_WIDTH / WINDOW_HEIGHT / FPS are inherited from ../config.py (1100 x 720 @ 60)

# =============================================================================
# Songs and charts
# =============================================================================
SONGS_DIR = HERE / "songs"
CHART_SUFFIX = ".chart.json"             # songs/<name>.chart.json next to the audio file
AUDIO_EXTENSIONS = (".wav", ".ogg", ".mp3")

# Which beats of each bar YOU hit the ball on, for every song (overrides the charts' patterns
# and sections while you play; the chart files are never changed). Only two values exist:
#   [4]     you hit on beat 4; the opponent hits on beat 2 (2-beat flights each way)
#   [2, 4]  you hit on beats 2 and 4; the opponent hits on 1 and 3 (1-beat flights)
#   None    use each chart's own pattern / sections (the charts default to [4] everywhere)
# Anything else is an error at startup.
HIT_BEATS = None
BEATS_PER_BAR = 4                        # every song is 4/4 (do not change)
SCORES_PATH = HERE / "scores.json"       # best accuracy / max streak per song (local)

# =============================================================================
# Audio engine
#   "sounddevice": our own mixer in a PortAudio callback. The song clock is the number
#                  of samples written, tied to PortAudio's "this buffer reaches the
#                  speaker at time T" timestamp -> sample-accurate. Cue sounds are mixed
#                  in at exact sample positions, so they never drift with the frame rate.
#   "pygame":      fallback (pygame.mixer). Clock = time since play() + AUDIO_PYGAME_LATENCY_MS.
#                  Less precise (start jitter up to one buffer); cue sounds are triggered
#                  from the game loop.
#   "null":        no sound at all, clock only (tests / no audio device).
# =============================================================================
AUDIO_BACKEND = "sounddevice"
AUDIO_HOSTAPI = "WASAPI"                 # "WASAPI" (low latency), "MME", "DirectSound"; None = system default
AUDIO_DEVICE = None                      # output device index or name; None = the host API's default output
AUDIO_LATENCY = "low"                    # PortAudio latency hint: "low", "high" or seconds
AUDIO_BLOCKSIZE = 0                      # frames per callback; 0 = let the driver choose (~10 ms on WASAPI)
AUDIO_FALLBACK_SAMPLE_RATE = 48000       # used when the device doesn't report one
AUDIO_EXTRA_OUTPUT_LATENCY_MS = 0.0      # latency PortAudio can't see (e.g. Bluetooth); calibration
                                         # absorbs it anyway, so normally leave at 0
AUDIO_PYGAME_LATENCY_MS = 60.0           # DEFAULT assumed output latency of the pygame fallback
AUDIO_CLOCK_SMOOTHING = 0.05             # EMA weight per callback for the sample-clock -> wall-clock offset
AUDIO_CLOCK_RESYNC_MS = 15.0             # a jump larger than this (underrun, device change) resets the clock

MUSIC_VOLUME = 0.85                      # 0..1
CUE_SOUNDS_ENABLED = True                # warning cues before flagged cues
CUE_VOLUME = 0.75                        # 0..1, separate from the music
TEMPO_TICK_ENABLED = True                # tick with each tempo-change LED blink (speed-ups and slowdowns)
TEMPO_TICK_VOLUME = 1.0                  # 0..1, separate from music and warning cues (a metronome tick)
HIT_SOUND_ENABLED = True                 # soft "tock" on a successful return (plays when the hit is
HIT_SOUND_VOLUME = 0.35                  # registered, so with the IMU it can sound slightly late)
METRONOME_VOLUME = 0.8                   # calibration clicks and countdown clicks
COUNTDOWN_CLICKS = "preroll"             # "preroll" = click the countdown only where the song is still
                                         # silent (before it starts), "always", or "off"

# =============================================================================
# Timing judgment (ms). At 160-200 BPM one beat is 375-300 ms.
# =============================================================================
PERFECT_WINDOW_MS = 45                   # DEFAULT |error| <= this = Perfect
GOOD_WINDOW_MS = 100                     # DEFAULT |error| <= this = Good (else the swing doesn't count)
MISS_GRACE_MS = 120                      # extra wait for a late-arriving IMU event before calling a Miss
GRADE_POINTS = {"Perfect": 1.0, "Good": 0.5, "Miss": 0.0}   # accuracy weight of each grade
STRAY_SWING_PENALTY = False              # True = a swing with no ball near it counts as a Miss
GRADE_POPUP_S = 0.65                     # how long Perfect / Good / Miss stays on screen

# Rank on the results screen: first threshold reached (accuracy %) wins.
RANKS = [("S", 95.0), ("A", 85.0), ("B", 70.0), ("C", 55.0), ("D", 0.0)]

# =============================================================================
# Latency offsets (defaults; calibration.json overrides them, see the bottom of this file)
#   INPUT_OFFSET_MS:  how late your swings register on average (BLE + detection + you).
#                     Subtracted from every swing time. Measured per input source.
#   VISUAL_OFFSET_MS: positive = draw everything later (use if the ball looks EARLY
#                     compared with the music), negative = earlier.
# =============================================================================
INPUT_OFFSET_MS = {"imu": 0.0, "keyboard": 0.0}
VISUAL_OFFSET_MS = 0.0
CALIBRATION_PATH = HERE / "calibration.json"

# Calibration routine
CALIB_BPM = 100                          # metronome tempo for the input calibration
CALIB_COUNT_IN_BEATS = 4                 # clicks before measuring starts
CALIB_BEATS = 16                         # measured beats (swing on every click)
CALIB_MIN_SWINGS = 8                     # need at least this many usable swings
CALIB_OUTLIER_MS = 120                   # swings further than this from the median are dropped
CALIB_MAX_ABS_MS = 400                   # swings further than this from any click are ignored
CALIB_AV_BPM = 100                       # tempo of the visual (A/V) calibration
CALIB_AV_STEP_MS = 5                     # Left/Right step (hold Shift for 1 ms)

# =============================================================================
# Swing detection for the rhythm game (IMU thresholds/axes come from ../config.py)
#   The swing time is the IMU sample with the PEAK strength (the "contact" moment).
#   The event is sent as soon as the strength falls below SWING_PEAK_DROP x peak,
#   or after RHYTHM_SWING_PEAK_WINDOW_S at the latest -- shorter = less delay.
# =============================================================================
RHYTHM_SWING_COOLDOWN_S = 0.22           # DEFAULT; auto-shortened per song if the chart needs it
RHYTHM_SWING_MIN_COOLDOWN_S = 0.08       # never below this (one swing would count twice)
RHYTHM_SWING_PEAK_WINDOW_S = 0.09        # longest wait for the peak after onset
SWING_PEAK_DROP = 0.6                    # peak is over once strength < this fraction of it
RHYTHM_SWING_THRESHOLD_SCALE = 1.0       # x the ping-pong swing thresholds: raise (e.g. 1.5) if small
                                         # movements count as swings, lower if hard swings are missed

# =============================================================================
# On-screen paddle twist (visual only: spin is not scored in this game)
#   Swinging every beat leaves the roll filter few still moments to correct gyro drift,
#   so the shown twist is calmed: it holds during swings and haptic pulses, ignores small
#   twists, and slow drift fades back to neutral. C (during a song) re-zeroes it.
# =============================================================================
ROLL_DISPLAY_ENABLED = True              # False = the paddle never twists on screen
ROLL_DISPLAY_GAIN = 0.6                  # shown twist = (real twist - deadzone) x this
ROLL_DISPLAY_DEADZONE_DEG = 12.0         # twists smaller than this show as straight
ROLL_DISPLAY_MAX_DEG = 50.0
ROLL_DISPLAY_RECENTER_S = 4.0            # drift (and a twist held still) fades back to neutral this fast
ROLL_DISPLAY_SWING_HOLD_S = 0.30         # the twist holds this long after each swing

# =============================================================================
# Game modes (M on song select cycles them; --mode on the command line). Your paddle always moves
# as in ping pong: a continuous slider following your wrist (webcam pose), or hold Left / Right
# to glide it (KEYBOARD_PADDLE_SPEED_M_S from ../config.py).
#   "aim"     the ball arrives at varied spots; a hit needs good swing timing (IMU) AND the ball
#             actually meeting your paddle: the drawn blade must overlap the drawn ball when the
#             ball reaches you (an unknown paddle position, e.g. the webcam lost you, is no hit)
#   "timing"  the ball arrives at varied spots; only timing counts
#   "follow"  the ball always flies to your paddle; only the timing of your flick counts
# Best scores are kept per mode.
# =============================================================================
GAME_MODES = ("aim", "timing", "follow")
GAME_MODE = "aim"                        # the mode the game starts in
GAME_MODE_NAMES = {"aim": "Aim", "timing": "Timing", "follow": "Follow"}
GAME_MODE_HELP = {"aim": "timing + paddle at the ball", "timing": "timing only",
                  "follow": "the ball comes to your paddle; timing only"}
ARRIVAL_X_SPREAD_M = 0.45                # aim / timing: balls reach your end at x in +/- this (varies per hit)
AIM_CONTACT_M = None                     # aim: max |paddle x - ball x| for contact; None = what you see:
                                         # half the drawn blade + the drawn ball (about 0.14 m)
AIM_LOOKBACK_MS = 30                     # aim: contact is checked from this long before the ball reaches
AIM_LOOKAHEAD_MS = 80                    # your paddle to this long after (webcam samples arrive late)

# =============================================================================
# Ball flight. See README "Timing engine" for the rule: the ball spends equal time in the air
# each way, so the opponent hits halfway between your hits ([4]: opponent on 2, you on 4;
# [2, 4]: opponent on 1 and 3, you on 2 and 4). Flight times come from the tempo map.
# =============================================================================
BOUNCE_LEAD_BEATS = 0.5                  # the ball bounces on your side this long before arriving
                                         # (or halfway through shorter flights)
BOUNCE_Y_NEAR_M = 0.70                   # bounce spot on your half (m from your end)
BOUNCE_Y_FAR_M = TABLE_LENGTH_M - 0.70   # bounce spot on the opponent's half
ARC_GRAVITY_M_S2 = 9.81                  # arc height of a segment = g * duration^2 / 8 (real physics)
ARC_MIN_NET_CLEARANCE_M = 0.06           # arcs are raised if needed to clear the net by this much
ARC_MAX_HEIGHT_M = 0.55
RETURN_TARGET_SPREAD_M = 0.40            # your returns land at x in +/- this (varies per cue)
MISS_OVERSHOOT_TAU_S = 0.07              # an unanswered ball eases to a stop just past your paddle...
MISS_FALL_S = 0.55                       # ...then (once it's a Miss) drops off the table
LATE_HIT_BLEND_S = 0.08                  # a late-registered hit blends onto the return path this fast
SERVE_FADE_S = 0.12                      # a new ball fades in at the opponent's paddle

# =============================================================================
# Haptic pulse on a successful hit (motor in ../config.py: HAPTIC_SPEED / HAPTIC_MOTOR)
#   The pulse shakes the handle, so the IMU is ignored during it + the settle time.
#   That quiet time must end before the NEXT cue's timing window opens; the pulse
#   is shortened (or skipped) per hit to make sure, and checked when a chart loads.
# =============================================================================
RHYTHM_HAPTIC_ENABLED = True
RHYTHM_HAPTIC_PULSE_MS = 60              # DEFAULT longest pulse
RHYTHM_HAPTIC_SETTLE_MS = 60             # DEFAULT IMU ignored this long after the pulse ends
RHYTHM_HAPTIC_MIN_PULSE_MS = 25          # a pulse shorter than this is skipped
RHYTHM_HAPTIC_SAFETY_MS = 40             # extra margin before the next window opens (also covers the
                                         # swing's onset, which comes a little before its peak)

# =============================================================================
# Warning cues
# =============================================================================
WARNING_LEAD_BEATS = 1.0                 # cue sound plays this long before a flagged cue
UPBEAT_GLOW_COLOR = (250, 190, 50)       # upbeat balls (future charts) get an amber halo
ARRIVAL_HINT = True                      # aim / timing: soft spot where the next ball reaches your end
ARRIVAL_HINT_BEATS = 1.0                 # (in aim mode it is as wide as the reach); shown this long before

# =============================================================================
# Tempo change indicator: before each SUDDEN tempo change (faster or slower), a round LED blinks
# TEMPO_LED_BLINKS times AT THE NEW TEMPO (with a tick each time), landing 3, 2, 1 new-tempo
# beats before the change, so the new tempo's first beat is the next count.
#   Sudden = the tempo map rises or falls by MORE than TEMPO_JUMP_MIN_BPM within
#   TEMPO_SUDDEN_WITHIN_BEATS beats (one segment boundary is instant; small steps in the same
#   direction close together add up). Gradual drift never triggers it.
# =============================================================================
TEMPO_JUMP_MIN_BPM = 5.0
TEMPO_SUDDEN_WITHIN_BEATS = 2.0
TEMPO_LED_BLINKS = 3
TEMPO_LED_POS = (0.5, 0.20)              # centre, as fractions of the window width / height
TEMPO_LED_RADIUS_PX = 18
TEMPO_LED_COLOR = (255, 160, 40)         # lit lens before a speed-up (amber); unlit = a dark version
TEMPO_LED_SLOW_COLOR = (80, 160, 255)    # lit lens before a slowdown (blue)
TEMPO_LED_BLINK_BEATS = 0.45             # each blink lasts this many new-tempo beats
TEMPO_LED_SHOW_S = 0.6                   # the LED housing fades in this long before the first blink
TEMPO_LED_LABEL = True                   # show the new tempo ("160 BPM") under the LED

# =============================================================================
# Countdown before the first cue
# =============================================================================
COUNTDOWN_BEATS = 4                      # "3", "2", "1", "GO!" on the beats before the first cue
PREROLL_MIN_S = 0.6                      # silence added before the song if the countdown needs it

# =============================================================================
# Look: plain and realistic -- a dark backboard, a floor, the table. Colours are RGB.
# =============================================================================
TABLE_TOP_COLOR = (24, 86, 160)          # tournament blue: the white ball stands out on it
TABLE_LINE_COLOR = (245, 245, 245)
TABLE_EDGE_COLOR = (14, 46, 96)
TABLE_LEG_COLOR = (30, 30, 34)
NET_COLOR = (232, 232, 236)
NET_POST_COLOR = (40, 40, 46)
BALL_COLOR = (250, 250, 246)
BALL_SHADE_COLOR = (150, 156, 170)
BALL_OUTLINE_COLOR = (40, 40, 48)
SHADOW_COLOR = (8, 14, 30)
BALL_VISUAL_SCALE = 1.5
TRAIL_ENABLED = True
TRAIL_LENGTH_S = 0.10
TRAIL_DOTS = 5
TRAIL_COLOR = (230, 232, 236)

# My paddle: yellow / black / wood handle with the mirrored faces (PADDLE_FACE_INVERT),
# inherited from ../config.py -- without ping pong's glowing rim (not needed on a blue table).
PADDLE_RIM_GLOW = False
SWING_ANIM_S = 0.22

# The opponent: a floating paddle at the far end (no character).
OPPONENT_PADDLE_FRONT = (190, 30, 36)    # red rubber toward you
OPPONENT_PADDLE_BACK = (24, 24, 28)      # black rubber
OPPONENT_SWING_S = 0.24                  # swing animation length around each of its hits

THEME = "plain"
PLAIN_COLORS = {
    "wall_top": (26, 28, 33), "wall_bottom": (44, 47, 54),          # the backboard
    "floor_far": (50, 52, 58), "floor_near": (70, 72, 78), "baseboard": (20, 21, 25),
    "panel": (18, 20, 24), "panel_border": (78, 84, 96), "text": (236, 238, 242),
    "accent": (110, 170, 245), "accent2": (170, 196, 226), "dim": (168, 174, 186),
    "button": (40, 44, 52), "button_selected": (110, 170, 245), "start": (60, 170, 100),
    "good": (90, 200, 120), "warn": (230, 190, 70), "bad": (230, 90, 90), "off": (120, 124, 132),
    "perfect": (250, 210, 80), "goodgrade": (110, 190, 255), "miss": (235, 95, 95),
}

# Fonts: plain sans-serif (the first one installed wins).
FONT_TITLE = "segoeui,helvetica,arial"
FONT_BODY = "segoeui,helvetica,arial"

SHOW_CAMERA_PREVIEW = True               # PREVIEW_WIDTH inherited
CONTROLS_BAR_DEFAULT = True              # toggle with H

# =============================================================================
# Chart helper (make_chart.py): tempo-map detection and preview
# =============================================================================
CHART_BPM_RANGE = (90, 185)              # tempos searched (a song's beat is assumed inside this)
CHART_WINDOW_S = 12.0                    # local tempo is measured in windows this long...
CHART_WINDOW_HOP_S = 3.0                 # ...this far apart
CHART_MIN_SEGMENT_S = 15.0               # a tempo must last this long to become its own segment
                                         # (shorter wobbles are detection jitter)
CHART_MERGE_BPM = 1.5                    # neighbouring segments closer than this are merged
CHART_RUN_PURITY = 0.8                   # a segment needs this share of its windows to agree on its
                                         # tempo; less = a groove fitting two tempos, not a change
CHART_BARLINE_SNAP_MS = 20               # a mid-bar tempo change moves to a bar line within a bar if
                                         # the old and new beat grids meet there within this
CHART_CHANGE_FIT = 0.6                  # a tempo change starts where the old beat grid's fit falls
                                         # below this x its usual level (and stays there 2 bars)
CHART_INTRO_BARS = 2                     # hits start this many bars after the music gets going...
CHART_OUTRO_BARS = 1                     # ...and stop this many bars before it fades out
CHART_LOUD_FRACTION = 0.45               # "the music is going" = bar loudness >= this x the median
PREVIEW_CLICK_VOLUME = 0.55              # preview: click on every beat
PREVIEW_ACCENT_VOLUME = 0.9              # preview: louder click on beat 1 of each bar
PREVIEW_STEP_MS = 5                      # preview: Left/Right offset step (Shift = 1 ms)
PREVIEW_SEEK_BARS = 4                    # preview: PgUp / PgDn jump

# =============================================================================
# Debug / dev
# =============================================================================
DEBUG_OVERLAY_DEFAULT = False            # F1


# -- load the saved calibration (overrides the defaults above) -----------------------------
def load_calibration(path=None):
    """Apply calibration.json to this module. Returns the dict that was loaded (or {})."""
    global VISUAL_OFFSET_MS
    path = Path(path or CALIBRATION_PATH)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    for src, ms in (data.get("input_offset_ms") or {}).items():
        if isinstance(ms, (int, float)):
            INPUT_OFFSET_MS[src] = float(ms)
    if isinstance(data.get("visual_offset_ms"), (int, float)):
        VISUAL_OFFSET_MS = float(data["visual_offset_ms"])
    return data


load_calibration()
