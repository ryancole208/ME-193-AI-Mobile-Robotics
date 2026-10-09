# Ping Pong: Human vs Robot (human side)

You hold a LEGO Education **Double Motor** as the paddle: the hub *is* the handle. Its IMU detects
*when* you swing, and **twisting the handle puts spin on the ball**. A webcam with MediaPipe pose
tracking detects *where* your paddle is and which *direction* you swing. The pygame window shows a
pseudo-3D, Halloween-themed scene from behind and above your end: an orange table on legs, a
cartoon skeleton opponent that grips its paddle, ghosts and pumpkins. The current streak of
continuous hits is published over MQTT as a float to `ME193/RQ-D2/<PLAYER_NAME>`. The opponent is
simulated (Easy / Medium / Hard) or driven by a **learning AI (Dynamic)**, behind an interface that
the physical robot can replace later.

## Setup

Everything runs in the repo's existing virtual environment `../my_env` (Python 3.14).
`legoeducation` needs Python 3.14+, and MediaPipe 1.0.1 supports 3.14, so one venv and one
process are enough.

```powershell
cd "ME 193 AI Mobile Robotics\Midterm"
..\my_env\Scripts\python.exe -m pip install -r requirements.txt
```

The PoseLandmarker model (`models/pose_landmarker_lite.task`, about 6 MB) downloads automatically
the first time you run the game.

### Bluetooth (Windows 11)
1. Settings → Bluetooth & devices: turn **Bluetooth on**. The PC needs a Bluetooth 4.0+ (BLE) adapter.
2. **Do not pair** the motor in Windows Settings. The library connects directly. If it was paired before, remove it.
3. Close the LEGO Education app and Coding Canvas, since only one program can hold the connection.
4. Turn the motor on. If it reports a firmware/package RPC mismatch, update the firmware at https://code.legoeducation.com/.

### Camera (Windows 11)
Settings → Privacy & security → **Camera** → allow desktop apps. If you have several cameras, set
`CAMERA_INDEX` in `config.py`.

## Running

```powershell
..\my_env\Scripts\python.exe main.py              # motor + webcam + MQTT
..\my_env\Scripts\python.exe main.py --keyboard   # no hardware at all
..\my_env\Scripts\python.exe main.py --no-mqtt    # stay offline
```

When it starts, the game scans for Double Motors, connects to the **nearest** one (strongest RSSI)
and prints its name and address. **Hold the paddle still for about a second after it connects**,
because that is when it calibrates gravity. **Hold your neutral grip when you press Start** (paddle
face vertical, **BLACK side toward the screen, yellow toward you**): that grip becomes zero twist.
Press **C** to re-zero it.

| Key | Action |
|---|---|
| 1 / 2 / 3 / 4 (or click) | Easy / Medium / Hard / **Dynamic** (the learning AI opponent) |
| Enter (or START) | start |
| Esc | back to menu (also saves what the Dynamic AI has learned) |
| R (in the menu, with Dynamic selected) | reset the Dynamic AI's learning |
| H | show / hide the controls bar (it wraps to fit any window width) |
| Space | swing (keyboard fallback) |
| hold Q / E while pressing Space | curve left / right (sidespin); the paddle also twists on screen |
| hold W / S while pressing Space | topspin / backspin (combine with Q/E) |
| ← / → | move paddle (used when pose isn't tracking) |
| C | re-zero the handle twist (hold your neutral grip) |
| F1 | debug overlay (FPS, live IMU, raw vs filtered roll, wrist position, last hit/miss and its spin) |
| **Ctrl+Q** / close window / Ctrl+C in the console | quit (disconnects BLE, releases the camera, disconnects MQTT) |

**A hit needs both:** a swing that starts within `HIT_WINDOW_BEFORE_S`/`HIT_WINDOW_AFTER_S` of the
ball reaching your paddle plane, **and** your paddle within `LATERAL_HIT_TOLERANCE_M` of the ball.
Wrist motion (left/center/right) around the swing angles the return. Swing strength scales the
return speed, within limits. A miss resets the streak to `0.0`. A successful hit pulses the motor
(haptics). For the pulse plus `HAPTIC_SETTLE_S` the IMU is ignored for swing detection and the twist
filter stops trusting the accelerometer, so the buzz never counts as a swing or a twist.

### Spin
| You do | Spin | Effect |
|---|---|---|
| twist the handle quickly during the swing | sidespin | the ball curves left/right in flight (Magnus push) and lands off its aim line |
| swing upward (brush up) | topspin | lower arc, bounces earlier, low fast kick forward after the bounce |
| swing downward (chop) | backspin | floats higher, bounces later, sits up and slows down |
| small twist / gentle swing | none | inside the dead zones nothing happens |

- **Visuals:** strong spin shows a label like **CURVE LEFT!** or **TOPSPIN!**. The ball's seam
  visibly rotates about the spin axis. The trail is tinted: lime for side, magenta for top,
  cyan for back, white for none. Each trail dot has a faint dark halo so it shows on the orange table.
- **Opponent:** heavy spin makes the opponent miss more often and place its returns less
  accurately. The effect is biggest on Easy and smallest on Hard (`OPPONENT_SPIN_PENALTY`).
- **Signal:** `SPIN_INPUT` chooses the sidespin signal:
  - `"rate"` (default): how fast you twist during the swing. It's quick and unaffected by slow drift.
  - `"angle"`: how far the paddle is twisted when the swing starts.
  - `"both"`: the two added together.
- **Limits:** spin is clamped to `SPIN_MAX`, and a curve can never push the ball off the table.

### Your paddle on screen (mirrored faces)
The camera looks over your shoulder, so it shows the face of the paddle that points at **you**:

| You hold | The game shows toward you |
|---|---|
| BLACK toward the screen (neutral grip) | YELLOW |
| YELLOW toward the screen | BLACK |

Only the colours are swapped. Twisting is the same physical rotation whichever face points
where, so the twist direction (and `IMU_ROLL_SIGN`) is unchanged: twist the handle right and the
on-screen paddle turns right. Set `PADDLE_FACE_INVERT = False` in `config.py` to go back to the old
mapping. The paddle has a purple glowing rim (`PADDLE_RIM_COLOR`) so it stands out on the orange table.

### The skeleton's paddle
The skeleton holds its paddle in its hand in every frame (fingers wrap the handle), and the paddle
moves as a rigid extension of its forearm. When the ball comes to its paddle side it plays a
**forehand** (purple face); otherwise a **backhand**, with the arm across its body (lime face).
It winds up before the ball arrives, swings through contact and follows through. From contact on
(earlier with the Dynamic AI, which picks its reply in advance), the paddle face turns toward
where the shot is going and tilts for its spin: closed for topspin, open for backspin, angled for
sidespin. It only reads the game state, so it animates the same for every opponent, including the
robot later. Tunables: `SKELETON_*` in `config.py`.

## Tests

```powershell
..\my_env\Scripts\python.exe -m pytest tests            # unit tests, no hardware needed
..\my_env\Scripts\python.exe -m pytest tests --hardware # + real motor, webcam, MQTT broker
```

| File | Covers |
|---|---|
| `test_doublemotor_connection.py` | RSSI transport registration, nearest-device choice, connect, IMU streaming, reconnect after a drop, haptic pulse, Connection Card filters. Hardware: real connect and sample rate |
| `test_imu_swing.py` | IMU parsing, gravity calibration, gravity removal, thresholds and modes, cooldown, peak strength |
| `test_pose.py` | mirroring, wrist→paddle mapping, smoothing, swing direction, dominant hand. Hardware: model load, live webcam tracking |
| `test_game_rules.py` | hit window, lateral tolerance, streak and score floats, rallies, direction and strength of returns, keyboard fallback, perspective camera |
| `test_spin.py` | roll complementary filter (gyro mid-swing, accel correction when still, haptic freeze, wrap, history), twist ≠ swing, twist rate / vertical motion capture, spin mapping and dead zones, curve/top/back flights, spin labels, opponent spin penalty |
| `test_opponent.py`, `test_selectors.py`, `test_mqtt_client.py`, `test_config.py`, `test_render.py` | interfaces, AprilTag stub logic, MQTT wrapper (hardware: real broker round trip), config sanity, headless rendering of both themes, skeleton reactions, frame-time budget, the controls bar and 4-button menu at several window sizes, the Dynamic panel |
| `test_rl_agent.py` | the Dynamic AI: TD / successor-feature maths (hand-checked updates, bootstrap vs terminal, weights acting without retraining, joint/separate blend, bounds), the epsilon schedule (session and rally modes) and unlocks, difficulty score and rewards, the target-band rule, saving / loading / refusing old models / reset, learning tests against simulated players (a combination only the joint table can learn, set-ups that need gamma > 0, getting harder as the player improves, the challenging-but-winnable goal), full rallies inside the game, speed |
| `test_visuals.py` | mirrored paddle faces and twist direction, the skeleton's grip in every frame, rigid forearm, forehand/backhand faces, aim and spin tilt, no elbow flips, decorations never overlapping each other or gameplay, colour contrast |

## Tuning (all values are in `config.py`)

1. **IMU units aren't documented by LEGO.** Run `..\my_env\Scripts\python.exe imu_probe.py`,
   hold still, then swing normally and gently. It prints gravity-removed `|a|` in **g**, `|gyro|`
   in **raw** units, and their peaks.
   - Set `SWING_ACCEL_THRESHOLD_G` a bit below a gentle swing's peak `|a|`, and well above the
     values you see while just moving the paddle around.
   - Set `SWING_GYRO_THRESHOLD_RAW` the same way, using the gyro peaks.
   - `SWING_MODE`: use `"either"` to begin with. Switch to `"both"` if you get false triggers.
   - Raise `SWING_COOLDOWN_S` if one swing registers twice.
   - If the console says *"FALLBACK - paddle moved during calibration"*, reconnect while holding
     still, or set `IMU_ACCEL_RAW_PER_G_FALLBACK` to the 1 g value the probe reports.
2. **Paddle position.** Press F1 and watch `wrist x` while you reach comfortably left and right.
   Put those two extremes into `POSE_X_RANGE`. Raise `POSE_SMOOTHING` for less jitter, or lower
   it for less lag.
3. **Swing direction.** If too many swings read as "center", lower `POSE_DIRECTION_THRESHOLD`
   or raise `POSE_DIRECTION_FRAMES`.
4. **Timing.** The debug overlay shows `dt` (swing onset minus ball arrival) and `err`
   (lateral error) for each attempt. Widen `HIT_WINDOW_*` or `LATERAL_HIT_TOLERANCE_M` if the
   game feels too strict.
5. **Haptics.** Set `HAPTIC_ENABLED`, `HAPTIC_PULSE_MS`, `HAPTIC_SPEED`, `HAPTIC_MOTOR` and
   `HAPTIC_SETTLE_S`. Raise the settle time if the debug overlay ever shows a swing right after a hit.

## IMU axis test and spin calibration

The LEGO library doesn't say how the IMU axes sit in the hub, nor what the gyro units are. The
`IMU_ROLL_*` and `IMU_GYRO_DEG_PER_RAW` values in `config.py` are **placeholders** until you run:

```powershell
..\my_env\Scripts\python.exe imu_axis_test.py          # guided, about 30 s
..\my_env\Scripts\python.exe imu_axis_test.py --live   # just stream accel + gyro on all axes
```

1. **Neutral grip, still.** Paddle face vertical, BLACK side toward the screen (yellow toward you).
   This measures which way gravity points.
2. **Twist back and forth** like a doorknob, keeping the hub in place. The axis with the largest
   gyro response is the roll axis. The script prints each axis's share.
3. **Slowly turn the black (screen-side) face about 90° to your right**, then hold. This gives the
   sign (right = positive) and the gyro scale, by comparing the integrated gyro with how far
   gravity turned. (This is the same physical twist as the old "yellow toward the screen"
   instructions, so values you already measured stay valid.)

The script prints the `config.py` lines to copy (`IMU_ROLL_AXIS`, `IMU_ROLL_SIGN`,
`IMU_ACCEL_ROLL_SIGN`, `IMU_GYRO_DEG_PER_RAW`) plus starting values for `SPIN_RATE_FULL_DPS` and
`SPIN_RATE_DEADZONE_DPS`, based on your fastest twist.

**Then tune spin in the game (F1 overlay):**
- **Twist display.** `roll filt` should follow your twist smoothly, and the on-screen paddle should
  turn the same way. If it turns the wrong way, flip `IMU_ROLL_SIGN`. If `raw(accel)` moves the
  opposite way to `filt` while you twist slowly, flip `IMU_ACCEL_ROLL_SIGN`.
- **Drift.** If the paddle creeps while you hold still, raise `ROLL_GYRO_DEADZONE_DPS` or lower
  `ROLL_FILTER_ALPHA` (more accelerometer correction). If it wobbles during swings, raise
  `ROLL_FILTER_ALPHA` or lower `ROLL_STILL_TOLERANCE_G`.
- **Handle pointing straight down.** Gravity then lies along the handle and can't measure twist,
  so the filter runs on the gyro only. `ROLL_MIN_GRAVITY_FRACTION` sets that cut-off. Press **C**
  to re-zero if it drifts.
- **Sidespin strength.** The overlay shows `spin side / top` and the curve for each hit. Adjust
  `SPIN_RATE_DEADZONE_DPS` / `SPIN_RATE_FULL_DPS` (or `SPIN_ANGLE_*` with `SPIN_INPUT = "angle"`),
  and use `SPIN_SENSITIVITY` to scale everything.
- **Top/backspin.** Set `SPIN_TOP_DEADZONE_G` / `SPIN_TOP_FULL_G` from the vertical motion of your
  normal swings. Set `SPIN_TOP_SOURCE = "off"` to disable it.
- **Physics.** Tune `MAGNUS_STRENGTH`, `TOPSPIN_*` / `BACKSPIN_*` and `OPPONENT_SPIN_*`.

## Look, theme and performance (all in `config.py`)

| Group | Settings |
|---|---|
| Camera | `VIEW_CAMERA_X_M`, `VIEW_CAMERA_BACK_M`, `VIEW_CAMERA_HEIGHT_M`, `VIEW_CAMERA_PITCH_DEG`, `VIEW_FOV_DEG`, `VIEW_CENTER_Y_PX` |
| Table | `TABLE_TOP_COLOR` (orange), `TABLE_LINE_COLOR` (white), `TABLE_EDGE_COLOR`, `TABLE_LEG_COLOR`, `NET_COLOR`, `NET_POST_COLOR`, `TABLE_HEIGHT_M`, `TABLE_THICKNESS_M` |
| Paddles | `PLAYER_PADDLE_FRONT` (yellow), `PLAYER_PADDLE_BACK` (black), `PADDLE_FACE_INVERT`, `PADDLE_WOOD`, `PADDLE_RIM_GLOW` / `PADDLE_RIM_COLOR` (purple), `OPPONENT_PADDLE_*`, `PADDLE_DRAW_SCALE`, `SWING_ANIM_S` |
| Ball | `BALL_COLOR` (white), `BALL_SHADE_COLOR`, `BALL_OUTLINE_COLOR`, `BALL_SEAM_COLOR`, `SHADOW_COLOR`, `BALL_VISUAL_SCALE`, `BALL_SPIN_VIS_RPS`, `TRAIL_ENABLED`, `TRAIL_LENGTH_S`, `TRAIL_DOTS`, `TRAIL_COLORS`, `TRAIL_HALO_ALPHA` |
| Decorations | `THEME_SEED` (same seed = same scene), `PUMPKIN_SPOTS` / `GHOST_SPOTS` (place them by hand), `DECOR_MIN_GAP_PX`, `DECOR_AVOID_HUD` |
| Controls bar | `CONTROLS_BAR_DEFAULT` (H toggles it while playing) |
| Theme | `THEME` (`"halloween"` or `"classic"`), `HALLOWEEN_COLORS`, `CLASSIC_COLORS`, `FONT_TITLE`, `FONT_BODY` |
| Ambient | `AMBIENT_ANIMATION`, `AMBIENT_BATS`, `AMBIENT_LEAVES`, `GHOST_COUNT`, `PUMPKIN_COUNT`, `BAT_COUNT`, `LEAF_COUNT`, `BACKGROUND_SOFTEN`, `BACKGROUND_BLUR`, `THEME_SEED` |
| Skeleton | `SKELETON_*` colours, `SKELETON_TRACK_SPEED_M_S`, `SKELETON_BEHIND_HIT_PLANE_M`, grip / wrist / stance / aim / spin-tilt angles |

- **Decorations.** Pumpkins and ghosts are placed from `THEME_SEED`, so the scene looks the same
  every launch. Each spot is checked in screen space: decorations never overlap each other and
  never cover the table, the ball's flight space, your paddle or anywhere the skeleton can reach.
  They also try to stay out from under the HUD panels (only if no other room is left may one sit
  under a panel). If not all of `PUMPKIN_COUNT` fit, fewer are drawn. Hand-placed spots in
  `PUMPKIN_SPOTS` / `GHOST_SPOTS` are checked the same way, and any that overlap are skipped with a
  console message.
- **Performance.** The sky, scenery, floor, table and net are drawn **once** at startup, because
  the camera doesn't move. A frame takes about 3 ms on a laptop, so 60 FPS has plenty of headroom.
  If a machine struggles, turn off `TRAIL_ENABLED`, `AMBIENT_LEAVES`, `AMBIENT_BATS` or
  `AMBIENT_ANIMATION`, or set `THEME = "classic"`. FPS is the first line of the F1 overlay.
- **Original art.** All artwork is drawn in code (`theme.py`, `scene.py`, `paddle.py`,
  `skeleton.py`). There are no image files.
- **Fonts.** These are fonts already installed on the PC (Kristen ITC / Segoe Print / Ink Free).
  None are bundled, and the first one installed is used.

## Dynamic difficulty (the learning AI opponent)

**Goal: the AI is not trying to beat you.** It looks for the hardest shots you can still return,
so the game gets harder as you get better while the rallies keep going. Select it with **4** (or
the menu button). The score (your streak) is published over MQTT exactly as in the other modes.

### What it controls
Every time the skeleton hits the ball (serves included), the AI picks one **shot command**:

| Part | Choices (`config.py`) |
|---|---|
| placement (where it lands on your side) | left / center / right (`RL_PLACEMENT_X_M`) |
| speed | slow / medium / fast (`RL_SPEEDS_M_S`) |
| spin | none / curve left / curve right / topspin / backspin (`RL_SPIN_AMOUNT`) |

That is 45 combinations. For sidespin the aim is shifted so the curve brings the ball onto the
chosen spot. A `ShotCommand` has the labels and the numbers (`to_dict()`), so the same agent can
drive the physical robot later.

### What it sees (state)
| Feature | Buckets |
|---|---|
| your paddle zone when it decides | left / centre / right (`RL_PADDLE_ZONE_M`) |
| where its previous shot in this rally went | serve / left / centre / right |
| rally length | short / long (`RL_RALLY_LONG`) |
| your rolling return rate vs. the target band | below / in / above |

That gives 72 states. Your paddle zone and its previous shot let it learn set-ups, e.g. pull you
wide and then hit to the open side.

### Rewards
Each shot gets a **difficulty score D** (0 to 1): a weighted mix of its speed, how far it lands
from your paddle, and its spin (`REWARD["difficulty_weights"]`). Then:

| You... | Reward |
|---|---|
| return it | `base_return` (small) + `hard_return_bonus` × D² (the main reward: D=0.3 earns 0.09, D=0.9 earns 0.81) + a small rally bonus (`rally_weight`) |
| miss it | −`miss_penalty` (the shot was too hard) |

Missing is never rewarded. All reward settings live in **one block, `REWARD`, in `config.py`**.
Edit the numbers there to change what the AI wants.

**Target band.** The AI tracks your return rate over your last `window` (20) shots. Every
`adjust_every` (5) shots it compares that rate with `target_band` (70 to 85%):
- above the band (too easy): `hard_return_bonus` × 1.03
- below the band (too hard): `miss_penalty` × 1.03
- inside the band: both move 1.5% back toward their defaults

Both stay within `weight_limits` × their defaults. The band acts as a dead zone, the steps are
small and the window smooths the signal, so it doesn't oscillate.

### How it learns
- **Successor features.** The reward is a weighted sum of parts (returned, D² when returned,
  missed, rally term). For each state and shot the AI learns ψ, the discounted expected sum of
  each part, with a TD update: `ψ(s,a) ← ψ(s,a) + α [φ + γ ψ(s′,a*) − ψ(s,a)]`. The shot's value is
  `Q = w · ψ` using the **current** weights, so band adjustments, and your own edits to the four
  weights, take effect immediately without relearning. A rally end (your miss, or the skeleton's
  miss) is terminal, so there is no bootstrap. `RL_GAMMA` (0.7) makes it value set-ups over the next
  few shots.
- **Note:** ψ was learned under the AI's past greedy choices, and those choices depend on the
  weights. After a weight change its values can therefore drift slightly until it re-learns. This
  is expected.
- **Combinations, with a fallback.** A joint table holds every (state, full shot) combination, so
  it can learn things like "fast to your backhand is too hard, but fast to your forehand is fine".
  Three small separate tables (placement, speed, spin) generalise across combinations it hasn't
  tried yet. The estimate blends them as `β·joint + (1−β)·separate`, with `β = visits / (visits + RL_BACKOFF_K)`.
- **Learning rate.** Each entry first averages its samples (1/visits), then uses a constant
  `RL_ALPHA`, so it keeps tracking you as you improve.
- **Explore → exploit.** ε starts at `RL_EPSILON_START` (0.9, mostly random shots). Every
  `RL_HITS_PER_STEP` (5) of your hits it is multiplied by `RL_EPSILON_DECAY` (0.85), down to
  `RL_EPSILON_MIN`. `RL_HIT_COUNT_MODE = "session"` counts all your hits, so it keeps progressing
  after a miss; `"rally"` counts only hits in the current rally.
- **Gradual unlocks** (`RL_PROGRESSIVE_UNLOCK`). At first it only plays slow/medium shots with no
  spin or sidespin. Fast shots unlock below `RL_UNLOCK_FAST_EPSILON`, and topspin/backspin below
  `RL_UNLOCK_TOPBACK_EPSILON`.
- **Speed.** A decision plus an update takes well under a millisecond (numpy), and saving happens
  only on Esc-to-menu and on quit, so the frame rate is unaffected.

### Saving and resetting
- With `RL_PERSIST = True`, everything is saved to `RL_SAVE_PATH` (`models/dynamic_agent.json`) on
  Esc-to-menu and on quit, and loaded at the next start: the tables, visit counts, ε, the hit
  counter and the adjusted weights.
- **Old models are never loaded silently.** If the file comes from another version or reward
  design, or from other settings that change what was learned (γ, actions, states, the difficulty
  score's exponent/weights/distance, `rally_cap`), the game says so in the console and on the
  Dynamic menu/panel, renames the file to `*.bak` and starts fresh. Changing only the four reward
  weights keeps the model.
- **Reset:** in the menu, with Dynamic selected, press **R**. Or delete the file.

### The Dynamic panel
While you play in Dynamic mode, a panel shows:
- ε and γ
- whether the last shot was EXPLORE or EXPLOIT, and which shot it was
- that shot's difficulty and reward
- whether its estimate came mostly from the joint table or the separate tables
- your return rate against the target band
- the current bonus and penalty weights

Set `RL_PANEL_DEBUG_ONLY = True` to show it only with the F1 overlay.

## Architecture

```
MotorIMU thread ──(legoeducation BLE loop)──► SwingDetector ──► queue ─┐
PoseTracker thread (OpenCV + PoseLandmarker) ──► wrist history ────────┤
Keyboard (space / arrows) ─────────────────────────────────────────────┤
                                                                       ▼
main.py pygame loop ─► GameLogic (rules + spin flights) ─► Renderer (camera3d, scene, theme,
                          │   ▲                                         paddle, skeleton)
                          ▼   │ Opponent interface: ModeOpponent → SimulatedOpponent (Easy/Medium/Hard)
                          │   │                                  → RLOpponent (Dynamic) → DynamicAgent
                    on_score ─► ScorePublisher ─► ../mqttlib.py ─► ME193/RQ-D2/<Name>
```

| File | Role |
|---|---|
| `config.py` | every tunable value |
| `main.py` | wiring, input handling, clean shutdown |
| `motor_imu.py` | BLE nearest-device selection, connect/reconnect, IMU calibration, swing detection (twist excluded), handle-roll filter, haptics + IMU quiet window |
| `spin.py` | `RollFilter` (complementary filter), swing → spin mapping, spin → flight shape (Magnus curve, bounce) |
| `pose_tracker.py` | MediaPipe Tasks PoseLandmarker, mirroring, paddle mapping, direction |
| `player_input.py` | merges pose and keyboard paddle sources |
| `game.py` | `GameLogic` and `Flight` (pure, tested); `status_color` (also used by ../bowling and ../minigolf) |
| `renderer.py` | per-frame drawing: ball, shadow, trail, spin seam, my paddle, HUD, menu, debug |
| `camera3d.py` | perspective camera (position, height, pitch, FOV) |
| `scene.py` | pre-rendered floor shadows, table (legs, frame, thickness, lines) and net |
| `theme.py` | `Theme` base + `HalloweenTheme` / `ClassicTheme`: colours, fonts, sky, scenery, ambient animation |
| `paddle.py` | 3D table-tennis paddle (blade, edge, handle, rim glow) and the swing animation |
| `skeleton.py` | the cartoon skeleton opponent (visual only, driven by game state) and its arm/paddle kinematics: stances, swing keyframes, pole-vector elbow, grip, face aim and spin tilt |
| `opponent.py` | `Opponent` interface (with optional learning hooks), `SimulatedOpponent`, and `ModeOpponent` (routes each difficulty to its opponent) |
| `rl_agent.py` | the Dynamic AI: `ShotCommand`, `State`, difficulty score, `RewardModel` (weights + target band), `DynamicAgent` (successor-feature Q-learning with factored backoff, save/load) |
| `rl_opponent.py` | `RLOpponent`: the agent behind the `Opponent` interface (decides shots, learns from your returns/misses, panel info) |
| `difficulty_selector.py` | `DifficultySource` interface and the manual selector |
| `apriltag_selector.py` | **stub**: tag→command mapping and debounce are done; detection is left to do |
| `mqtt_client.py` | thin wrapper over `../mqttlib.py` |
| `imu_probe.py` | live IMU readout for tuning swing thresholds |
| `imu_axis_test.py` | finds the handle's roll axis, sign and gyro scale |

### Swapping in the real robot later
Write `RobotOpponent(Opponent)` in a new file:
- `ball_incoming()` publishes the incoming ball (x, arrival time, speed) to the robot's topic.
- `poll()` returns a `Shot` once the robot reports a hit, or `MISS` once it reports a miss. If
  the robot never answers, `OPPONENT_TIMEOUT_S` counts it as a miss.

Then put it into the `ModeOpponent` in `main.py`, either as the default or for a difficulty of
its own. `IncomingBall.spin` carries the spin I put on the ball, and `Shot.spin` (optional, default
none) lets the robot put spin back. The skeleton only reads game state, so it animates for the robot
exactly as it does now. To let the Dynamic AI drive the robot, send each
`ShotCommand.to_dict()` (placement, speed, spin, plus their numeric values) to the robot instead
of launching the simulated shot.

### Adding AprilTags later
1. Implement `AprilTagSelector._detect_tag_ids(frame)`, for example with `cv2.aruco` and
   `DICT_APRILTAG_36h11`; see `../AprilTagTracking`.
2. Feed it camera frames.
3. Set `APRILTAG_ENABLED = True`.

Tag IDs map to commands in `APRILTAG_COMMANDS` (0 = start, 1/2/3/4 = Easy/Medium/Hard/Dynamic).
The game code doesn't change.

## Concert Rally (rhythm game)
`rhythm/` holds a separate rhythm game built on the same paddle, IMU and pose code: the ball
arrives on the beat of a song, and you swing on time in the right lane. Run it with
`..\..\my_env\Scripts\python.exe rhythm_main.py` from inside `rhythm/`.

It reuses this folder's modules by import and changes none of them. It never uses MQTT.

[rhythm/README.md](rhythm/README.md) covers:
- songs, the chart format and `make_chart.py`;
- tap-to-chart;
- the test song;
- latency calibration;
- scoring, controls and config.

## Notes on `mqttlib.py`
`../mqttlib.py` is **unchanged**. `mqtt_client.py` imports it from the parent folder and adds three things:
- **Background connect with retry.** `mqttlib`'s connect raises an error if the broker is unreachable, so the wrapper keeps retrying in the background.
- **UI connection status.** The wrapper attaches paho `on_connect`/`on_disconnect` callbacks to the paho client inside `MQTTClient`, reached through its private `_client` attribute.
- **Score resend.** After a reconnect, the latest score is sent again.
