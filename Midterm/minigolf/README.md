# Spooky Minigolf (Wii Sports putting style)

You putt with the same LEGO Education **Double Motor** and webcam as the ping-pong and bowling games.
Hold the motor like a putter. **Twist it to turn the putter face** (your aim), then **swing quickly to
putt**. The putt fires on the quick swing, and its speed sets the power. The webcam adds a little
**push/pull** from sideways wrist motion around impact. Each game is **5 freshly generated
Halloween holes**. Holes 1–2 are open corridors, and holes 3–5 add obstacles. The running stroke
total is published over MQTT as a float to `ME193/RQ-D2/<PLAYER_NAME>`.

Nothing in `../` (the ping-pong game) or `../bowling` was changed. This folder reuses
`../motor_imu.py`, `../pose_tracker.py`, `../mqtt_client.py`, `../difficulty_selector.py` and
`../game.py` (for its status lights) by import. `config.py` loads `../config.py` first, so the motor,
webcam, pose, MQTT and haptic settings tuned for ping pong carry over. Override them here if needed.

## Setup
Use the same venv and requirements as ping pong (see `../README.md`). Nothing new to install.

## Running
```powershell
cd "ME 193 AI Mobile Robotics\Midterm\minigolf"
..\..\my_env\Scripts\python.exe golf.py              # motor + webcam + MQTT
..\..\my_env\Scripts\python.exe golf.py --keyboard   # no hardware at all
..\..\my_env\Scripts\python.exe golf.py --no-mqtt    # stay offline
```
`--no-motor` and `--no-camera` turn off one device. Hold the motor still for about a second
after it connects so it can calibrate gravity.

| Key | Action |
|---|---|
| 1 / 2 (or click) | Easy / Hard |
| Enter (or START / Play again) | start a round; during a hole's flyover, skip it |
| Esc | back to the menu |
| twist the motor | turn the putter face (aim) |
| C | recenter the twist (however you hold the motor now = straight at the suggested line) |
| swing the motor quickly | putt |
| ← / → or A / D | turn the aim too (hold **Shift** for fine aim) |
| hold **Space**, release | keyboard putt: the power meter sweeps up and down while held |
| hold **V** | overhead view of the whole hole |
| F1 | debug overlay (live IMU, aim, last putt, ball state) |
| Q / close window / Ctrl+C | quit |

**Easy** has a bigger cup, a long aim guide and slower ghosts and spinners. **Hard** has a
regulation 4.25 in cup, a short guide and faster obstacles.

## How a putt works
1. **Aim.** Each hole opens with a flyover, then the camera drops in behind the ball. The yellow
   guide starts on a suggested line: at the cup if it's in sight, otherwise at the farthest part
   of the corridor in sight, avoiding static obstacles when possible.
   - **Twist the motor** to turn the putter face off that line, up to `AIM_TWIST_MAX_DEG`
     either way. The "twist" bar under the info panel shows how far you've turned it.
   - The twist resets to zero every time a new putt is set up, so however you're holding the
     motor then counts as straight. Press **C** to reset it by hand, for example if it has drifted.
   - With `TWIST_AXIS = "vertical"` the twist is the rotation around gravity: turning a putter
     face with the motor hanging down, however it sits in your hand. The forward-back swing
     barely changes it, and the twist stops counting while you swing.
   - ←/→ still turn the aim for big turns, such as banking around a corner.
2. **Swing.** Take a slow backswing, then swing forward **quickly**. The putt fires the moment the
   swing speed passes `PUTT_GYRO_THRESHOLD_RAW` for `PUTT_CONFIRM_SAMPLES` readings in a row, so a
   single bump doesn't count. Twisting never fires a putt. Keep the backswing slow so it stays
   under the threshold.
3. **Launch.**
   - **Direction** is your aim (suggested line + twist + arrow keys) when the swing fired, plus up
     to `PUSH_MAX_DEG` from sideways wrist motion on the webcam.
   - **Power** is the peak swing speed over the next `PUTT_PEAK_WINDOW_S` (0.12 s). It sets the
     ball speed (`PUTT_SPEED_*`). The power bar fills live as you swing.
   - Follow-through and motion in the next `PUTT_COOLDOWN_S` (1 s) are ignored.
4. **Roll.** The ball rolls with rolling resistance, bounces off walls and obstacles, and drops if
   it's slow enough for how centred it is. Too fast or too far off-centre and it **lips out**.
   The motor buzzes when the ball drops.
5. **Score.** You get a callout (HOLE IN ONE!, Birdie!, Par, Bogey, …) and the scorecard updates.
   After `MAX_STROKES` (8) you pick up.

The old wrist-joystick aim (hold your wrist left or right of centre to turn) is still there.
Set `AIM_POSE_JOYSTICK = True` to use it on top of the twist.

## The holes
Each hole is a corridor of 1.25 m cells laid out by a random walk from the tee: straight runs
joined by 90° turns, never touching itself. It is closed by orange-topped walls. Longer holes
have more turns, and later holes may get a second lane on a long straight. Par is 2–4, depending
on length, turns and obstacles. Outside the walls are dead trees, gravestones, iron fences and
jack-o'-lanterns, all under a full moon with bats.

| Hole | Layout | Obstacles (the first is always there) |
|---|---|---|
| 1 | straight | none |
| 2 | one turn | none |
| 3 | 1–2 turns | **tombstone**, pumpkin, slime |
| 4 | 2 turns | **ghost**, pumpkin, tombstone, cauldron |
| 5 | 2–3 turns | **bone spinner**, ghost, tombstone, cauldron, slime |

- **Pumpkin**: a round bumper.
- **Tombstone**: a slab jutting in from a wall, blocking about half the lane.
- **Ghost**: floats back and forth across the lane, and the ball bounces off it. Time your putt.
- **Bone spinner**: a bone spinning around a skull post. It bats the ball away.
- **Slime**: a sticky puddle with about 6× the rolling friction.
- **Witch's cauldron**: +1 stroke, and the ball goes back to where you putted from.

Every obstacle sits to one side of its cell or moves, so there is always a way through. The tests
check this on many random courses.

## Tests
```powershell
..\..\my_env\Scripts\python.exe -m pytest tests     # run from this folder; no hardware needed
```
| File | Covers |
|---|---|
| `test_course.py` | 5 reproducible holes, corridor rules, walls closing every edge, obstacles only on holes 3–5, fair gaps, suggested aim |
| `test_physics.py` | rolling distance, wall bounces, cup capture vs. lip-out, each obstacle, hazards, the ball never escaping, every generated hole sinkable |
| `test_putt_detector.py` | quick swing fires at once with peak power, slow backswing and bumps ignored, cooldown, twist angle (integration, sign, drift deadzone, frozen while swinging, vertical vs axis), `GolfMotor` with a fake device |
| `test_golfer_input.py` | twist aim (offset, clamp, recenter, history), keyboard aim, optional wrist joystick, push/pull, the keyboard power meter |
| `test_game_flow.py` | intro/skip, full rounds, score names, penalty strokes, pick-ups, MQTT stroke totals, arming, real-physics round |
| `test_config_and_render.py` | config inheritance (and that `../config.py` is untouched), camera projection, headless rendering of every state |

Run the ping-pong (`..`) and bowling (`../bowling`) tests separately. Each folder has a module
named `config`, so they can't share one pytest run.

## Tuning (`config.py`)
Run `..\..\my_env\Scripts\python.exe putt_probe.py`. It shows the live swing speed, twist rate and
twist angle, and prints each putt's strength, ball speed and roughly how far it would roll.
Press Enter to re-zero the twist.

1. **Twist.**
   - Twist the motor about 90°. If the angle doesn't read about 90, adjust `TWIST_DEG_PER_RAW_S`.
     The LEGO gyro units aren't documented, and 0.1 is a guess.
   - If the aim turns the wrong way, set `TWIST_SIGN = -1.0`.
   - If the angle creeps while you hold still, raise `TWIST_DEADZONE_RAW`.
   - If twisting barely registers, you're holding the motor so the twist isn't around vertical.
     Set `TWIST_AXIS` to `"x"`, `"y"` or `"z"`: the raw `g=(x,y,z)` column that changes most
     while you twist.
   - `AIM_TWIST_GAIN` scales twist to aim (2.0 = a small twist turns the aim a lot).
     `AIM_TWIST_MAX_DEG` limits it.
2. **Swing.**
   - Gentle putts don't fire: lower `PUTT_GYRO_THRESHOLD_RAW`.
   - Your backswing fires the putt: raise it (or swing back slower).
   - The power is the swing speed relative to that threshold. Use `PUTT_SPEED_PER_STRENGTH` to
     scale how far your normal putt goes.
   - `PUTT_MODE` is `"gyro"` (swing rotation only), so linear bumps don't fire putts.
3. **Aim from the webcam.** `PUSH_DEG_PER_UNIT` and `PUSH_MAX_DEG` set the push/pull. Set the
   push gain to 0 to aim with the twist only.
4. **Feel.**
   - `ROLL_DECEL_M_S2` and `ROLL_DRAG_PER_S` set how fast the green is.
   - `DIFFICULTIES[...]` sets the cup size, capture speed, guide length and obstacle speed.
   - `HOLE_PLANS` and `HOLE_OBSTACLES` set each hole's size, turns and obstacles.

## Files
| File | Role |
|---|---|
| `golf.py` | entry point: wiring, input handling, clean shutdown |
| `config.py` | loads `../config.py`, then the minigolf values |
| `putt_detector.py` | `PuttDetector` (quick swing → putt + power, twist → putter angle) and `GolfMotor(MotorIMU)` |
| `golfer_input.py` | aim = suggested line + motor twist + keys, push/pull, keyboard power meter |
| `course.py` | random hole generation, walls, Halloween obstacles, decorations, suggested aim |
| `physics.py` | `PuttSim`: 2D rolling, bounces, obstacles, cauldrons, cup capture and lip-outs |
| `golf_game.py` | game flow (menu → intro → aiming → rolling → … → game over), scoring |
| `golf_render.py` | pygame view: camera behind the ball, night sky, course, obstacles, Mii golfer, scorecard, minimap |
| `putt_probe.py` | live readout of swing, twist and putts for tuning |
