# Bowling (Wii Sports style)

You bowl with the same LEGO Education **Double Motor** and webcam as the ping-pong game.
The motor's IMU detects the **throw**, its **speed** (swing strength) and its **hook** (twisting
the motor at release). The webcam decides **where you stand** and adds a little **aim** from
sideways wrist motion. One bowler plays 10 frames with standard scoring. You choose
**Bumpers** or **No Bumpers** first. The running total is published over MQTT as a float to
`ME193/RQ-D2/<PLAYER_NAME>`.

Nothing in `../` (the ping-pong game) was changed. This folder reuses `../motor_imu.py`,
`../pose_tracker.py`, `../mqtt_client.py`, `../difficulty_selector.py` and `../game.py`
(for its status lights) by import. `config.py` loads `../config.py` first, so motor, webcam,
pose, MQTT and haptic settings tuned for ping pong carry over. Override them here if needed.

## Setup
Use the same venv and requirements as ping pong (see `../README.md`). Nothing new to install.

## Running
```powershell
cd "ME 193 AI Mobile Robotics\Midterm\bowling"
..\..\my_env\Scripts\python.exe bowling.py              # motor + webcam + MQTT
..\..\my_env\Scripts\python.exe bowling.py --keyboard   # no hardware at all
..\..\my_env\Scripts\python.exe bowling.py --no-mqtt    # stay offline
```
`--no-motor` and `--no-camera` turn off one device. Hold the motor still for about a second
after it connects so it can calibrate gravity.

| Key | Action |
|---|---|
| 1 / 2 (or click) | Bumpers / No Bumpers |
| Enter (or START / Play again) | start a game |
| Esc | back to the menu |
| ← / → | move along the foul line (used when the webcam isn't tracking you) |
| A / D | turn your aim (like A + D-pad on the Wii) |
| Space | throw (keyboard fallback); ↑ / ↓ set its power; hold **Z** / **C** for a left / right hook |
| F1 | debug overlay (live IMU, last throw's twist per axis, launch values) |
| Q / close window / Ctrl+C | quit |

## How a throw works
1. **Aim.** Stand where you want to release. The webcam maps your wrist's sideways position
   to a spot on the foul line, using the same `POSE_X_RANGE` as ping pong. A/D turn the
   yellow guide.
2. **Swing.** Swing the motor like a real bowling ball: backswing, then forward swing, then
   follow-through. The throw starts when the swing thresholds trip and ends once the motor
   has been quiet for `THROW_END_QUIET_S`. The **release** is the fastest moment of the swing.
3. **Launch.** The ball leaves from where you stood when the swing started.
   - **Speed** comes from swing strength (`BALL_SPEED_*`).
   - **Angle** is your A/D aim plus up to `SWING_AIM_MAX_DEG` from sideways wrist motion
     around release.
   - **Hook** comes from your twist rate around `SPIN_GYRO_AXIS` at release.
4. **Roll.** The ball barely hooks on the oiled front 12 m and breaks on the dry back end.
   - Pins fall when hit hard enough, and fallen pins take out their neighbours.
   - Without bumpers, a ball off the edge drops into the gutter. With bumpers, it bounces back.
   - The camera follows the ball, and the motor buzzes when the ball reaches the pins.
5. **Score.** You get a Wii-style callout (STRIKE!, Double!, Turkey!, SPARE!, Split!,
   Gutter...), the scorecard updates, and the pinsetter respots the remaining pins or resets
   the rack.

## Tests
```powershell
..\..\my_env\Scripts\python.exe -m pytest tests     # run from this folder; no hardware needed
```
| File | Covers |
|---|---|
| `test_scoring.py` | strikes, spares, 10th frame, marks, pending bonuses, running total, perfect/gutter games |
| `test_lane.py` | pin layout, split detection, pocket strike, thin hits, gutter vs bumpers, hook direction, determinism |
| `test_throw_detector.py` | pendulum → one throw, release at the peak, twist → spin, twitches, cooldown, `BowlingMotor` with a fake device |
| `test_bowler_input.py` | keyboard/pose stand position, preset and swing aim |
| `test_game_flow.py` | full games, callouts, rack resets, throw arming, MQTT scores, a full game on real physics |
| `test_config_and_render.py` | config inheritance (and that `../config.py` is untouched), headless rendering of every state |

Run the ping-pong tests separately from `..` (`pytest tests`). Both suites have a module named
`config`, so they can't share one pytest run.

## Tuning (`config.py`)
1. **Throw detection.** Run `..\..\my_env\Scripts\python.exe throw_probe.py` and bowl.
   - The start thresholds default to ping pong's `SWING_*` values. Raise
     `THROW_*_THRESHOLD_*` if just walking up triggers a throw.
   - If one throw splits in two (it ends at the top of the backswing), raise
     `THROW_END_QUIET_S` or lower `THROW_END_FRACTION`.
   - Use `BALL_SPEED_PER_STRENGTH` to scale how fast your normal throw is.
2. **Hook axis.** In the probe, compare the `twist x/y/z` values of straight throws with
   throws where you turn the motor like a doorknob at release. Set `SPIN_GYRO_AXIS` to the
   axis that changes most and `SPIN_FULL_RAW` to a strong twist's value. Flip `SPIN_SIGN` if
   the ball hooks the wrong way.
3. **Aim.** Adjust `SWING_AIM_DEG_PER_UNIT` and `SWING_AIM_MAX_DEG`, or set the gain to 0 to
   aim with A/D only.
4. **Feel.**
   - `HOOK_ACCEL_M_S2` and `OIL_LENGTH_M` control the hook.
   - `PIN_TOPPLE_SPEED`, `PIN_FRICTION_M_S2` and `PIN_PIN_RESTITUTION` control pin action:
     lower friction means more strikes.

## Files
| File | Role |
|---|---|
| `bowling.py` | entry point: wiring, input handling, clean shutdown |
| `config.py` | loads `../config.py`, then the bowling values |
| `throw_detector.py` | `ThrowDetector` (pendulum throw, release, twist spin) and `BowlingMotor(MotorIMU)` |
| `bowler_input.py` | stand position and aim from pose snapshots or the keyboard |
| `lane.py` | lane and pin geometry, `RollSim` 2D physics (ball, hook, pins, gutters, bumpers) |
| `scoring.py` | `ScoreSheet`: ten-pin rules, marks and totals |
| `bowling_game.py` | game flow (menu → aiming → rolling → result → … → game over) |
| `bowling_render.py` | pygame view: follow camera, lane, pins, Mii-like bowler, scorecard, pin diagram |
| `throw_probe.py` | live readout of IMU throws for tuning |
