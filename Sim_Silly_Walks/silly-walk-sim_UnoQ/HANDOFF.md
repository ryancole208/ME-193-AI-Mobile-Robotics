# HANDOFF: Silly Walk Sim (sim model → real robot), 2026-11-17

> **⚠️ OUT OF TIME, NO MORE TESTS POSSIBLE.** The owner has run out of time and **cannot run any more tests**:
> no motor-speed measurement, no ONNX export, no sim-vs-real comparison, and **no trial run of the walking policy
> on the robot**. Everything below is ready to run, but the policy loop on the real robot has **never been run**.
> Treat the first run as untested.

## What this project is (and isn't)
- **Goal:** run the trained sim model (`model_499.pt`, from mjlab + MuJoCo) on the real robot (Arduino UNO Q, two TT
  motors on a Cytron Maker Drive, MPU-6050 IMU).
- **No new model and no training.** The trained weights from `model_499.pt` were copied unchanged into
  `sketch/policy_weights.h`, and `sketch/policy.h` does the same math as the sim's network. On the STM32 it matches
  the original to within 1.7e-6.
- **Do not modify `drive_task.py`, `model_499.pt` or `robot.xml`.** It's a working sim.
- Motor torque is fine (the robot walks with the driver's test buttons), so don't revisit the motor model.
- Ask the owner before installing packages on the UNO Q.

## Current state of the board
- The app `silly-walk-sim` is **running** with the final code (policy sketch + small local web server).
- Checked: the terminal → web server → Bridge → sketch chain works (`robot.py stop` returned `{'ok': True}`).
  The motors haven't run under the policy.
- `finger-safe` (the owner's other app) is stopped. Restart it with `arduino-app-cli app start ~/ArduinoApps/finger-safe`.

## How to run the sim model on the robot (all from the UNO Q terminal, no browser needed)
1. Robot on the floor, caster at the back, room to walk. Battery on (6 V).
2. If the app isn't running: `arduino-app-cli app start ~/ArduinoApps/silly-walk-sim` (the motors stay off).
   If it's running old code, use `app stop` then `app start` (`app restart` has failed with "Is Running").
3. Run:
   ```bash
   python3 ~/ArduinoApps/silly-walk-sim/robot.py start 1.0 10    # full power (= sim), 10 s
   python3 ~/ArduinoApps/silly-walk-sim/robot.py start 0.5 10    # gentler: all PWM x 0.5
   python3 ~/ArduinoApps/silly-walk-sim/robot.py stop
   python3 ~/ArduinoApps/silly-walk-sim/robot.py status
   ```
   Keep the robot still for the first 1 s (gyro bias, yaw = 0). The run stops on the timeout, on tilt > 60° (the
   sim's termination), on Ctrl+C, or with `robot.py stop`.
4. `status` shows `starting` until the first run. That's harmless: the sketch said "idle" before Python was listening.

## What runs where
- **STM32 (sketch), every 20 ms** (= sim timestep 0.005 s × decimation 4):
  1. Read the IMU → angular velocity (rad/s, body frame) and yaw (rad, zeroed at start).
  2. Build the observation `[wx, wy, wz, sin(yaw), cos(yaw), prev_action_L, prev_action_R]`.
  3. Run the model.
  4. Set each motor: `duty = clip(action × 8.0 / NO_LOAD_SPEED, ±1) × scale`, with motor signs from step 3.
  5. Keep the **unclipped** action as the next previous action, like the sim.
- **Linux (python/main.py):** passes start/stop to the sketch and saves a record of each run.
- **Run log (optional):** a CSV per run in `~/ArduinoApps/silly-walk-sim/logs/`, sent after the run ends, so it
  doesn't affect the robot. It was requested for checking/comparison (steps 8–9) and can be removed if unwanted.
- **Web page:** `assets/index.html` at `http://<board-ip>:7000` does the same as `robot.py` from another device.
  The owner can't use it, so it's optional.

## Step status
| Step | Status |
|---|---|
| 1. Check obs/actions | ✅ Actor takes 7 IMU-only inputs, so the model can run on the robot as is. Details in `DEPLOYMENT.md`. |
| 2. Motor no-load speed | ❌ **Skipped (no time).** `NO_LOAD_SPEED_L/R = 8.0` (= sim's `MAX_WHEEL_SPEED`), so duty = clipped action. |
| 3. Motor direction | ✅ Tested: both signs `+1` turn the arms forward. |
| 4. IMU | ✅ Tested: mounted upside down, fixed with `AXIS_SIGN = {+1,-1,-1}`; axis signs checked by hand; yaw drift 0.06° / 10 s. |
| 5. Model on the STM32 | ✅ Checked on the STM32 vs Python: max error 1.7e-6, 0.46 ms per step. ❌ ONNX export skipped (no training machine). |
| 6. Where it runs | ✅ On the STM32 at 50 Hz. |
| 7. Policy loop | ⚠️ **Running on the board, never run with the motors.** |
| 8. Run log | ⚠️ Written, never exercised. |
| 9–10. Sim-vs-real comparison, training suggestions | ❌ Not started. |

## Known untested points (check first if the robot misbehaves)
- **The model often outputs |action| > 1** (the sim never clips). The real PWM is clipped at 100 %, a known
  sim-to-real difference.
- Motor speed was never measured. If the real arms turn much faster than 8 rad/s (76 rpm) at full PWM, the robot
  will move more aggressively than the sim. Fix: measure, then set `NO_LOAD_SPEED_L/R` in `sketch/sketch.ino`
  (sketch: `test_sketches/step2_motor_speed.ino`, analysis: `tools/analyze_motor_speed.py`).
- Normalizer eps = 0.01 (rsl_rl default) is assumed, not checked against rsl_rl.

## Board quirks
- `arduino-app-cli monitor` printed nothing when run non-interactively. `python3 tools/monitor.py [seconds] [file]` works.
- Never call `pinMode(OUTPUT)` on the motor PWM pins on this Zephyr core (it breaks PWM).
- If something fails: `arduino-app-cli app logs ~/ArduinoApps/silly-walk-sim --tail 50`.

## Files
| File | What |
|---|---|
| `robot.py` | **Start/stop the robot from the terminal** |
| `sketch/sketch.ino` | Active policy loop (copy of `test_sketches/step7_policy_run.ino`) |
| `sketch/policy.h`, `sketch/policy_weights.h` | The trained model in C (weights unchanged from `model_499.pt`) |
| `sketch/imu.h`, `sketch/motors.h` | IMU (filter, yaw, axis mapping) and motor driver (pins D10/D11 left, D6/D5 right, signs) |
| `python/main.py`, `assets/index.html` | Start/stop server, run logs, optional web page |
| `DEPLOYMENT.md` | Observation order/units/normalizer, action mapping, timing, tilt limit, IMU wiring |
| `test_sketches/` | Test sketches for steps 2, 3, 4, 5 and the policy run |
| `tools/` | Checkpoint reader (no PyTorch), header generator, checkers, monitor reader, ONNX export script |
