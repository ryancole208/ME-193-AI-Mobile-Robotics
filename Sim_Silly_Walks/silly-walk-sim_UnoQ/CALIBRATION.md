# Step 2: motor calibration

Goal: measure what the real motors do at full PWM on your battery, so the sim's motor model can match them
(see DEPLOYMENT.md section 3).

## Procedure (about 10 minutes)

1. Check the motor pins at the top of `sketch/sketch.ino` (defaults come from your `motor-test` app).
2. Prop the robot up so both arms spin freely in the air. Put a piece of tape on each arm.
3. Charge the battery. Have a multimeter on the battery terminals if you can.
4. In one terminal: `arduino-app-cli monitor`.
   In another: `arduino-app-cli app start ~/ArduinoApps/silly-walk-sim` (this stops whatever app is running).
5. Film the arms with your phone (slow motion if you have it). The built-in LED is on during each timed run.
   For each motor the sketch runs:
   * **A** 100 % PWM for 20 s: count revolutions.
   * **B** 50 % PWM for 20 s: count revolutions.
   * **C** ramp 0→50 %: note the duty printed when the arm first starts turning.
6. During run A, read the battery voltage.
7. Optional, for the stall torque (the most important missing number): clamp the chassis so one arm points
   horizontally. Hang a small bag on the arm at a known distance `r` from the axle, and add weight (coins, water)
   until the motor at 100 % can no longer lift it. Then `τ_stall ≈ m · 9.81 · r`. To get a constant 100 % for this
   test, temporarily change run A's time to 60 s.

Speed in rad/s = revolutions / 20 s × 2π. (For example, 25 revolutions in 20 s = 7.85 rad/s = 75 rpm.)

## Results (fill in)

| | Left | Right |
|---|---|---|
| Battery voltage under load (V) | | |
| A: revolutions in 20 s at 100 % | | |
| → no-load speed ω_noload (rad/s) | | |
| B: revolutions in 20 s at 50 % | | |
| → speed at 50 % (rad/s) | | |
| C: duty where the arm starts (%) | | |
| Optional: stall torque (N·m) | | |

## What to compare

* **ω_noload vs `MAX_WHEEL_SPEED` = 8.0 rad/s** in `drive_task.py`. A typical 1:48 TT motor runs at ~150–200 rpm
  (16–21 rad/s) at 5–6 V. If yours measures far from 8.0, change `MAX_WHEEL_SPEED`.
* **Speed at 50 % ≈ half of ω_noload?** If it is much lower, or the deadband in C is big (> 15 %), plain
  "duty = action" won't reproduce small actions. Then we add a deadband offset on the robot
  (`duty = sign(a)·(d0 + (1−d0)·|a|)`) instead of changing the sim.
* **Left vs right differ by > 10 %?** Use the slower one for `MAX_WHEEL_SPEED`, and scale the faster motor's
  PWM down on the robot so both match.
