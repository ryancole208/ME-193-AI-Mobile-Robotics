# Deploying the Drive-Straight policy on the real robot

Source of truth: `drive_task.py` and `model_499.pt` in this folder, as copied from the training machine.
Re-check this file whenever either of them changes.

## 1. What the policy needs (checked against the checkpoint)

`model_499.pt` (rsl_rl, iteration 499):

| Network | Input size | Layers |
|---|---|---|
| actor  | **7**  | normalizer → Linear(7,64) → ELU → Linear(64,64) → ELU → Linear(64,2) |
| critic | 13 (training only, not deployed) | lin_vel 3 + ang_vel 3 + heading/offset 3 + wheel_vel 2 + last_action 2 |

The actor only uses things an IMU can measure, so we can deploy it without retraining.

### Observation vector (this order, no extra scaling in `drive_task.py`)

| # | Name | Sim source | Frame / units | Real robot | Normalizer mean | std |
|---|---|---|---|---|---|---|
| 0 | ω_x (roll rate)  | `base_ang_vel` | chassis body frame, rad/s | gyro X − bias | +0.0676 | 3.4519 |
| 1 | ω_y (pitch rate) | `base_ang_vel` | chassis body frame, rad/s | gyro Y − bias | −0.3205 | 3.1825 |
| 2 | ω_z (yaw rate)   | `base_ang_vel` | chassis body frame, rad/s | gyro Z − bias | +0.0623 | 2.0598 |
| 3 | sin(yaw) | `heading_only` | yaw of the chassis in the world, rad | integrated yaw, zero at start | −0.0029 | 0.2075 |
| 4 | cos(yaw) | `heading_only` | " | " | +0.9770 | 0.0489 |
| 5 | previous action, left  | `last_action` | unitless, **raw** policy output | what the policy output last step | +0.0081 | 1.9300 |
| 6 | previous action, right | `last_action` | " | " | +0.6471 | 1.1053 |

* **Body frame** (from `robot.xml`): x forward (away from the caster), y to the left (`left_wheel` side), z up.
  Positive rates follow the right-hand rule: +ω_z means turning left (counter-clockwise seen from above).
* **Yaw**: the sim computes yaw from the chassis quaternion (`atan2(2(wz+xy), 1−2(y²+z²))`), the heading angle
  in the world. `START_YAW = 0`, so every sim episode starts at yaw = 0. On the robot we zero yaw at the start of
  each run and integrate the gyro. The roll/pitch rates are large (std ≈ 3.3 rad/s: the robot rocks a lot), so
  we integrate the **world** yaw rate `(sin(roll)·ω_y + cos(roll)·ω_z) / cos(pitch)`, using roll/pitch from the
  complementary filter. Integrating only the body-frame ω_z would drift while the robot rocks.
* **Normalization** (rsl_rl `EmpiricalNormalization`): `obs_n = (obs − mean) / (std + 0.01)`. The `+0.01`
  matters here: cos(yaw) has std 0.0489. Step 5 checks this against the real checkpoint.
* **Previous action** is the policy's raw output, before any clipping. mjlab's `last_action` returns the raw
  action and the sim never clips it. It is 0 on the first step of a run (that is how the sim resets it).
* No observation noise was used in training. `enable_corruption` is on, but no term defines any noise.

### Action
* The actor outputs 2 numbers `[a_left, a_right]`. At deployment, use the network output directly (the mean).
  The trained Gaussian std (0.088, 0.069) is only for exploration during training.
* Sim mapping (`JointVelocityActionCfg`, scale = `MAX_WHEEL_SPEED` = 8.0, offset 0, **no clipping**):
  target arm speed = 8.0 · a rad/s, positive = rolls the robot forward (+x).
* Sim motor: `torque = clip(WHEEL_KD · (target − ω), ±WHEEL_TORQUE_LIMIT) = clip(0.5 · (8a − ω), ±3) N·m`.
* Real mapping: `PWM duty = clip(a, −1, 1)`, with a sign per motor found in step 3.
* The policy often outputs |a| > 1 (left previous action std = 1.93), so clipping is a real sim-to-real
  difference. See section 3.

### Timing and termination
* Control period = `timestep × decimation` = 0.005 s × 4 = **0.020 s (50 Hz)**.
* Training episode length: 10 s (`EPISODE_S`), so the real run timeout defaults to 10 s.
* Tilt termination: `bad_orientation`, limit **60°** between the chassis z axis and vertical. The real
  safety stop uses the same 60° from the complementary filter: `tilt = acos(cos(roll)·cos(pitch))`.
* The 5 m off-line termination cannot be measured without position sensing, so the robot doesn't use it.

## 2. Direction convention (checked in robot.xml)

Both arm bodies have `quat="0.707107 -0.707107 0 0"` (−90° about x), which turns the joint axis `0 0 1` into the
chassis **+y** axis, for both arms. For positive rotation about +y, the bottom of the arm moves backward
(−x) relative to the chassis, so the ground pushes the robot forward. In other words, **for a positive joint velocity
the top of each arm moves forward, like a wheel rolling forward.** That matches your description. Note that the
two motors face opposite ways, so the same PWM sign probably turns them in opposite directions (step 3 finds out).

## 3. Motor mapping (sim is final: drive_task.py, robot.xml and model_499.pt are not changed)

* The real motors (TT gear motors, Cytron Maker Drive, 6 V battery) walk the robot fine, so the robot uses the
  sim as it is: `PWM duty = clip(a, -1, 1)`, with a sign per motor from step 3.
* The previous-action observation still uses the raw (unclipped) action, as in the sim.

## 4. IMU (MPU-6050 on a GY-521)

* Mounting (checked in step 4): **upside down**, chip facing the floor, printed X arrow forward, Y arrow to the
  robot's right. So body x = +chip X, body y = −chip Y, body z = −chip Z (`AXIS_SIGN = {+1, −1, −1}` in `imu.h`).
* Gyro range: **±2000 °/s** (≈ ±35 rad/s). The default ±250 °/s (4.4 rad/s) would saturate: the sim's roll/pitch
  rates have a std of about 3.3 rad/s.
* Accelerometer ±8 g. On-chip low-pass filter (DLPF) at 44 Hz: it removes motor buzz and adds only ~5 ms of delay.
* Gyro bias: average 1–2 s of samples with the robot still, at startup, and subtract.

### Wiring the GY-521 to the UNO Q
The UNO Q's microcontroller runs at **3.3 V logic**. The GY-521 has its own 3.3 V regulator, and its I²C pull-ups
go to that 3.3 V rail, so the I²C lines stay at 3.3 V whichever supply you use.

| GY-521 | UNO Q (header) | or Qwiic connector |
|---|---|---|
| VCC | 3.3V pin (safest; 5V also works through the module's regulator) | 3.3 V (red) |
| GND | GND | GND (black) |
| SDA | SDA pin (top-left header, next to AREF) → `Wire` | SDA (blue) → `Wire1` |
| SCL | SCL pin → `Wire` | SCL (yellow) → `Wire1` |
| AD0 | leave open or GND → address 0x68 | |
| INT, XDA, XCL | not connected | |

Use short wires and route them away from the motor leads. Unplug power before you change the wiring.

## 5. Policy export (step 5)

| File | What | Made by |
|---|---|---|
| `sketch/policy_weights.h` | normalizer mean/std + all actor weights (float32) for the STM32 | `tools/make_policy_header.py` (on the board, no PyTorch) |
| `sketch/policy.h` | `policyRun(obs, act)`: the network in C, ~0.46 ms per step on the STM32 | by hand |
| `policy.onnx` | actor including the normalizer (input = raw obs) | `tools/export_policy.py` (training machine) |
| `tools/policy_numpy.py` | NumPy re-implementation | by hand |

Checks: STM32 C code vs pure-Python reference on 20 fixed vectors: max error 1.7e-6 (PASS).
NumPy vs pure Python: 9e-16 (PASS). Checkpoint vs ONNX vs NumPy (and rsl_rl's normalizer, to confirm eps = 0.01):
run `tools/export_policy.py` on the training machine.

## 6. Where things run (step 6)

The policy runs **on the STM32** at 50 Hz: IMU read → observation → `policyRun` → PWM, all in one fixed-rate
loop with no Bridge round trip in the control path. Linux (Python) only sends start/stop and writes the log file.
Bridge latency doesn't affect control, so it wasn't measured.
