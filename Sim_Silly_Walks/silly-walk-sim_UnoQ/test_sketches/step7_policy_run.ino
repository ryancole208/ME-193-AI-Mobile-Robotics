// =====================================================================================
// Silly Walk Sim - STEP 7 + 8: RUN THE TRAINED POLICY (on the STM32, 50 Hz)
//
// Every 20 ms (= sim timestep 0.005 s x decimation 4):
//   1. read the IMU                         -> angular velocity (rad/s, body frame), yaw (rad, 0 at start)
//   2. build the observation like the sim   -> [wx, wy, wz, sin(yaw), cos(yaw), prevL, prevR]
//   3. run the policy (policy.h)            -> raw action a (NOT clipped; can be > 1)
//   4. action -> PWM, per motor:  duty = clip(a * MAX_WHEEL_SPEED / NO_LOAD_SPEED, -1, 1) * pwmScale
//      (a * MAX_WHEEL_SPEED is the sim's target arm speed; dividing by the motor's real top speed
//       gives the PWM that reaches it, so only targets beyond the real top speed get clipped)
//   5. remember the RAW (unclipped) action as the next "previous action", like the sim
//   6. store a log row in RAM (sent to Linux after the run -> no slow serial traffic in the loop)
//
// Safety: the run stops on (a) tilt > 60 deg (the sim's termination), (b) the timeout,
// (c) the STOP button. The motors are also off whenever no run is active.
//
// Commands from Linux (python/main.py) over the Bridge:
//   start(scalePct, timeoutS)  -> 1 s still for gyro bias, then run
//   stop()
// Messages to Linux: "state"(text), "log_line"(CSV row), "run_done"(reason, rows)
// =====================================================================================

#include <Arduino_RouterBridge.h>
#include "imu.h"
#include "motors.h"
#include "policy.h"

// ---- Constants from drive_task.py (do not change unless the sim changes) ----
const float MAX_WHEEL_SPEED = 8.0f;              // rad/s: action 1.0 = this arm speed in the sim
const float TILT_LIMIT = 60.0f * PI / 180.0f;    // bad_orientation limit_angle
const unsigned long PERIOD_US = 20000;           // 0.005 s x 4 = 50 Hz
const float DT = PERIOD_US * 1e-6f;

// ---- Measured on the real robot (step 2): arm speed at 100 % PWM, no load, 6 V battery ----
// NOT MEASURED (no time): 8.0 = MAX_WHEEL_SPEED, so duty = action (clipped). Replace with measured values later.
const float NO_LOAD_SPEED_L = 8.0f;   // rad/s, left motor
const float NO_LOAD_SPEED_R = 8.0f;   // rad/s, right motor

// ---- Start checks ----
const float START_TILT_MAX = 30.0f * PI / 180.0f;   // refuse to start if not roughly upright
const float STILL_NOISE_MAX = 0.05f;                // rad/s gyro std during the bias measurement
const int MAX_TIMEOUT_S = 30;

// ---- Log buffer: one row per control step ----
struct LogRow {
  float t;            // s since run start
  float obs[POLICY_OBS];
  float act[POLICY_ACT];   // raw policy output
  float duty[2];           // final PWM duty (-1..1), after clip and scale
  float roll, pitch, tilt, yaw;
  uint16_t computeUs;      // time spent in this step (IMU + policy + PWM)
};
const int MAX_ROWS = MAX_TIMEOUT_S * 50 + 10;
LogRow logRows[MAX_ROWS];
int nRows = 0;

enum State { IDLE, CALIBRATING, RUNNING, SENDING };
volatile State state = IDLE;
volatile bool startRequested = false, stopRequested = false;
volatile int reqScalePct = 50, reqTimeoutS = 10;

Imu imu;
bool imuOk = false;
float pwmScale = 0.5f;
unsigned long runTimeoutUs = 10000000UL;
unsigned long runStartUs = 0, nextUs = 0;
float prevAction[POLICY_ACT] = {0, 0};
const char* stopReason = "";
int sendIndex = 0;

void ledOn(bool on) { digitalWrite(LED_BUILTIN, on ? LOW : HIGH); }  // active-low LED

// ---- Bridge handlers (provide_safe: they run between loop() calls, never mid-step) ----
bool cmdStart(int scalePct, int timeoutS) {
  if (state != IDLE || !imuOk) return false;
  reqScalePct = constrain(scalePct, 5, 100);
  reqTimeoutS = constrain(timeoutS, 1, MAX_TIMEOUT_S);
  startRequested = true;
  return true;
}

bool cmdStop() {
  stopRequested = true;
  stopArms();          // cut the motors right away too
  return true;
}

void sendState(const char* s) { Bridge.notify("state", String(s)); }

void finishRun(const char* reason) {
  stopArms();
  ledOn(false);
  stopReason = reason;
  sendIndex = 0;
  state = SENDING;
  sendState("sending log");
}

void startRun() {
  startRequested = false;
  stopRequested = false;
  pwmScale = reqScalePct / 100.0f;
  runTimeoutUs = reqTimeoutS * 1000000UL;

  state = CALIBRATING;
  sendState("calibrating: keep still");
  float noise = imu.calibrate(1000);   // gyro bias, robot must sit still (1 s)
  imu.reset();                         // level filter from gravity, yaw = 0 (sim starts at yaw 0)
  nRows = 0;
  if (noise > STILL_NOISE_MAX) { finishRun("aborted: robot moved during calibration"); return; }
  if (imu.tilt > START_TILT_MAX) { finishRun("aborted: robot not upright"); return; }
  if (stopRequested) { finishRun("stop button"); return; }

  prevAction[0] = prevAction[1] = 0;   // the sim resets last_action to 0
  state = RUNNING;
  sendState("running");
  ledOn(true);
  runStartUs = micros();
  nextUs = runStartUs;
}

// One control step. Returns false when the run must end (reason in stopReason).
bool controlStep() {
  unsigned long t0 = micros();
  imu.update(DT);

  if (stopRequested) { stopReason = "stop button"; return false; }
  if (imu.tilt > TILT_LIMIT) { stopReason = "tilt > 60 deg"; return false; }
  if (t0 - runStartUs >= runTimeoutUs) { stopReason = "timeout"; return false; }
  if (nRows >= MAX_ROWS) { stopReason = "log full"; return false; }

  // Observation, same order/units/frames as drive_task.py's actor group
  float obs[POLICY_OBS] = {imu.gx, imu.gy, imu.gz, sinf(imu.yaw), cosf(imu.yaw), prevAction[0], prevAction[1]};
  float act[POLICY_ACT];
  policyRun(obs, act);

  // Action -> PWM duty (per motor), then the gentle-test scale
  float dutyL = constrain(act[0] * MAX_WHEEL_SPEED / NO_LOAD_SPEED_L, -1.0f, 1.0f) * pwmScale;
  float dutyR = constrain(act[1] * MAX_WHEEL_SPEED / NO_LOAD_SPEED_R, -1.0f, 1.0f) * pwmScale;
  setArms(dutyL, dutyR);

  prevAction[0] = act[0];   // UNCLIPPED, like mjlab's last_action
  prevAction[1] = act[1];

  LogRow& r = logRows[nRows++];
  r.t = (t0 - runStartUs) * 1e-6f;
  memcpy(r.obs, obs, sizeof(obs));
  memcpy(r.act, act, sizeof(act));
  r.duty[0] = dutyL; r.duty[1] = dutyR;
  r.roll = imu.roll; r.pitch = imu.pitch; r.tilt = imu.tilt; r.yaw = imu.yaw;
  r.computeUs = (uint16_t)min(micros() - t0, 65535UL);
  return true;
}

// Send a few log rows per loop() so the Bridge stays responsive.
void sendSomeRows() {
  char line[240];
  for (int k = 0; k < 4 && sendIndex < nRows; k++, sendIndex++) {
    const LogRow& r = logRows[sendIndex];
    snprintf(line, sizeof(line),
             "%d,%.3f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%u",
             sendIndex, r.t, r.obs[0], r.obs[1], r.obs[2], r.obs[3], r.obs[4], r.obs[5], r.obs[6],
             r.act[0], r.act[1], r.duty[0], r.duty[1], r.roll, r.pitch, r.tilt, r.yaw, r.computeUs);
    Bridge.notify("log_line", String(line));
  }
  if (sendIndex >= nRows) {
    Bridge.notify("run_done", String(stopReason), nRows);
    state = IDLE;
    sendState("idle");
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  ledOn(false);
  stopArms();
  Bridge.begin();
  Monitor.begin();
  Bridge.provide_safe("start", cmdStart);
  Bridge.provide_safe("stop", cmdStop);

  imuOk = imu.begin();
  if (imuOk) { imu.calibrate(1000); imu.reset(); }
  Monitor.println(imuOk ? "policy sketch ready" : "ERROR: IMU not found");
  sendState(imuOk ? "idle" : "ERROR: IMU not found");
  nextUs = micros();
}

void loop() {
  if (state == IDLE) {
    stopArms();
    if (startRequested) startRun();
    else delay(5);
    return;
  }
  if (state == SENDING) {
    stopArms();
    sendSomeRows();
    return;
  }
  if (state == RUNNING) {
    // wait for the next 20 ms tick: sleep while > 1 ms is left (lets the Bridge thread run), then spin
    while ((long)(nextUs - micros()) > 1500) delay(1);
    while ((long)(nextUs - micros()) > 0) {}
    nextUs += PERIOD_US;
    if (!controlStep()) finishRun(stopReason);
  }
}
