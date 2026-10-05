#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <math.h>
#include <string.h>

// ===================== PID TUNING — EDIT HERE =====================
const float KP = 0.6f;
const float KI = 0.05f;
const float KD = 0.15f;
// ====================================================================

// ----- Drive & safety limits -----
const int MAX_PWM = 150;           // top PWM (0-255) sent to a motor
const int MIN_PWM = 40;            // below this the motor just stalls; jump straight to it once moving
const float DEADZONE_PX = 5.0f;    // |dx| below this counts as "centered": stop instead of hunting
const float INTEGRAL_MAX = 200.0f; // anti-windup clamp on the accumulated I term

// Flip to -1 if the car drives away from center instead of toward it
// (depends on which way "forward" points relative to the camera view).
const int DIRECTION_SIGN = -1;

// ----- Timing -----
const unsigned long WATCHDOG_PERIOD_MS = 20;  // how often we check for a timed-out target / refresh the matrix
const unsigned long LOST_GRACE_MS = 250;      // keep driving on the last heading for 0.25s after losing the minifig
const float FIRST_UPDATE_DT_S = 0.1f;         // assumed dt for the very first PID update (matches the broadcaster's ~10 Hz)

// ----- Motor pins: forward/reverse PWM pair per motor, both motors always move together -----
// Maker Drive wiring: M2A=10, M2B=11, M1A=5, M1B=6. The test-button check showed
// pressing M2A spins motor 2 forward and pressing M1B spins motor 1 forward, so
// those are each motor's "forward" pin (the two motors are mounted mirrored).
const int MOTOR_A_FWD = 10; // M2A
const int MOTOR_A_REV = 11; // M2B
const int MOTOR_B_FWD = 6;  // M1B
const int MOTOR_B_REV = 5;  // M1A

// ----- LED matrix -----
Arduino_LED_Matrix matrix;
const uint8_t FRAME_ROWS = 8;
const uint8_t FRAME_COLS = 13;
uint8_t frame[FRAME_ROWS * FRAME_COLS];

// ----- Shared tracking state, updated from Python via Bridge.call("update_target", ...) -----
volatile int targetCol = 0;
volatile int targetRow = 0;
volatile unsigned long lastSeenMs = 0;
volatile bool everSeen = false;
volatile bool timedOut = true; // true once we've stopped the motors for a lost target (latch, avoids repeat writes)

// ----- PID state: only ever touched from update_target(), once per real new measurement -----
float integralTerm = 0.0f;
float lastError = 0.0f;
unsigned long lastPidMs = 0;
bool pidInitialized = false;

int lastAppliedPwm = 0;
bool motorsInitialized = false;

void setMotors(int pwm) {
    pwm = constrain(pwm, -MAX_PWM, MAX_PWM);
    int mag = abs(pwm);
    if (mag > 0 && mag < MIN_PWM) {
        mag = MIN_PWM;
    }
    int signedMag = (pwm > 0) ? mag : (pwm < 0 ? -mag : 0);

    // Skip redundant analogWrite() calls when the command hasn't actually changed.
    // The UNO Q's Zephyr core has documented quirks where PWM pins can interfere
    // with each other's pin-mux state; re-issuing the same 4 analogWrite calls on
    // every single MQTT update (~10 Hz) is unnecessary churn and a plausible
    // contributor to one motor intermittently dropping out.
    if (motorsInitialized && signedMag == lastAppliedPwm) {
        return;
    }
    lastAppliedPwm = signedMag;
    motorsInitialized = true;

    if (pwm > 0) {
        analogWrite(MOTOR_A_FWD, mag);
        analogWrite(MOTOR_A_REV, 0);
        analogWrite(MOTOR_B_FWD, mag);
        analogWrite(MOTOR_B_REV, 0);
    } else if (pwm < 0) {
        analogWrite(MOTOR_A_FWD, 0);
        analogWrite(MOTOR_A_REV, mag);
        analogWrite(MOTOR_B_FWD, 0);
        analogWrite(MOTOR_B_REV, mag);
    } else {
        analogWrite(MOTOR_A_FWD, 0);
        analogWrite(MOTOR_A_REV, 0);
        analogWrite(MOTOR_B_FWD, 0);
        analogWrite(MOTOR_B_REV, 0);
    }
}

// Exposed to Python: runs the PID update exactly once per real detection update, using the
// actual elapsed time since the previous one (NOT a fixed tick) for the I and D terms. The
// resulting motor command is then held steady by setMotors() until the next call arrives.
void update_target(bool found, int col, int row, int dx) {
    if (!found) {
        return; // no new measurement: keep driving the last commanded trajectory
    }

    unsigned long now = millis();
    targetCol = col;
    targetRow = row;
    lastSeenMs = now;
    everSeen = true;
    timedOut = false;

    float error = (float)dx;

    if (fabs(error) < DEADZONE_PX) {
        integralTerm = 0.0f;
        lastError = 0.0f;
        setMotors(0);
    } else {
        float dt = pidInitialized ? (now - lastPidMs) / 1000.0f : FIRST_UPDATE_DT_S;
        if (dt <= 0.0f) {
            dt = FIRST_UPDATE_DT_S; // guard against two updates landing on the same millisecond
        }

        integralTerm = constrain(integralTerm + error * dt, -INTEGRAL_MAX, INTEGRAL_MAX);
        float derivative = (error - lastError) / dt;
        lastError = error;

        float output = KP * error + KI * integralTerm + KD * derivative;
        setMotors(DIRECTION_SIGN * (int)output);
    }

    pidInitialized = true;
    lastPidMs = now;
}

void setup() {
    Monitor.begin(115200);

    // NOTE: deliberately no pinMode() on the motor pins — on the UNO Q's Zephyr core,
    // calling pinMode(OUTPUT) on a PWM-capable pin before analogWrite() breaks PWM on
    // that pin (stuck low / glitchy), since analogWrite() configures the pin itself.
    setMotors(0);

    matrix.begin();
    matrix.setGrayscaleBits(3); // brightness levels 0-7
    matrix.clear();

    Bridge.begin();
    Bridge.provide("update_target", update_target);
}

void loop() {
    unsigned long now = millis();
    static unsigned long lastWatchdogMs = 0;
    if (now - lastWatchdogMs < WATCHDOG_PERIOD_MS) {
        return;
    }
    lastWatchdogMs = now;

    bool tracking = everSeen && (now - lastSeenMs <= LOST_GRACE_MS);

    // ---- LED matrix: lit pixel at the minifig's last known position ----
    // (the onboard matrix is monochrome brightness-only, so the "dot" is a
    // full-brightness pixel rather than an actual color)
    memset(frame, 0, sizeof(frame));
    if (tracking) {
        frame[targetRow * FRAME_COLS + targetCol] = 7;
    }
    matrix.draw(frame);

    // ---- Stop once the grace period after losing the target has elapsed ----
    if (!tracking && !timedOut) {
        setMotors(0);
        integralTerm = 0.0f;
        lastError = 0.0f;
        pidInitialized = false;
        timedOut = true;
    }
}
