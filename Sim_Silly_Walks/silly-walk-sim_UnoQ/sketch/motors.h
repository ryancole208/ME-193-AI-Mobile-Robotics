// motors.h - open-loop PWM for the two arm motors (Cytron Maker Drive).
//
// setArms(left, right) takes the sim's convention: +1 = full speed in the sim's POSITIVE joint direction,
// which rolls the robot FORWARD (the top of the arm moves forward). LEFT_SIGN / RIGHT_SIGN
// translate that into the driver's A/B inputs. Step 3 (direction test) finds the right signs.
#pragma once
#include <Arduino.h>

// ---- Pins (same as your motor-test app). All must be PWM pins (~). ----
const int LEFT_A  = 10;   // Maker Drive M2A
const int LEFT_B  = 11;   // Maker Drive M2B
const int RIGHT_A = 6;    // Maker Drive M1B
const int RIGHT_B = 5;    // Maker Drive M1A

// ---- Direction signs (+1 or -1). Verified in step 3 (2026-11-17): both +1 turn the top of each
//      arm forward = the sim's positive direction. ----
const int LEFT_SIGN  = +1;
const int RIGHT_SIGN = +1;

const int PWM_MAX = 255;  // analogWrite() is 8-bit on the UNO Q: 255 = 100 % duty

// Drive one motor's two driver inputs. duty -1..+1: positive = PWM on A, negative = PWM on B.
// No pinMode() on these pins: on the UNO Q Zephyr core it breaks PWM (found in motor-test).
inline void driveRaw(int pinA, int pinB, float duty) {
  duty = constrain(duty, -1.0f, 1.0f);
  int pwm = (int)(fabsf(duty) * PWM_MAX + 0.5f);
  if (duty >= 0) { analogWrite(pinB, 0); analogWrite(pinA, pwm); }
  else           { analogWrite(pinA, 0); analogWrite(pinB, pwm); }
}

// Sim convention: +1 = forward (sim's positive joint direction) for each arm.
inline void setArms(float left, float right) {
  driveRaw(LEFT_A, LEFT_B, LEFT_SIGN * left);
  driveRaw(RIGHT_A, RIGHT_B, RIGHT_SIGN * right);
}

inline void stopArms() { setArms(0, 0); }
