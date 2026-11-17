// =====================================================================================
// Silly Walk Sim - STEP 3: DIRECTION TEST
//
// Goal: find LEFT_SIGN and RIGHT_SIGN in motors.h so that setArms(+, +) turns each arm in the
// sim's POSITIVE direction: THE TOP OF THE ARM MOVES FORWARD (toward the front, away from
// the caster), like a wheel rolling the robot forward.
//
// Sequence (runs once, then the motors stay off):
//   1. LEFT arm alone,  setArms(+D, 0)  for 3 s   -> does the top of the LEFT arm move forward?
//   2. RIGHT arm alone, setArms(0, +D)  for 3 s   -> does the top of the RIGHT arm move forward?
//   3. BOTH,            setArms(+D, +D) for 4 s   -> put the robot on the floor for this one:
//                                                    it should walk forward, caster at the back.
//
// Do 1 and 2 with the robot propped up (arms in the air). The sketch waits 10 s before step 3
// so you can put it on the floor. If an arm turned the wrong way, flip its sign in motors.h
// (+1 <-> -1) and run again.
// =====================================================================================

#include <Arduino_RouterBridge.h>   // Monitor -> `arduino-app-cli monitor`
#include "motors.h"

const float D = 0.5f;   // gentle: 50 % PWM

void ledOn(bool on) { digitalWrite(LED_BUILTIN, on ? LOW : HIGH); }  // active-low LED

void countdown(const char* what, int seconds) {
  Monitor.print("\nNEXT: ");
  Monitor.println(what);
  for (int s = seconds; s > 0; s--) {
    Monitor.print("  in ");
    Monitor.println(s);
    delay(1000);
  }
}

void run(float l, float r, unsigned long ms) {
  ledOn(true);
  setArms(l, r);
  delay(ms);
  stopArms();
  ledOn(false);
}

void setup() {
  Monitor.begin();
  pinMode(LED_BUILTIN, OUTPUT);
  ledOn(false);
  stopArms();
  delay(5000);   // time to open the monitor

  Monitor.println("\n=== STEP 3: DIRECTION TEST ===");
  Monitor.print("LEFT_SIGN = "); Monitor.print(LEFT_SIGN);
  Monitor.print("   RIGHT_SIGN = "); Monitor.println(RIGHT_SIGN);
  Monitor.println("Robot propped up, arms free. Watch the TOP of each arm.");

  countdown("LEFT arm only, +50 %. Top of LEFT arm should move FORWARD", 5);
  run(D, 0, 3000);
  Monitor.println("  -> wrong way? flip LEFT_SIGN in motors.h");

  countdown("RIGHT arm only, +50 %. Top of RIGHT arm should move FORWARD", 3);
  run(0, D, 3000);
  Monitor.println("  -> wrong way? flip RIGHT_SIGN in motors.h");

  countdown("BOTH +50 %. PUT THE ROBOT ON THE FLOOR, caster at the back. It should walk FORWARD", 10);
  run(D, D, 4000);

  Monitor.println("\n=== DONE. Motors off. Restart the app to repeat. ===");
}

void loop() {
  stopArms();
  delay(1000);
}
