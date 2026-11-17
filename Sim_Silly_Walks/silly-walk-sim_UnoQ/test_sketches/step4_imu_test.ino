// =====================================================================================
// Silly Walk Sim - STEP 4: IMU TEST (motors stay OFF)
//
//   1. Checks the MPU-6050 answers on I2C.
//   2. Measures the gyro bias: KEEP THE ROBOT STILL on the table for the first ~2 s.
//   3. Drift test: KEEP IT STILL 10 s more. Prints how far the integrated yaw drifted.
//   4. Then prints, 10 times per second, the values the policy will see:
//        wx wy wz [rad/s]   yaw [rad]   (and roll, pitch, tilt in degrees for you)
//      Move the robot by hand and check (sim body frame: x forward, y left, z up):
//        * turn it LEFT (counter-clockwise seen from above) -> wz > 0, yaw increases
//        * lift its LEFT side (roll right side down)        -> wx > 0, roll increases
//        * tip its NOSE DOWN (front toward the floor)       -> wy > 0, pitch increases
//        * flat on the table                                -> roll ~ 0, pitch ~ 0, tilt ~ 0
//      If a sign is wrong, fix AXIS_SIGN / AXIS_IDX in imu.h.
// =====================================================================================

#include <Arduino_RouterBridge.h>
#include "imu.h"
#include "motors.h"

Imu imu;
const unsigned long PERIOD_US = 20000;   // 50 Hz, same as the sim's control rate
const float DT = PERIOD_US * 1e-6f;
const float RAD2DEG = 180.0f / PI;

unsigned long nextUs;
unsigned long step = 0;
char summary[160];   // setup results, re-printed every 5 s in case the monitor connected late

void setup() {
  Monitor.begin();
  stopArms();
  delay(5000);   // time to open the monitor
  Monitor.println("\n=== STEP 4: IMU TEST ===");

  if (!imu.begin()) {
    while (true) {   // repeat so a late-connecting monitor still sees it
      Monitor.println("ERROR: no answer from the IMU at 0x68. Check VCC, GND, SDA, SCL (and AD0 not tied to 3.3V).");
      delay(1000);
    }
  }
  Monitor.print("WHO_AM_I = 0x");
  Monitor.print(imu.whoAmI, HEX);
  Monitor.println(imu.whoAmI == 0x68 ? " (genuine MPU-6050)" : " (not 0x68: a clone chip; usually still works)");

  Monitor.println("Measuring gyro bias - KEEP STILL...");
  float noise = imu.calibrate(2000);
  Monitor.print("gyro noise (std) = ");
  Monitor.print(noise, 4);
  Monitor.println(noise < 0.02f ? " rad/s  OK" : " rad/s  HIGH: was the robot moving? restart to redo");

  // ---- Drift test: 10 s still ----
  Monitor.println("Drift test: KEEP STILL for 10 s...");
  imu.reset();
  nextUs = micros();
  for (int i = 0; i < 500; i++) {          // 500 x 20 ms = 10 s
    while ((long)(micros() - nextUs) < 0) {}
    nextUs += PERIOD_US;
    imu.update(DT);
  }
  float drift = fabsf(imu.yaw) * RAD2DEG;
  Monitor.print("Yaw drift over 10 s = ");
  Monitor.print(imu.yaw * RAD2DEG, 2);
  Monitor.print(" deg  -> ");
  Monitor.println(drift < 3.0f ? "OK (a few degrees or less)" : "TOO MUCH (> 3 deg): tell Claude");
  snprintf(summary, sizeof(summary), "# SUMMARY: WHO_AM_I=0x%02X  gyro_noise=%.4f rad/s  yaw_drift_10s=%+.2f deg",
           imu.whoAmI, noise, imu.yaw * RAD2DEG);

  Monitor.println("\nNow move the robot by hand. Columns:");
  Monitor.println("wx wy wz [rad/s] | yaw [rad] | yaw roll pitch tilt [deg]");
  imu.reset();   // yaw = 0 again
  nextUs = micros();
}

void loop() {
  // Fixed 50 Hz loop: wait for the next 20 ms tick.
  while ((long)(micros() - nextUs) < 0) {}
  nextUs += PERIOD_US;
  imu.update(DT);

  if (step % 250 == 0) Monitor.println(summary);   // every 5 s
  if (++step % 5 == 0) {   // print at 10 Hz so the monitor keeps up
    char line[128];
    snprintf(line, sizeof(line), "%+6.2f %+6.2f %+6.2f | %+6.3f | %+7.1f %+6.1f %+6.1f %5.1f",
             imu.gx, imu.gy, imu.gz, imu.yaw,
             imu.yaw * RAD2DEG, imu.roll * RAD2DEG, imu.pitch * RAD2DEG, imu.tilt * RAD2DEG);
    Monitor.println(line);
  }
}
