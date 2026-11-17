// =====================================================================================
// Silly Walk Sim - STEP 2: NO-LOAD SPEED OF EACH MOTOR AT 100 % PWM
//
// Robot PROPPED UP, arms free in the air. For each motor (LEFT, then RIGHT):
//   * 100 % PWM forward for 20 s (built-in LED on) -> COUNT THE ARM'S REVOLUTIONS
//   * meanwhile the IMU is recorded at 100 Hz into RAM; afterwards it is printed
//     ("D ..." lines) so tools/analyze_motor_speed.py can find the arm's rotation
//     frequency from the chassis vibration (a second, automatic estimate).
// The motors stay off at the end. Restart the app to repeat.
// =====================================================================================

#include <Arduino_RouterBridge.h>
#include "imu.h"
#include "motors.h"

const unsigned long RUN_MS = 20000;
const unsigned long SAMPLE_US = 10000;              // 100 Hz
const int N_SAMPLES = RUN_MS * 1000 / SAMPLE_US;    // 2000
float buf[N_SAMPLES][6];                            // gx gy gz ax ay az (body frame)
Imu imu;

void ledOn(bool on) { digitalWrite(LED_BUILTIN, on ? LOW : HIGH); }  // active-low LED

void measure(const char* name, float left, float right) {
  for (int s = 3; s > 0; s--) {
    Monitor.print("# "); Monitor.print(name); Monitor.print(" motor 100 % in "); Monitor.println(s);
    delay(1000);
  }
  Monitor.print("# GO "); Monitor.print(name); Monitor.println(" - count revolutions for 20 s");
  ledOn(true);
  setArms(left, right);
  unsigned long next = micros();
  for (int i = 0; i < N_SAMPLES; i++) {
    while ((long)(micros() - next) < 0) {}
    next += SAMPLE_US;
    imu.update(SAMPLE_US * 1e-6f);
    buf[i][0] = imu.gx; buf[i][1] = imu.gy; buf[i][2] = imu.gz;
    buf[i][3] = imu.ax; buf[i][4] = imu.ay; buf[i][5] = imu.az;
  }
  stopArms();
  ledOn(false);
  Monitor.print("# STOP "); Monitor.println(name);

  // Print the recording (takes a few seconds).
  char line[96];
  Monitor.print("# BEGIN "); Monitor.println(name);
  for (int i = 0; i < N_SAMPLES; i++) {
    snprintf(line, sizeof(line), "D %d %.4f %.4f %.4f %.3f %.3f %.3f", i,
             buf[i][0], buf[i][1], buf[i][2], buf[i][3], buf[i][4], buf[i][5]);
    Monitor.println(line);
  }
  Monitor.print("# END "); Monitor.println(name);
  delay(3000);
}

void setup() {
  Monitor.begin();
  pinMode(LED_BUILTIN, OUTPUT);
  ledOn(false);
  stopArms();
  delay(5000);
  Monitor.println("# === STEP 2: MOTOR NO-LOAD SPEED ===");
  if (!imu.begin()) Monitor.println("# WARNING: IMU not found - only your revolution count will work");
  imu.calibrate(1000);
  imu.reset();
  measure("LEFT", 1.0f, 0.0f);
  measure("RIGHT", 0.0f, 1.0f);
  Monitor.println("# === DONE. Motors off. ===");
}

void loop() {
  stopArms();
  delay(1000);
}
