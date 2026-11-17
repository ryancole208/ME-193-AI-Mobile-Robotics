// imu.h - MPU-6050 (GY-521) reader + attitude estimate, written to match the sim's observation.
//
// Outputs, all in the sim's chassis BODY frame (x forward, y left, z up):
//   gx, gy, gz : angular velocity in rad/s, gyro bias removed         (= sim base_ang_vel)
//   roll, pitch: from a complementary filter (gyro + accelerometer), rad
//   yaw        : heading in rad, integrated from the gyro, zero at reset (= sim yaw)
//   tilt       : angle between body z and vertical, rad              (= sim bad_orientation angle)
//
// Talks to the chip directly over I2C (no extra library needed).
#pragma once
#include <Arduino.h>
#include <Wire.h>

// SDA/SCL header pins = Wire. If you use the Qwiic connector instead, change this to Wire1.
#define IMU_WIRE Wire

// ---- Axis mapping: chip axes -> body axes. Index 0/1/2 = chip X/Y/Z. ----
// The GY-521 is mounted UPSIDE DOWN (chip facing the floor), printed X arrow forward,
// Y arrow to the robot's RIGHT, so chip Z points down. That is a 180 deg turn about x:
//   body x = +chip X,  body y = -chip Y,  body z = -chip Z
// (Found in step 4: flat on the table the unmapped reading was roll -168 deg.)
const int   AXIS_IDX[3]  = {0, 1, 2};      // body x,y,z take chip axis ...
const float AXIS_SIGN[3] = {+1, -1, -1};   // ... times this sign

class Imu {
 public:
  float gx = 0, gy = 0, gz = 0;        // rad/s, body frame, bias removed
  float ax = 0, ay = 0, az = 0;        // m/s^2 (specific force; about +9.81 on z when flat)
  float roll = 0, pitch = 0, yaw = 0;  // rad
  float tilt = 0;                      // rad
  float yawBodyOnly = 0;               // for comparison only: plain integral of gz
  uint8_t whoAmI = 0;

  static const uint8_t ADDR = 0x68;    // AD0 low/unconnected

  // Configure the chip. Returns false if nothing answers on I2C.
  bool begin() {
    IMU_WIRE.begin();
    IMU_WIRE.setClock(400000);
    if (!write(0x6B, 0x80)) return false;  // PWR_MGMT_1: reset
    delay(100);
    write(0x6B, 0x01);   // PWR_MGMT_1: wake up, clock from gyro X PLL (more stable)
    write(0x1A, 0x03);   // CONFIG: digital low-pass ~44 Hz (cuts motor buzz, ~5 ms delay)
    write(0x19, 0x00);   // SMPLRT_DIV: 1 kHz internal sample rate
    write(0x1B, 0x18);   // GYRO_CONFIG: +-2000 deg/s (sim rates reach several rad/s; 250 deg/s would clip)
    write(0x1C, 0x10);   // ACCEL_CONFIG: +-8 g
    delay(50);
    whoAmI = read8(0x75);
    return true;
  }

  // Average the gyro while the robot sits still. Returns the gyro noise (rad/s, std) so the
  // caller can check the robot really was still.
  float calibrate(int ms = 2000) {
    double s[3] = {0, 0, 0}, s2 = 0;
    int n = 0;
    unsigned long t0 = millis();
    while (millis() - t0 < (unsigned long)ms) {
      float g[3], a[3];
      readRaw(a, g);
      for (int i = 0; i < 3; i++) s[i] += g[i];
      s2 += g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
      n++;
      delay(2);
    }
    for (int i = 0; i < 3; i++) bias[i] = s[i] / n;
    float meanSq = bias[0] * bias[0] + bias[1] * bias[1] + bias[2] * bias[2];
    return sqrtf(fmaxf(0.0f, (float)(s2 / n) - meanSq) / 3.0f);
  }

  // Start of a run: level the filter from the accelerometer and zero the yaw (sim starts at yaw = 0).
  void reset() {
    float g[3], a[3];
    readRaw(a, g);
    applyMap(a, g);
    roll = atan2f(ay, az);
    pitch = atan2f(-ax, sqrtf(ay * ay + az * az));
    yaw = 0;
    yawBodyOnly = 0;
    tilt = acosf(constrain(cosf(roll) * cosf(pitch), -1.0f, 1.0f));
  }

  // Read the sensor and advance the filter by dt seconds (call once per control step).
  void update(float dt) {
    float g[3], a[3];
    readRaw(a, g);
    for (int i = 0; i < 3; i++) g[i] -= bias[i];
    applyMap(a, g);

    // Gyro: Euler-angle rates (roll, pitch, yaw order matches the sim's yaw formula)
    float sr = sinf(roll), cr = cosf(roll);
    float cp = cosf(pitch), tp = tanf(pitch);
    if (fabsf(cp) < 0.1f) cp = copysignf(0.1f, cp);  // avoid divide-by-zero near 90 deg pitch
    float rollDot  = gx + sr * tp * gy + cr * tp * gz;
    float pitchDot = cr * gy - sr * gz;
    float yawDot   = (sr * gy + cr * gz) / cp;        // WORLD yaw rate (not just body gz)

    // Complementary filter: trust the gyro short-term, the accelerometer (gravity) long-term.
    const float ALPHA = 0.98f;                        // ~1 s time constant at 50 Hz
    float rollAcc  = atan2f(ay, az);
    float pitchAcc = atan2f(-ax, sqrtf(ay * ay + az * az));
    roll  = ALPHA * (roll + rollDot * dt) + (1 - ALPHA) * rollAcc;
    pitch = ALPHA * (pitch + pitchDot * dt) + (1 - ALPHA) * pitchAcc;

    yaw += yawDot * dt;                               // no magnetometer: yaw drifts slowly
    yawBodyOnly += gz * dt;
    tilt = acosf(constrain(cosf(roll) * cosf(pitch), -1.0f, 1.0f));
  }

 private:
  float bias[3] = {0, 0, 0};   // gyro bias in CHIP axes, rad/s

  void applyMap(const float a[3], const float g[3]) {
    float A[3], G[3];
    for (int i = 0; i < 3; i++) {
      A[i] = AXIS_SIGN[i] * a[AXIS_IDX[i]];
      G[i] = AXIS_SIGN[i] * g[AXIS_IDX[i]];
    }
    ax = A[0]; ay = A[1]; az = A[2];
    gx = G[0]; gy = G[1]; gz = G[2];
  }

  // One burst read of accel (0x3B..0x40) + temp + gyro (0x43..0x48), converted to SI units, chip axes.
  void readRaw(float a[3], float g[3]) {
    uint8_t b[14] = {0};
    IMU_WIRE.beginTransmission(ADDR);
    IMU_WIRE.write(0x3B);
    IMU_WIRE.endTransmission(false);
    IMU_WIRE.requestFrom((int)ADDR, 14);
    for (int i = 0; i < 14 && IMU_WIRE.available(); i++) b[i] = IMU_WIRE.read();
    const float ACC = 9.80665f / 4096.0f;                 // +-8 g: 4096 LSB per g
    const float GYR = (PI / 180.0f) / 16.4f;              // +-2000 deg/s: 16.4 LSB per deg/s
    for (int i = 0; i < 3; i++) {
      a[i] = (int16_t)((b[2 * i] << 8) | b[2 * i + 1]) * ACC;
      g[i] = (int16_t)((b[8 + 2 * i] << 8) | b[9 + 2 * i]) * GYR;
    }
  }

  bool write(uint8_t reg, uint8_t val) {
    IMU_WIRE.beginTransmission(ADDR);
    IMU_WIRE.write(reg);
    IMU_WIRE.write(val);
    return IMU_WIRE.endTransmission() == 0;
  }

  uint8_t read8(uint8_t reg) {
    IMU_WIRE.beginTransmission(ADDR);
    IMU_WIRE.write(reg);
    IMU_WIRE.endTransmission(false);
    IMU_WIRE.requestFrom((int)ADDR, 1);
    return IMU_WIRE.available() ? IMU_WIRE.read() : 0;
  }
};
