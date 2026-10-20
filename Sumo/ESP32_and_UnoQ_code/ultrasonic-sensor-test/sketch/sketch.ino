// Ultrasonic Sensor Test - standalone sketch, no MQTT/laptop involved.
//
// Pings the HC-SR04, filters it, and draws the result straight on the LED
// matrix: a box centered on the matrix that grows as something gets
// closer, a single dot when nothing's in range, and an X if the sensor
// stops responding. This reuses the exact same measuring/filtering logic
// as the "RQ-D2 Vision" app's sketch, just without any of the MQTT/Bridge
// icon-selection wiring - so it's a quick way to confirm the sensor is
// wired correctly and see its detection range live, without needing a
// laptop, broker, or behavior tree running at all.
//
// Also pushes each reading to Python over the Bridge purely so it can be
// logged to the console (see python/main.py) - handy if you have App Lab
// open, but not required: the LED matrix alone is enough to test with.

#include "Arduino_RouterBridge.h"
#include <Arduino_LED_Matrix.h>
#include <string.h>
#include <math.h>

// ===================== CONFIG =====================
// Same defaults as the "RQ-D2 Vision" app's config.py. This is a throwaway
// test tool, not the competition build, so these are just constants here -
// edit them directly and re-upload if you want to try different numbers.
const int TRIG_PIN = 9;
const int ECHO_PIN = 8;
const unsigned long PING_INTERVAL_MS = 60;
const unsigned long ECHO_TIMEOUT_US = 25000UL;
const int MEDIAN_WINDOW = 5;

const float MIN_VALID_DISTANCE_CM = 5.0f;
const float MAX_RANGE_CM = 100.0f;         // "detect" threshold
const float LOSE_RANGE_CM = 115.0f;        // "lose" threshold (hysteresis)
const int DEBOUNCE_COUNT = 3;

const int STALE_AFTER_PINGS = 8;           // ~0.5s: stop trusting the last distance
const int SENSOR_FAULT_AFTER_PINGS = 100;  // ~6s: sensor itself isn't responding

// ===================== Measurement state =====================
float validReadings[MEDIAN_WINDOW];
int validCount = 0;
int validHead = 0;

int consecutiveTimeouts = 0;
bool sensorOk = true;
bool haveValidDistance = false;
float currentMedianCm = -1.0f;
float currentConfidence = 0.0f;

bool detected = false;
bool tentativeState = false;
int debounceCounter = 0;

unsigned long lastPingMs = 0;

// ===================== LED matrix =====================
Arduino_LED_Matrix matrix;
const uint8_t FRAME_ROWS = 8;
const uint8_t FRAME_COLS = 13;
const uint8_t FRAME_SIZE = FRAME_ROWS * FRAME_COLS;

uint8_t FRAME_SENSOR_ERROR[FRAME_SIZE] = { // "X" - sensor not responding
  7,7,0,0,0,0,0,0,0,0,0,7,7,
  0,0,7,7,0,0,0,0,0,7,7,0,0,
  0,0,0,0,7,7,0,7,7,0,0,0,0,
  0,0,0,0,0,7,7,7,0,0,0,0,0,
  0,0,0,0,0,7,7,7,0,0,0,0,0,
  0,0,0,0,7,7,0,7,7,0,0,0,0,
  0,0,7,7,0,0,0,0,0,7,7,0,0,
  7,7,0,0,0,0,0,0,0,0,0,7,7
};

uint8_t FRAME_IDLE[FRAME_SIZE] = { // single centered dot - nothing in range
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,7,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0
};

// Growing/shrinking box, same scheme as the main app's "chasing" icon.
const uint8_t CHASE_HALF_SIZE[] = { 0, 1, 2, 3 };
const uint8_t CHASE_SIZE_STEPS = sizeof(CHASE_HALF_SIZE) / sizeof(CHASE_HALF_SIZE[0]);
const uint8_t CHASE_CENTER_ROW = 3;
const uint8_t CHASE_CENTER_COL = 6;

void drawChasingFrame(int sizeStep) {
  if (sizeStep < 0) sizeStep = 0;
  if (sizeStep >= CHASE_SIZE_STEPS) sizeStep = CHASE_SIZE_STEPS - 1;
  uint8_t half = CHASE_HALF_SIZE[sizeStep];

  uint8_t frame[FRAME_SIZE];
  memset(frame, 0, sizeof(frame));

  int rowLo = max(0, CHASE_CENTER_ROW - half);
  int rowHi = min(FRAME_ROWS - 1, CHASE_CENTER_ROW + half);
  int colLo = max(0, CHASE_CENTER_COL - half);
  int colHi = min(FRAME_COLS - 1, CHASE_CENTER_COL + half);
  for (int r = rowLo; r <= rowHi; r++) {
    for (int c = colLo; c <= colHi; c++) {
      frame[r * FRAME_COLS + c] = 7;
    }
  }
  matrix.draw(frame);
}

int sizeStepForDistance(float distanceCm) {
  float frac = 1.0f - fmaxf(0.0f, fminf(1.0f, distanceCm / MAX_RANGE_CM));
  int step = (int)roundf(frac * (CHASE_SIZE_STEPS - 1));
  if (step < 0) step = 0;
  if (step >= CHASE_SIZE_STEPS) step = CHASE_SIZE_STEPS - 1;
  return step;
}

// Only redraws when the picture actually needs to change.
int lastDrawnMode = -1;    // 0=error, 1=idle, 2=chasing
int lastDrawnSizeStep = -1;

void updateDisplay() {
  int mode;
  int sizeStep = 0;
  if (!sensorOk) {
    mode = 0;
  } else if (detected) {
    mode = 2;
    sizeStep = sizeStepForDistance(currentMedianCm);
  } else {
    mode = 1;
  }

  if (mode == lastDrawnMode && sizeStep == lastDrawnSizeStep) {
    return;
  }
  lastDrawnMode = mode;
  lastDrawnSizeStep = sizeStep;

  if (mode == 0) {
    matrix.draw(FRAME_SENSOR_ERROR);
  } else if (mode == 2) {
    drawChasingFrame(sizeStep);
  } else {
    matrix.draw(FRAME_IDLE);
  }
}

// ===================== Measurement (same as the main app's sketch) =====================

float pingOnce() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Manual micros()-based timing, not pulseIn() - see the main app's README
  // for why: pulseIn() didn't reliably honor its timeout on this board.
  unsigned long waitStart = micros();
  while (digitalRead(ECHO_PIN) == LOW) {
    if (micros() - waitStart >= ECHO_TIMEOUT_US) {
      return -1.0f;
    }
  }
  unsigned long pulseStart = micros();
  while (digitalRead(ECHO_PIN) == HIGH) {
    if (micros() - pulseStart >= ECHO_TIMEOUT_US) {
      return -1.0f;
    }
  }
  unsigned long durationUs = micros() - pulseStart;
  return (durationUs * 0.0343f) / 2.0f;
}

void pushValidReading(float cm) {
  validReadings[validHead] = cm;
  validHead = (validHead + 1) % MEDIAN_WINDOW;
  if (validCount < MEDIAN_WINDOW) {
    validCount++;
  }
}

float medianOf(const float *values, int n) {
  float sorted[MEDIAN_WINDOW];
  for (int i = 0; i < n; i++) sorted[i] = values[i];
  for (int i = 1; i < n; i++) {
    float key = sorted[i];
    int j = i - 1;
    while (j >= 0 && sorted[j] > key) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = key;
  }
  return sorted[n / 2];
}

float computeConfidence(float medianCm) {
  if (validCount == 0) return 0.0f;
  float tolerance = fmaxf(10.0f, medianCm * 0.15f);
  int agree = 0;
  for (int i = 0; i < validCount; i++) {
    if (fabsf(validReadings[i] - medianCm) <= tolerance) agree++;
  }
  return (float)agree / (float)validCount;
}

void doPingCycle() {
  float raw = pingOnce();

  if (raw < 0.0f || raw < MIN_VALID_DISTANCE_CM) {
    consecutiveTimeouts++;
  } else {
    consecutiveTimeouts = 0;
    pushValidReading(raw);
  }

  sensorOk = consecutiveTimeouts < SENSOR_FAULT_AFTER_PINGS;
  haveValidDistance = (validCount > 0) && (consecutiveTimeouts < STALE_AFTER_PINGS);

  if (haveValidDistance) {
    currentMedianCm = medianOf(validReadings, validCount);
    currentConfidence = computeConfidence(currentMedianCm);
  } else {
    currentMedianCm = -1.0f;
    currentConfidence = 0.0f;
  }

  bool rawInRange;
  if (!haveValidDistance) {
    rawInRange = false;
  } else if (!detected) {
    rawInRange = currentMedianCm <= MAX_RANGE_CM;
  } else {
    rawInRange = currentMedianCm <= LOSE_RANGE_CM;
  }

  if (rawInRange == tentativeState) {
    if (debounceCounter < 255) debounceCounter++;
  } else {
    tentativeState = rawInRange;
    debounceCounter = 1;
  }
  if (debounceCounter >= DEBOUNCE_COUNT) {
    detected = tentativeState;
  }

  updateDisplay();

  // Push to Python purely for console logging (see python/main.py) - no
  // Bridge.call ever happens in response to this, so there's no risk of
  // the Bridge-thread deadlock documented in the main app's README.
  Bridge.notify("vision_update", detected, haveValidDistance ? currentMedianCm : -1.0f, currentConfidence, sensorOk);

  static unsigned long lastPrintMs = 0;
  if (millis() - lastPrintMs >= 1000) {
    lastPrintMs = millis();
    Monitor.print("distance=");
    if (haveValidDistance) {
      Monitor.print(currentMedianCm);
      Monitor.print("cm");
    } else {
      Monitor.print("?");
    }
    Monitor.print(" detected=");
    Monitor.print(detected ? "true" : "false");
    Monitor.print(" sensorOk=");
    Monitor.println(sensorOk ? "true" : "false");
  }
}

void setup() {
  Monitor.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  matrix.draw(FRAME_IDLE);

  Bridge.begin();
}

void loop() {
  unsigned long now = millis();
  if (now - lastPingMs >= PING_INTERVAL_MS) {
    lastPingMs = now;
    doPingCycle();
  }
}
