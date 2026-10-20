// RQ-D2 Vision - microcontroller sketch
//
// Runs on the UNO Q's STM32U585 (the "MCU side"). Three jobs:
//   1. Ping the HC-SR04 ultrasonic sensor at ~16 Hz, filter the readings,
//      and decide "is something in the ring ahead of us?" using hysteresis
//      + debouncing. The result is pushed to Python over the Bridge.
//   2. Draw status icons on the 8x13 LED matrix - but only when Python
//      tells us to (Python owns the decision of *which* icon to show; the
//      sketch just knows how to draw each one).
//   3. Play the 5-note "sad song" on the buzzer when Python asks for it,
//      without ever blocking (no delay() in loop()) so Bridge calls and
//      LED updates keep working while the song plays.
//
// All three jobs run from the same tiny loop() - see the bottom of this
// file - using millis()-based timers instead of delay(), which is what
// makes it possible to do all three at once without any of them stalling
// the others.

#include "Arduino_RouterBridge.h"
#include <Arduino_LED_Matrix.h>
#include <string.h>
#include <math.h>

// ===================================================================
// ===== CONFIG - defaults used until Python pushes its own values =====
// ===================================================================
// Python's config.py is the single place a student edits to retune the
// robot (see that file for the full explanation). At startup, main.py
// calls Bridge.call("configure", ...) to overwrite the detection numbers
// below with whatever config.py says. The values here are only the
// fallback used before that first call arrives (or forever, if for some
// reason it never does) - they intentionally match config.py's defaults.

// ----- Pin wiring (fixed in hardware; changing these means re-wiring AND
//       editing this file - they can't be changed from config.py) -----
const int TRIG_PIN = 9;     // HC-SR04 Trig (output pulse)
const int ECHO_PIN = 8;     // HC-SR04 Echo (input pulse) - see README for the voltage-protection note
const int BUZZER_PIN = 5;   // passive buzzer signal (PWM-capable pin)

// ----- Ultrasonic timing -----
const unsigned long PING_INTERVAL_MS = 60;     // ~16.7 Hz; HC-SR04 needs >=60ms between pings so echoes don't overlap
const unsigned long ECHO_TIMEOUT_US = 25000UL;  // 25ms ~= ~4.3m round trip; pulseIn() gives up after this
const int MEDIAN_WINDOW = 5;                    // "median of the last 5 readings" per the assignment

// ----- Detection thresholds (overridable live from config.py) -----
float g_minValidCm = 5.0f;     // readings closer than this are treated as sensor noise, not a real object
float g_maxRangeCm = 100.0f;   // "detect" threshold: ring diameter - farther than this isn't in the ring
float g_loseRangeCm = 115.0f;  // "lose" threshold (hysteresis): must retreat past THIS before we call it lost
int g_debounceCount = 3;       // consecutive agreeing readings required before flipping detected state

// How long without a *valid* echo before we stop trusting the last distance
// (still probably a working sensor, e.g. genuinely nothing within ~4m) vs.
// how long before we conclude the sensor itself isn't responding at all
// (X icon). The second threshold is deliberately much longer: a few seconds
// of "nothing within 4m" is physically plausible indoors, but a sensor that
// NEVER sees an echo for that long is almost certainly unplugged/miswired.
const int STALE_AFTER_PINGS = 8;          // ~0.5s @ ~16Hz
const int SENSOR_FAULT_AFTER_PINGS = 100; // ~6s @ ~16Hz

// ===================================================================
// ===== Ultrasonic measurement state =====
// ===================================================================
float validReadings[MEDIAN_WINDOW];
int validCount = 0;   // how many slots below are actually filled (caps at MEDIAN_WINDOW)
int validHead = 0;    // next slot to overwrite (circular buffer)

int consecutiveTimeouts = 0;
bool sensorOk = true;
bool haveValidDistance = false;
float currentMedianCm = -1.0f;
float currentConfidence = 0.0f;

bool detected = false;      // the debounced, published "visible" state
bool tentativeState = false; // the raw hysteresis state, waiting to be confirmed by debounceCounter
int debounceCounter = 0;

unsigned long lastPingMs = 0;

// ----- LED matrix -----
Arduino_LED_Matrix matrix;
const uint8_t FRAME_ROWS = 8;
const uint8_t FRAME_COLS = 13;
const uint8_t FRAME_SIZE = FRAME_ROWS * FRAME_COLS;

// ===================================================================
// ===== Sad song (reusable melody player, non-blocking) =====
// ===================================================================
// Verified on this board: the arduino:zephyr core implements tone()/
// noTone() (see the "Buzzer Melody" app in this same App Lab install),
// so we use those instead of bit-banging a square wave by hand.

const int NOTE_G4 = 392;
const int NOTE_F4 = 349;
const int NOTE_E4 = 330;
const int NOTE_D4 = 294;
const int NOTE_C4 = 262;

// Melody data kept separate from the playback engine below, so adding a
// new song later is just a new array passed to playMelody().
const int SAD_SONG_NOTES[] = { NOTE_G4, NOTE_F4, NOTE_E4, NOTE_D4, NOTE_C4 };
const int SAD_SONG_LENGTH = sizeof(SAD_SONG_NOTES) / sizeof(SAD_SONG_NOTES[0]);

const int NOTE_SLOT_MS = 1000; // total time budget per note: tone + silence
const int NOTE_ON_MS = 900;    // audible portion
const int NOTE_OFF_MS = NOTE_SLOT_MS - NOTE_ON_MS; // silent gap before the next note

enum MelodyPhase { PHASE_ON, PHASE_OFF };

bool melodyPlaying = false;
const int *melodyNotes = nullptr;
int melodyLength = 0;
int melodyIndex = 0;
MelodyPhase melodyPhase = PHASE_ON;
unsigned long melodyPhaseStartMs = 0;

// Starts (or restarts) a melody. Returns immediately - playback is advanced
// one step at a time by stepMelody(), called from loop() every iteration,
// so a whole 5-second song never blocks the Bridge or the LED matrix.
void playMelody(const int *notes, int length) {
  melodyNotes = notes;
  melodyLength = length;
  melodyIndex = 0;
  melodyPhase = PHASE_ON;
  melodyPhaseStartMs = millis();
  melodyPlaying = true;
  tone(BUZZER_PIN, melodyNotes[0]);
}

void stepMelody() {
  if (!melodyPlaying) {
    return;
  }
  unsigned long elapsed = millis() - melodyPhaseStartMs;

  if (melodyPhase == PHASE_ON && elapsed >= (unsigned long)NOTE_ON_MS) {
    noTone(BUZZER_PIN);
    melodyPhase = PHASE_OFF;
    melodyPhaseStartMs = millis();
  } else if (melodyPhase == PHASE_OFF && elapsed >= (unsigned long)NOTE_OFF_MS) {
    melodyIndex++;
    if (melodyIndex >= melodyLength) {
      melodyPlaying = false;
      noTone(BUZZER_PIN); // make sure the buzzer is silent once the song ends
      Bridge.notify("song_finished"); // tell Python the song is done
      return;
    }
    tone(BUZZER_PIN, melodyNotes[melodyIndex]);
    melodyPhase = PHASE_ON;
    melodyPhaseStartMs = millis();
  }
}

// Exposed to Python: Bridge.call("play_sad_song") from main.py.
// Does NOT run on startup - it only ever runs when Python asks for it.
void play_sad_song() {
  playMelody(SAD_SONG_NOTES, SAD_SONG_LENGTH);
}

// ===================================================================
// ===== LED matrix icons =====
// ===================================================================
// Python decides WHICH icon to show (it knows the MQTT state); these
// functions just know HOW to draw each one. Each is only called when
// Python's chosen icon actually changes, so the matrix is only redrawn
// on change as required.

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

uint8_t FRAME_NO_NETWORK[FRAME_SIZE] = { // circle-with-slash - no internet/broker
  0,0,7,0,0,7,7,7,0,0,0,0,0,
  0,0,0,7,7,0,0,0,7,0,0,0,0,
  0,0,0,7,7,0,0,0,0,7,0,0,0,
  0,0,0,7,0,0,7,0,0,7,0,0,0,
  0,0,0,7,0,0,0,7,0,7,0,0,0,
  0,0,0,0,7,0,0,0,7,0,0,0,0,
  0,0,0,0,0,7,7,7,0,7,0,0,0,
  0,0,0,0,0,0,0,0,0,0,7,0,0
};

uint8_t FRAME_WAITING_LAPTOP[FRAME_SIZE] = { // "..." - waiting for the laptop
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,7,0,0,7,0,0,7,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0
};

uint8_t FRAME_ARMED[FRAME_SIZE] = { // hollow diamond - armed, waiting for the start lever
  0,0,0,0,0,0,7,0,0,0,0,0,0,
  0,0,0,0,0,7,0,7,0,0,0,0,0,
  0,0,0,0,7,0,0,0,7,0,0,0,0,
  0,0,0,7,0,0,0,0,0,7,0,0,0,
  0,0,0,7,0,0,0,0,0,7,0,0,0,
  0,0,0,0,7,0,0,0,7,0,0,0,0,
  0,0,0,0,0,7,0,7,0,0,0,0,0,
  0,0,0,0,0,0,7,0,0,0,0,0,0
};

uint8_t FRAME_REVERSING[FRAME_SIZE] = { // left arrow - driving backward
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,7,0,0,0,0,0,0,0,
  0,0,0,0,7,7,0,0,0,0,0,0,0,
  0,0,7,7,7,7,7,7,7,7,7,7,0,
  0,0,7,7,7,7,7,7,7,7,7,7,0,
  0,0,0,0,7,7,0,0,0,0,0,0,0,
  0,0,0,0,0,7,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0
};

uint8_t FRAME_TELEOP[FRAME_SIZE] = { // game-controller silhouette - human driving
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,7,0,0,0,7,0,0,0,0,
  0,0,0,0,7,7,7,7,7,0,0,0,0,
  0,0,7,7,7,7,7,7,7,7,7,0,0,
  0,0,7,7,7,7,7,7,7,7,7,0,0,
  0,0,0,0,7,7,7,7,7,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0
};

uint8_t FRAME_IDLE[FRAME_SIZE] = { // single centered dot - idle
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,7,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0
};

uint8_t FRAME_LOST[FRAME_SIZE] = { // sad face - opponent (or us) pushed out
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,7,0,0,0,7,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,7,0,0,0,7,0,0,0,0,
  0,0,0,0,0,7,7,7,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0
};

// ----- "chasing" is drawn, not a fixed bitmap: a box centered on the
// matrix (a single ultrasonic sensor can't tell left from right, so we
// always center it) that grows as the opponent gets closer. -----
const uint8_t CHASE_HALF_SIZE[] = { 0, 1, 2, 3 }; // half-size in pixels per step; must match LED_CHASE_SIZE_STEPS in config.py
const uint8_t CHASE_SIZE_STEPS = sizeof(CHASE_HALF_SIZE) / sizeof(CHASE_HALF_SIZE[0]);
const uint8_t CHASE_CENTER_ROW = 3; // 0-indexed; rows 3/4 are the two middle rows of 8
const uint8_t CHASE_CENTER_COL = 6; // middle column of 13

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

// ----- Exposed to Python: one function per icon (see main.py's _update_led) -----
void show_icon_sensor_error() { matrix.draw(FRAME_SENSOR_ERROR); }
void show_icon_no_network()   { matrix.draw(FRAME_NO_NETWORK); }
void show_icon_waiting_laptop() { matrix.draw(FRAME_WAITING_LAPTOP); }
void show_icon_armed()        { matrix.draw(FRAME_ARMED); }
void show_icon_reversing()    { matrix.draw(FRAME_REVERSING); }
void show_icon_teleop()       { matrix.draw(FRAME_TELEOP); }
void show_icon_idle()         { matrix.draw(FRAME_IDLE); }
void show_icon_lost()         { matrix.draw(FRAME_LOST); }
void show_icon_chasing(int sizeStep) { drawChasingFrame(sizeStep); }

// ===================================================================
// ===== Bridge: configuration push from Python =====
// ===================================================================
// Exposed to Python: Bridge.call("configure", ...) from main.py, called
// once at startup with whatever config.py says. Until that first call
// lands (or if it never does), the defaults declared above are used.
void configure(float minValidCm, float maxRangeCm, float loseRangeCm, int debounceCount) {
  g_minValidCm = minValidCm;
  g_maxRangeCm = maxRangeCm;
  g_loseRangeCm = loseRangeCm;
  g_debounceCount = debounceCount;
  Monitor.println("[sketch] configuration received from Python");
}

// ===================================================================
// ===== Ultrasonic measurement =====
// ===================================================================

// Fires one ultrasonic ping and returns the measured distance in cm, or
// -1.0 if no echo came back within ECHO_TIMEOUT_US (nothing in range, or
// the sensor isn't responding).
//
// This measures the echo by hand with micros() + digitalRead() instead of
// pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US). pulseIn() was tried first (it
// compiles fine on this board's arduino:zephyr core), but testing with no
// sensor connected - Echo permanently LOW - showed it does NOT reliably
// honor its timeout in that case: the sketch stopped responding to Bridge
// calls entirely (they timed out after 10s) and ultrasonic updates stopped
// arriving, which means pulseIn() was blocking far longer than its stated
// timeout while waiting for a rising edge that never came. Polling
// digitalRead() ourselves, with an explicit micros()-based deadline
// checked on every single loop iteration of BOTH phases (waiting for Echo
// to rise, then waiting for it to fall), guarantees this function can
// never block longer than ECHO_TIMEOUT_US no matter what the pin is doing -
// which matters a lot here, since "Echo never responds" is exactly the
// real-world case of a disconnected/broken sensor that this app needs to
// detect and recover from, not hang on.
float pingOnce() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long waitStart = micros();
  while (digitalRead(ECHO_PIN) == LOW) {
    if (micros() - waitStart >= ECHO_TIMEOUT_US) {
      return -1.0f; // Echo never rose: nothing in range, or sensor not responding
    }
  }

  unsigned long pulseStart = micros();
  while (digitalRead(ECHO_PIN) == HIGH) {
    if (micros() - pulseStart >= ECHO_TIMEOUT_US) {
      return -1.0f; // Echo got stuck HIGH: treat the same as a timeout
    }
  }
  unsigned long durationUs = micros() - pulseStart;

  // Speed of sound ~0.0343 cm/us; round trip there-and-back, so halve it.
  return (durationUs * 0.0343f) / 2.0f;
}

void pushValidReading(float cm) {
  validReadings[validHead] = cm;
  validHead = (validHead + 1) % MEDIAN_WINDOW;
  if (validCount < MEDIAN_WINDOW) {
    validCount++;
  }
}

// Small insertion-sort median - n is at most MEDIAN_WINDOW (5), so a
// simple O(n^2) sort is plenty fast and avoids pulling in <algorithm>.
float medianOf(const float *values, int n) {
  float sorted[MEDIAN_WINDOW];
  for (int i = 0; i < n; i++) {
    sorted[i] = values[i];
  }
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

// "How consistent are the last few readings?" - the fraction of the
// buffered readings that land within tolerance of the median.
float computeConfidence(float medianCm) {
  if (validCount == 0) {
    return 0.0f;
  }
  float tolerance = fmaxf(10.0f, medianCm * 0.15f); // 15%, or 10cm, whichever is bigger
  int agree = 0;
  for (int i = 0; i < validCount; i++) {
    if (fabsf(validReadings[i] - medianCm) <= tolerance) {
      agree++;
    }
  }
  return (float)agree / (float)validCount;
}

void doPingCycle() {
  float raw = pingOnce();

  if (raw < 0.0f || raw < g_minValidCm) {
    // Timeout, or "too close to be real" (HC-SR04's own blind zone / an
    // echo bouncing off the robot's own chassis) - treat both like a miss.
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

  // ---- hysteresis: which threshold applies depends on the CURRENT
  // debounced state, not the tentative one, so a single stray reading
  // crossing a threshold can't both detect AND lose in two back-to-back
  // cycles. ----
  bool rawInRange;
  if (!haveValidDistance) {
    rawInRange = false;
  } else if (!detected) {
    rawInRange = currentMedianCm <= g_maxRangeCm;
  } else {
    rawInRange = currentMedianCm <= g_loseRangeCm;
  }

  // ---- debounce: only flip the published state after g_debounceCount
  // consecutive cycles agree. ----
  if (rawInRange == tentativeState) {
    if (debounceCounter < 255) {
      debounceCounter++;
    }
  } else {
    tentativeState = rawInRange;
    debounceCounter = 1;
  }
  if (debounceCounter >= g_debounceCount) {
    detected = tentativeState;
  }

  // Notify is throttled to a steady heartbeat instead of firing on every
  // single ~60ms ping, as a precaution against flooding the Bridge with
  // more traffic than it needs - `detected` changing is always sent
  // immediately regardless, since that's the one thing that genuinely
  // needs to be snappy. The ping/detection bookkeeping above always keeps
  // running at full rate either way - only how often we TELL Python about
  // it is throttled. (The main fix for the Bridge call hangs seen during
  // testing turned out to be on the Python side - see main.py's
  // _on_vision_update docstring and the README.)
  static bool lastNotifiedDetected = false;
  static unsigned long lastNotifyMs = 0;
  const unsigned long NOTIFY_MIN_INTERVAL_MS = 100; // ~10Hz heartbeat cap

  bool detectedChanged = (detected != lastNotifiedDetected);
  bool heartbeatDue = (millis() - lastNotifyMs >= NOTIFY_MIN_INTERVAL_MS);

  if (!melodyPlaying && (detectedChanged || heartbeatDue)) {
    Bridge.notify("vision_update", detected, haveValidDistance ? currentMedianCm : -1.0f, currentConfidence, sensorOk);
    lastNotifiedDetected = detected;
    lastNotifyMs = millis();
  }
}

// ===================================================================
void setup() {
  Monitor.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  // NOTE: no pinMode() on BUZZER_PIN - tone()/noTone() configure the pin
  // themselves, and calling pinMode(OUTPUT) first has caused PWM glitches
  // on this board's Zephyr core on other pins in this same project.

  matrix.begin();
  matrix.setGrayscaleBits(3); // brightness values 0-7
  matrix.clear();
  show_icon_waiting_laptop(); // reasonable default: nothing is known yet at boot

  Bridge.begin();
  Bridge.provide("configure", configure);
  Bridge.provide("play_sad_song", play_sad_song);
  Bridge.provide("show_icon_sensor_error", show_icon_sensor_error);
  Bridge.provide("show_icon_no_network", show_icon_no_network);
  Bridge.provide("show_icon_waiting_laptop", show_icon_waiting_laptop);
  Bridge.provide("show_icon_armed", show_icon_armed);
  Bridge.provide("show_icon_reversing", show_icon_reversing);
  Bridge.provide("show_icon_chasing", show_icon_chasing);
  Bridge.provide("show_icon_teleop", show_icon_teleop);
  Bridge.provide("show_icon_idle", show_icon_idle);
  Bridge.provide("show_icon_lost", show_icon_lost);
}

void loop() {
  unsigned long now = millis();
  if (now - lastPingMs >= PING_INTERVAL_MS) {
    lastPingMs = now;
    doPingCycle();
  }
  stepMelody(); // advances the sad song, if one is playing - never blocks
}
