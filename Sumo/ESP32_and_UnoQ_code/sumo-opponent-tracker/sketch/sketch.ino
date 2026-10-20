// Sumo Opponent Tracker - microcontroller sketch.
//
// Runs on the UNO Q's STM32 MCU side. Three jobs, all from one tiny loop():
//   1. Draw status icons on the 8x13 LED matrix - but only when Python
//      tells us to (Python owns the decision of *which* icon to show, by
//      combining camera health, MQTT/laptop status, and robot/state from
//      the laptop's behavior tree; the sketch just knows HOW to draw each
//      one). Most icons are fixed bitmaps; "searching"/"chasing" (the
//      camera-driven ones) are drawn procedurally since their content
//      changes every frame.
//   2. Play a 5-note "sad song" on the buzzer when Python asks for it,
//      WITHOUT ever blocking (no delay() in loop()) - stepMelody() advances
//      the song one step per loop() call instead, using millis() timers,
//      the same pattern the "searching" sweep already used. This matters
//      because a blocking 5-second song would freeze LED updates and
//      Bridge calls for the whole song.
//   3. Tell Python when the song finishes (Bridge.notify), so main.py knows
//      when to publish the event/song_done MQTT message.
//
// The buzzer/icon code below is adapted from this same App Lab install's
// "Buzzer Melody" and "RQ-D2 Vision" apps (verified working there first),
// not written from scratch - see those apps for the original standalone
// versions and more detail on why tone()/noTone() and the non-blocking
// melody design were chosen.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <string.h>

Arduino_LED_Matrix matrix;

const uint8_t FRAME_ROWS = 8;
const uint8_t FRAME_COLS = 13;
const uint8_t FRAME_SIZE = FRAME_ROWS * FRAME_COLS;
const uint8_t CENTER_ROW = 3; // 0-indexed; rows 0-7, so 3/4 are the two middle rows

// ===================================================================
// ===== Fixed-bitmap icons =====
// ===================================================================
// Each is read top row to bottom row, left to right, 13 numbers per row.
// 0 = pixel off, 7 = pixel at full brightness (3-bit grayscale).

uint8_t FRAME_NO_CAMERA[FRAME_SIZE] = { // "X" - ESP32-CAM stream missing/stale
    7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7,
    0, 0, 7, 7, 0, 0, 0, 0, 0, 7, 7, 0, 0,
    0, 0, 0, 0, 7, 7, 0, 7, 7, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 7, 0, 7, 7, 0, 0, 0, 0,
    0, 0, 7, 7, 0, 0, 0, 0, 0, 7, 7, 0, 0,
    7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7
};

uint8_t FRAME_NO_NETWORK[FRAME_SIZE] = { // circle-with-slash - no internet/can't reach the MQTT broker
    0, 0, 7, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 7, 7, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 7, 7, 0, 0, 0, 0, 7, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 0, 0, 0,
    0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 7, 7, 0, 7, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0
};

uint8_t FRAME_WAITING_LAPTOP[FRAME_SIZE] = { // "..." - laptop not connected / laptop/status offline or missing
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

uint8_t FRAME_ARMED[FRAME_SIZE] = { // hollow diamond - armed, waiting for the start lever push
    0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0,
    0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0
};

uint8_t FRAME_REVERSING[FRAME_SIZE] = { // left arrow - driving backward
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0,
    0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0,
    0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

uint8_t FRAME_TELEOP[FRAME_SIZE] = { // game-controller silhouette - a person is driving
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0,
    0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0,
    0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0,
    0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

uint8_t FRAME_IDLE[FRAME_SIZE] = { // single centered dot - robot/state running, no particular branch
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

uint8_t FRAME_LOST[FRAME_SIZE] = { // sad face - event/lost received, or robot/state.phase == "lost"
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// ===================================================================
// ===== "searching" animation (camera alive, opponent not currently seen) =====
// ===================================================================
// A single dot sweeps back and forth across the middle row so it's obvious
// at a glance that the app is actively looking, not stuck.
bool searching = false;
unsigned long lastSearchMs = 0;
const unsigned long SEARCH_INTERVAL_MS = 80; // lower = faster sweep
int searchCol = 0;
int searchDir = 1;

void drawSearchingFrame() {
    uint8_t frame[FRAME_SIZE];
    memset(frame, 0, sizeof(frame));
    frame[CENTER_ROW * FRAME_COLS + searchCol] = 7;
    matrix.draw(frame);
}

// ===================================================================
// ===== "chasing" box (camera alive, opponent currently seen) =====
// ===================================================================
// half-size (in pixels, each direction) for each size step - step 0 is a
// single pixel, step 3 is as large as the matrix's height comfortably
// allows. Add more entries here for finer size steps if you want them;
// just keep LED_SIZE_STEPS in main.py's TUNABLE section matching this
// array's length.
const uint8_t TARGET_HALF_SIZE[] = {0, 1, 2, 3};
const uint8_t TARGET_SIZE_STEPS = sizeof(TARGET_HALF_SIZE) / sizeof(TARGET_HALF_SIZE[0]);

void drawTargetFrame(int col, int sizeStep) {
    if (sizeStep < 0) sizeStep = 0;
    if (sizeStep >= TARGET_SIZE_STEPS) sizeStep = TARGET_SIZE_STEPS - 1;
    uint8_t half = TARGET_HALF_SIZE[sizeStep];

    uint8_t frame[FRAME_SIZE];
    memset(frame, 0, sizeof(frame));

    int rowLo = max(0, CENTER_ROW - half);
    int rowHi = min(FRAME_ROWS - 1, CENTER_ROW + half);
    int colLo = max(0, col - half);
    int colHi = min(FRAME_COLS - 1, col + half);

    for (int r = rowLo; r <= rowHi; r++) {
        for (int c = colLo; c <= colHi; c++) {
            frame[r * FRAME_COLS + c] = 7;
        }
    }
    matrix.draw(frame);
}

// ===================================================================
// ===== Sad song (reusable melody player, non-blocking) =====
// ===================================================================
// Verified on this board (see the "Buzzer Melody" app in this same App Lab
// install): the arduino:zephyr core implements tone()/noTone(), so no
// manual square-wave bit-banging is needed.

const int BUZZER_PIN = 5; // passive buzzer signal (S), PWM-capable pin. + -> 5V, - -> GND.

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

// Exposed to Python: Bridge.call("play_sad_song") from main.py, on
// event/lost. Does NOT run on startup - only ever runs when Python asks.
void play_sad_song() {
    playMelody(SAD_SONG_NOTES, SAD_SONG_LENGTH);
}

// ===================================================================
// ===== EXPOSED TO PYTHON: one function per icon =====
// ===================================================================
// Python owns the decision of *which* icon to show (see main.py's
// _update_led) - these just know HOW to draw each one. Showing any
// non-"searching" icon stops the searching sweep's own timer.

void show_no_camera() {
    searching = false;
    matrix.draw(FRAME_NO_CAMERA);
}

void show_searching() {
    if (!searching) {
        searching = true;
        searchCol = 0;
        searchDir = 1;
        lastSearchMs = millis();
        drawSearchingFrame();
    }
}

void show_target(int col, int sizeStep) {
    searching = false;
    drawTargetFrame(col, sizeStep);
}

void show_icon_no_network()     { searching = false; matrix.draw(FRAME_NO_NETWORK); }
void show_icon_waiting_laptop() { searching = false; matrix.draw(FRAME_WAITING_LAPTOP); }
void show_icon_armed()          { searching = false; matrix.draw(FRAME_ARMED); }
void show_icon_reversing()      { searching = false; matrix.draw(FRAME_REVERSING); }
void show_icon_teleop()         { searching = false; matrix.draw(FRAME_TELEOP); }
void show_icon_idle()           { searching = false; matrix.draw(FRAME_IDLE); }
void show_icon_lost()           { searching = false; matrix.draw(FRAME_LOST); }

void setup() {
    Monitor.begin(115200);

    matrix.begin();
    matrix.setGrayscaleBits(3); // brightness values 0-7
    matrix.clear();
    show_icon_waiting_laptop(); // reasonable default: nothing is known yet at boot

    // NOTE: no pinMode() on BUZZER_PIN - tone()/noTone() configure the pin
    // themselves; calling pinMode(OUTPUT) first has caused PWM glitches on
    // this board's Zephyr core on other pins in this same project.

    Bridge.begin();
    Bridge.provide("show_no_camera", show_no_camera);
    Bridge.provide("show_searching", show_searching);
    Bridge.provide("show_target", show_target);
    Bridge.provide("show_icon_no_network", show_icon_no_network);
    Bridge.provide("show_icon_waiting_laptop", show_icon_waiting_laptop);
    Bridge.provide("show_icon_armed", show_icon_armed);
    Bridge.provide("show_icon_reversing", show_icon_reversing);
    Bridge.provide("show_icon_teleop", show_icon_teleop);
    Bridge.provide("show_icon_idle", show_icon_idle);
    Bridge.provide("show_icon_lost", show_icon_lost);
    Bridge.provide("play_sad_song", play_sad_song);
}

void loop() {
    // Advance the searching sweep on its own timer; fixed-bitmap icons and
    // the target box are drawn once inside their own Bridge functions, so
    // there's nothing to do for those here.
    if (searching && millis() - lastSearchMs >= SEARCH_INTERVAL_MS) {
        lastSearchMs = millis();
        searchCol += searchDir;
        if (searchCol >= FRAME_COLS - 1 || searchCol <= 0) {
            searchDir = -searchDir;
        }
        drawSearchingFrame();
    }
    stepMelody(); // advances the sad song, if one is playing - never blocks
}
