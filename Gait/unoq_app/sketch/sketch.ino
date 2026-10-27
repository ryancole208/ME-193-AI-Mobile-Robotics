// Gait Activity Classifier - MCU sketch (STM32, Arduino core on Zephyr).
//
// This side does ONE job: draw the 8x13 LED matrix. All animation timing
// lives here, so the stick figures stay smooth even when Python is busy
// computing features. Python only sends small "what to show" commands over
// the Router Bridge:
//
//   show_activity(class_id)        base screen = stick-figure animation
//   show_model(model_id, banner)   column-12 indicator; banner=1 also shows "1D"/"6D"
//   show_status(code)              status screen or overlay flag (codes below)
//   show_setup(step, progress)     setup / diag screens (codes below)
//   show_value(kind, value)        queued transient: scrolling accuracy, battery bar
//   display_version()              returns DISPLAY_PROTOCOL_VERSION (handshake)
//
// Every handler is registered with provide_safe(), so it runs on the loop
// thread between two loop() calls and can never race the renderer.
//
// Screen layout while classifying:
//   columns 0-10  animation
//   column  11    blank (row 7 blinks while the redo-setup window is open)
//   column  12    active model: top 2 pixels = A ("1D"), bottom 2 = B ("6D")
//
// The art is stored as ASCII strings: '#' = full brightness, '+' = dim,
// '.' = off. Edit them by hand and check the result without flashing with
//   python3 tools/dev.py preview
// (tools/matrix_preview.py parses this file).

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <string.h>

// =====================================================================
// Display protocol. KEEP IN SYNC with python/gait/constants.py.
// =====================================================================
#define DISPLAY_PROTOCOL_VERSION 1

// activity classes (index into config.json "classes")
#define CLASS_STILL   0
#define CLASS_WALKING 1
#define CLASS_RUNNING 2
#define CLASS_BIKING  3

// models
#define MODEL_A 0  // "1D": max(a_y)
#define MODEL_B 1  // "6D": full IMU

// show_status(code)
#define ST_SEARCHING       1   // base: scanning dot (looking for GaitNode)
#define ST_LINK_LOST       2   // base: big X (no data for > 2 s)
#define ST_UNSURE          3   // base: big ? (unsure / not enough data)
#define ST_CFG_ERROR       4   // transient: "CFG" then "X"
#define ST_LEG_MISMATCH    5   // transient: "LEG" then "?"
#define ST_FLASH3          6   // transient: whole matrix flashes 3 times
#define ST_CHECK           7   // transient: check mark
#define ST_LOW_BATT_ON     8   // flag: flash battery icon every 30 s
#define ST_LOW_BATT_OFF    9   // flag off
#define ST_REDO_WINDOW_ON  10  // flag: blinking marker, redo-setup window open
#define ST_REDO_WINDOW_OFF 11  // flag off
#define ST_FLASH1          12  // transient: single flash (gesture acknowledged)
#define ST_IDLE            13  // base: blank except indicator

// show_setup(step, progress)
#define STEP_STAND          0   // progress = seconds remaining
#define STEP_COUNTDOWN_BASE 10  // 10 + class, progress = seconds remaining
#define STEP_RECORD_BASE    20  // 20 + class, progress = percent done
#define STEP_GESTURE_CHECK  30  // progress = gestures detected so far (0..5)
#define STEP_BUILD          31  // progress ignored (spinner)
#define STEP_PARTIAL        32  // progress = class index still missing
#define STEP_SET_CONFIRM    33  // scrolling "SET?"
#define STEP_DIAG           34  // progress = check number being run (1..7)
#define STEP_DIAG_DONE      35  // progress = 1 all passed, 0 something failed
#define STEP_HOLDOUT_DONE   36  // held-out recording finished

// show_value(kind, value)
#define VAL_ACC_A   1  // value = percent; scrolls "A 87%"
#define VAL_ACC_B   2  // value = percent; scrolls "B 91%"
#define VAL_BATTERY 3  // value = percent 0..100, or -1 = reading unavailable ("N/A")

// =====================================================================
// Art. Activity frames are 11 columns x 8 rows (columns 0-10).
// =====================================================================
#define ART_W 11
#define ROWS 8
#define COLS 13

// Still: standing figure. Frame 1 expands the chest ("breathing"); each
// frame is held for 2 s.
static const char* const STILL_FRAMES[2][ROWS] = {
  { ".....#.....",
    "....###....",
    "...#.#.#...",
    "...#.#.#...",
    ".....#.....",
    "....#.#....",
    "....#.#....",
    "...##.##..." },
  { ".....#.....",
    "...#####...",
    "...#.#.#...",
    "...#.#.#...",
    ".....#.....",
    "....#.#....",
    "....#.#....",
    "...##.##..." },
};

// Walking: upright body, short stride, arms swing opposite to the legs.
// 4 frames at ~3 fps.
static const char* const WALK_FRAMES[4][ROWS] = {
  { ".....#.....",
    "....###....",
    "...#.#.#...",
    "..#..#..#..",
    ".....#.....",
    "....#.#....",
    "...#...#...",
    "..##...##.." },
  { ".....#.....",
    "....###....",
    "....##.....",
    "....##.....",
    ".....#.....",
    ".....##....",
    ".....#.#...",
    "....##.#..." },
  { ".....#.....",
    "....###....",
    "...#.#.#...",
    "..#..#..#..",
    ".....#.....",
    "....#.#....",
    "...#...#...",
    "..##...##.." },
  { ".....#.....",
    "....###....",
    ".....##....",
    ".....##....",
    ".....#.....",
    "....##.....",
    "...#.#.....",
    "...#.##...." },
};

// Running: body leans forward, long stride, bent arms. 4 frames at ~6 fps.
static const char* const RUN_FRAMES[4][ROWS] = {
  { ".......#...",
    "....####.#.",
    "...#..#.#..",
    "..#..#.....",
    ".....#.....",
    "....#.##...",
    "..##....#..",
    ".#.......#." },
  { ".......#...",
    ".....###...",
    "....#.#.#..",
    ".....#.#...",
    ".....#.....",
    ".....##....",
    "....#.#....",
    "....#..#..." },
  { ".......#...",
    ".#..####...",
    "..#.#.#....",
    ".....#..#..",
    ".....#.....",
    "....##.#...",
    "...#....##.",
    "..#........" },
  { ".......#...",
    ".....###...",
    "....#.#.#..",
    ".....#.#...",
    ".....#.....",
    "....##.....",
    "....#.#....",
    "...#..#...." },
};

// Biking: seated rider on a two-wheeled bike (wheels dim). The pedal leg
// goes round the crank in 4 frames at ~5 fps.
static const char* const BIKE_FRAMES[4][ROWS] = {
  { ".....#.....",
    ".....##....",
    "....#..##..",
    "...#####.#.",
    "...#.#...#.",
    ".+++.#.+++.",
    "+...+.+...+",
    ".+++...+++." },
  { ".....#.....",
    ".....##....",
    "....#..##..",
    "...#####.#.",
    "...#..#..#.",
    ".+++..#+++.",
    "+...+.#...+",
    ".+++...+++." },
  { ".....#.....",
    ".....##....",
    "....#..##..",
    "...#####.#.",
    "...#.#...#.",
    ".+++.#.+++.",
    "+...+#+...+",
    ".+++.#.+++." },
  { ".....#.....",
    ".....##....",
    "....#..##..",
    "...#####.#.",
    "...##....#.",
    ".+++#..+++.",
    "+...#.+...+",
    ".+++...+++." },
};

// Status icons (11 x 8, drawn in columns 0-10).
static const char* const ICON_X[ROWS] = {
  "..#.....#..",
  "...#...#...",
  "....#.#....",
  ".....#.....",
  "....#.#....",
  "...#...#...",
  "..#.....#..",
  "..........." };

static const char* const ICON_QUESTION[ROWS] = {
  "....###....",
  "...#...#...",
  ".......#...",
  "......#....",
  ".....#.....",
  ".....#.....",
  "...........",
  ".....#....." };

// Full-width icons (13 x 8).
static const char* const ICON_CHECK[ROWS] = {
  ".............",
  "..........#..",
  ".........#...",
  "........#....",
  "..#....#.....",
  "...#..#......",
  "....##.......",
  "............." };

static const char* const ICON_BATTERY[ROWS] = {
  ".............",
  ".##########..",
  ".#........#..",
  ".#+.......##.",
  ".#+.......##.",
  ".#........#..",
  ".##########..",
  "............." };

// =====================================================================
// 3x5 font (W and N are wider). Each row is a bit pattern, MSB = left.
// =====================================================================
struct Glyph {
  char c;
  uint8_t w;
  uint8_t rows[5];
};

static const Glyph FONT[] = {
  {'0', 3, {0b111, 0b101, 0b101, 0b101, 0b111}},
  {'1', 3, {0b010, 0b110, 0b010, 0b010, 0b111}},
  {'2', 3, {0b111, 0b001, 0b111, 0b100, 0b111}},
  {'3', 3, {0b111, 0b001, 0b011, 0b001, 0b111}},
  {'4', 3, {0b101, 0b101, 0b111, 0b001, 0b001}},
  {'5', 3, {0b111, 0b100, 0b111, 0b001, 0b111}},
  {'6', 3, {0b111, 0b100, 0b111, 0b101, 0b111}},
  {'7', 3, {0b111, 0b001, 0b010, 0b010, 0b010}},
  {'8', 3, {0b111, 0b101, 0b111, 0b101, 0b111}},
  {'9', 3, {0b111, 0b101, 0b111, 0b001, 0b111}},
  {'A', 3, {0b010, 0b101, 0b111, 0b101, 0b101}},
  {'B', 3, {0b110, 0b101, 0b110, 0b101, 0b110}},
  {'C', 3, {0b011, 0b100, 0b100, 0b100, 0b011}},
  {'D', 3, {0b110, 0b101, 0b101, 0b101, 0b110}},
  {'E', 3, {0b111, 0b100, 0b110, 0b100, 0b111}},
  {'F', 3, {0b111, 0b100, 0b110, 0b100, 0b100}},
  {'G', 3, {0b011, 0b100, 0b101, 0b101, 0b011}},
  {'H', 3, {0b101, 0b101, 0b111, 0b101, 0b101}},
  {'L', 3, {0b100, 0b100, 0b100, 0b100, 0b111}},
  {'N', 4, {0b1001, 0b1101, 0b1011, 0b1001, 0b1001}},
  {'R', 3, {0b110, 0b101, 0b110, 0b101, 0b101}},
  {'S', 3, {0b011, 0b100, 0b010, 0b001, 0b110}},
  {'T', 3, {0b111, 0b010, 0b010, 0b010, 0b010}},
  {'W', 5, {0b10001, 0b10001, 0b10101, 0b10101, 0b01010}},
  {'X', 3, {0b101, 0b101, 0b010, 0b101, 0b101}},
  {'?', 3, {0b110, 0b001, 0b010, 0b000, 0b010}},
  {'%', 3, {0b101, 0b001, 0b010, 0b100, 0b101}},
  {'/', 3, {0b001, 0b001, 0b010, 0b100, 0b100}},
  {'-', 3, {0b000, 0b000, 0b111, 0b000, 0b000}},
  {' ', 2, {0, 0, 0, 0, 0}},
};

static const char CLASS_LETTERS[4] = {'S', 'W', 'R', 'B'};

// =====================================================================
// State
// =====================================================================
Arduino_LED_Matrix matrix;

static uint8_t fb[ROWS * COLS];       // frame being composed
static uint8_t shown[ROWS * COLS];    // last frame sent to the matrix

enum BaseMode { BASE_IDLE, BASE_SEARCHING, BASE_LINK_LOST, BASE_UNSURE, BASE_ACTIVITY, BASE_SETUP };

static BaseMode baseMode = BASE_SEARCHING;
static int baseClass = CLASS_STILL;
static unsigned long baseSince = 0;     // animation phase reference
static int setupStep = 0;
static int setupProgress = 0;
static int activeModel = MODEL_A;
static bool lowBattFlag = false;
static unsigned long lastBattFlash = 0;
static bool redoWindowFlag = false;

// Transients: short screens that interrupt the base screen, played in order.
enum TransientKind { TR_TEXT, TR_SCROLL, TR_FLASH, TR_ICON_CHECK, TR_ICON_X, TR_ICON_BATTERY, TR_BATTERY_LEVEL };

struct Transient {
  TransientKind kind;
  char text[12];
  int value;
  unsigned long durationMs;
};

#define QUEUE_LEN 8
static Transient queue[QUEUE_LEN];
static int qHead = 0, qCount = 0;
static unsigned long transientStart = 0;
static bool transientRunning = false;

// =====================================================================
// Drawing helpers
// =====================================================================
static void clearFb() { memset(fb, 0, sizeof(fb)); }

static void setPx(int r, int c, uint8_t v) {
  if (r >= 0 && r < ROWS && c >= 0 && c < COLS) fb[r * COLS + c] = v;
}

static uint8_t artValue(char ch) {
  if (ch == '#') return 7;
  if (ch == '+') return 3;
  return 0;
}

static void drawArt(const char* const art[ROWS], int width, int col0) {
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < width; c++) setPx(r, col0 + c, artValue(art[r][c]));
  }
}

static const Glyph* findGlyph(char c) {
  for (unsigned i = 0; i < sizeof(FONT) / sizeof(FONT[0]); i++) {
    if (FONT[i].c == c) return &FONT[i];
  }
  return &FONT[sizeof(FONT) / sizeof(FONT[0]) - 1];  // unknown -> space
}

// Width in pixels of a string, with 1 blank column between glyphs.
static int textWidth(const char* s) {
  int w = 0;
  for (int i = 0; s[i]; i++) w += findGlyph(s[i])->w + (s[i + 1] ? 1 : 0);
  return w;
}

// Draw text with its left edge at column x (may be negative while scrolling).
static void drawText(const char* s, int x, int row0) {
  for (int i = 0; s[i]; i++) {
    const Glyph* g = findGlyph(s[i]);
    for (int r = 0; r < 5; r++) {
      for (int c = 0; c < g->w; c++) {
        if (g->rows[r] & (1 << (g->w - 1 - c))) setPx(row0 + r, x + c, 7);
      }
    }
    x += g->w + 1;
  }
}

static void drawTextCentered(const char* s, int row0) {
  drawText(s, (COLS - textWidth(s)) / 2, row0);
}

// Bottom-row progress bar, 0..100 percent of the 13 columns.
static void drawBar(int percent) {
  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;
  int lit = (percent * COLS + 50) / 100;
  for (int c = 0; c < COLS; c++) setPx(7, c, c < lit ? 7 : 1);
}

static void drawModelIndicator() {
  if (activeModel == MODEL_A) {
    setPx(0, 12, 7);
    setPx(1, 12, 7);
  } else {
    setPx(6, 12, 7);
    setPx(7, 12, 7);
  }
}

// A readable "A"/"B" in the top-left corner, on top of whatever else is drawn
// there (e.g. the activity animation) - clear its 3x5 patch first so it stays
// legible instead of blending into the art.
static void drawModelLetter() {
  for (int r = 0; r < 5; r++) {
    for (int c = 0; c < 3; c++) setPx(r, c, 0);
  }
  drawText(activeModel == MODEL_A ? "A" : "B", 0, 0);
}

static void drawNumberRight(int value, int row0) {
  char buf[6];
  snprintf(buf, sizeof(buf), "%d", value);
  drawText(buf, COLS - textWidth(buf), row0);
}

// =====================================================================
// Base screens
// =====================================================================
static void drawActivity(unsigned long now) {
  unsigned long t = now - baseSince;
  switch (baseClass) {
    case CLASS_STILL:   drawArt(STILL_FRAMES[(t / 2000) % 2], ART_W, 0); break;
    case CLASS_WALKING: drawArt(WALK_FRAMES[(t / 333) % 4], ART_W, 0); break;
    case CLASS_RUNNING: drawArt(RUN_FRAMES[(t / 167) % 4], ART_W, 0); break;
    case CLASS_BIKING:  drawArt(BIKE_FRAMES[(t / 200) % 4], ART_W, 0); break;
    default:            drawArt(ICON_QUESTION, ART_W, 0); break;
  }
}

// A dot sweeping back and forth across the animation area.
static void drawSearching(unsigned long now) {
  int period = 2 * (ART_W - 1);
  int pos = (int)((now / 90) % period);
  int col = pos < ART_W ? pos : period - pos;
  setPx(3, col, 7);
  setPx(4, col, 7);
  // faint trail so the direction is visible
  int trail = pos < ART_W ? col - 1 : col + 1;
  setPx(3, trail, 2);
  setPx(4, trail, 2);
}

static void drawSpinner(unsigned long now) {
  static const int ring[8][2] = {{2, 5}, {2, 6}, {2, 7}, {3, 7}, {4, 7}, {4, 6}, {4, 5}, {3, 5}};
  int head = (now / 100) % 8;
  for (int i = 0; i < 8; i++) {
    int age = (head - i + 8) % 8;
    setPx(ring[i][0], ring[i][1], age == 0 ? 7 : (age < 3 ? 3 : 1));
  }
}

static void drawLetter(int classId, int col0) {
  char s[2] = {CLASS_LETTERS[classId & 3], 0};
  drawText(s, col0, 0);
}

static void drawSetup(unsigned long now) {
  int step = setupStep;
  int p = setupProgress;
  if (step == STEP_STAND) {
    // countdown in the middle, shrinking bar underneath (10 s = full)
    char buf[6];
    snprintf(buf, sizeof(buf), "%d", p);
    drawTextCentered(buf, 1);
    drawBar(p * 10);
  } else if (step >= STEP_COUNTDOWN_BASE && step < STEP_COUNTDOWN_BASE + 4) {
    drawLetter(step - STEP_COUNTDOWN_BASE, 0);
    drawNumberRight(p, 0);
    drawBar(p * 10);
  } else if (step >= STEP_RECORD_BASE && step < STEP_RECORD_BASE + 4) {
    drawLetter(step - STEP_RECORD_BASE, 0);
    if ((now / 500) % 2) {  // blinking "recording" dot
      setPx(0, 12, 7);
      setPx(1, 12, 7);
      setPx(0, 11, 7);
      setPx(1, 11, 7);
    }
    drawBar(p);
  } else if (step == STEP_GESTURE_CHECK) {
    drawText("G", 0, 0);
    drawNumberRight(p, 0);
    for (int i = 0; i < 5; i++) {
      setPx(7, 1 + i * 2, i < p ? 7 : 1);
      setPx(6, 1 + i * 2, i < p ? 7 : 0);
    }
  } else if (step == STEP_BUILD) {
    drawSpinner(now);
  } else if (step == STEP_PARTIAL) {
    drawLetter(p, 1);
    drawText("?", 8, 1);
  } else if (step == STEP_SET_CONFIRM) {
    const char* s = "SET?";
    int w = textWidth(s);
    int span = w + COLS;
    int off = (int)((now / 90) % span);
    drawText(s, COLS - off, 1);
  } else if (step == STEP_DIAG) {
    drawText("D", 0, 0);
    drawNumberRight(p, 0);
    drawBar(p * 100 / 7);
  } else if (step == STEP_DIAG_DONE || step == STEP_HOLDOUT_DONE) {
    if (step == STEP_HOLDOUT_DONE || p == 1) {
      drawArt(ICON_CHECK, COLS, 0);
    } else {
      drawArt(ICON_X, ART_W, 1);
    }
  }
}

// =====================================================================
// Transients
// =====================================================================
static void enqueue(TransientKind kind, const char* text, int value, unsigned long durationMs) {
  if (qCount >= QUEUE_LEN) return;  // drop rather than block
  Transient& t = queue[(qHead + qCount) % QUEUE_LEN];
  t.kind = kind;
  strncpy(t.text, text ? text : "", sizeof(t.text) - 1);
  t.text[sizeof(t.text) - 1] = 0;
  t.value = value;
  t.durationMs = durationMs;
  if (kind == TR_SCROLL) t.durationMs = (unsigned long)(textWidth(t.text) + COLS) * 90UL;
  qCount++;
}

static void enqueueText(const char* text, unsigned long ms) {
  if (textWidth(text) <= COLS) {
    enqueue(TR_TEXT, text, 0, ms);
  } else {
    enqueue(TR_SCROLL, text, 0, 0);
  }
}

// Returns true while a transient owns the screen.
static bool drawTransient(unsigned long now) {
  if (qCount == 0) return false;
  if (!transientRunning) {
    transientRunning = true;
    transientStart = now;
  }
  Transient& t = queue[qHead];
  unsigned long el = now - transientStart;
  if (el >= t.durationMs) {
    qHead = (qHead + 1) % QUEUE_LEN;
    qCount--;
    transientRunning = false;
    return drawTransient(now);
  }
  switch (t.kind) {
    case TR_TEXT:
      drawTextCentered(t.text, 1);
      break;
    case TR_SCROLL:
      drawText(t.text, COLS - (int)(el / 90), 1);
      break;
    case TR_FLASH:  // value = number of flashes, 150 ms on / 150 ms off
      if ((el / 150) % 2 == 0) memset(fb, 7, sizeof(fb));
      break;
    case TR_ICON_CHECK:
      drawArt(ICON_CHECK, COLS, 0);
      break;
    case TR_ICON_X:
      drawArt(ICON_X, ART_W, 1);
      break;
    case TR_ICON_BATTERY:
      drawArt(ICON_BATTERY, COLS, 0);
      break;
    case TR_BATTERY_LEVEL:
      if (t.value < 0) {
        drawTextCentered("N/A", 1);
      } else {
        drawArt(ICON_BATTERY, COLS, 0);
        int lit = (t.value * 8 + 50) / 100;  // inside of the icon: columns 2..9
        for (int c = 0; c < 8; c++) {
          for (int r = 2; r <= 5; r++) setPx(r, 2 + c, c < lit ? 7 : 0);
        }
      }
      break;
  }
  return true;
}

// =====================================================================
// Bridge handlers (run on the loop thread via provide_safe)
// =====================================================================
static void setBase(BaseMode m) {
  if (baseMode != m) baseSince = millis();
  baseMode = m;
}

void show_activity(int classId) {
  if (baseMode != BASE_ACTIVITY || baseClass != classId) baseSince = millis();
  baseMode = BASE_ACTIVITY;
  baseClass = classId;
}

void show_model(int modelId, int banner) {
  activeModel = modelId == MODEL_B ? MODEL_B : MODEL_A;
  if (banner) enqueueText(activeModel == MODEL_A ? "1D" : "6D", 1200);
}

void show_status(int code) {
  switch (code) {
    case ST_SEARCHING:       setBase(BASE_SEARCHING); break;
    case ST_LINK_LOST:       setBase(BASE_LINK_LOST); break;
    case ST_UNSURE:          setBase(BASE_UNSURE); break;
    case ST_IDLE:            setBase(BASE_IDLE); break;
    case ST_CFG_ERROR:
      enqueueText("CFG", 1000);
      enqueue(TR_ICON_X, "", 0, 1000);
      break;
    case ST_LEG_MISMATCH:
      enqueueText("LEG", 1000);
      enqueueText("?", 1000);
      break;
    case ST_FLASH3:          enqueue(TR_FLASH, "", 3, 900); break;
    case ST_FLASH1:          enqueue(TR_FLASH, "", 1, 300); break;
    case ST_CHECK:           enqueue(TR_ICON_CHECK, "", 0, 1500); break;
    case ST_LOW_BATT_ON:
      if (!lowBattFlag) lastBattFlash = millis() - 30000UL;  // flash right away
      lowBattFlag = true;
      break;
    case ST_LOW_BATT_OFF:    lowBattFlag = false; break;
    case ST_REDO_WINDOW_ON:  redoWindowFlag = true; break;
    case ST_REDO_WINDOW_OFF: redoWindowFlag = false; break;
    default:
      Monitor.print("show_status: unknown code ");
      Monitor.println(code);
  }
}

void show_setup(int step, int progress) {
  if (baseMode != BASE_SETUP || setupStep != step) baseSince = millis();
  baseMode = BASE_SETUP;
  setupStep = step;
  setupProgress = progress;
}

void show_value(int kind, int value) {
  char buf[12];
  if (kind == VAL_ACC_A || kind == VAL_ACC_B) {
    snprintf(buf, sizeof(buf), "%c %d%%", kind == VAL_ACC_A ? 'A' : 'B', value);
    enqueue(TR_SCROLL, buf, 0, 0);
  } else if (kind == VAL_BATTERY) {
    enqueue(TR_BATTERY_LEVEL, "", value, 2000);
  }
}

int display_version() { return DISPLAY_PROTOCOL_VERSION; }

// =====================================================================
// setup / loop
// =====================================================================
void setup() {
  Monitor.begin(115200);
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness values 0-7
  matrix.clear();
  memset(shown, 0xFF, sizeof(shown));  // force the first draw

  Bridge.begin();
  Bridge.provide_safe("show_activity", show_activity);
  Bridge.provide_safe("show_model", show_model);
  Bridge.provide_safe("show_status", show_status);
  Bridge.provide_safe("show_setup", show_setup);
  Bridge.provide_safe("show_value", show_value);
  Bridge.provide_safe("display_version", display_version);
  Monitor.println("gait display ready");
}

void loop() {
  unsigned long now = millis();

  // Low ankle battery: flash the battery icon for 1 s every 30 s.
  if (lowBattFlag && baseMode != BASE_SETUP && now - lastBattFlash >= 30000UL) {
    lastBattFlash = now;
    enqueue(TR_ICON_BATTERY, "", 0, 1000);
  }

  clearFb();
  if (!drawTransient(now)) {
    switch (baseMode) {
      case BASE_IDLE:      break;
      case BASE_SEARCHING: drawSearching(now); break;
      case BASE_LINK_LOST: drawArt(ICON_X, ART_W, 0); break;
      case BASE_UNSURE:    drawArt(ICON_QUESTION, ART_W, 0); break;
      case BASE_ACTIVITY:  drawActivity(now); break;
      case BASE_SETUP:     drawSetup(now); break;
    }
    if (baseMode != BASE_SETUP) {
      drawModelIndicator();
      drawModelLetter();
      if (redoWindowFlag && (now / 400) % 2) setPx(7, 11, 7);
    }
  }

  // Only push to the matrix when something changed.
  if (memcmp(fb, shown, sizeof(fb)) != 0) {
    memcpy(shown, fb, sizeof(fb));
    matrix.draw(shown);
  }
  delay(15);  // ~60 Hz; Bridge requests are serviced between loop() calls
}
