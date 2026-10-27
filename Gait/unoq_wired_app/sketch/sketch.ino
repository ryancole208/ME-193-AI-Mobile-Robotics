// Gait Activity Classifier, WIRED test version - MCU sketch (STM32, Arduino core on Zephyr).
//
// Three jobs, all non-blocking (no delay() in loop()):
//   1. Read the MPU-6050 FIFO over I2C about every 100 ms.
//   2. Pack the samples into the SAME packet format as the Bluetooth app's ankle
//      node and send them to Python with Bridge.notify("imu_packet", bytes).
//      Once per second also notify "imu_status" (sensor state + I2C error counts).
//   3. Draw the 8x13 LED matrix (animations, status and setup screens) from small
//      commands Python sends (show_activity / show_model / show_status / show_setup /
//      show_value / display_version), registered with provide_safe so they run on
//      the loop thread and never race the renderer.
//
// Wiring (3.3 V logic, GY-521 style breakout):
//   VCC -> 3.3V   GND -> GND   SDA -> SDA (D20 / PB11)   SCL -> SCL (D21 / PB10)
// Those header pins are `Wire` (I2C2) on the UNO Q. The Qwiic connector is `Wire1`
// (I2C4) and A4/A5 are `Wire2` (I2C3). The LED matrix uses PF0-PF10, so there is
// no conflict with any of them.
//
// Packet (little-endian, packed):
//   uint16 seq; uint32 t0_us; uint8 n; int16 raw[n][6]  (ax, ay, az, gx, gy, gz counts)
// t0_us is SENSOR time: (samples so far) x 10000 us. After a FIFO overflow the
// lost samples are added to that count, so t0_us jumps while seq does not.
//
// The frame art is stored as ASCII strings: '#' = full brightness, '+' = dim, '.' = off.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>
#include <string.h>

// =====================================================================
// IMU configuration - change these if you wire it differently
// =====================================================================
#define IMU_WIRE        Wire     // header SDA/SCL. Use Wire1 for the Qwiic connector, Wire2 for A4/A5
#define IMU_BUS_HZ      100000   // 100 kHz: reliable on a 0.5-1 m cable. 400000 only with short wires
#define MPU_ADDR        0x68     // AD0 low or unconnected (0x69 if AD0 is tied high)
#define READ_PERIOD_MS  100      // drain the FIFO this often (holds ~0.85 s at 100 Hz)

// MPU-6000/MPU-6050 Register Map rev 4.0 (RM-MPU-6000A-00)
#define REG_SMPLRT_DIV    0x19   // Sample Rate = 1 kHz (DLPF on) / (1 + 9) = 100 Hz
#define REG_CONFIG        0x1A   // DLPF_CFG[2:0] = 3 -> accel 44 Hz, gyro 42 Hz bandwidth
#define REG_GYRO_CONFIG   0x1B   // FS_SEL[4:3] = 3 -> +-2000 dps (16.4 LSB/dps)
#define REG_ACCEL_CONFIG  0x1C   // AFS_SEL[4:3] = 3 -> +-16 g (2048 LSB/g)
#define REG_FIFO_EN       0x23   // XG|YG|ZG|ACCEL = 0x78 -> 12 bytes per sample
#define REG_INT_STATUS    0x3A   // bit 4 FIFO_OFLOW_INT, cleared by reading
#define REG_USER_CTRL     0x6A   // bit 6 FIFO_EN, bit 2 FIFO_RESET (only while FIFO_EN = 0)
#define REG_PWR_MGMT_1    0x6B   // bit 7 DEVICE_RESET; CLKSEL = 1 (PLL, X gyro), SLEEP = 0
#define REG_FIFO_COUNTH   0x72   // FIFO_COUNT, read high byte first
#define REG_FIFO_R_W      0x74
#define REG_WHO_AM_I      0x75   // reads 0x68

#define VAL_SMPLRT_DIV    9
#define VAL_CONFIG        0x03
#define VAL_GYRO_CONFIG   0x18
#define VAL_ACCEL_CONFIG  0x18
#define VAL_FIFO_EN       0x78
#define BIT_FIFO_OFLOW    0x10
#define BIT_USER_FIFO_EN  0x40
#define BIT_FIFO_RESET    0x04
#define FIFO_SIZE         1024
#define SAMPLE_BYTES      12
#define SAMPLES_PER_PKT   10
#define SAMPLE_PERIOD_US  10000UL

// imu_status state codes (python/gait/imu_link.py)
#define IMU_BUS_ERROR  0
#define IMU_NO_DEVICE  1
#define IMU_WRONG_ID   2
#define IMU_OK         3

// =====================================================================
// Display protocol. KEEP IN SYNC with python/gait/constants.py.
// =====================================================================
#define DISPLAY_PROTOCOL_VERSION 2   // 2 = wired app

#define CLASS_STILL   0
#define CLASS_WALKING 1
#define CLASS_RUNNING 2
#define CLASS_BIKING  3

#define MODEL_A 0  // "1D": max(a_y)
#define MODEL_B 1  // "6D": full IMU

// show_status(code)
#define ST_NO_DATA         2   // base: "IMU" then "X" (no IMU samples for > 2 s)
#define ST_UNSURE          3   // base: big ? (unsure / not enough data)
#define ST_CFG_SETUP       4   // transient: "CFG" then "?" (no usable config_wired.json)
#define ST_LEG_MISMATCH    5   // transient: "LEG" then "?"
#define ST_FLASH3          6   // transient: whole matrix flashes 3 times
#define ST_CHECK           7   // transient: check mark
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
#define STEP_DIAG           34  // progress = check number being run (1..5)
#define STEP_DIAG_DONE      35  // progress = 1 all passed, 0 something failed
#define STEP_HOLDOUT_DONE   36  // held-out recording finished

// show_value(kind, value)
#define VAL_ACC_A   1  // value = percent; scrolls "A 87%"
#define VAL_ACC_B   2  // value = percent; scrolls "B 91%"

#define ROWS 8
#define COLS 13
#define ART_W 11

// =====================================================================
// Art. Activity frames are 11 columns x 8 rows (columns 0-10).
// =====================================================================

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

// =====================================================================
// 3x5 font (W and M are wider). Each row is a bit pattern, MSB = left.
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
  {'I', 3, {0b111, 0b010, 0b010, 0b010, 0b111}},
  {'L', 3, {0b100, 0b100, 0b100, 0b100, 0b111}},
  {'M', 5, {0b10001, 0b11011, 0b10101, 0b10001, 0b10001}},
  {'R', 3, {0b110, 0b101, 0b110, 0b101, 0b101}},
  {'S', 3, {0b011, 0b100, 0b010, 0b001, 0b110}},
  {'T', 3, {0b111, 0b010, 0b010, 0b010, 0b010}},
  {'U', 3, {0b101, 0b101, 0b101, 0b101, 0b111}},
  {'W', 5, {0b10001, 0b10001, 0b10101, 0b10101, 0b01010}},
  {'X', 3, {0b101, 0b101, 0b010, 0b101, 0b101}},
  {'?', 3, {0b110, 0b001, 0b010, 0b000, 0b010}},
  {'%', 3, {0b101, 0b001, 0b010, 0b100, 0b101}},
  {' ', 2, {0, 0, 0, 0, 0}},
};

static const char CLASS_LETTERS[4] = {'S', 'W', 'R', 'B'};

// =====================================================================
// Display state
// =====================================================================
Arduino_LED_Matrix matrix;

static uint8_t fb[ROWS * COLS];       // frame being composed
static uint8_t shown[ROWS * COLS];    // last frame sent to the matrix

enum BaseMode { BASE_IDLE, BASE_NO_DATA, BASE_UNSURE, BASE_ACTIVITY, BASE_SETUP };

static BaseMode baseMode = BASE_NO_DATA;
static int baseClass = CLASS_STILL;
static unsigned long baseSince = 0;     // animation phase reference
static int setupStep = 0;
static int setupProgress = 0;
static int activeModel = MODEL_A;
static bool redoWindowFlag = false;

// Transients: short screens that interrupt the base screen, played in order.
enum TransientKind { TR_TEXT, TR_SCROLL, TR_FLASH, TR_ICON_CHECK, TR_ICON_X };

struct Transient {
  TransientKind kind;
  char text[12];
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

// "IMU" for 1 s, then a big X for 1 s, repeating.
static void drawNoData(unsigned long now) {
  if (((now - baseSince) / 1000) % 2 == 0) {
    drawTextCentered("IMU", 1);
  } else {
    drawArt(ICON_X, ART_W, 1);
  }
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
    int span = textWidth(s) + COLS;
    int off = (int)((now / 90) % span);
    drawText(s, COLS - off, 1);
  } else if (step == STEP_DIAG) {
    drawText("D", 0, 0);
    drawNumberRight(p, 0);
    drawBar(p * 100 / 5);
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
static void enqueue(TransientKind kind, const char* text, unsigned long durationMs) {
  if (qCount >= QUEUE_LEN) return;  // drop rather than block
  Transient& t = queue[(qHead + qCount) % QUEUE_LEN];
  t.kind = kind;
  strncpy(t.text, text ? text : "", sizeof(t.text) - 1);
  t.text[sizeof(t.text) - 1] = 0;
  t.durationMs = durationMs;
  if (kind == TR_SCROLL) t.durationMs = (unsigned long)(textWidth(t.text) + COLS) * 90UL;
  qCount++;
}

static void enqueueText(const char* text, unsigned long ms) {
  if (textWidth(text) <= COLS) {
    enqueue(TR_TEXT, text, ms);
  } else {
    enqueue(TR_SCROLL, text, 0);
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
    case TR_TEXT:       drawTextCentered(t.text, 1); break;
    case TR_SCROLL:     drawText(t.text, COLS - (int)(el / 90), 1); break;
    case TR_FLASH:      if ((el / 150) % 2 == 0) memset(fb, 7, sizeof(fb)); break;  // 150 ms on / off
    case TR_ICON_CHECK: drawArt(ICON_CHECK, COLS, 0); break;
    case TR_ICON_X:     drawArt(ICON_X, ART_W, 1); break;
  }
  return true;
}

// =====================================================================
// Bridge display handlers (run on the loop thread via provide_safe)
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
    case ST_NO_DATA:         setBase(BASE_NO_DATA); break;
    case ST_UNSURE:          setBase(BASE_UNSURE); break;
    case ST_IDLE:            setBase(BASE_IDLE); break;
    case ST_CFG_SETUP:
      enqueueText("CFG", 1000);
      enqueueText("?", 1000);
      break;
    case ST_LEG_MISMATCH:
      enqueueText("LEG", 1000);
      enqueueText("?", 1000);
      break;
    case ST_FLASH3:          enqueue(TR_FLASH, "", 900); break;
    case ST_FLASH1:          enqueue(TR_FLASH, "", 300); break;
    case ST_CHECK:           enqueue(TR_ICON_CHECK, "", 1500); break;
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
    enqueue(TR_SCROLL, buf, 0);
  }
}

int display_version() { return DISPLAY_PROTOCOL_VERSION; }

static void renderMatrix(unsigned long now) {
  clearFb();
  if (!drawTransient(now)) {
    switch (baseMode) {
      case BASE_IDLE:     break;
      case BASE_NO_DATA:  drawNoData(now); break;
      case BASE_UNSURE:   drawArt(ICON_QUESTION, ART_W, 0); break;
      case BASE_ACTIVITY: drawActivity(now); break;
      case BASE_SETUP:    drawSetup(now); break;
    }
    if (baseMode != BASE_SETUP) {
      if (baseMode != BASE_NO_DATA) drawModelIndicator();
      if (redoWindowFlag && (now / 400) % 2) setPx(7, 11, 7);
    }
  }
  if (memcmp(fb, shown, sizeof(fb)) != 0) {  // only push when something changed
    memcpy(shown, fb, sizeof(fb));
    matrix.draw(shown);
  }
}

// =====================================================================
// MPU-6050
// =====================================================================
static int imuState = IMU_NO_DEVICE;
static uint8_t whoAmI = 0;
static unsigned long i2cErrors = 0, reinits = 0, fifoOverflows = 0;
static bool imuStreaming = false;
static unsigned long resetAtMs = 0;         // DEVICE_RESET issued, configure after 100 ms
static bool resetPending = false;
static unsigned long nextInitMs = 0;        // retry time after a failure
static unsigned long lastReadMs = 0;
static unsigned long lastDrainUs = 0;       // micros() when the FIFO was last emptied
static bool everStreamed = false;

static uint32_t sampleCount = 0;            // sensor samples so far (incl. estimated lost ones)
static uint16_t seq = 0;
static int16_t pending[SAMPLES_PER_PKT][6];
static int nPending = 0;
static uint32_t pendingStart = 0;           // sampleCount of pending[0]

static bool writeReg(uint8_t reg, uint8_t val) {
  IMU_WIRE.beginTransmission(MPU_ADDR);
  IMU_WIRE.write(reg);
  IMU_WIRE.write(val);
  return IMU_WIRE.endTransmission() == 0;
}

static bool readRegs(uint8_t reg, uint8_t* buf, size_t len) {
  IMU_WIRE.beginTransmission(MPU_ADDR);
  IMU_WIRE.write(reg);
  if (IMU_WIRE.endTransmission(false) != 0) return false;
  if (IMU_WIRE.requestFrom((uint8_t)MPU_ADDR, len) != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)IMU_WIRE.read();
  return true;
}

static void putLE16(MsgPack::bin_t<uint8_t>& v, uint16_t x) {
  v.push_back(x & 0xFF);
  v.push_back(x >> 8);
}

// Send whatever samples are pending as one packet (n = 1..10).
static void sendPacket() {
  if (nPending == 0) return;
  MsgPack::bin_t<uint8_t> pkt;
  pkt.reserve(7 + SAMPLE_BYTES * nPending);
  putLE16(pkt, seq);
  uint32_t t0 = pendingStart * SAMPLE_PERIOD_US;  // wraps at 2^32 like the protocol says
  putLE16(pkt, (uint16_t)(t0 & 0xFFFF));
  putLE16(pkt, (uint16_t)(t0 >> 16));
  pkt.push_back((uint8_t)nPending);
  for (int s = 0; s < nPending; s++) {
    for (int k = 0; k < 6; k++) putLE16(pkt, (uint16_t)pending[s][k]);
  }
  Bridge.notify("imu_packet", pkt);
  seq++;
  nPending = 0;
}

static void addSample(const uint8_t* b) {
  if (nPending == 0) pendingStart = sampleCount;
  // FIFO order (by register number): ACCEL_X/Y/Z then GYRO_X/Y/Z, each high byte first
  for (int k = 0; k < 6; k++) pending[nPending][k] = (int16_t)((b[2 * k] << 8) | b[2 * k + 1]);
  nPending++;
  sampleCount++;
  if (nPending == SAMPLES_PER_PKT) sendPacket();
}

// Samples generated since the FIFO was last emptied are gone: count them so
// t0_us jumps (Python sees the gap from t0_us while seq stays continuous).
static void accountLostSamples() {
  sendPacket();  // flush what we have before the jump
  unsigned long now = micros();
  sampleCount += (uint32_t)((now - lastDrainUs + SAMPLE_PERIOD_US / 2) / SAMPLE_PERIOD_US);
  lastDrainUs = now;
}

static void imuFail() {
  i2cErrors++;
  if (imuStreaming) {
    sendPacket();
    imuStreaming = false;
  }
  if (imuState == IMU_OK || imuState == IMU_WRONG_ID) imuState = IMU_NO_DEVICE;
  resetPending = false;
  nextInitMs = millis() + 1000;  // retry once a second, without rebooting
}

// Step 1 of (re)initialisation: check WHO_AM_I and issue DEVICE_RESET.
static void imuBeginInit() {
  if (!readRegs(REG_WHO_AM_I, &whoAmI, 1)) {
    whoAmI = 0;
    imuState = IMU_NO_DEVICE;
    i2cErrors++;
    nextInitMs = millis() + 1000;
    return;
  }
  if (!writeReg(REG_PWR_MGMT_1, 0x80)) { imuFail(); return; }
  resetPending = true;
  resetAtMs = millis();
}

// Step 2, 100 ms after the reset: configure, verify, start the FIFO.
static void imuFinishInit() {
  resetPending = false;
  struct { uint8_t reg, val; } cfg[] = {
    {REG_PWR_MGMT_1, 0x01},  // wake, PLL with X-gyro reference
    {REG_SMPLRT_DIV, VAL_SMPLRT_DIV},
    {REG_CONFIG, VAL_CONFIG},
    {REG_GYRO_CONFIG, VAL_GYRO_CONFIG},
    {REG_ACCEL_CONFIG, VAL_ACCEL_CONFIG},
  };
  for (auto& c : cfg) {
    uint8_t back = 0;
    if (!writeReg(c.reg, c.val) || !readRegs(c.reg, &back, 1) || back != c.val) { imuFail(); return; }
  }
  uint8_t st;
  if (!writeReg(REG_USER_CTRL, 0x00) ||                 // FIFO off ...
      !writeReg(REG_USER_CTRL, BIT_FIFO_RESET) ||       // ... reset it (only allowed while off)
      !writeReg(REG_FIFO_EN, VAL_FIFO_EN) ||
      !writeReg(REG_USER_CTRL, BIT_USER_FIFO_EN) ||
      !readRegs(REG_INT_STATUS, &st, 1)) {              // clears a stale overflow flag
    imuFail();
    return;
  }
  if (everStreamed) {
    reinits++;
    accountLostSamples();  // the time spent re-initialising is a gap
  } else {
    lastDrainUs = micros();
  }
  everStreamed = true;
  imuStreaming = true;
  imuState = whoAmI == 0x68 ? IMU_OK : IMU_WRONG_ID;
  Monitor.print("MPU-6050 streaming, WHO_AM_I 0x");
  Monitor.println(whoAmI, HEX);
}

static void resetFifo() {
  if (!writeReg(REG_USER_CTRL, 0x00) || !writeReg(REG_USER_CTRL, BIT_FIFO_RESET) ||
      !writeReg(REG_USER_CTRL, BIT_USER_FIFO_EN)) {
    imuFail();
    return;
  }
  accountLostSamples();
}

// Empty the FIFO: called every READ_PERIOD_MS while streaming.
static void imuDrain() {
  uint8_t b[2];
  if (!readRegs(REG_INT_STATUS, b, 1)) { imuFail(); return; }
  bool overflow = b[0] & BIT_FIFO_OFLOW;
  if (!readRegs(REG_FIFO_COUNTH, b, 2)) { imuFail(); return; }
  unsigned count = ((unsigned)b[0] << 8) | b[1];
  // An overflow (or a count that is not whole samples, i.e. misaligned data)
  // means the oldest bytes were overwritten: drop the FIFO and count the gap.
  if (overflow || count >= FIFO_SIZE - SAMPLE_BYTES || count % SAMPLE_BYTES) {
    fifoOverflows++;
    resetFifo();
    return;
  }
  unsigned n = count / SAMPLE_BYTES;
  uint8_t buf[SAMPLE_BYTES * SAMPLES_PER_PKT];
  while (n > 0) {
    unsigned chunk = n < SAMPLES_PER_PKT ? n : SAMPLES_PER_PKT;  // <= 120 bytes (Wire buffer is 256)
    if (!readRegs(REG_FIFO_R_W, buf, chunk * SAMPLE_BYTES)) { imuFail(); return; }
    for (unsigned s = 0; s < chunk; s++) addSample(buf + s * SAMPLE_BYTES);
    n -= chunk;
  }
  lastDrainUs = micros();
}

static void imuTask(unsigned long now) {
  if (imuStreaming) {
    if (now - lastReadMs >= READ_PERIOD_MS) {
      lastReadMs = now;
      imuDrain();
    }
  } else if (resetPending) {
    if (now - resetAtMs >= 100) imuFinishInit();
  } else if ((long)(now - nextInitMs) >= 0) {
    imuBeginInit();
  }
}

// =====================================================================
// setup / loop
// =====================================================================
static unsigned long lastStatusMs = 0;
static unsigned long lastFrameMs = 0;

void setup() {
  Monitor.begin(115200);
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness values 0-7
  matrix.clear();
  memset(shown, 0xFF, sizeof(shown));  // force the first draw
  enqueueText("W", 1000);              // "W" = the wired app is running

  IMU_WIRE.begin();
  IMU_WIRE.setClock(IMU_BUS_HZ);       // the board's devicetree default is 400 kHz

  Bridge.begin();
  Bridge.provide_safe("show_activity", show_activity);
  Bridge.provide_safe("show_model", show_model);
  Bridge.provide_safe("show_status", show_status);
  Bridge.provide_safe("show_setup", show_setup);
  Bridge.provide_safe("show_value", show_value);
  Bridge.provide_safe("display_version", display_version);
  Monitor.println("gait wired ready");
}

void loop() {
  unsigned long now = millis();
  imuTask(now);
  if (now - lastStatusMs >= 1000) {
    lastStatusMs = now;
    Bridge.notify("imu_status", imuState, (int)whoAmI, (int)i2cErrors, (int)reinits, (int)fifoOverflows,
                  (int)IMU_BUS_HZ);
  }
  if (now - lastFrameMs >= 15) {  // ~60 Hz redraw; Bridge requests run between loop() calls
    lastFrameMs = now;
    renderMatrix(now);
  }
}
