// Neural-Net Step Counter - MCU sketch (STM32U585, Arduino core on Zephyr).
//
// Two jobs, both non-blocking (no delay() anywhere in loop()):
//
// 1. HIP SAMPLING. A GY-61 (ADXL335, analog 3-axis, +-3 g) is wired to A0/A1/A2
//    and powered from 3.3 V. Every 10 ms exactly (micros() scheduling) each axis
//    is read 4 times and averaged. 10 samples make one packet, sent to Python with
//    Bridge.notify("hip_packet", bytes) - the same mechanism the wired app uses
//    for its IMU packets. Packet, little-endian, packed:
//        uint16 seq; uint32 t0_us; uint8 n; uint16 raw[n][3]   (14-bit ADC counts)
//    t0_us is SENSOR time = (index of the first sample) x 10000 us, so it is
//    built from the sample schedule, not from when the packet happened to leave.
//    Once per second: Bridge.notify("hip_status", late, skipped, adc_errors,
//    adc_bits, samples) for diag.
//
// 2. LED MATRIX (8 x 13). Python says WHAT to show, this sketch knows HOW and
//    owns every animation timer (scrolling, blinking, flashes):
//        show_value(text, display_mode)  base screen: the count or the distance
//        show_status(code)               status screens and transient messages
//        show_setup(step, progress)      setup and confirmation screens
//        show_banner(text)               a one-off message (e.g. "STEP", "RESET")
//        display_version()               handshake
//    All are registered with provide_safe(), so they run on the loop thread
//    between two loop() calls and can never race the renderer.
//
// UNO Q gotcha: do NOT call pinMode() on A0-A2. On this board it reconfigures
// the pin away from the ADC and analogRead() then returns a constant.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <string.h>

// =====================================================================
// Types. They must come BEFORE the first function: the Arduino builder
// inserts auto-generated prototypes for every function above the first
// function definition, and those prototypes use these types.
// =====================================================================
struct Glyph {
  char c;
  uint8_t w;
  uint8_t rows[5];
};

enum BaseMode { BASE_IDLE, BASE_SEARCHING, BASE_LINK_LOST, BASE_UNSURE, BASE_VALUE, BASE_SETUP };

// Transients: short screens that interrupt the base screen, played in order.
enum TransientKind { TR_TEXT, TR_SCROLL, TR_FLASH, TR_ICON_CHECK, TR_ICON_X };

struct Transient {
  TransientKind kind;
  char text[16];
  unsigned long durationMs;
};

// =====================================================================
// Hip sampling
// =====================================================================
#define HIP_PERIOD_US      10000UL  // 100 Hz
#define HIP_OVERSAMPLE     4        // reads per axis per sample, averaged (less noise)
#define HIP_LATE_US        2000UL   // a sample taken > 2 ms after its slot counts as late
#define HIP_MAX_BEHIND     10       // more than 10 samples behind -> skip ahead (a gap Python can see)
#define HIP_SAMPLES_PER_PKT 10
#define ADC_BITS           14

static const pin_size_t HIP_PINS[3] = {A0, A1, A2};   // X, Y, Z

static uint32_t hipNextUs = 0;          // scheduled time of the next sample
static uint32_t hipIndex = 0;           // sample index (including skipped ones)
static uint16_t hipSeq = 0;
static uint16_t hipPending[HIP_SAMPLES_PER_PKT][3];
static int hipNPending = 0;
static uint32_t hipPendingStart = 0;    // hipIndex of hipPending[0]
static uint16_t hipLast[3] = {8192, 8192, 8192};
static uint32_t hipLate = 0, hipSkipped = 0, hipAdcErrors = 0;

static void putLE16(MsgPack::bin_t<uint8_t>& v, uint16_t x) {
  v.push_back(x & 0xFF);
  v.push_back(x >> 8);
}

// Send whatever samples are pending as one packet (n = 1..10; normally 10,
// fewer only right before a skip so every packet holds consecutive samples).
static void hipSendPacket() {
  if (hipNPending == 0) return;
  MsgPack::bin_t<uint8_t> pkt;
  pkt.reserve(7 + 6 * hipNPending);
  putLE16(pkt, hipSeq);
  uint32_t t0 = hipPendingStart * HIP_PERIOD_US;  // wraps at 2^32 like the ankle protocol
  putLE16(pkt, (uint16_t)(t0 & 0xFFFF));
  putLE16(pkt, (uint16_t)(t0 >> 16));
  pkt.push_back((uint8_t)hipNPending);
  for (int s = 0; s < hipNPending; s++) {
    for (int k = 0; k < 3; k++) putLE16(pkt, hipPending[s][k]);
  }
  Bridge.notify("hip_packet", pkt);
  hipSeq++;
  hipNPending = 0;
}

static void hipTakeSample() {
  uint32_t sum[3] = {0, 0, 0};
  // Interleave the axes (X Y Z X Y Z ...) so all three see the same instant on average.
  for (int r = 0; r < HIP_OVERSAMPLE; r++) {
    for (int k = 0; k < 3; k++) {
      int v = analogRead(HIP_PINS[k]);
      if (v < 0) {            // Zephyr returns a negative errno on an ADC error
        hipAdcErrors++;
        v = hipLast[k];
      }
      sum[k] += (uint32_t)v;
    }
  }
  if (hipNPending == 0) hipPendingStart = hipIndex;
  for (int k = 0; k < 3; k++) {
    hipLast[k] = (uint16_t)((sum[k] + HIP_OVERSAMPLE / 2) / HIP_OVERSAMPLE);
    hipPending[hipNPending][k] = hipLast[k];
  }
  hipNPending++;
  hipIndex++;
  if (hipNPending == HIP_SAMPLES_PER_PKT) hipSendPacket();
}

static void hipTask() {
  // next += period (never now + period): the schedule cannot drift, a late
  // sample is followed by an early one. At most 3 catch-up samples per call so
  // the matrix keeps rendering.
  for (int caught = 0; caught < 3; caught++) {
    uint32_t now = micros();
    int32_t behind = (int32_t)(now - hipNextUs);
    if (behind < 0) return;
    if (behind > (int32_t)(HIP_MAX_BEHIND * HIP_PERIOD_US)) {
      // Something blocked us for > 100 ms. Don't fake a burst of samples:
      // skip the missed slots (t0_us jumps while seq continues, which Python
      // reports as a gap, exactly like a FIFO overflow on the ankle node).
      uint32_t skip = (uint32_t)behind / HIP_PERIOD_US;
      hipSendPacket();
      hipIndex += skip;
      hipNextUs += skip * HIP_PERIOD_US;
      hipSkipped += skip;
      continue;
    }
    if ((uint32_t)behind > HIP_LATE_US) hipLate++;
    hipTakeSample();
    hipNextUs += HIP_PERIOD_US;
  }
}

// =====================================================================
// Display protocol. KEEP IN SYNC with python/display.py.
// =====================================================================
#define DISPLAY_PROTOCOL_VERSION 3   // 3 = step counter

// show_status(code)
#define ST_SEARCHING   1   // base: scanning dot (no sensor data yet)
#define ST_LINK_LOST   2   // base: big X (no data from either sensor)
#define ST_UNSURE      3   // base: big ?
#define ST_IDLE        4   // base: blank
#define ST_CFG_ERROR   10  // transient: "CFG" then X
#define ST_LEG_MISMATCH 11 // transient: "LEG" then ?
#define ST_REC         12  // transient: "REC"
#define ST_NN_FAIL     13  // transient: "NN" then X (model missing / failed its check)
#define ST_HIP_FAIL    14  // transient: "HIP" then X (hip sensor implausible)
#define ST_ANKLE_LOST  15  // transient: X for 1 s (ankle link lost, still counting from the hip)
#define ST_FLASH1      16  // transient: one flash (gesture acknowledged)
#define ST_FLASH3      17  // transient: three flashes (step finished)
#define ST_CHECK       18  // transient: check mark

// show_value display modes (column-12 indicator)
#define DM_STEPS 0          // top 3 LEDs of column 12
#define DM_DIST  1          // bottom 3 LEDs

// show_setup(step, progress)
#define STEP_STAND         0   // progress = seconds remaining
#define STEP_FACE_BASE     1   // 1..6 = face +X -X +Y -Y +Z -Z; progress = percent, -1 = waiting
#define STEP_GESTURE_CHECK 20  // progress = gestures detected so far (0..5)
#define STEP_BUILD         21  // spinner
#define STEP_SET_CONFIRM   22  // scrolling "SET?"
#define STEP_RESET_CONFIRM 23  // scrolling "RESET?"

#define ROWS 8
#define COLS 13
#define TEXT_COLS 12         // columns 0-11 for values, column 12 = mode indicator
#define SCROLL_MS 125        // ~8 columns per second

// =====================================================================
// Art ('#' = bright, '+' = dim, '.' = off), copied from the gait apps
// =====================================================================
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
// 3x5 font (M, N, W are wider, '.' is 1 wide). Rows are bit patterns, MSB = left.
// =====================================================================
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
  {'I', 3, {0b111, 0b010, 0b010, 0b010, 0b111}},
  {'K', 3, {0b101, 0b101, 0b110, 0b101, 0b101}},
  {'L', 3, {0b100, 0b100, 0b100, 0b100, 0b111}},
  {'M', 5, {0b10001, 0b11011, 0b10101, 0b10001, 0b10001}},
  {'N', 4, {0b1001, 0b1101, 0b1011, 0b1001, 0b1001}},
  {'O', 3, {0b010, 0b101, 0b101, 0b101, 0b010}},
  {'P', 3, {0b110, 0b101, 0b110, 0b100, 0b100}},
  {'R', 3, {0b110, 0b101, 0b110, 0b101, 0b101}},
  {'S', 3, {0b011, 0b100, 0b010, 0b001, 0b110}},
  {'T', 3, {0b111, 0b010, 0b010, 0b010, 0b010}},
  {'U', 3, {0b101, 0b101, 0b101, 0b101, 0b111}},
  {'V', 3, {0b101, 0b101, 0b101, 0b101, 0b010}},
  {'W', 5, {0b10001, 0b10001, 0b10101, 0b10101, 0b01010}},
  {'X', 3, {0b101, 0b101, 0b010, 0b101, 0b101}},
  {'Y', 3, {0b101, 0b101, 0b010, 0b010, 0b010}},
  {'Z', 3, {0b111, 0b001, 0b010, 0b100, 0b111}},
  {'?', 3, {0b110, 0b001, 0b010, 0b000, 0b010}},
  {'%', 3, {0b101, 0b001, 0b010, 0b100, 0b101}},
  {'/', 3, {0b001, 0b001, 0b010, 0b100, 0b100}},
  {'-', 3, {0b000, 0b000, 0b111, 0b000, 0b000}},
  {'+', 3, {0b000, 0b010, 0b111, 0b010, 0b000}},
  {'.', 1, {0b0, 0b0, 0b0, 0b0, 0b1}},
  {' ', 2, {0, 0, 0, 0, 0}},
};

// =====================================================================
// Display state
// =====================================================================
Arduino_LED_Matrix matrix;

static uint8_t fb[ROWS * COLS];       // frame being composed
static uint8_t shown[ROWS * COLS];    // last frame sent to the matrix

static BaseMode baseMode = BASE_SEARCHING;
static unsigned long baseSince = 0;     // animation / scroll phase reference
static char valueText[16] = "0";
static int displayMode = DM_STEPS;
static int setupStep = 0;
static int setupProgress = 0;

#define QUEUE_LEN 8
static Transient tq[QUEUE_LEN];
static int qHead = 0, qCount = 0;
static unsigned long transientStart = 0;
static bool transientRunning = false;

// =====================================================================
// Drawing helpers (from the gait apps)
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
  if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';
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

// Draw text with its left edge at column x (may be negative while scrolling),
// clipped to columns [0, maxCol).
static void drawTextClip(const char* s, int x, int row0, int maxCol) {
  for (int i = 0; s[i]; i++) {
    const Glyph* g = findGlyph(s[i]);
    for (int r = 0; r < 5; r++) {
      for (int c = 0; c < g->w; c++) {
        int col = x + c;
        if (col < 0 || col >= maxCol) continue;
        if (g->rows[r] & (1 << (g->w - 1 - c))) setPx(row0 + r, col, 7);
      }
    }
    x += g->w + 1;
  }
}

static void drawText(const char* s, int x, int row0) { drawTextClip(s, x, row0, COLS); }

static void drawTextCentered(const char* s, int row0) {
  drawText(s, (COLS - textWidth(s)) / 2, row0);
}

// Text that scrolls in from the right, endlessly, inside columns [0, width).
static void drawScrollingLoop(const char* s, unsigned long el, int width, int row0) {
  int span = textWidth(s) + width;
  int off = (int)((el / SCROLL_MS) % span);
  drawTextClip(s, width - off, row0, width);
}

// Bottom-row progress bar, 0..100 percent of the 13 columns.
static void drawBar(int percent) {
  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;
  int lit = (percent * COLS + 50) / 100;
  for (int c = 0; c < COLS; c++) setPx(7, c, c < lit ? 7 : 1);
}

// Column 12: top 3 LEDs = steps mode, bottom 3 LEDs = distance mode.
static void drawModeIndicator() {
  int r0 = displayMode == DM_STEPS ? 0 : 5;
  for (int r = r0; r < r0 + 3; r++) setPx(r, 12, 7);
}

// =====================================================================
// Base screens
// =====================================================================
static void drawSearching(unsigned long now) {
  // A dot sweeping back and forth across columns 0-11.
  int period = 2 * (TEXT_COLS - 1);
  int pos = (int)((now / 90) % period);
  int col = pos < TEXT_COLS ? pos : period - pos;
  setPx(3, col, 7);
  setPx(4, col, 7);
  int trail = pos < TEXT_COLS ? col - 1 : col + 1;  // faint trail shows the direction
  setPx(3, trail, 2);
  setPx(4, trail, 2);
}

// Up to 3 digits sit still (they fit in columns 0-11); anything wider scrolls.
static void drawValue(unsigned long now) {
  int w = textWidth(valueText);
  if (w <= TEXT_COLS) {
    drawTextClip(valueText, (TEXT_COLS - w) / 2, 1, TEXT_COLS);
  } else {
    drawScrollingLoop(valueText, now - baseSince, TEXT_COLS, 1);
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

static const char* const FACE_NAMES[6] = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};

static void drawSetup(unsigned long now) {
  int step = setupStep;
  int p = setupProgress;
  if (step == STEP_STAND) {
    char buf[6];
    snprintf(buf, sizeof(buf), "%d", p);
    drawTextCentered(buf, 1);
    drawBar(p * 10);                       // 10 s = full, shrinks as it counts down
  } else if (step >= STEP_FACE_BASE && step < STEP_FACE_BASE + 6) {
    drawTextCentered(FACE_NAMES[step - STEP_FACE_BASE], 1);
    if (p < 0) {
      if ((now / 400) % 2) for (int c = 0; c < COLS; c++) setPx(7, c, 2);  // waiting: blink the bar
    } else {
      drawBar(p);
    }
  } else if (step == STEP_GESTURE_CHECK) {
    drawText("G", 0, 0);
    char buf[4];
    snprintf(buf, sizeof(buf), "%d", p);
    drawText(buf, COLS - textWidth(buf), 0);
    for (int i = 0; i < 5; i++) {
      setPx(7, 1 + i * 2, i < p ? 7 : 1);
      setPx(6, 1 + i * 2, i < p ? 7 : 0);
    }
  } else if (step == STEP_BUILD) {
    drawSpinner(now);
  } else if (step == STEP_SET_CONFIRM) {
    drawScrollingLoop("SET?", now - baseSince, COLS, 1);
  } else if (step == STEP_RESET_CONFIRM) {
    drawScrollingLoop("RESET?", now - baseSince, COLS, 1);
  }
}

// =====================================================================
// Transients
// =====================================================================
static void enqueue(TransientKind kind, const char* text, unsigned long durationMs) {
  if (qCount >= QUEUE_LEN) return;  // drop rather than block
  Transient& t = tq[(qHead + qCount) % QUEUE_LEN];
  t.kind = kind;
  strncpy(t.text, text ? text : "", sizeof(t.text) - 1);
  t.text[sizeof(t.text) - 1] = 0;
  t.durationMs = durationMs;
  if (kind == TR_SCROLL) t.durationMs = (unsigned long)(textWidth(t.text) + COLS) * SCROLL_MS;
  qCount++;
}

// Short text is shown still for `ms`; text wider than the matrix scrolls once.
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
  Transient& t = tq[qHead];
  unsigned long el = now - transientStart;
  if (el >= t.durationMs) {
    qHead = (qHead + 1) % QUEUE_LEN;
    qCount--;
    transientRunning = false;
    return drawTransient(now);
  }
  switch (t.kind) {
    case TR_TEXT:       drawTextCentered(t.text, 1); break;
    case TR_SCROLL:     drawText(t.text, COLS - (int)(el / SCROLL_MS), 1); break;
    case TR_FLASH:      if ((el / 150) % 2 == 0) memset(fb, 7, sizeof(fb)); break;  // 150 ms on / off
    case TR_ICON_CHECK: drawArt(ICON_CHECK, COLS, 0); break;
    case TR_ICON_X:     drawArt(ICON_X, 11, 1); break;
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

void show_value(String text, int mode) {
  char buf[sizeof(valueText)];
  text.toCharArray(buf, sizeof(buf));
  // Restart the scroll only when the text really changes width class or mode,
  // so a count ticking up while scrolling does not jump back to the start.
  if (baseMode != BASE_VALUE || mode != displayMode) baseSince = millis();
  baseMode = BASE_VALUE;
  displayMode = mode == DM_DIST ? DM_DIST : DM_STEPS;
  strncpy(valueText, buf, sizeof(valueText) - 1);
  valueText[sizeof(valueText) - 1] = 0;
}

void show_status(int code) {
  switch (code) {
    case ST_SEARCHING:    setBase(BASE_SEARCHING); break;
    case ST_LINK_LOST:    setBase(BASE_LINK_LOST); break;
    case ST_UNSURE:       setBase(BASE_UNSURE); break;
    case ST_IDLE:         setBase(BASE_IDLE); break;
    case ST_CFG_ERROR:    enqueueText("CFG", 1000); enqueue(TR_ICON_X, "", 1000); break;
    case ST_LEG_MISMATCH: enqueueText("LEG", 1000); enqueueText("?", 1000); break;
    case ST_REC:          enqueueText("REC", 1000); break;
    case ST_NN_FAIL:      enqueueText("NN", 1000); enqueue(TR_ICON_X, "", 1000); break;
    case ST_HIP_FAIL:     enqueueText("HIP", 1000); enqueue(TR_ICON_X, "", 1000); break;
    case ST_ANKLE_LOST:   enqueue(TR_ICON_X, "", 1000); break;
    case ST_FLASH1:       enqueue(TR_FLASH, "", 300); break;
    case ST_FLASH3:       enqueue(TR_FLASH, "", 900); break;
    case ST_CHECK:        enqueue(TR_ICON_CHECK, "", 1500); break;
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

void show_banner(String text) {
  char buf[16];
  text.toCharArray(buf, sizeof(buf));
  enqueueText(buf, 1200);
}

int display_version() { return DISPLAY_PROTOCOL_VERSION; }

static void renderMatrix(unsigned long now) {
  clearFb();
  if (!drawTransient(now)) {
    switch (baseMode) {
      case BASE_IDLE:      break;
      case BASE_SEARCHING: drawSearching(now); break;
      case BASE_LINK_LOST: drawArt(ICON_X, 11, 0); break;
      case BASE_UNSURE:    drawArt(ICON_QUESTION, 11, 0); break;
      case BASE_VALUE:     drawValue(now); break;
      case BASE_SETUP:     drawSetup(now); break;
    }
    if (baseMode == BASE_VALUE) drawModeIndicator();
  }
  if (memcmp(fb, shown, sizeof(fb)) != 0) {  // only push when something changed
    memcpy(shown, fb, sizeof(fb));
    matrix.draw(shown);
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
  matrix.setGrayscaleBits(3);          // brightness values 0-7
  matrix.clear();
  memset(shown, 0xFF, sizeof(shown));  // force the first draw
  enqueueText("S", 1000);              // flash "S" = the step counter app is running

  analogReadResolution(ADC_BITS);      // NO pinMode() on A0-A2 (see the top of this file)

  Bridge.begin();
  Bridge.provide_safe("show_value", show_value);
  Bridge.provide_safe("show_status", show_status);
  Bridge.provide_safe("show_setup", show_setup);
  Bridge.provide_safe("show_banner", show_banner);
  Bridge.provide_safe("display_version", display_version);
  Monitor.println("step counter ready");
  hipNextUs = micros();
}

void loop() {
  hipTask();
  unsigned long now = millis();
  if (now - lastStatusMs >= 1000) {
    lastStatusMs = now;
    Bridge.notify("hip_status", (int)hipLate, (int)hipSkipped, (int)hipAdcErrors, (int)ADC_BITS,
                  (int)(hipIndex & 0x7FFFFFFF));
  }
  hipTask();
  if (now - lastFrameMs >= 20) {  // ~50 Hz redraw; Bridge requests run between loop() calls
    lastFrameMs = now;
    renderMatrix(now);
  }
}
