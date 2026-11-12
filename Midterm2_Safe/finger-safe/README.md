# 🔐 Finger Safe

The UNO Q version of the Finger-Count Safe (`Midterm2_Safe/` in the course repo).
Push the button, then hold up three different numbers of fingers (1–5). The
right code opens the servo latches. The model runs on the UNO Q.

**Camera source** (`SOURCE` in `main.py`, default `"laptop"`):
- `laptop`: the DICCIK is on the laptop; the UNO Q stays on the laptop's USB-C
  port. Run `..\my_env\Scripts\python.exe stream_to_unoq.py` in `Midterm2_Safe/`
  (close `app.py` first). It does `adb forward tcp:8770 tcp:8770`, sends raw
  224×224 letterboxed frames, and shows the board's answer (B = button, Q = quit).
- `usb`: the camera plugs into the board through a powered USB-C hub.

```
python/main.py              camera -> ONNX model -> smoothing -> lock logic -> Bridge
python/infer_standalone.py  copy of Midterm2_Safe/infer_standalone.py (cv2 + numpy + onnxruntime only)
python/model/               finger_cnn.onnx + model_config.json (run r20261111-092822)
sketch/sketch.ino           button -> Python; LED matrix; 2 servo latches
```

## Wiring (check `sketch.ino` before the first run)

| Part | Pin | Setting in `sketch.ino` |
|---|---|---|
| Push button | D2 to GND (internal pull-up). Not wired yet: `AUTO_WAKE = True` in `main.py` | `BUTTON_PIN` |
| Servo 1 signal | D9 | `SERVO_PINS` |
| Servo 2 signal | D10 | `SERVO_PINS` |
| Servo power | external 5 V supply; its GND joined to a board GND | — |
| Latch angles | 0° locked, 90° open (each servo), open for 8 s | `SERVO_LOCKED_DEG`, `SERVO_OPEN_DEG`, `OPEN_MS` |
| Camera | USB via a powered USB-C hub (found automatically) | `CAMERA_INDEX` env var to force one |

Both servos open and close together. They are driven with plain 50 Hz pulses (no
Servo library). Pulses stop 1 s after each move so they don't buzz; the latch
has to hold by friction. `/dev/video0` and `/dev/video1` on the UNO Q are the
Qualcomm video encoder, not a camera; `main.py` skips them.

## LED matrix

| State | Matrix |
|---|---|
| Asleep | one dim pixel blinking in the corner |
| Awake | top row: 3 code slots (bright = entered). Below: one bar per finger seen; bars grow while you hold |
| Unlocked | check mark (latch opens) |
| Wrong code | X |
| No model / no camera | blinking "!" |

## Adding the model (only after it has been verified on the laptop)

From the course repo:

```powershell
mkdir ArduinoApps\finger-safe\python\model
copy Midterm2_Safe\models\finger_cnn.onnx  ArduinoApps\finger-safe\python\model\
copy Midterm2_Safe\models\model_config.json ArduinoApps\finger-safe\python\model\
cd ArduinoApps
python tools\deploy.py finger-safe
```

Without the model the app still starts, shows "!" and checks again every 5 s.
The secret code comes from `model_config.json` (3, 1, 4); set `SECRET_CODE` in
`main.py` to override it.

## Laptop dry run (no board)

With the model in `python/model/`, `python main.py` from `python/` runs the same
loop on the laptop (Bridge calls are printed). In the default `SOURCE="laptop"` mode it
listens on port 8770, so `stream_to_unoq.py`-style clients can connect to localhost.

## Status on the board (2026-11-11)

- Deployed over USB. `deploy.py` needs `adb` on PATH; App Lab's copy is at
  `%LOCALAPPDATA%\Arduino15\packages\arduino\tools\adb\32.0.0\adb.exe`.
- Sketch compiles and flashes. `onnxruntime` 1.31.1 installs; OpenCV and NumPy
  come with the board image (listing `opencv-python-headless` breaks the install).
- The model loads on the board. Laptop stream over USB-C: 120 val frames,
  119 right (99.2%); 22 fps round trip; the model takes 27 ms/frame on the board.
- **Not yet tested:** the live camera through the stream, the servos and the
  button, and `SOURCE="usb"` with a camera on the board.
