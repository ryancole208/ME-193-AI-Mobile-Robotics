# Finger-Count Safe

An overhead USB webcam (DICCIK) watches a pad. A from-scratch PyTorch CNN counts
the fingers (0–5). Push the button, then hold up three different numbers (1–5);
the right code (`config.SECRET_CODE = (3, 1, 4)`) unlocks the safe. The laptop
runs a live web demo; the UNO Q app is in `ArduinoApps/finger-safe/`.

## Workflow (run from `Midterm2_Safe/` with `..\my_env\Scripts\python.exe`)

| Step | Command | Output |
|---|---|---|
| Live demo | `python app.py` → <http://127.0.0.1:5000/> | count, confidence, lock demo (key **B** = button) |
| Record data | <http://127.0.0.1:5000/collect> | keys **0–5** pick the count, **Space** records a burst into `data/raw/<class>/<session>/` |
| Check data | `python -m ml.data_report` | counts per class × session, `reports/sheet_<class>.jpg` |
| Train | `python -m ml.train` | `models/finger_cnn.pt`, `models/train_meta.json`, `reports/<run_id>/` |
| Export | `python -m ml.export` | `models/finger_cnn.onnx`, `models/model_config.json`, ONNX parity check + speed |
| UNO Q (camera on laptop) | `python stream_to_unoq.py` | sends frames over USB-C to the `finger-safe` app on the UNO Q; shows its answer |
| Standalone check | `python infer_standalone.py --dir data/raw --session val1` | accuracy using only cv2 + numpy + onnxruntime |

Sessions whose name starts with `val` are validation; everything else trains.
Training defaults: color input 96×96, 40 epochs, ±25° rotation, horizontal flip,
white-balance jitter (`--color-jitter 0.3`, for different lighting).

The model input is **1×3×96×96 color in OpenCV BGR order**, not the spec's
1×1×96×96 grayscale: in grayscale, finger shadows look like fingers (≈83% vs 98%).
`infer_standalone.py` is the reference preprocessing for the UNO Q; never swap to RGB.

The sections below describe the camera/web-viewer layer (Stage 1).

```
Midterm2_Safe/
├── requirements.txt
├── list_cameras.py      # probes camera indexes and reports what works
├── app.py               # Flask server and entry point
├── camera.py            # threaded camera reader (one shared instance)
├── processing.py        # frame-processing hook (pass-through for now)
├── templates/index.html
└── static/style.css, app.js
```

## Setup

Requires Python 3.10+. The project uses the repo's shared virtual environment,
`my_env`, in the repository root (one level above this folder). Run these from
the repository root, then `cd Midterm2_Safe`.

**Windows (PowerShell)**
```powershell
python -m venv my_env          # skip if my_env already exists
my_env\Scripts\Activate.ps1
pip install -r Midterm2_Safe\requirements.txt
cd Midterm2_Safe
```

**macOS / Linux**
```bash
python3 -m venv my_env         # skip if my_env already exists
source my_env/bin/activate
pip install -r Midterm2_Safe/requirements.txt
cd Midterm2_Safe
```

The pins match what's already in `my_env`, so installing won't upgrade
packages that other projects (e.g. torch) depend on.

## Find the right camera

```bash
python list_cameras.py
```

This tries indexes 0–5, asks each camera for 1920×1080, and prints a table:

```
Index  Opens   Frame   Resolution  Note
----------------------------------------------------
0      yes     yes     1920x1080   <- likely DICCIK (Full HD)
1      yes     yes     1920x1080   <- likely DICCIK (Full HD)
2      no      -       -
```

The DICCIK usually reports 1920×1080. If the built-in camera also does (as on
the development laptop), try each index in the app and see which view you get.
Indexes can change after a reboot or re-plug (the DICCIK has been both 1 and 0),
so the app picks the camera by its Windows name instead: `config.CAMERA_NAME =
"icspring camera"` (the DICCIK). `python camera_names.py` prints each index's name.

## Run

```bash
python app.py
```

Then open <http://127.0.0.1:5000>. Press Ctrl+C to stop; the camera is released.

| Option | Default | Meaning |
|---|---|---|
| `--camera-name` | `icspring camera` | Camera name (or part of it) from `camera_names.py` |
| `--camera` | none | Fixed index; overrides `--camera-name` |
| `--width` / `--height` | 1280 / 720 | Requested capture resolution |
| `--port` | 5000 | Web server port |
| `--host` | 127.0.0.1 | Use `0.0.0.0` to view from a phone on the same Wi-Fi |

### Endpoints

- `GET /` — the webpage
- `GET /video_feed` — MJPEG stream (`multipart/x-mixed-replace`) of processed frames
- `GET /status` — JSON, for example:
  ```json
  {"connected": true, "camera_index": 0, "camera_name": "icspring camera",
   "resolution": {"width": 1280, "height": 720}, "fps": 29.8, "info": {}}
  ```
  `info` is whatever `processing.process_frame` returns. The page's Detection
  panel lists its keys automatically.

### How it works

One `Camera` thread reads frames continuously and keeps only the newest one, so
the stream never lags. `process_frame` runs once per frame in that thread, so a
neural net added later runs once per frame no matter how many tabs are open. If
the camera disappears, the thread logs it and retries every 3 s. The page polls
`/status` every second, shows an overlay while the camera is down, and reopens
the stream once it's back.

## Troubleshooting

- **macOS:** the Terminal (or your IDE) needs camera access:
  System Settings → Privacy & Security → Camera → enable it, then restart the Terminal.
- **Windows:** only one app can use a webcam at a time. Close Zoom, Teams, the
  Camera app, OBS, or browser tabs using the camera.
- **Linux:** if the camera opens with permission errors, add yourself to the
  `video` group and log out and back in:
  `sudo usermod -aG video $USER`. `v4l2-ctl --list-devices` shows connected cameras.
- **Low FPS:** webcams lower their frame rate in dim light (auto-exposure gets
  longer). More light gives up to 30 FPS. The app requests MJPG, because
  uncompressed video only manages about 5 FPS at 720p over USB 2.0.
- **Black or wrong camera:** run `list_cameras.py` again and try the other index.
- **"Camera N not available" repeating:** the index is wrong, or another app has the camera.
