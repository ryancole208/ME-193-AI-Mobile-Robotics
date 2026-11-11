# Handoff Log — Finger-Count Safe (formerly Gesture Safe)

## Current State
**Project:** Finger-Count Safe (replaced the pinch/V direction-gesture plan on 2026-11-10; assignment deadline about 2026-11-11 evening). Show a 3-number code (1–5 fingers, three different numbers) under the overhead webcam; the safe unlocks or stays locked. The CNN is from scratch in PyTorch and returns the number of fingers.

**Stage (2026-11-11, Session 14):** Retrained on old + new-lighting data (run **r20261111-092822**, val 97.9%, gate PASS). Step 7 done (ONNX export, parity PASS, `model_config.json`, `infer_standalone.py`). Step 8 done (README, requirements). The UNO Q app is written at `ArduinoApps/finger-safe/` but **NOT deployed and the model is NOT in it**: the user wants to verify the new model live on the laptop first.
**NEXT (after Session 16):** camera stays on the LAPTOP for now. Run `stream_to_unoq.py` (close app.py first) and test live counts + the code on the board; then wire the servos (D9/D10, external 5 V, shared GND) and later the button (D2, then AUTO_WAKE=False). See Sessions 15–16.

**How to run (laptop):** from `Midterm2_Safe/`:
- App: `..\my_env\Scripts\python.exe app.py` → <http://127.0.0.1:5000/> (live count + lock demo; Push button or key **B**; code `config.SECRET_CODE = (3,1,4)`) and <http://127.0.0.1:5000/collect> (keys **0–5** pick the count, **Space** starts a burst).
- Data check: `..\my_env\Scripts\python.exe -m ml.data_report`
- Train: `..\my_env\Scripts\python.exe -m ml.train` (defaults reproduce the shipped recipe: color, 40 epochs, ±25° rotation, H-flip, `--color-jitter 0.3`). Each run overwrites `models/finger_cnn.pt` + `models/train_meta.json`; every run's weights are also copied to `reports/<run_id>/finger_cnn.pt`. **After training, re-run `python -m ml.export`.**
- Export: `..\my_env\Scripts\python.exe -m ml.export`. Standalone check: `..\my_env\Scripts\python.exe infer_standalone.py --dir data/raw --session val1`.

### The shipped model (do not lose these facts)
- Files: `models/finger_cnn.pt` (state_dict), `models/train_meta.json`, `models/finger_cnn.onnx` (830 KB; input `input` 1×3×96×96, output `logits`, opset 17), `models/model_config.json`. Run **r20261111-092822** (also in `reports/r20261111-092822/`). The previous model r20261110-230127 is kept in `reports/`.
- Architecture: `ml/model.py` `FingerCNN(n_classes=6, in_ch=3)`: 4 blocks of 2×(Conv3×3 no-bias → BN → ReLU) + MaxPool2, widths 16/32/64/96, AdaptiveAvgPool(1), Flatten, Dropout 0.3, Linear(96→6). **211,510 params.**
- **Input: 1×3×96×96, COLOR in OpenCV BGR order** (no RGB swap), NOT the spec's 1×1 grayscale (grayscale topped out ~83%: shadows look like fingers).
- Preprocessing (`preprocess.py` = `infer_standalone.py`, verified identical): raw 1280×720 BGR frame → crop x 0–0.98, y 0–0.96 → **letterbox** to 224×224 (INTER_AREA, centered, fill 0) → resize 96×96 INTER_AREA → /255 → `(x - mean)/std` with ONE scalar: **mean = 0.19157075881958008, std = 0.24253405630588531** (changed from the old model's 0.0966/0.1123 because the new data is brighter; always read it from `model_config.json`) → CHW → batch dim.
- Output: logits for `["none","one","two","three","four","five"]` → counts 0–5. Softmax for confidence.
- Results (train1+train2 → val1, 1920 val images, half old lighting, half new): **val acc 97.9%, macro-F1 0.979; recall none 100 / one 97.8 / two 99.7 / three 96.6 / four 93.4 / five 100.** Most confused: four→five (21, all in old-light burst b20261110-225152). Some new-light "four" frames are borderline 4 vs 3 (e.g. 53%/46%), so a "four" may need a steady, spread hand to pass the 0.80 threshold.
- Speed: ONNX Runtime on the laptop CPU, model only: 5.0 ms (1 thread), 3.2 ms (default threads). ONNX vs PyTorch on 50 val images: max |diff| 2.4e-6, argmax 50/50.
- Live logic unchanged: `SMOOTH_FRAMES=5`, `CONF_THRESHOLD=0.80`, `HOLD_SECONDS=1.0`, `RESULT_SECONDS=4`, `AWAKE_TIMEOUT_SECONDS=30`.

### Data
- `data/raw/<class>/<session>/<burst_id>_<nnn>.png`, 224×224 letterboxed BGR. Bursts are 80 frames.
- Old lighting = bursts `b20261110-*` (evening). New lighting = `b20261111-*` (morning of 2026-11-11). The user added new-light bursts to **train1** (1–3 per class), a new **train2** session (4 per class, all new light) and **val1** (2 per class). Totals: train none 688 / one 1040 / two 1120 / three 1040 / four 960 / five 880; val1 320 per class (2 old + 2 new bursts).
- The new-light **val1** bursts (09:14–09:18) have a strong red/orange cast (the pad looks tan) that no training burst has (the new-light train bursts still look green). Without color jitter the model got both new-light val "four" bursts 0/80 (→ three); white-balance jitter fixed it.

### Files
- `gestures.py`: `CLASSES`, `CLASS_TO_COUNT`, `COUNT_TO_CLASS`, `CODE_COUNTS=[1..5]`, display names, how-to text. Single source of truth.
- `config.py`: camera name, crop + letterbox, `RAW_SAVE_SIZE=224`, `INPUT_SIZE=96`, **`GRAYSCALE=False`**, burst settings, `TARGET_PER_CLASS=250`, `VAL_SESSION_PREFIX="val"`, `SMOOTH_FRAMES`, `HOLD_SECONDS`, `CONF_THRESHOLD`, `SECRET_CODE=(3,1,4)`, `RESULT_SECONDS`, `AWAKE_TIMEOUT_SECONDS`.
- `preprocess.py`: `crop_box`, `crop_rect`, `letterbox`, `frame_to_raw` (→ 224 saved form), `raw_to_unit` (→ 96, /255; color unless GRAYSCALE), `normalize` (→ CHW), `preprocess_frame(frame, mean, std)` → (1,3,96,96) float32 NumPy, `settings()`. OpenCV + NumPy only (imports `config`).
- `lock.py`: `LockDemo`, `validate_code`. Pure Python (imports `config`, `gestures`). Reusable on the UNO Q.
- `inference.py`: `FingerCounter` (PyTorch; checks model meta against config), `predict_count(frame) -> (count, confidence, probabilities)`, `LiveWorker` thread (newest raw frame → model → smoothing → `LockDemo`).
- `app.py` (Flask): `/`, `/video_feed`, `/status`, `/live`, `/lock/press`, `/lock/reset`, `/collect` + collect routes. Starts `LiveWorker` only if `models/finger_cnn.pt` exists. Templates in `templates/`, JS/CSS in `static/`. `processing.py` is a pass-through hook.
- `camera.py` + `camera_names.py`: opens the DICCIK **by name** on every (re)open; never falls back to the laptop camera.
- `collector.py`: burst capture + collect overlay.
- `ml/data_report.py`, `ml/dataset.py` (`split_sessions`, `load`, GPU `augment(rot_deg, vflip)`), `ml/model.py`, `ml/train.py` (options `--epochs --batch --lr --weight-decay --seed --rot --vflip --gray --input-size --cpu`).
- `reports/`: `data_report.png`, `sheet_<class>.jpg`, one folder per training run (metrics.json, confusion.png, curves.png, misclassified.png, weights).
- `archive_gestures/`: the whole old gesture project (data, model, reports, code). **Never delete.**
- `ml/export.py` (`python -m ml.export`): ONNX + `model_config.json` + parity check + benchmark.
- `infer_standalone.py`: cv2 + numpy + onnxruntime only; `FingerModel`, `Smoother`, `LockLogic` (port of `lock.py`, verified identical), `open_camera`; CLI `--image/--dir/--camera`. A copy lives in `ArduinoApps/finger-safe/python/` (keep the two identical).
- `ArduinoApps/finger-safe/` (in the separate `ArduinoApps/` git repo): the UNO Q app, see Session 14.

### Environment
- Windows 11, Python 3.14.7, shared venv `my_env` in the repo root (`..\my_env\Scripts\python.exe`). Never create a per-project venv; don't upgrade shared packages.
- In `my_env`: torch 2.11.0+cu128 (RTX 5070 Laptop GPU), torchvision 0.26.0+cu128, opencv-python 5.0.0.93, numpy 2.5.3, flask 3.1.3, matplotlib 3.11.1.
- onnx 1.23.2 and onnxruntime 1.31.1 installed in `my_env` (Session 14). `requirements.txt` pins all of these.

### Camera
- DICCIK Windows name **"icspring camera"**; its index is NOT stable (was 1, then 0). 1280×720 MJPG, ~20–30 fps. Overhead, pointing down at a green pad; fingers enter from the top edge (the user's side).

**Other files in the folder, not part of the safe:** `V_code.py`, `annacode-v.tsv`, `relu_fit.png` (a separate class exercise).

## Next Session Plan (Step 7 → UNO Q handoff)
The spec rules still apply: the laptop stage has no servo/arm code; keep UNO Q-side dependencies to OpenCV + NumPy + onnxruntime.
1. **Read this file, summarize it to the user, wait for OK** (handoff protocol).
2. **Install** `onnx` and `onnxruntime` into `my_env` (pre-approved). Python 3.14 wheels may be missing; if so, tell the user and look for a compatible version before anything drastic.
3. **Export script** `ml/export.py` (`python -m ml.export`): load `models/finger_cnn.pt` into `FingerCNN(in_ch=3)`, eval mode, `torch.onnx.export` to **`models/finger_cnn.onnx`** with a FIXED input **1×3×96×96** (input name `input`, output `logits`, no dynamic axes, opset ~17; use the legacy exporter `dynamo=False` if the new one has trouble). Note in the README and here that the spec said 1×1×96×96 and why it is 3 channels.
4. **Parity check**: 50 val images through `preprocess` → PyTorch vs onnxruntime CPU; **max abs diff < 1e-4** and argmax agreement. **Benchmark** onnxruntime CPU ms/frame (1 thread and default).
5. **Keep** `models/finger_cnn.pt` as the fallback (already there).
6. **Write `models/model_config.json`**: class order, class→count mapping, input shape [1,3,96,96], channel order "BGR", grayscale false, crop rectangle, letterbox (size 224, fill 0, centered, INTER_AREA), input resize 96 INTER_AREA, scale /255, mean/std (the scalars above), smoothing frames, hold seconds, confidence threshold, secret code, result/awake timeouts, run ID, val accuracy + per-class recall, ONNX file name, parity/benchmark numbers.
7. **`infer_standalone.py`** (the UNO Q will copy this): imports ONLY `cv2`, `numpy`, `onnxruntime` and the standard library. No Flask, torch, `config.py`, `preprocess.py` or `ml/`. Re-implement the preprocessing from `model_config.json` (crop → letterbox → resize → /255 → normalize → CHW), open the camera (`--camera` index, default 0; there is no name lookup on Linux), print the finger count + confidence per frame (and the 5-frame smoothed count), with an `--image` option for a saved frame. **Verify on the laptop that it gives the same tensor as `preprocess.preprocess_frame` (max diff ~0) and the same counts as `inference.predict_count`** on the same frames/PNGs. Optionally port the lock logic (copy `lock.py` without the config import).
8. Step 8: update `README.md` (collect, train, live demo, export, standalone commands) and `requirements.txt`.
9. Update this file: Current State + a new Session Log entry with the parity and benchmark numbers.
10. **UNO Q stage (only when the user asks):** copy `models/finger_cnn.onnx`, `models/model_config.json` and `infer_standalone.py` to the board. Read the user's auto-memory **no-live-device-tests** first: check ports, and never send test commands that could reach the user's UNO Q without permission. The UNO Q is an Arm Cortex-A53 Linux board: check that onnxruntime installs there (aarch64 wheel) before relying on it; the board may need `opencv-python-headless`.

## Known Issues
- **Color input** deviates from the spec (grayscale) and makes the ONNX input 1×3×96×96 instead of 1×1×96×96. Keep BGR order everywhere; an RGB swap on the UNO Q would silently hurt accuracy.
- **Lighting:** the old model failed in the new (morning) light. The new model trains on both, plus white-balance jitter. If another light fails, record a train session there, retrain, and re-export.
- **UNO Q unknowns (untested; the user said not to touch the board):** camera access from the App Lab Python container, aarch64 wheels for onnxruntime/opencv-python-headless, sketch compile (no compiler on the laptop), speed on the board, and the button/servo pins (defaults D2/D9).
- "one" is the weakest class (91.2% recall, confused with "two"), often frames with a strong finger shadow.
- Letterboxing makes the pad only ~53 of 96 input rows tall (fingers ~4–5 px wide). 128×128 was tried and was no better.
- Camera driver settings (zoom, focus, exposure) and the mount must not change; the crop (`CROP_*`) is baked into saved PNGs and the model. Record any change here.
- On DirectShow, FOURCC MJPG must be set **after** width/height (handled in `camera.py`). Manual exposure set via OpenCV persists in the driver; restore with `CAP_PROP_AUTO_EXPOSURE=0.75`.
- No formal live test results are recorded (per-count positions/hands, correct and wrong code); the user said they like the model. Ask if they want to record one.
- The whole `Midterm2_Safe/` folder is untracked in git; nothing has been committed. Ask the user before committing anything, especially `data/` (thousands of PNGs) and `archive_gestures/`.

## Session Log

### Session 1 — 2026-11-05 — Stage 1: Webcam viewer (reconstructed)
*Written in Session 2 from the code and earlier notes; Stage 1 had no handoff log.*
**Goal:** Stream the DICCIK webcam to a local webpage as the base for later stages.
**Actions taken:** Created `app.py`, `camera.py`, `processing.py`, `list_cameras.py`, `templates/index.html`, `static/app.js`, `static/style.css`, `requirements.txt`, `README.md`. Later added a temporary `CAP_PROP_SETTINGS` call in `camera.py` to inspect driver settings (zoom).
**Decisions and why:** A single camera thread with newest-frame-only buffering, so slow clients never lag and the future NN runs once per frame no matter how many tabs are open. DirectShow on Windows (faster and more reliable than MSMF). MJPG for 30 fps over USB 2.0.
**Test results:** The user confirmed the stream works with the DICCIK at index 1.
**Hardware and settings discovered:** DICCIK = index 1; MJPG-after-size quirk; exposure persistence (see Known Issues).
**Unfinished or next steps:** Stage 2 (gesture CNN).

### Session 2 — 2026-11-10 — Stage 2: Gesture neural network
**Goal:** Begin Stage 2. Follow the handoff protocol, verify Stage 1, then build `gestures.py`, `config.py`, `preprocess.py`, and the `/collect` page.
**Actions taken:**
- Created `handoff.md` (it didn't exist).
**Decisions and why:** (in progress)
**Test results:** (in progress)
**Hardware and settings discovered:** `onnx` and `onnxruntime` are not in `my_env`.
**Unfinished or next steps:** Waiting for the user's OK on the plan.

### Session 3 — 2026-11-10 — Stage 2: core modules + /collect page
**Goal:** Steps 0–3 of the Stage 2 build order: handoff, check Stage 1, write `gestures.py`, `config.py`, `preprocess.py`, build `/collect`, then stop for data session 1.
**Actions taken:**
1. Read `handoff.md` and summarized it to the user; got OK and answers.
2. Created `gestures.py`, `config.py`, `preprocess.py`, `collector.py`, `templates/collect.html`, `static/collect.js`.
3. Changed `camera.py` (raw-frame copy, `get_raw_frame`), `app.py` (collect routes, overlay stream, defaults from config), `templates/index.html` (title, nav), `static/style.css` (collect styles).
4. Tested with a fake camera and the Flask test client (scratchpad script, not in repo): a 2 s × 8 fps burst saved 16 PNGs at 224×224 in 1.9 s; counts, undo, double-undo rejection and bad class/session rejection all OK; saved-PNG and live-frame preprocessing identical; overlay image checked visually.
**Decisions and why:**
- Burst capture runs on the server from the camera thread, not via browser requests, so timing is steady and frames are full-res and overlay-free.
- The crop is set as fractions of the frame (resolution-independent). Default is the centered full-height square, because the pad fills the frame and the hand enters from the top.
- Live preprocessing goes through the same 224 resize as the saved PNGs (frame → 224 → 96), so live and training pixels match exactly instead of 720 → 96 vs 720 → 224 → 96.
- `preprocess_frame` returns NumPy, not a torch tensor, so the UNO Q can use it with onnxruntime and no torch.
- The overlay is drawn server-side only on `/video_feed?overlay=collect`, so the main page and saved data stay clean.
- Undo only touches bursts from the current server run (safer than guessing from files on disk).
- Added keyboard shortcuts (1–8, Space), since one hand is under the camera.
**Test results:** See action 4. The real camera was not opened this session; the user confirmed it works at index 1.
**Hardware and settings discovered:** Forward = DOWN in the image (user screenshot). The camera sees a green pad filling the 16:9 frame; two fingers span about half the frame height.
**Unfinished or next steps:**
- The user records data session 1 using the protocol given in chat (also in the README later). Then: confirm the "your left" hint is on the correct side, run Step 2 (`ml/data_report.py`, not written yet), then Steps 4–5.
- The user's decision on the `CAP_PROP_SETTINGS` line.


### Session 4 — 2026-11-10 — Camera selection fix
**Goal:** The app was showing the laptop camera. Make it use the overhead DICCIK, and remove the camera-settings dialog.
**Actions taken:**
1. Grabbed a frame from each index: index 0 = green pad (DICCIK), index 1 = laptop camera facing the user. Windows had swapped the order.
2. Added `camera_names.py` (lists DirectShow device names with ctypes, no new packages; `find_camera_index(name)`).
3. `camera.py`: `Camera(..., name=...)` resolves the index by name on every open (so reconnects survive reordering); missing name → warning + retry, never another camera. Removed the TEMPORARY `CAP_PROP_SETTINGS` line (the user's decision). `/status` now also reports `camera_name`.
4. `config.py`: `CAMERA_INDEX` replaced by `CAMERA_NAME = "icspring camera"`. `app.py`: `--camera-name` (default from config), `--camera` index overrides. `static/app.js` shows "index (name)". README and `list_cameras.py` updated.
**Decisions and why:** Select by name because indexes are unstable; no fallback so wrong-camera images can't enter the dataset.
**Test results:** `camera_names.py` → `0 icspring camera`, `1 Integrated Webcam`. `Camera(name="icspring camera")` opened index 0, 1280×720, ~20 fps, frame shows the green pad. Bad name → not connected, no fallback. Full Flask app not run.
**Hardware and settings discovered:** DICCIK's Windows name is "icspring camera"; index order can change. In the test frame the pad covered only the bottom ~2/3 of the image (dark band on top), unlike the earlier "pad fills the frame" note. Check the framing before recording.
**Unfinished or next steps:** Same as Session 3: the user records data session 1, confirms the "your left" hint side, then Step 2 (`ml/data_report.py`).

### Session 5 — 2026-11-10 — Pad-shaped crop + left hint fix
**Goal:** The user asked for (1) the collect crop to match the green rectangle instead of a square, for room to move the fingers, and (2) the "your left" hint is on their right.
**Actions taken:**
1. Took a frame from the user's running app (`/video_feed`) and found the pad by colour: it now fills the frame except ~3% at the bottom and ~2% at the right (the camera was re-framed since Session 4).
2. `config.py`: square crop (`CROP_CENTER_*`, `CROP_SIDE`) replaced by `CROP_X0/Y0/X1/Y1 = 0, 0, 0.98, 0.96`; `RAW_SAVE_SIZE=(320,176)`, `INPUT_SIZE=(160,88)`; `IMAGE_MIRRORED=True`.
3. `preprocess.py`: `crop_box` returns (x0, y0, w, h); `crop_square` → `crop_rect`; sizes are (w, h); `settings()` records the rectangle.
4. `collector.py` overlay: rectangle outline/dimming; the forward arrow goes on the edge opposite the "your left" hint so they don't overlap.
**Decisions and why:** Keep the pad's aspect (~1.82) in saved images and the model input, rather than squashing it to a square, so fingers keep their shape. 160×88 is about the same pixel count as the old 96×96 ×1.5; the CNN (not written yet) takes any fixed size.
**Test results:** crop box (0,0,1254,691); saved PNG 176×320; model input (1,1,88,160); saved-PNG vs live preprocessing max diff 0.0; overlay checked visually (hint left, arrow right, outline on the pad). Running app NOT restarted, so the change only takes effect after the user restarts `app.py`.
**Hardware and settings discovered:** Image is mirrored relative to the user's left/right. Pad position as above.
**Unfinished or next steps:** The user restarts the app and records data session 1. The crop is now frozen. Then Step 2 (`ml/data_report.py`).

### Session 6 — 2026-11-10 — Data check + training code
**Goal:** Data session 1 is recorded; check it (Step 2) and write the training code (Steps 4–5).
**Actions taken:**
1. `ml/data_report.py` (`python -m ml.data_report`): counts per class/burst/session, flags unreadable/wrong-size/too dark/too bright images and short bursts, writes `reports/sheet_<class>.jpg` contact sheets (one row per burst).
2. `ml/dataset.py`: `split_bursts` (per class ~20% of BURSTS to val, seed 0, never split by frame), `load` (through `preprocess.raw_to_unit`), `augment` on GPU (±10° rotation, ±10% zoom, ±8% shift, brightness ±0.07, contrast ±30%, gamma, noise; **no flips**, since direction is the label).
3. `ml/model.py`: `GestureCNN`, 4 blocks of 2×(conv3-BN-ReLU)+maxpool (16/32/48/64), global avg pool, dropout 0.3, linear 8. 116,504 params; ONNX-friendly ops only.
4. `ml/train.py` (`python -m ml.train`): AdamW + OneCycle, 60 epochs, batch 64, label smoothing 0.05, mean/std from train set, keeps best val epoch → `models/gesture_cnn.pt`, `models/model_config.json` (classes, preprocess settings, mean/std, input shape, val bursts), `reports/training_curves.png`, `reports/confusion_val.png`.
**Data session 1 (`20261110_2138`):** 400–464 per class, 5–8 bursts per class; bursts are 48 or 80 frames (the 80-frame ones are 10 s bursts; config says 6 s; the user may have changed it in the running app). Report: 0 problems, mean gray 45–60 (dim). Contact sheets checked by eye: directions are diagonal, "left" = image left, consistent with Session 5. **Mislabels found:** `no_hand/.../b20261110-214705` is a fist/knuckles (belongs in `other`); `other/.../b20261110-215611` and `b20261110-215539` contain some empty-pad frames; the end of `pinch_left/.../b20261110-214928` rotates toward forward.
**Test results:** model forward OK (2,8); augmentation viewed; 2-epoch smoke run on the RTX 5070: val 63% (expected for 2 epochs); smoke outputs deleted.
**Unfinished or next steps:** The user decides on the mislabeled bursts, then runs `..\my_env\Scripts\python.exe -m ml.train`. Then live inference on the page (Detection panel) and ONNX export (Step 9).

### Session 7 — 2026-11-10 — First training run
**Goal:** Review the user's first full training run.
**Actions taken:** The user moved `no_hand/b20261110-214705` (fist, 48 files) to `other`, then ran `python -m ml.train` (60 epochs). Changed `ml/train.py` to save the FINAL epoch instead of the best-val epoch (the user has not re-run it yet).
**Test results (run 1):** final epoch val 93.0%, train 99.0%; the best-val epoch was 11 (95.5%), a spike while val was still swinging 70–95%, and that is the one saved in `models/gesture_cnn.pt`. Val = 9 bursts, 560 images (about 1 burst per class), so each class's recall is basically one burst. Errors at epoch 11: pinch_forward → other 15/80 (burst 215134), other → v_right 9/80 (burst 215627, 3–4 fingers).
**Decisions and why:** Save the last epoch: with ~1 val burst per class, "best val" picks noise; OneCycle ends at low LR, so the last epoch is stable. The val score is optimistic (same session, same lighting).
**Unfinished or next steps:** The user re-runs `ml.train`. Then live inference on the main page to test with the real camera; collect a 2nd session if live accuracy is poor.

### Session 8 — 2026-11-10 — Switch to finger counting (build-order Steps 0–2)
**Goal:** New plan from the user: count fingers 0–5 instead of pinch/V directions. Steps 0–2: handoff, check what exists, switch classes + collect page, then stop for data.
**Actions taken:**
1. Read `handoff.md`, summarized it, asked two questions. The user answered: archive yes; crop = **letterbox** (keep the whole pad, pad to a square), not the center square.
2. Moved `data/raw/*` (3456 PNGs, count checked before/after), `models/`, `reports/` into `archive_gestures/`, and copied the gesture-era `gestures.py`, `config.py`, `preprocess.py`, `collector.py`, `ml/` into `archive_gestures/code/`. Nothing deleted.
3. Rewrote `gestures.py` (6 count classes + mappings), `config.py` (letterbox, 224/96, `TARGET_PER_CLASS=250`, `VAL_SESSION_PREFIX`; removed `FORWARD_DIRECTION`/`IMAGE_MIRRORED`), `preprocess.py` (`letterbox`).
4. `collector.py`: overlay without arrows. `app.py`: page config sends count mapping + val prefix; titles renamed. `collect.html`/`collect.js`: keys 0–5, session default `train1` with train/val guidance. `index.html` title.
**Decisions and why:** Letterbox per the user (room to move, square input for the ±180° rotation the spec asks for). Validation chosen by session-ID prefix `val`, so the split is by session as the spec requires. Kept the pad crop rectangle from Session 5.
**Test results:** letterbox of a real frame → 224×224 with 101 black band rows; `preprocess_frame` → (1,1,96,96). Flask test client with a fake camera: `/collect` renders with the new classes; a 1 s burst of `three/train1` saved 8 PNGs at 224×224; counts and undo OK; old class names rejected (400).
**Hardware and settings discovered:** none new.
**Unfinished or next steps:** The user records `train1`, `train2`, `val1` (protocol in Current State). Then Step 3: rewrite `ml/data_report.py` (counts per class × session, `reports/data_report.png` grid, flag mislabels) and show the user before training.

### Session 9 — 2026-11-10 — Data check (Step 3) + training code (Step 4)
**Goal:** Check the user's recording, then prepare training.
**Actions taken:**
1. The user recorded only session `train1` (no `train2`, no `val1`). Bursts were 80 frames (10 s, set on the page; config says 6 s). Counts: none 320 (4 bursts), one 480 (6), two/three/four/five 400 (5 each).
2. Rewrote `ml/data_report.py` for the count classes and ran it: 0 image problems; 1 problem = no validation session. Checked every burst sheet by eye: all bursts show the labelled count; `none` has empty pad, fists, and an arm. Fingers mostly enter from the top edge, pointing down, one hand.
3. Rewrote `ml/dataset.py` (session split, spec augmentation), `ml/model.py` (`FingerCNN`), `ml/train.py` (spec training + reports + gate check).
**Test results:** `FingerCNN` 211,222 params, 3.3 ms/frame on CPU (1 thread). Augmentation viewed (rotation/flip/black fill OK). Smoke run (3 epochs, fake val = last burst per class from train1, outputs redirected to the scratchpad, nothing written to `models/`/`reports/`): all report files produced; accuracy meaningless at 3 epochs. No real training run yet.
**Hardware and settings discovered:** none new.
**Unfinished or next steps:** The user records `val1` (2 bursts per class, different lighting if possible). Then `python -m ml.data_report` (confirm val counts), then `python -m ml.train` and check the gate (acc ≥ 90%, every recall ≥ 85%). If it fails, diagnose and ask for targeted data before changing the model.

### Session 10 — 2026-11-10 — First training runs (Steps 4–5), gate failed
**Goal:** Train on train1, validate on val1, pass the gate.
**Actions taken:** The user recorded `val1` (2 bursts × 80 frames per class = 160/class; 5 very dark `none` frames = arm against the lens, kept). The user ran `python -m ml.train` (25 epochs); I looked at misclassified.png / curves.png / burst sheets, then ran `python -m ml.train --epochs 60` to test underfitting.
**Test results:**
| Run | Epochs | Train imgs/class (none/1/2/3/4/5) | Val acc | Macro-F1 | Recall none/1/2/3/4/5 | Most confused | Gate |
|---|---|---|---|---|---|---|---|
| r20261110-225403 | 25 (best ep 12) | 320/480/400/400/400/400 | 78.3% | 0.773 | 100/78.8/73.8/28.7/91.9/96.9 | three→two (106) | FAIL |
| r20261110-225523 | 60 (best ep 12) | same | 82.0% | 0.819 | 100/81.9/53.8/65.0/91.2/100 | two→one (54) | FAIL |
Val = 160/class from val1. Run 2: train acc (augmented) 96% vs final val 77% → generalization gap, not underfitting.
**Decisions and why:** Diagnosis: (1) val1 "three" bursts often have fingers close/touching while train1 "three" is spread; (2) 1/2 errors are low-confidence frames with heavy shadows; (3) all train1 bursts come in from the top with one hand, little variety; (4) letterboxed fingers are only ~4–5 px wide at 96×96. Per the user's rule, asked for targeted data (1, 2, 3) before making the model or input bigger. 4/5/none are fine.
**Unfinished or next steps:** The user records `train2`: 3 bursts each of one/two/three (varied fingers, directions, left hand, some less-spread), optionally 1–2 of four/five. Retrain with `--epochs 40`. If 2/3 still fail: consider a larger input (e.g. 128) to give fingers more pixels — ask the user first.

### Session 11 — 2026-11-10 — Pass the accuracy gate without new data
**Goal:** The user wants a working model now with the existing data (train1 already includes left-hand bursts; more data with other lighting later).
**Actions taken:** Added `ml/train.py` options `--rot`, `--vflip`, `--input-size`, `--gray`, and `augment(rot_deg, vflip)`; `load` handles color. Ran 4 experiments (40 epochs each), then made the passing recipe the default: `config.GRAYSCALE=False`, `--rot 25`, no vertical flip, 40 epochs. Copied run r20261110-230127's weights to `models/finger_cnn.pt` and its metrics (minus history) to `models/train_meta.json` (the 128 run had been written last).
**Test results:** (train1 → val1; train 320/480/400/400/400/400, val 160/class)
| Run | Input | Aug | Val acc | Macro-F1 | Recall none/1/2/3/4/5 | Most confused | Gate |
|---|---|---|---|---|---|---|---|
| r20261110-225851 | gray 96 | ±25°, H-flip | 82.8% | 0.829 | 100/79.4/66.2/68.1/83.8/99.4 | two→three (34) | FAIL |
| r20261110-225942 | gray 128 | ±25°, H-flip | 83.4% | 0.838 | 98.8/73.8/63.1/68.1/96.9/100 | two→one (54) | FAIL |
| **r20261110-230127** | **color 96** | ±25°, H-flip | **97.9%** | 0.979 | 100/91.2/100/96.2/100/100 | one→two (14) | **PASS** |
| r20261110-230227 | color 128 | ±25°, H-flip | 97.2% | 0.972 | 100/86.9/100/96.2/100/100 | one→two (21) | PASS |
**Decisions and why:** Color was the fix: in grayscale, shadows of fingers read as extra fingers (the 1↔2 and 2↔3 errors). Kept 96×96 (spec size; 128 was no better). Light rotation because the hand always enters from the user's side. The ONNX input will be 1×3×96×96, not 1×1×96×96 (spec), because of color.
**Unfinished or next steps:** Step 6: live prediction + lock demo on the main page, then the user tests live.

### Session 12 — 2026-11-10 — Step 6: live inference + lock demo
**Goal:** Live finger count on the main page and the on-screen lock demo.
**Actions taken:** Created `lock.py`, `inference.py`. Changed `config.py` (`SECRET_CODE=(3,1,4)`, `RESULT_SECONDS=4`, `AWAKE_TIMEOUT_SECONDS=30`), `app.py` (LiveWorker startup, `/live`, `/lock/press`, `/lock/reset`), `templates/index.html` (Detection panel replaced by Finger count + Safe panels), `static/app.js` (polls `/live` every 125 ms; key B = push button), `static/style.css`.
**Decisions and why:** Inference in its own thread reading the camera's raw frame, so the MJPEG stream never waits. Lock logic is a separate pure module so it can be unit-tested and reused on the UNO Q. A digit only counts while awake, with the same count (1–5) on top at ≥ 0.80 smoothed confidence for 1 s; a repeat of the previous digit shows "Change your number" (even after the hand leaves in between). Auto-sleep after 30 s idle and 4 s after the result.
**Test results:** Lock simulation: 3-1-4 → unlocked; 3-1-5 → locked; repeat refused; interrupted hold not counted; low confidence not counted; back to asleep after the result; codes (3,3,4), (0,1,2), (1,2) rejected. Predictor on val1 PNGs via the inference path: 97.9% (matches training). Empty-pad camera frame → 0 (98.4%). `predict_count` 7.3 ms/frame (preprocess + model, CPU 2 threads). Flask test client + LiveWorker with a fake camera: `/live` ready, 11.8 ms, 25 fps; press → awake; reset → asleep; page renders; app without a model → `model_loaded: false`. **Not yet tested with the real camera.**
**Unfinished or next steps:** The user restarts `app.py` and tests live (record per-count results and the two code attempts here). Then Step 7: install onnx/onnxruntime (approved), export `models/finger_cnn.onnx` (input 1×3×96×96 because color), parity check on 50 images, benchmark, `models/model_config.json`, `infer_standalone.py`. Then README (Step 8).

### Session 13 — 2026-11-10 — End of chat: handoff for export / UNO Q
**Goal:** The user is happy with the model and is ending this chat; record everything the next session needs for Step 7 (ONNX export, `model_config.json`, `infer_standalone.py`) and the UNO Q.
**Actions taken:** Rewrote "Current State" (shipped-model facts including the exact preprocessing and mean/std, data, files, environment) and "Known Issues", and added "Next Session Plan". No code changes in this entry.
**Decisions and why:** Ship run r20261110-230127 as is; no retraining unless the user asks. The standalone script must re-implement preprocessing from `model_config.json` so the UNO Q needs no project code.
**Test results:** none new (see Sessions 11–12: val 97.9%, gate PASS; inference path verified at 97.9%; lock logic simulated).
**Hardware and settings discovered:** none new.
**Unfinished or next steps:** "Next Session Plan" steps 1–9; copying to the UNO Q (step 10) only when the user asks.

### Session 14 — 2026-11-11 — Retrain with new lighting, export, UNO Q app (not deployed)
**Goal:** The shipped model failed in new lighting; the user recorded more data. Train on old + new data, and build the UNO Q app. The user said: do NOT upload the model to the UNO Q yet (verify it first).
**User decisions:** camera plugs into the UNO Q; outputs = servo latch + LED matrix; wake = physical button; do nothing on the board this session.
**Actions taken:**
1. Fixed `ml/data_report.py` (crashed on color tiles). Data report: no new problems (the same 5 dark `none` val frames).
2. `python -m ml.train` (old default recipe) → r20261111-092211: 90.5%, FAIL. Per-burst check: both new-light val "four" bursts 0/80 → three. Cause: the new-light val bursts have a red/orange cast that is not in training.
3. Added `--color-jitter` (`ml/dataset.augment`: per-channel gain e^±cj, then saturation 1±cj) and made 0.3 the default → **r20261111-092822: 97.9%, PASS** (shipped).
4. Installed onnx/onnxruntime. Wrote `ml/export.py` and `infer_standalone.py`. Updated README + requirements (Step 8).
5. Wrote `ArduinoApps/finger-safe/` (app.yaml, python/main.py, python/infer_standalone.py copy, python/requirements.txt, sketch/sketch.ino + sketch.yaml, README). No `python/model/` folder yet.
**Test results (val1 = 1920 images, half old light, half new):**
| Run | Aug | Val acc | Recall none/1/2/3/4/5 | Gate |
|---|---|---|---|---|
| r20261110-230127 (old model) | — | 80.1% | 100/89.1/50.0/91.6/50.0/100 | FAIL |
| r20261111-092211 | old default | 90.5% | 100/97.8/99.4/98.4/48.1/99.1 | FAIL |
| **r20261111-092822** | + color jitter 0.3 | **97.9%** | 100/97.8/99.7/96.6/93.4/100 | **PASS** |
ONNX parity: max |diff| 2.4e-6, argmax 50/50. `infer_standalone` vs `preprocess.preprocess_frame`: tensor diff 0.0 on 45 frames (random + upscaled PNGs); counts 45/45 equal to `inference.predict_count`; `LockLogic` = `lock.LockDemo` over 200 random sequences; standalone on val1 = 97.9%. `main.py` dry run with a fake camera: 3-1-4 → mode 2 (UNLOCKED), 3-1-5 → mode 3 (STAYS LOCKED), then asleep; no model → mode 4. **Not yet tested with the real camera; nothing run on the UNO Q.**
**Decisions and why:** White-balance jitter rather than moving val bursts into training, so val1 still measures an unseen color cast. The sketch drives the servo with hand-made 50 Hz pulses (no dependency on Servo library support in the Zephyr core) and stops pulsing 1 s after a move. Sketch → Python button uses `Bridge.call(...).result()` like the ip-address app; Python → sketch `update_state(mode, digits, count, hold%)` like door-to-door. Model files are kept OUT of the app folder because `deploy.py` uploads everything in it.
**Unfinished or next steps:** see Current State, NEXT.

### Session 15 — 2026-11-11 — Model + app deployed to the UNO Q
**Goal:** The user tested the new model live on the laptop and OK'd putting it on the UNO Q. They have 2 servos (no pins yet) and no button yet.
**Actions taken:** Sketch now drives 2 servos (D9, D10; both move together, per-servo angles). `main.py`: `AUTO_WAKE = True` (wakes itself while no button; set False once a button is on D2), auto camera search (skips the Qualcomm "venus" encoder at /dev/video0-1). Copied `finger_cnn.onnx` + `model_config.json` into `ArduinoApps/finger-safe/python/model/`. Deployed with `tools/deploy.py finger-safe` over USB adb (App Lab's adb at `%LOCALAPPDATA%\Arduino15\packages\arduino\tools\adb\32.0.0`; Wi-Fi hosts did not answer). `requirements.txt` = onnxruntime only.
**Test results:** Sketch compiled (95.7 KB flash) and flashed. onnxruntime 1.31.1 installed (Python 3.13.14 in the container). Model r20261111-092822 loads on the board. **No USB camera is attached to the board** (only the encoder), so no live test yet.
**Hardware and settings discovered:** The board image pins its own OpenCV build (`opencv-python-headless==4.13.0+1ddb20b`); requiring opencv in requirements.txt makes the install fail. cv2 + numpy are preinstalled. The container gets `/dev` + video group, so a UVC webcam will work once plugged in. /dev/video0-1 = Qualcomm Venus encoder.
**Unfinished or next steps:** Connect the DICCIK to the UNO Q through a powered USB-C hub, check the logs say "Found camera", test counts live on the board, and measure fps. Wire servos (signal D9/D10, external 5 V, shared GND), tune angles. Add the button on D2 and set AUTO_WAKE = False.

### Session 16 — 2026-11-11 — Camera through the laptop
**Goal:** The user keeps the UNO Q on the laptop's USB-C port and the camera on another laptop USB port.
**Actions taken:** `ArduinoApps/finger-safe/python/main.py` now has `SOURCE = "laptop"` (default) or `"usb"`. In laptop mode it is a TCP server on port 8770 (app.yaml `ports: [8770]`, published by Docker). Protocol: header `<4sBHHI` = b"FSF1", flags (bit 0 = button), h, w, nbytes, then raw BGR uint8; one JSON reply line per frame (count, confidence, raw_count, state, entered, progress, message, result, model_ms). New `Midterm2_Safe/stream_to_unoq.py`: finds adb (PATH or App Lab's copy), `adb forward tcp:8770 tcp:8770`, opens the DICCIK by name, sends `frame_to_raw` 224×224 frames (exactly the training form, lossless), shows a window with the board's answer; B = button, Q = quit. Port 8765 avoided (step-counter app).
**Test results:** Laptop loopback (dry run): 3-1-4 → UNLOCKED. On the board through USB-C: 120 val1 frames (every 16th), raw count 119/120 = 99.2%; round trip 45.9 ms = 21.8 fps; board ONNX model median 26.8 ms/frame (2 threads). Live camera through the stream not yet tested by the user.
**Unfinished or next steps:** see Current State NEXT.

### Session 17 — 2026-11-11 — Hardware wired, servos calibrated, Finger Safe redeployed
**Wiring (user confirmed wired):** button D2↔GND (internal pull-up); servo 1 signal D9, servo 2 signal D10; servos on a separate 5 V supply with GND shared with the board.
**Actions taken:** `ArduinoApps/servo-button-test/` (start-up sweep, button toggles servos, matrix shows button/servo state). The first sweep (0°/90°) drove the arms past where they can go, so I added `ArduinoApps/servo-calibrate/` (servos start limp; TCP 8771) + `Midterm2_Safe/servo_calibrate.py` (keys nudge each servo; L/O save; P writes the angles into finger-safe + servo-button-test sketches). The user calibrated: **LOCKED {30, 67}, OPEN {0, 97}** (servo 1, servo 2). `AUTO_WAKE = False` (button wakes the safe). Finger Safe redeployed (SOURCE=laptop) and waiting for the stream.
**Unfinished or next steps:** Full live test: `stream_to_unoq.py`, press the button, 3-1-4 → latches open 8 s then close; a wrong code stays locked. Record the results here.

### Session 18 — 2026-11-11 — 6-position unlock sequence
**Change (user's design):** positions (servo 1, servo 2): locked (0, new_angle1) → (0,97) → (30,97) → (30,67) → (0,67) → open (0,97); one servo moves per step. new_angle1 = servo 2 with the arm resting above the door. User choices: re-lock = the same list in REVERSE, 8 s after fully open; 0.7 s per step; find new_angle1 with the tool.
**Actions taken:** `finger-safe/sketch/sketch.ino`: `LOCK_REST_DEG` (placeholder 97 = open, until calibrated), `UNLOCK_SEQ` table, `STEP_MS=700`, `OPEN_MS=8000`; the old SERVO_LOCKED/OPEN_DEG arrays are gone. `servo-button-test`: the button runs the sequence forward/back. `servo_calibrate.py`: N saves servo 2's angle as new_angle1, T toggles servo 2 between it and 97, P writes `LOCK_REST_DEG` into both sketches. Servo-calibrate app deployed; finger-safe + servo-button-test NOT yet redeployed (new sketches not compiled yet).
**Unfinished or next steps:** the user finds new_angle1 → P → deploy servo-button-test (check the sequence with the button) → deploy finger-safe → full test with stream_to_unoq.py.

### Session 19 — 2026-11-11 — Everything together
The user saved **new_angle1 = 75** (`LOCK_REST_DEG = 75` in both sketches) and confirmed the button works. Finger Safe deployed with the 6-step sequence (sketch compiled, 96.8 KB), AUTO_WAKE=False, SOURCE=laptop; the model loaded; waiting for `stream_to_unoq.py`. Full live test pending (record results here).

### Session 20 — 2026-11-11 — Standalone on the UNO Q
**Goal:** run only on the UNO Q, camera plugged into the board.
**Actions taken:** `main.py` SOURCE default = "auto" (`AutoSource`: board camera when found, re-checked every 5 s; otherwise the laptop stream on 8770; a connected laptop keeps priority). Deployed. `arduino-app-cli properties set default /home/arduino/ArduinoApps/finger-safe` → Finger Safe starts at boot. The model was already on the board in `python/model/`.
**Test results:** dry run on the laptop: auto → laptop stream works (port 8790), and a fake camera appearing switches to "board camera". On the board: the model loads, source auto, waiting. Board camera NOT yet tested (the user is plugging it in).
**Note:** while `adb forward tcp:8770` is active, laptop port 8770 goes to the board; use another port for local tests.
**Unfinished or next steps:** the camera goes through a powered USB-C hub (the board's only USB-C). Matrix "!" = no camera/stream; dim corner blink = camera OK, asleep. Check the speed on the board (MJPG decode + crop + model).
