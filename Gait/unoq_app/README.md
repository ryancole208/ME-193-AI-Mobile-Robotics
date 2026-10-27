# Gait Activity Classifier (UNO Q waist unit)

An Arduino App Lab app for the **Arduino UNO Q**, worn on the waist. It receives IMU data over
Bluetooth LE from a Pico W ankle node ("GaitNode"), classifies the wearer's activity as **still,
walking, running or biking**, and shows the result on the built-in 8×13 LED matrix as an
animated stick figure.

Two classifiers run side by side. Both are **k-nearest-neighbours (kNN)**, and there are **no
neural networks**:

| | Model A, "1D" | Model B, "Full IMU" / "6D" |
|---|---|---|
| Input | one number: `max(a_y)` in a window (peak shin-axis acceleration) | up to 30 features from all six IMU channels |
| Shown as | `1D`, top two pixels of column 12 | `6D`, bottom two pixels of column 12 |

A 4-tap foot gesture switches which model is shown. There is no button. Calibration happens
**once**: a guided setup writes `config.json`, and every later start loads it and goes straight
to classifying.

---

## Quick start

1. **Deploy** (from the board's terminal, or VS Code connected to the board):
   ```bash
   arduino-app-cli app start ~/ArduinoApps/unoq_app
   ```
   The first start needs internet so App Lab can install `bleak`, the Bluetooth library. Later
   starts work offline.
2. **Run `diag`** with the ankle pod switched on and lying flat on a table:
   ```bash
   python3 ~/ArduinoApps/unoq_app/tools/dev.py diag
   ```
   It runs seven checks and ends with a PASS/FAIL table in the log and in `reports/diag.md`.
   Fix any FAIL first; each one comes with its likely cause.
3. **Do the one-time setup outdoors.** Restart the app normally (`arduino-app-cli app restart
   ~/ArduinoApps/unoq_app`). With no `config.json`, setup starts by itself. Follow the matrix
   (see [One-time setup](#one-time-setup-outdoors-step-by-step)).
4. **Wear it.** When setup finishes, the matrix shows a ✓ and both accuracies, then the stick
   figure. Tap your heel 4 times to switch between `1D` and `6D`.

---

## What's in this folder

```
unoq_app/
├── app.yaml                  App Lab manifest (lists the local brick ble_link)
├── setup.yaml                options for setup (leg, Model A axis, recording length, ...)
├── sketch/sketch.ino         MCU: all LED-matrix drawing + animation timing, frame art as ASCII
├── sketch/sketch.yaml        build profile
├── python/main.py            entry point (App Lab runs this)
├── python/gait/              the classifier: protocol, sensor, windowing, features, knn,
│                             gesture, training, setup_mode, live, diag, replay, evaluate, ...
├── bricks/ble_link/          local brick: BLE relay container + the app-side client
├── tools/                    dev.py (entry point), matrix_preview, replay, evaluate
├── data/                     setup + holdout recordings (gitignored)
├── logs/                     per-run CSV logs (gitignored)
└── reports/                  setup_report.md, diag.md, evaluate.md, matrix_preview.html (gitignored)
```

The files that appear at run time are `config.json` (the trained models) and `state.json` (which
model was last shown), plus a one-shot `RUN_MODE` file when you ask for a special mode.

## How it fits together

```
 Pico W ankle node ──BLE──► ble-relay container ──Unix socket──► app container (python/main.py)
  (GaitNode, 100 Hz)        bricks/ble_link/relay.py              packet parse → gaps → units → leg
                            bleak + BlueZ over D-Bus              → gesture detector (every sample)
                                                                  → windows → features → kNN A + kNN B
                                                                  → smoothing → display commands
                                                                        │ Router Bridge
                                                                        ▼
                                                      sketch.ino on the STM32: draws + animates
```

**Why a relay container?** App Lab runs the Python app in a Docker container. That container
gets no D-Bus socket, and Linux only allows Bluetooth sockets in the host's network namespace,
so `bleak` cannot work inside it. The local brick's `brick_compose.yaml` adds a small second
container that:

- uses the **same image and the same Python environment** App Lab already built for this app;
- mounts the system D-Bus socket read-only (BlueZ's own policy already allows any user to talk
  to `org.bluez`);
- forwards the GaitNode notifications to the app over a Unix socket in `.cache/ble_link/`.

Nothing is installed on the board and no system settings change. The other apps are not
affected.

---

## Deploying

**Command line** (on the board, e.g. in VS Code's terminal):
```bash
arduino-app-cli app start   ~/ArduinoApps/unoq_app   # stops whatever app was running
arduino-app-cli app logs    ~/ArduinoApps/unoq_app --follow
arduino-app-cli monitor                              # sketch log ("gait display ready")
arduino-app-cli app stop    ~/ArduinoApps/unoq_app
```
To make it start on boot: `arduino-app-cli properties set default user:unoq_app`.

**App Lab GUI:** open App Lab, find **Gait Activity Classifier 🏃** under *My Apps*, and
press **Run**. The *Bricks* tab shows **BLE Link (GaitNode)**, whose two variables are the
advertised name and the socket path. The console shows the same log as `app logs`.

If the sketch acts stale after an edit: `arduino-app-cli app clean-cache user:unoq_app --force`,
then start again. This also deletes `.cache/`, so the next start reinstalls `bleak` and needs
internet.

### Special modes (RUN_MODE)

Write one word into the file `RUN_MODE` in the app folder, then restart the app. The file is
deleted when it is read, so the start after that is normal again. `tools/dev.py` does both steps
for you, after asking:

| Command | What it does |
|---|---|
| `python3 tools/dev.py diag` | hardware check (below) |
| `python3 tools/dev.py mode setup` | redo setup from scratch: backs up `config.json`, archives all old recordings |
| `python3 tools/dev.py mode "redo gestures"` | redo **only** the named part(s) (comma-separated: `stand`, `gestures`, `still`, `walking`, `running`, `biking`); everything else is kept and `config.json` stays valid until the shorter redo finishes and rebuilds it whole |
| `python3 tools/dev.py mode holdout` | records each activity once more into `data/holdout/` for `evaluate` |
| `python3 tools/dev.py mode "replay data/setup/walking.csv"` | plays a recording through the live pipeline **on the real matrix** |

---

## diag (first real power-on)

Put the pod flat on a table and run `python3 tools/dev.py diag`. You don't need to do anything
else. The checks run in this order:

1. **Bluetooth access.** The relay container is up, BlueZ answers over D-Bus, and the adapter is powered.
2. **GaitNode link.** The node is found, connected and subscribed, and its MTU and samples per packet (n) are reported.
3. **30 s of streaming.** Effective sample rate (expect 100 Hz), packet loss, and gaps found from `seq` and from `t0_us`.
4. **Pod still.** `|a|` is about 1.00 g and the gyro noise is low. This catches byte-order and scaling bugs (≈8 g means ±2 g range; wild values mean byte order).
5. **Battery.** A battery notification arrived, after a `PING`. 0 means *reading unavailable*.
6. **Matrix.** The display handshake succeeds, then every animation and status screen plays once. Watch the matrix.
7. **config.json.** It is valid, or "will start setup".

The table goes to the log and to **`reports/diag.md`**. Paste that file back when asking for help.

---

## One-time setup (outdoors, step by step)

You don't need a laptop. Before you go, set `leg:` in `setup.yaml` (default `right`). Strap the
pod to that ankle with the lid arrow (+Y) pointing up the shin, and start the app. If there is
no `config.json`, setup starts on its own.

| Step | Matrix | What you do |
|---|---|---|
| 0 | scanning dot | wait for the ankle node to connect |
| 1 | big countdown number, shrinking bar | **stand still for 10 s** (stores the gravity direction and sensor noise) |
| 2 | `S` + countdown, then `S` + progress bar | get ready, then **stand / sit still for 120 s** |
| 3 | `G` + count, five dots | **gesture check:** do the switch gesture (4 heel taps) 5 times; wait for the flash after each |
| 4 | `W` + countdown, then `W` + bar | **walk for 120 s**, slow and fast |
| 5 | `R` + countdown, then `R` + bar | **run for 120 s**, easy and hard |
| 6 | `B` + countdown, then `B` + bar | **bike for 120 s**, at different cadences |
| 7 | spinner | the models are built (a few seconds) |
| 8 | ✓, then `A 87%` and `B 91%` scroll past | done. Live mode starts. |

- Each countdown is 10 s. Use it to get moving. The bar along the bottom row shows recording progress.
- At the end of every recording the whole matrix **flashes 3 times**. That recording is
  saved straight away to `data/setup/<activity>.csv`.
- **Skipping an activity** (e.g. to bike on another trip): during its 10 s countdown, do
  the 4-tap gesture. The matrix flashes once and moves on. At the end it shows the missing
  letter and `?`. Next time you start the app, setup **resumes from the first missing
  activity** and keeps what was already recorded.
- If the link drops, the matrix shows `X` and the countdown or recording pauses until data
  flows again.
- Setup then drops the first and last 5 s of each recording, cuts the rest into windows,
  computes features, chooses k by cross-validation, and writes `config.json` plus
  `reports/setup_report.md`. The report has a confusion matrix per model, the k scores, the
  Model B features kept, Model A's per-class ranges, and the tuned gesture thresholds.

If setup shows a big **X** at the end, something was wrong with the data. The log line starting
`SETUP FAILED` says what happened and how to fix it.

## Gestures

All gestures are heel (or toe) taps on the ground **with the pod's foot, while standing still**.

| Gesture | How | Effect |
|---|---|---|
| **Switch** | 4 quick taps, 200–600 ms apart (like knocking), still before and after | switches between Model A (`1D`) and Model B (`6D`); the banner shows for about 1.2 s; the choice survives a reboot (`state.json`) |
| **Long** | 8 taps in the same rhythm | shows the ankle battery bar for 2 s (`N/A` if the node reports 0 = unavailable) |
| **Redo setup** | 8 taps **within 30 s of the node first connecting** (a blinking dot at the bottom of column 11 means this window is open), then `SET?` scrolls: **4 more taps within 5 s** | confirms redoing setup; anything else, or nothing, cancels |

Why this doesn't trigger while you move: the same foot lands only about once a second when
walking or running, so 200–600 ms between taps is far faster than any stride. Also, the 1.5 s
before the first tap and the 1 s after the last must be **still**: the variation of `|a|` and
`|g|` must be below 3 × the noise measured in setup. After a switch, further gestures are
ignored for 3 s. A tap count within `gesture.taps_tolerance` (default 1) of 4 or 8 still counts -
real taps occasionally pick up an extra bounce or get undercounted, and this absorbs that without
widening the timing/stillness checks that guard against accidental triggers. Every gesture **and
every near-miss** (wrong count, wrong rhythm, not still) is
logged to `logs/gestures-*.csv` with the reason, so you can tune the thresholds in
`config.json → "gesture"`.

**Fallback:** if gestures don't work for you, set `switch_mode: auto_alternate` in `setup.yaml`
before setup, or `"display": {"switch_mode": "auto_alternate"}` in `config.json`. The display
then shows Model A for 4 s and Model B for 4 s, using the column-12 indicator.

## What the matrix shows

| Screen | Meaning |
|---|---|
| standing figure, chest "breathes" every 2 s | **still** |
| upright walker, 4 frames at ~3 fps | **walking** |
| leaning runner with bent arms, 4 frames at ~6 fps | **running** |
| rider on a two-wheeled bike, pedalling at ~5 fps | **biking** |
| small `A` / `B` in the top-left corner | which model is active (always on, overlays the animation's corner) |
| column 12: top two pixels / bottom two pixels | Model A (`1D`) / Model B (`6D`) is shown |
| `1D` / `6D` for about 1.2 s | you just switched models |
| dot sweeping left and right | searching for the ankle node |
| big `X` | link lost (no data for more than 2 s) |
| big `?` | not enough data yet, windows keep having gaps (>5 % missing, 3 in a row), or "unsure" (if enabled) |
| blinking pixel, bottom of column 11 | redo-setup window is open (first 30 s after the node connects) |
| battery outline, 1 s every 30 s | ankle battery below `low_battery_mV` (never shown for 0 = unavailable) |
| battery bar / `N/A` for 2 s | answer to the long gesture |
| `CFG` then `X` | `config.json` is broken. It is renamed to `config.broken-<date>.json` and setup starts. |
| `LEG` then `?` | `setup.yaml` names a different leg than setup used. It keeps classifying; see the log. |
| letter + number, letter + bar, `G` + dots, spinner, ✓ | setup screens (see above) |
| letter + `?` | setup paused: this activity is still missing |
| `D` + number, then ✓ or `X` | diag running, then all passed / something failed |
| scrolling `SET?` | confirm redo-setup with 4 taps |

Check the art without flashing: `python3 tools/dev.py preview` prints every frame and writes
`reports/matrix_preview.html`, where the animations play at their real speed. The frames are
the ASCII strings in `sketch/sketch.ino` (`#` = bright, `+` = dim, `.` = off). Edit them there.

## Redoing setup on purpose

Only these ways ever trigger setup. Nothing recalibrates automatically.

1. **Gesture:** 8 taps in the first 30 s after the node connects, then 4 taps when `SET?` scrolls.
2. **`python3 tools/dev.py mode setup`**, which writes `RUN_MODE=setup` and restarts.
3. **Delete or rename `config.json`** through App Lab or SSH, then restart.

In every case the old config is kept as `config.backup-<date>.json`. The old recordings move
to `data/setup-archive-<date>/`, so setup records afresh rather than rebuilding from them.
(Partial setups, where an activity was skipped, are resumed instead. See above.)

4. **Recalibrate only part of it:** `python3 tools/dev.py mode "redo gestures"` (or `stand`,
   `still`, `walking`, `running`, `biking`, comma-separated for several). Only the named
   recording(s) are archived; everything else - including `config.json` itself - is left alone
   while setup re-records just that part. `config.json` stays valid for live mode the whole time,
   so an interrupted partial redo just means nothing changed yet, not a broken config; rerun the
   command to pick it back up. When setup finishes it rebuilds `config.json` from scratch anyway
   (all recordings are always re-read from disk), so the kept recordings produce the same
   signal/model sections and only the redone part's section actually changes.

## Live mode and logs

- Every window (2.56 s, every 1.28 s) is classified by **both** models, and both are logged. The
  gesture only changes which one is shown, so switching is instant and the comparison stays fair.
- Smoothing: a majority vote over the last 3 windows. The figure changes only after the smoothed
  result has differed from the shown one for 2 hops in a row. Windows containing a tap gesture
  are logged but not used for smoothing, so tapping doesn't flash "walking".
- `logs/run-<time>.csv` has one row per window: time, raw and smoothed prediction of A and B,
  shown model, what was displayed, gesture flag, nearest-neighbour distances, packet loss %,
  RSSI and ankle battery mV.
- `logs/gestures-<time>.csv` logs every gesture and near-miss.
- `logs/raw-<time>.csv` holds the raw IMU stream (turn it off with `log_raw: false`). You can
  replay and evaluate it.

---

## Tools

On the UNO Q the board's own Python has no numpy, so `tools/dev.py` runs every tool in a
throwaway container of the app's own image. Nothing is installed on the board. On a laptop with
numpy, the same commands run directly.

```bash
cd ~/ArduinoApps/unoq_app
python3 tools/dev.py preview     # matrix art
python3 tools/dev.py replay logs/raw-20261022-101500.csv     # through the live pipeline, prints display commands
python3 tools/dev.py evaluate    # data/holdout/*.csv -> reports/evaluate.md (+ PNG plots if matplotlib exists)
```

- **`preview`**: prints every matrix frame and writes `reports/matrix_preview.html`.
- **`replay`**: feeds any recording through live mode, with gestures and smoothing. To watch it
  on the matrix, use `tools/dev.py mode "replay <csv>"`.
- **`evaluate`**: confusion matrices, per-class precision, recall and F1, accuracy and
  balanced accuracy for both models on held-out recordings. Where matplotlib is installed
  (e.g. on a laptop), it also draws a histogram of Model A's feature per class with
  `class_ranges` on it.

### Verification during development

The test suite and the simulated ankle node were removed from the board to save space
(about 400 MB of test-only packages). Before removal, all 112 tests passed:
- **Features:** a 2 Hz sine gives a 2 Hz dominant frequency; a constant 1 g gives zero std and
  zero jerk; gyro features ignore a gyro offset.
- **kNN:** agrees with scikit-learn for every k.
- **Packet parser:** `seq` and `t0_us` wraparound, n = 1..10, dropped packets, a FIFO overflow
  where `t0_us` jumps but `seq` doesn't, duplicates and restarts.
- **Leg transform:** a synthetic 3-D motion of a right shank against the mirror-image left
  shank, for both accel and gyro.
- **Config:** save, load and validate; broken configs are rejected with a clear message.
- **Gestures:** right and wrong rhythms, 3/4/8 taps, lockout, taps during running (no trigger),
  and the redo-setup sequence. Replaying 5 min of running and 5 min of biking gave zero false
  switches.
- **End to end**, with a simulated ankle node producing real protocol packets: setup,
  `config.json`, live mode and display commands, on both legs.

## Results

### Your data (fill this in)

After setup, the CV numbers are in `reports/setup_report.md`. For a fair test, record held-out
data on another day with `python3 tools/dev.py mode holdout`, then run
`python3 tools/dev.py evaluate`.

| | Model A "1D" | Model B "Full IMU" |
|---|---:|---:|
| Setup CV balanced accuracy | _from setup_report.md_ | |
| Held-out accuracy | _from evaluate.md_ | |
| Held-out balanced accuracy | | |
| Held-out, after re-mounting the cuff | | |
| Typical confusions | | |

Expect Model A to confuse **biking** with **still** or walking. Pedalling produces no heel strikes, so
peak `a_y` stays near 1.1–1.5 g. That's a result to report, not a bug: Model A deliberately uses
exactly one feature.

### Synthetic sanity check (not a result)

On data from a simulated ankle node used during development (120 s per activity, one held-out session, and one session with the
pod re-mounted at a different angle and gyro offset), both models score 100 %. The synthetic
activities are far cleaner than real movement. This only shows that the whole pipeline works;
it says nothing about real accuracy. Model A's synthetic class ranges (still ≈ 1.01 g, biking
1.1–1.5 g, walking 2.0–2.6 g, running 3.9–7.9 g) do show why biking is its weak spot.

### The cuff-off / cuff-on test

This shows that calibrating once is enough:
1. Do setup and wear the pod as usual. Record a held-out session: `tools/dev.py mode holdout`.
2. **Take the cuff off.** Walk around, let the pod warm up or cool down, then put it back on
   with the arrow up the shin but without being careful about the exact angle.
3. Record a second held-out session, and keep the two sets of files apart (move the first set
   into a sub-folder).
4. Run `python3 tools/dev.py evaluate <first files> --title "Before re-mounting"` and
   `... <second files> --title "After re-mounting" --out reports/remount`.
5. Compare the balanced accuracies. They should be close. If Model B drops a lot, the culprit
   is probably `pitch_mean`, the only orientation feature; see `reports/setup_report.md` for
   whether it was selected.

During development the same check on synthetic data (7° of re-mounting tilt and a different
gyro offset) showed no drop in accuracy.

---

## Design decisions

**kNN in plain numpy.** A prediction depends only on `config.json`: the training rows `X`,
labels `y`, the z-score mean and std, and `k`. It uses Euclidean distance on z-scored features
with 1/distance-weighted votes. If a neighbour is at distance exactly 0, only those vote, the
same rule scikit-learn uses (checked during development). scikit-learn isn't needed on the board.

**Calibrate once.** The pod sits at a slightly different angle every time, and the MPU-6050's
gyro offset drifts with temperature, so:
- **No raw gyro means anywhere.** Every gyro feature is de-meaned (std, range, RMS of the
  de-meaned signal, skew, kurtosis, jerk) or spectral or periodic. An offset cancels out
  exactly. So no gyro calibration is needed.
- **Accel features** are magnitudes, dynamics and spectra. The one orientation feature is
  **shank pitch** (mean and range), measured as an angle in the pod's x–y plane *relative to the
  gravity direction stored during setup's stand-still step*. The lid arrow keeps mounting
  consistent enough for that.

**Model B features** (88 candidates). They are computed for `ax, ay, az, gx, gy, gz, |a|, |g|`:
- **Time domain:** std, range, RMS of the de-meaned signal, skewness, excess kurtosis, mean
  absolute jerk.
- **Frequency domain** (FFT with a Hann window, 0.3–6 Hz): dominant frequency (zero-padded,
  with parabolic peak interpolation), its share of band power, normalised spectral entropy.
- **Periodicity:** autocorrelation peak height and lag (0.2–1.5 s).

Plus `|a|` mean, shank pitch mean and range, the correlations (ax,ay), (ay,az), (ax,az) and
(gx,gz), and the signal magnitude area of the de-meaned acceleration.

Setup drops near-constant features, ranks the rest by **ANOVA F-score**, and keeps the top
`model_b.top_n` (30), listed in the report. Missing samples inside a window (≤5 %) are filled
by linear interpolation.

**Model A**: exactly one feature, `max(a_y)` in g. `setup.yaml → model_a` can change the axis
or use `max(|a_y|)`. Nothing else is added.

**Choosing k without leaking.** Neighbouring windows overlap by 50 %, so ordinary random CV
would score a window against its near-copy. Instead:
- each recording is cut into 5 consecutive **time blocks**;
- block *b* of every recording forms fold *b*, so every fold contains every class (stratified)
  and folds are grouped by where in the recording the windows came from;
- any training window that overlaps a test window in time is dropped.

k comes from {1, 3, 5, 7, 9, 11} by CV balanced accuracy; on a tie, the larger k wins, because
it is smoother. Model B's feature selection is redone inside each fold, so its CV score is
honest. Each class keeps at most 200 windows, subsampled evenly through each recording.

**"Unsure" (optional, off by default).** For each training window, setup finds the distance to
its nearest neighbour from another time block, and stores the 99th percentile as
`unsure_distance`. With `unsure_enabled: true`, a window farther than that from all training
data shows `?`.

**Left leg.** The left pod is a mirror image, so the transform comes from mirroring across the
body's midline (the full derivation is in `python/gait/sensor.py`). Accelerations are polar
vectors and pick up only the reflection. Angular velocities are pseudovectors and pick up an
extra minus sign. The result is: accel `(ax, ay, az) → (−ax, ay, az)`, gyro
`(gx, gy, gz) → (gx, −gy, −gz)`. It was checked against a simulated 3-D motion, not just against
the formulas.

**Gaps.** The packet tracker turns `seq` and `t0_us` into a continuous sample index. A `seq`
jump means packets were lost over the air. A `t0_us` jump with a continuous `seq` means the Pico
W's FIFO overflowed. Both are counted separately; wraparound is handled; sensor time is never
compared with the UNO Q clock. Missing samples become NaN rows. Windows with more than 5 %
missing are skipped, and three in a row show `?`.

**Setup timing** counts *received samples*, not wall-clock time. If the link drops, the
countdown or recording pauses instead of silently recording nothing.

**MCU vs Python.** The sketch owns every pixel and every animation timer. Python sends a few
small commands through Router Bridge (`show_activity`, `show_model`, `show_status`,
`show_setup`, plus `show_value` for scrolling numbers and the battery bar, and a
`display_version` handshake). They are registered with `provide_safe`, so they run on the
sketch's loop thread and never race the renderer. The Python side sends them from a worker
thread, so a slow MCU never stalls classification.

**Failing loudly.** Every deliberate error says what it was doing, what went wrong, and the most
likely fix (`gait/errors.py`). A broken `config.json` shows `CFG X` and is kept as
`config.broken-<date>.json`. A broken `setup.yaml` shows `CFG X` and stops. A bug in the live
loop is logged with a full traceback, and the loop keeps running.

## config.json

It follows the brief's schema, plus a few additions. The additions are fully validated on load.

| Key | Contents |
|---|---|
| `signal.noise` | std of `|a|` and `|g|` from the stand-still step (used by the gesture stillness check) |
| `model_*.cv_accuracy`, `model_*.cv_scores_by_k` | for the report |
| `model_a.unsure_distance` | as for Model B |
| `model_b.selection` | ranking, top_n, dropped features, F-scores |
| `display.unsure_enabled` | the "unsure" switch |
| `link.battery_empty_mV`, `link.battery_full_mV` | for the battery bar |

Edit `"gesture"` there to retune tap thresholds without redoing setup, or see
[Redoing setup on purpose](#redoing-setup-on-purpose) to re-record just the gesture check.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Log: `BLE relay not reachable ...` for more than a minute | On the first start, the relay waits until App Lab has installed `bleak`, which needs internet. Check `docker ps` for a container named like `unoq_app-ble-relay-1`; `docker logs <it>` shows what it is doing. If it doesn't exist, App Lab didn't pick up `bricks/ble_link/brick_compose.yaml`; check `app.yaml` still lists `- ble_link: {}`. |
| Relay says BlueZ adapter powered off | `bluetoothctl power on` |
| Scanning dot forever | The Pico W isn't advertising as `GaitNode`, or it is out of range. A phone BLE scanner app helps. |
| `X` right after connecting | The Pico W isn't sending after the CCCD subscription; check its log |
| `|a|` ≈ 8 g or wild values in diag | Accel range or byte order on the Pico W (see diag's hint) |
| Many `?` while walking | Packet loss: diag check 3 shows the rate. Keep the UNO Q and the ankle on the same side of the body, or move closer. |
| Gestures don't register | `logs/gestures-*.csv` shows near-misses and why. Tap harder or more evenly, stand still first, or lower `gesture.tap_g`. |
| Gesture check stuck at 0 during setup, or `GaitNode disconnected` / `Pico W restart` repeating in the log | The ankle node is losing power or browning out on impact (check the battery and strap contact), not a software issue - a link gap under ~200 ms is tolerated and just drops the in-flight tap, but the node has to actually stay connected long enough to see 4 taps. |
| Matrix shows nothing / old sketch | Restart the app; if it's still stale, `arduino-app-cli app clean-cache user:unoq_app --force`. `arduino-app-cli monitor` should print `gait display ready`. |

---

## Assumptions

These are choices made without asking, each the option best supported by the board and the docs.

1. **Bluetooth goes through a relay container.** The app container has no D-Bus or Bluetooth
   access (Bluetooth sockets only work in the host network namespace). Rather than change system
   settings, the local brick `ble_link` adds a second container through `brick_compose.yaml`.
   It reuses the app's image and venv, with `/run/dbus` mounted read-only.
2. **First start needs internet** to install `bleak==3.0.2` (`bricks/ble_link/requirements.txt`),
   the same way other apps install their requirements.
3. **The relay's image tag** (`python-apps-base:0.12.0`) is pinned to the one App Lab 0.13 uses
   on this board. If App Lab updates its base image, update the tag in `brick_compose.yaml`.
4. **RSSI** comes from the advertisement seen just before connecting. BlueZ doesn't update RSSI
   for a connected device over D-Bus, so the log's RSSI column is per connection, not per window.
5. **MTU** is read from BlueZ with bleak's `_acquire_mtu()`, half a second after connecting, to
   give the Pico W's own MTU request time to finish. If that fails, the log says "default", and
   the samples per packet (n) of the first packet are logged instead. The UNO Q handles any n
   from 1 to 10.
6. **Special modes use a one-shot `RUN_MODE` file**, because App Lab apps take no command-line
   arguments. `tools/dev.py diag` writes it and restarts the app.
7. **diag stays on its result screen** until you restart the app. Continuing into setup with
   the pod lying on a table would record nonsense.
8. **Redo setup archives the old recordings**, so setup records afresh. Without this, setup
   would just rebuild the same models from the old CSVs. Partial setups are resumed instead,
   as the brief asks.
9. **Gesture lockout:** a gesture within 3 s of the previous one is ignored in live mode,
   **except** the redo-setup confirmation, which the brief requires within 5 s of the 8 taps.
10. **"Still before/after"** uses 1.5 s before and 1 s after, with a 0.3 s margin from the taps
    (the heel lifts slightly before the first tap). It is compared against max(noise from setup,
    0.01 g / 1 °/s) × 3, so an unusually quiet setup can't make the check impossible.
11. **Tap tuning in setup:** `tap_g` is set halfway between the still recording's
    99.9th-percentile `|a|` and the 20th-percentile tap peak, clipped to 1.3–4 g. The tap
    duration limit and gap range come from the recorded taps with margins. If fewer than 3
    clean gestures were recorded, the defaults are kept. Everything is in the report.
12. **Leg mismatch:** live mode keeps classifying using the leg in **setup.yaml**, because that
    is where the pod physically is, and shows `LEG ?`.
13. **The stand-still step is recorded once**, together with Still. If you finish biking on
    another trip, the gravity direction from the first session is kept.
14. **Battery bar** maps `battery_empty_mV` → 0 % and `battery_full_mV` → 100 %. The defaults,
    2000 and 3000, assume two AA cells feeding VSYS; set them in `setup.yaml` for other
    batteries.
15. **Unsure distance** is the 99th percentile of each training window's distance to its
    nearest neighbour *from a different time block*, which is what a window from a new session
    typically sees.
16. **Ties in smoothing:** with three different votes, the currently shown label wins if it is
    among them, otherwise the most recent one.
17. **Frame art is stored as ASCII strings inside `sketch.ino`** rather than in a separate
    header file, so the art is the data (you edit what you see) and the app contains only
    `.py`, `.yaml`, `.ino` and `.md` sources. `tools/matrix_preview.py` parses the same strings.
18. **The sketch logs with `Monitor`** (Router Bridge's monitor, shown by `arduino-app-cli
    monitor`), as the other sketches on this board do.
19. **`RESEND <seq>` is not used.** The brief marks it optional, and the tracker ignores late or
    duplicate packets safely if it is ever enabled on the Pico W.

## Sources relied on

The UNO Q is new, so everything was checked on this board (App Lab CLI 0.13.0, Zephyr core
1.0.0) rather than taken from memory:
- `arduino-app-cli` help, its embedded custom-brick template and API spec, and the generated
  `.cache/app-compose.yaml` of other apps (how containers, `/dev` and the Router socket are
  mounted).
- The base image `ghcr.io/arduino/app-bricks/python-apps-base:0.12.0`: `/run.sh` (venv, brick
  requirements, PYTHONPATH for `bricks/`), and the source of `arduino.app_utils` (`App`,
  `Bridge`, `@brick`, `Logger`).
- `Arduino_LED_Matrix` source in the Zephyr core (`begin`, `setGrayscaleBits(3)`,
  `draw(uint8_t[104])`, row-major 13 columns), and the bundled examples `mascot-jump-game`,
  `led-matrix-painter` and `keyword-spotting`.
- `Arduino_RouterBridge` 0.4.4 source (`provide` vs `provide_safe`, the loop hook).
- The installed `bleak` 3.0.2 source (BlueZ backend, `_acquire_mtu`, notification callbacks,
  `find_device_by_filter`) and BlueZ's D-Bus policy `/usr/share/dbus-1/system.d/bluetooth.conf`.
- docs.arduino.cc (UNO Q, App Lab bricks) and github.com/arduino/app-bricks-examples, for the
  conventions the on-board copies follow.
