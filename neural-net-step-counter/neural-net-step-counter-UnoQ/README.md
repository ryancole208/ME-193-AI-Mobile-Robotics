# 👣 Neural-Net Step Counter (ME 193)

Counts every footfall (right + left heel strikes) live on the UNO Q and estimates the
distance walked, each with a 95 % interval. A small temporal convolutional network
(StepTCN) is trained in PyTorch on the laptop and runs here in plain **numpy**.
The 8 × 13 LED matrix shows the steps or the distance. An HTTP API lets the laptop pull
results and recordings and upload new models.

Sensors:
- **Ankle:** the Pico W + MPU-6050 pod ("GaitNode", firmware unchanged). It sends
  100 Hz, 6-axis data over BLE through the `ble_link` relay brick, which is copied from `unoq_app`.
- **Hip:** a GY-61 (ADXL335) analog accelerometer inside the UNO Q case. The sketch
  samples it at 100 Hz.

Only one app can hold the BLE link to the Pico W, so **stop `unoq_app` before starting this one**.

## Hip sensor wiring (GY-61 → UNO Q)

| GY-61 | UNO Q |
|---|---|
| VCC | **3.3 V** (not 5 V) |
| GND | GND |
| X OUT | A0 |
| Y OUT | A1 |
| Z OUT | A2 |

Screw it rigidly to the floor of the case, with no foam. Never call `pinMode()` on A0–A2:
on the UNO Q that breaks `analogRead()`.

## First run: one-time setup

If `config.json` is missing, the app starts the guided setup, driven by the matrix and foot taps.
It is resumable after a restart.

1. **Stand still for 10 s** with everything worn normally. The matrix counts down.
2. **Hip 6-face calibration** (optional, recommended once). Take the case off the belt and
   hold it still with the shown axis pointing **up**: `+X`, `−X`, `+Y`, `−Y`, `+Z`, `−Z`,
   3 s each. The bar at the bottom fills while it records. **4 taps skip this step.**
3. **Gesture check:** do the 4-tap gesture up to 5 times. The matrix flashes for each one it recognises.

A check mark means `config.json` was written and counting starts.

## Gestures (heel taps with the ankle pod, standing still)

| Gesture | Effect |
|---|---|
| 4 taps | toggle **steps ↔ distance** (scrolls `STEP` / `DIST`) |
| 8 taps, then 4 more within 5 s, **within 30 s of first data** | redo setup (`SET?`) |
| 8 taps, then 4 more within 5 s, later | reset the count, start a new session (`RESET?` → `RESET`) |

Steps counted from 0.5 s before the first tap to 0.5 s after the last tap are deleted,
so the taps themselves are never counted. Set `gesture_debug: true` in `setup.yaml` to log
the reason for every rejected tap train.

## Matrix

- **Steps:** up to 999 is shown static; 1000+ scrolls. The top 3 LEDs of the rightmost column are lit.
- **Distance:** scrolls, e.g. `352FT`, `0.27MI` (or `M`, `KM`). The bottom 3 LEDs of the rightmost column are lit.
- `S` at start = this app. A sweeping dot = waiting for sensor data. `X` = no data from either sensor.
- `NN X` = no usable `model.npz`, so the baseline detector counts. `HIP X` = hip sensor implausible.
- `CFG X` = broken `setup.yaml` / `config.json`. `LEG ?` = `leg:` differs from setup.
- `REC` (every 10 s) = recording. A short `X` every 10 s = the ankle link is lost and counting continues from the hip.

## Modes and files

`setup.yaml` holds the settings (leg, mode, units, API token, calibration walks, gesture
thresholds). **Change `net.token`.**

- `mode: live` counts and displays.
- `mode: record` does the same, and also saves every session to `sessions/<YYYYMMDD-HHMMSS>/`
  for training. Each session folder holds `fused.npy`, `raw_ankle.bin`, `raw_hip.bin`,
  `events.csv`, `gestures.csv` and `meta.json`. When `sessions/` grows past 2 GB, the oldest
  sessions are deleted.
- Both modes can also be set over the API.

Created on the device and git-ignored: `config.json`, `model.npz` (from the laptop),
`state.json`, `sessions/`, `logs/`.

Without `model.npz`, the **baseline detector** counts, so you can record training data before
any model exists. The baseline always runs and is logged next to the network, for the report.

## Startup self-check

The app log prints PASS/FAIL for each item:

- **model:** loads, `prep_hash` matches, numpy output matches the PyTorch reference to < 1e-4
- **hip:** each axis is at 5–95 % of the ADC range, and |a| = 1 ± 0.25 g when still
- **ble:** ankle packets arrive gap-free for 3 s
- **http:** port bound and listed in `app.yaml`

Counting continues with whatever works.

## diag mode

```bash
echo diag > ~/ArduinoApps/neural-net-step-counter/RUN_MODE      # one-shot, read at the next start
arduino-app-cli app restart ~/ArduinoApps/neural-net-step-counter
arduino-app-cli app logs ~/ArduinoApps/neural-net-step-counter --follow
```

Every 2 s the log shows:
- **ankle:** packet rate, gaps, MTU, battery
- **hip:** rate, late samples, clipping %, calibrated |a|
- **time sync:** slope, offset and residual per source
- **network:** inference ms per member, latest `p_right` / `p_left`
- **gestures:** recent verdicts

The same snapshot is always at `GET /api/diag`. `echo setup > RUN_MODE` redoes setup at the next start.

## HTTP API (port 8765)

Every request needs the header `X-Token: <net.token>`.

```bash
H="X-Token: change-me"; U=http://<uno-q-ip>:8765
curl -H "$H" $U/api/status
curl -H "$H" "$U/api/events?since=0"
curl -H "$H" -X POST $U/api/mode -d '{"mode":"record"}'
curl -H "$H" -X POST $U/api/model --data-binary @model.npz    # checked first; a failing model is rejected
```

Endpoints:
- `GET`: `/api/status`, `/api/events?since=<id>`, `/api/stream?since=<t>`, `/api/files`,
  `/api/files/<session_id>/<file>`, `/api/diag`
- `POST`: `/api/session/start`, `/api/session/stop`, `/api/session/truth`, `/api/mode`,
  `/api/display`, `/api/reset`, `/api/model`

The JSON shapes are in the shared contract (brief section 9.1). An uploaded model only replaces the
active one if it passes every check; the previous one is kept as `model.prev.npz`.

## How it works (python/)

| File | Role |
|---|---|
| `main.py` | startup, self-check, main loop |
| `pipeline.py` | the engine: wires everything together every 0.5 s |
| `ble_client.py`, `hip_stream.py`, `protocol.py` | packets, gap detection (seq + t0 jumps) |
| `timesync.py` | sensor clock → host monotonic time (lower-envelope fit, drift-aware) |
| `fusion.py` | 100 Hz grid in host time, the 12 network channels (`build_channels`), `prep_hash` |
| `stepnet_np.py` | numpy StepTCN, model checks, streaming inference with trusted-centre stitching |
| `counter.py` | peak picking, commit, uncertainty (σ_N), baseline detector, event ids |
| `distance.py` | step length (L0 or Weinberg) and σ_D |
| `gestures.py`, `setup_mode.py`, `display.py`, `recorder.py`, `api.py`, `diag.py`, `settings.py` | as named |
