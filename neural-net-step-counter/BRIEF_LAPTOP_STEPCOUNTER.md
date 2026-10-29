# Claude Code brief 2 of 2: laptop training + web dashboard (ME 193)

Build everything in this brief **in one pass**. No test files or test folders. Built-in self-checks, a `diag` command and clear comments are wanted (class project: explain the "why").

This is one of two briefs. The other one (`BRIEF_UNOQ_STEPCOUNTER.md`) builds the UNO Q app, which counts live and serves an HTTP API. This laptop side does two jobs: **(1) trains the neural network in PyTorch and uploads it to the UNO Q, and (2) runs the live web dashboard and logs everything.** The two sides talk only through the **shared contract in section 8**. Implement your half of it exactly.

---

## 1. Goal

- **Train** StepTCN, a neural network that detects right and left heel strikes, on sessions recorded by the UNO Q. Steps = right + left.
- **Calibrate the uncertainty**, so the UNO Q's 95% intervals actually cover 95% of held-out sessions.
- **Export** `model.npz` (format 8.4) and upload it to the UNO Q.
- **Dashboard:** a local web page showing the live count and distance with uncertainty, signals, session controls, and a log on the laptop. It's optional for the live counter: the UNO Q keeps counting without it.

## 2. Folder layout

```
Gait/stepcount_laptop/
  stepnet.py                # one CLI: pull | label | train | export | upload | dash | diag | truth
  model.py                  # PyTorch StepTCN (8.4), the single source of truth
  stepnet_np.py             # numpy forward pass, used for the export check
  data.py                   # load sessions, windows, augmentation, the 8.3 channel text + hash
  labeling.py               # count-constrained auto-labelling
  baseline.py               # baseline detector (8.5)
  calibration.py            # uncertainty calibration
  unoq.py                   # API client (8.1): token, retries, timeouts
  dashboard/
    server.py               # Flask: polls the UNO Q, logs, serves the page and a small JSON API
    static/index.html       # plain HTML/JS/canvas, no CDN (must work offline)
  config.yaml               # unoq_url, token, units, paths
  requirements.txt          # torch, numpy, scipy, matplotlib, flask, requests, pyyaml
  data/                     # pulled sessions + live logs (gitignored)
  reports/                  # generated HTML/PNG for the write-up
  README.md                 # workflow + data-collection protocol (section 6)
```

## 3. CLI (`stepnet.py`)

- **`pull`**: download every session not already in `data/sessions/` (`GET /api/files`, then each file). Skip the session that's still recording. Resume partial downloads.
- **`truth <session_id> <steps> [--distance-in X] [--calib-walk] [--notes ...]`**: POST the ground truth to the UNO Q and update the local `meta.json`.
- **`label`**: count-constrained auto-labelling, for each session with a true count:
  1. Run the baseline detector (8.5) on `fused.npy` to propose right and left strikes. Drop proposals inside the windows in `gestures.csv`.
  2. **Sweep** `gz_thr` (60–200 °/s) and `hip_prom` (0.05–0.4 g), and keep the setting whose right + left total is closest to the true count.
  3. If it's still more than 2% (or 3 steps) off, mark the session `needs_review` and exclude it from training (`--include-review` overrides).
  4. Write `labels.npz` (strike times per foot) and `reports/label_<id>.html` (signals, labelled events, true vs labelled count).
  - Sessions with true count 0 (biking, sitting, standing, gestures) are pure negatives.
- **`train`**, see 4.
- **`export`**: write `model.npz` (8.4). Run `stepnet_np.py` on `ref_x` and assert it matches PyTorch (max abs diff < 1e-4) before saving.
- **`upload`**: POST `model.npz` to `/api/model` and print the UNO Q's check results.
- **`dash`**, see 5.
- **`diag`**: is the UNO Q reachable, is the token accepted, round-trip time, current status, and the numpy vs PyTorch check on the local `model.npz`.

## 4. Training (`train`)

- **Split by session**, never by window: about 20% of labelled sessions are held out, stratified so walking, running and negatives appear on both sides.
- Targets: a Gaussian bump (σ = 3 samples, peak 1.0) at each labelled strike, per foot. Loss: `BCEWithLogitsLoss(pos_weight=10)`.
- 4 s windows (400 samples) at random offsets, batch 64, AdamW lr 2e-3, cosine schedule, early stopping on held-out loss. CPU-friendly: a few minutes on a laptop.
- **Augmentation:**
  - gain ×[0.9, 1.1] per source
  - random ankle rotation ±10° about each axis
  - hip time shift ±50 ms (covers clock-sync error)
  - Gaussian noise
  - drop a whole source (ankle or hip) 15% of the time, zeroing its channels and present flag
  - time-stretch ±10% (cadence variety)
- **Ensemble:** 5 members, different seeds, each trained on a bootstrap resample of the training sessions.
- Tune `thr` on the held-out sessions to minimise the total absolute count error. Run peak picking exactly as the UNO Q does: local max ≥ thr, ≥ 0.30 s apart per foot.
- **Calibration** (`calibration.py`), on the held-out sessions:
  - `r` = RMS relative count error
  - `κ ≥ 1` = the smallest value for which 95% of held-out true counts fall inside `N ± 1.96 σ_N`, where `σ_N = sqrt(κ²·(Σc(1−c) + var across members) + (r·N)²)`
  - save a coverage plot (nominal vs observed at 50/80/90/95%)
  - warn that calibration from few sessions is approximate
- **Distance:**
  - `L0_m` = 0.7303 (115 in over 4 steps, from `config.yaml`)
  - if sessions marked `is_calib_walk` exist, fit Weinberg `K = D / Σ(a_max − a_min)^(1/4)` over their labelled steps, using hip vertical accel between consecutive steps
  - `r_L` = leave-one-out relative distance error if there are ≥ 2 calibration walks, else 0.05
- **Reports** in `reports/` (an HTML index plus PNGs, for the class write-up):
  - training curves for all 5 members
  - per-session true vs NN vs baseline counts and % error
  - event matching at ±60 ms tolerance: TP, FP, FN, precision, recall
  - example probability traces
  - the coverage plot
  - a network diagram with parameter counts
  - Print the held-out walking and running count errors (targets ≤ 2% and ≤ 4%).

## 5. Web dashboard (`dash`)

### 5.1 Server (`dashboard/server.py`, Flask, `localhost:8080`)

- Polls the UNO Q: `/api/status` every 0.5 s; `/api/events?since=` and `/api/stream?since=` every 1 s.
- **Backfill:** after any Wi-Fi drop, fetch `events?since=<last_id>`. Ids are unique, so nothing is duplicated.
- **Logs** to `data/live_logs/<date>/`: `events.csv` (deletions applied as rows with `deleted=1`), `status.csv` (one row per poll), `marks.csv`.
- Shows a clear "UNO Q unreachable — retrying" state; never crashes on timeouts.
- Serves `static/index.html` plus a small JSON API for the page: `/local/state`, `/local/history?minutes=2`, and pass-through POSTs for every UNO Q command in 8.1.
- If the UNO Q uses `net.mode: push`, also accept `POST /ingest` (same JSON shapes) and answer with the queued commands.

### 5.2 Page (`static/index.html`)

Plain HTML/JS/canvas, no external libraries, light and dark themes, works offline. Layout:

- **Headline:** step count, its 95% interval as text and as an error bar, and σ. Distance in the chosen units with its interval.
- **Variance split:** a small stacked bar of event / model / systematic, so the source of uncertainty is visible.
- **Status strip:** NN or baseline, model version, cadence, ankle link, ankle battery (0 = "unavailable", not flat), hip OK, inference ms, last update age.
- **Live plot:** the last 10 s, scrolling: ankle gz, hip vertical accel, `p_right` / `p_left`, with committed events marked (NN as filled dots, baseline as hollow rings).
- **Timeline:** steps per 5 s over the last 2 minutes, NN vs baseline.
- **Session controls:** start/stop session, enter the true step count and measured distance, notes, "this was a distance calibration walk" checkbox, mode (live/record), display (steps/distance), reset.
- **Mark** button: appends to `marks.csv` with a note.

## 6. README: workflow and data-collection protocol

Workflow: put the UNO Q in `record` mode → record sessions → `pull` → `label` → `train` → `export` → `upload` → switch back to `live`.

For every session: stand still 5 s, do the activity, stand still 5 s. Count every footfall of both feet with a phone tally-counter app, then enter it on the dashboard or with `truth`.

| Session | Repeats | About |
|---|---|---|
| Normal walk, with turns | 3 | 200 steps |
| Slow walk | 2 | 100 steps |
| Fast walk | 2 | 150 steps |
| Jog/run | 2 | 150 steps |
| Stairs up + down (optional) | 1 | count all |
| Walk–stop–walk, turning in place | 2 | 100 steps |
| Negatives (count 0): standing with fidgeting 2 min, sitting with leg movement 2 min, biking 3 min, tap gestures 1 min | 1 each | — |
| Distance calibration walk on a measured straight line (≥ 100 ft) | 2 | enter the distance, tick "calibration walk" |

That's about 2,500 labelled steps, in under an hour.

## 7. Do not

Add test files; use CDN scripts on the page; change the shared contract; train on windows split across train and held-out sessions.

---

## 8. Shared contract (identical in both briefs; only the section numbers differ)

### 8.1 HTTP API (UNO Q serves, laptop calls)

Stdlib `http.server` in a background thread on `net.port` (default 8765). JSON responses. Every request needs the header `X-Token: <net.token>`, otherwise 401.

- `GET /api/status` →
  `{"mode", "display", "steps", "steps_sigma", "steps_ci":[lo,hi], "dist_m", "dist_sigma_m", "units", "cadence_spm", "ankle_link":"ok|lost|searching", "ankle_batt_mv", "hip_ok", "model_version", "using":"nn|baseline", "infer_ms", "uptime_s", "session_id", "var_split":{"event","model","systematic"}}`
- `GET /api/events?since=<id>` → `{"events":[{"id", "t", "foot":"R|L", "conf", "src":"nn|baseline", "session_id", "deleted":bool}], "last_id"}`. Ids increase monotonically and are never reused. A gesture deletion is sent as the same id with `deleted: true`.
- `GET /api/stream?since=<t>` → the last ≤ 10 s at 25 Hz: `{"t":[...], "ankle_gz":[...], "hip_vert":[...], "p_right":[...], "p_left":[...]}`. `t` is host monotonic seconds.
- `POST /api/session/start` → `{"session_id"}`; `POST /api/session/stop`
- `POST /api/session/truth` `{"session_id", "steps":int, "distance_in":float|null, "is_calib_walk":bool, "notes":str}`
- `POST /api/mode` `{"mode":"live|record"}`; `POST /api/display` `{"show":"steps|distance"}`; `POST /api/reset`
- `GET /api/files` → `{"sessions":[{"session_id", "start", "stop", "bytes", "has_truth", "recording":bool}]}`
- `GET /api/files/<session_id>/<file>` → the raw file
- `POST /api/model` (body: `model.npz` bytes) → `{"ok", "checks":{...}, "active_version"}`

### 8.2 Session folder (`sessions/<session_id>/`, id = `YYYYMMDD-HHMMSS`)

- `fused.npy`: float32, shape (N, 13); column 0 = host time (s), columns 1–12 = the channels in 8.3 order, at 100 Hz
- `raw_ankle.bin`, `raw_hip.bin`: each record is `float64 host_recv_time; uint16 len; packet bytes`
- `events.csv`: `id,t,foot,conf,src,deleted`
- `gestures.csv`: `t_first_tap,t_last_tap,kind`
- `meta.json`: `{"session_id", "start", "stop", "leg", "mode", "model_version", "truth":{"steps", "distance_in", "is_calib_walk", "notes"}|null, "hip_cal":{...}, "gravity":{...}}`

### 8.3 The 12 network input channels (100 Hz)

1–6. ankle ax, ay, az (g); gx, gy, gz (°/s ÷ 500). Right-leg frame: mirror left-leg data with the existing `leg:` logic.
7. ankle |a| (g)
8. hip vertical accel (g, gravity removed): projection onto a running gravity estimate (low-pass of hip accel, τ = 3 s, started from the setup gravity)
9. hip horizontal accel magnitude (g)
10. hip |a| − 1 (g)
11. ankle_present (0/1)
12. hip_present (0/1)

Normalisation (per-channel mean and std) comes from `model.npz`. `prep_hash` = SHA-256 of this exact text, from "1–6." through "12. hip_present (0/1)" with whitespace collapsed. Both sides hard-code that text as a string constant, compute the hash at runtime, and compare.

### 8.4 StepTCN architecture and `model.npz`

- `Conv1d(12→32, k=5, d=1)` + ReLU
- 4 residual blocks: `x + ReLU(Conv1d(32→32, k=5, d))` for d = 2, 4, 8, 16
- `Conv1d(32→2, k=1)` → logits for (right, left); sigmoid gives probabilities
- "same" zero padding; no batch-norm; dropout only during training
- Receptive field 125 samples. About 22,600 parameters.

`model.npz` keys:
- `version` (str), `prep_hash` (str), `n_members` (int)
- `m{i}_c0_w` (32,12,5), `m{i}_c0_b` (32); `m{i}_r{j}_w` (32,32,5), `m{i}_r{j}_b` (32) for j = 0..3; `m{i}_out_w` (2,32,1), `m{i}_out_b` (2). Weight layout matches PyTorch `Conv1d.weight` (out, in, k).
- `ch_mean` (12), `ch_std` (12), `thr` (float), `kappa` (float), `r` (float)
- `weinberg_K` (float, optional), `r_L` (float, optional), `L0_m` (float)
- `ref_x` (12, 400) normalised input; `ref_y` (n_members, 2, 400) PyTorch sigmoid output

### 8.5 Baseline detector (same algorithm on both sides)

- Right heel strike: the first local minimum of ankle gz after a mid-swing gz peak above `gz_thr` (default 100 °/s), within 0.35 s of that peak. Auto-detect the gz sign: if the dominant walking peaks are negative, flip it.
- Left heel strike: the largest hip-vertical peak with prominence ≥ `hip_prom` (default 0.15 g) between two consecutive right strikes. If there's none, use the midpoint.
- Minimum 0.30 s between same-foot events.
