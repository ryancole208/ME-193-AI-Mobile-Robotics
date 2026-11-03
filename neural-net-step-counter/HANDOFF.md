# Handoff: NN step counter, laptop side (state as of 2026-11-02, ~21:45)

Read this first, then `README.md` in this folder. The working directory is
`neural-net-step-counter/` (the laptop half: train / export / upload / dashboard).

## Ground rules from the user

- **Run Python with the repo-root venv:** `..\my_env\Scripts\python.exe` (or activate
  `..\my_env\Scripts\Activate.ps1`). System Python lacks Flask/torch.
- **Never send commands that can reach the live UNO Q** (`diag`, `upload`, `pull`, `truth`,
  dashboard without `--demo`). The user runs those. Test offline (fake client, `--demo`, scratchpad paths).
- **`neural-net-step-counter/neural-net-step-counter/` is a read-only copy of the UNO Q app.**
  Never edit it. If the UNO Q needs a change, write instructions for the user's separate UNO Q session.
- **Don't run long training jobs in the background without asking.** The user prefers to run
  `python stepnet.py train` in their own terminal so they can see it. Never write experiments
  into the real `models/` or `reports/`.
- **Walking only.** The user removed running from scope.
- Don't delete session data without asking; move it under `data/excluded/` instead.

## Facts

- UNO Q: `http://10.247.137.33:8765`, token `RQ-D2` (in `config.yaml`). `unoq.local` doesn't resolve.
- The UNO Q only accepts width-32 models: its upload check hard-codes `HIDDEN = 32`.
- UNO Q inference already takes 77 ms median / 93 ms p95 / 144 ms max against a 100 ms budget
  (`data/live_logs/2026-11-02/status.csv`, `infer_ms`). Wider networks or more ensemble members
  would push it over.
- On upload the UNO Q checks the file, renames the old one to `model.prev.npz`, activates the new
  one immediately (no restart) and starts a new session (count resets to 0). A failed check keeps
  the old model.
- `pull` downloads anything not already in `data/sessions/`.

## Data layout now

- `data/sessions/`: 64 sessions (54 walks, 10 negatives); 58 are used for training, and 6 are
  flagged `needs_review` and skipped automatically.
- `data/excluded/no_truth/`: 33 sessions with no true count (one, `20261029-212950`, also has an
  unreadable `fused.npy`).
- `data/excluded/run/`: 2 run sessions (`20261103-010008`, `-010127`).
- `data/excluded/20261029-195030`, `-195432`, `-195552`: **stale duplicates**. The copies in
  `data/sessions/` label fine and are used. `data/excluded/20261102-235914` also exists, with a
  re-pulled duplicate in `data/sessions/` (flagged needs_review). The user agreed to delete these
  4 duplicates (the 3 stale excluded copies + the `sessions/` copy of 235914), but the permission
  check blocked the delete; the user may run it themselves:
  `Remove-Item -Recurse -Force data\sessions\20261102-235914, data\excluded\20261029-195030, data\excluded\20261029-195432, data\excluded\20261029-195552`

## Code changes this session (all uncommitted)

- `stepnet.py`: `pull` skips any session ID found anywhere under `data/excluded/` (new
  `paths.excluded`, default `data/excluded`) and reports how many it skipped. `train --width N` added.
- `data.py`: `Session.category` is now just `walk` / `negative` (the run category and its
  >140 steps/min cadence rule are gone; that rule was mislabelling short walks).
  `list_sessions` skips unreadable `fused.npy` (user's own earlier change).
  Sensor-drop augmentation `p_drop` is back at its original 0.15 (5% was tried and was worse).
- `labeling.py`: `GZ_SWEEP` widened to 60–400 deg/s (user's own change).
- `model.py`: `StepTCN(width=32)` parameter; `WIDTH = 32` is the only UNO Q-compatible value.
- `train.py`: width threaded through, saved in `ensemble.pt` / `train_summary.json`, shown in the
  report and network diagram; `export` refuses a checkpoint whose width isn't 32. Report/console no
  longer have a running target (walking <= 2% only).
- `README.md`: running removed from the protocol, `gz_thr` range corrected, `--width` documented.

## Model results and experiments

Current model: `steptcn-20261102-213823` (width 32, 5 members, 15% drop, seed 0).
`models/model.npz` is exported and passes all offline checks. Held-out walks: **2.93% pooled**,
16.03% mean per session, precision 0.807 / recall 0.785, thr 0.84, 3 false steps on negatives.
The user considers ~3% acceptable.

Tried / ruled out:
- **More data of the same kind:** not the bottleneck. The net undercounts its own training walks
  (e.g. 82->51, 120->86, 136->105), so it's not a data-quantity problem.
- **Width 48:** started, then stopped at the user's request before it finished. Not evaluated.
  Would also need a UNO Q code change and would blow the time budget.
- **More ensemble members (10/15):** rejected. Averaging reduces variance, not the shared
  undercount bias, and it multiplies UNO Q inference time.
- **Removing sessions under 10 steps:** the user declined.
- **Lighter augmentation (`p_drop` 0.15 -> 0.05):** pooled 0.53% but only by cancellation; total
  absolute miss rose 37 -> 44 of 375 and per-session error got worse. Reverted.

Most likely remaining cause of the undercount: label quality. Several undercounted walks had
ankle BLE dropouts that still passed the labeller, or needed unusual labeller thresholds.

Why sessions get flagged `needs_review`: mostly **ankle BLE dropouts** (`ANK_PRESENT` < 1;
6 of 8 flagged sessions lost the ankle for 22–73% of the recording), so the labeller misses steps
and the user's tally is higher than the labelled count. Fixing the Gait node's BLE link (Pico W,
`Gait/pico/`) would help both labelling and live counting.

## Where things stand / next step

As of 2026-11-03 ~09:00 the upload worked: `status.csv` shows `model_version =
steptcn-20261102-213823`, `using = nn`. The UNO Q rebooted twice that morning (08:50, 08:56).

Dashboard fixes (2026-11-03, uncommitted):
- **Graph froze after a UNO Q reboot.** Stream `t` is the UNO Q's host monotonic clock, which resets
  on reboot; the server kept asking `/api/stream?since=<old t>`, got nothing back, and never noticed.
  Now an empty reply triggers a full re-fetch, and any backwards clock clears the stream (and the
  in-memory events from the old clock). Tested offline with a fake client that reboots.
- Faster updates: server polls events/stream every 0.5 s (was 1 s; the UNO Q only republishes every
  0.5 s, `STEP_S`), page polls every 0.25 s (was 0.4), history every 1 s (was 2). The 10 s plot
  scrolls smoothly at frame rate, 0.75 s behind the newest sample.
- "Reset count" button in the Count card (`POST /api/reset`: new session, count 0).

Calibration (2026-11-03): `calibration.fit_ml` (maximum likelihood, kappa and r together) replaced
`fit_r` + `fit_kappa` in `train`, and `stepnet.py recalibrate` applies it to a saved model without
retraining. Ran it: kappa 2.24 -> 1.88, r 0.210 -> 0.061, version `steptcn-20261102-213823-ml`,
`models/model.npz` re-exported (old files kept as `*.before-recal.*`). Leave-one-out on the 12
held-out sessions: same coverage (11/12), intervals about half as wide. **Not uploaded yet**: the user
runs `python stepnet.py upload` (resets the count/session on the UNO Q).

Next: the user restarts `python stepnet.py dash` and checks the plot live, then does a ~100-step
hand-counted walk to sanity-check the model.
