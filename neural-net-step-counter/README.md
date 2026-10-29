# Neural-net step counter: laptop side

This half trains **StepTCN**, a small neural network that finds right and left heel strikes, and uploads it to the UNO Q. It also runs a live web dashboard. The UNO Q counts steps on its own, so the dashboard is only for watching, controlling sessions and logging.

The two sides only talk through the shared contract (section 8 of `BRIEF_LAPTOP_STEPCOUNTER.md`): the HTTP API, the session folder format, the 12 input channels and `model.npz`.

## Setup

```
pip install -r requirements.txt
```

Edit `config.yaml`: set `unoq_url` and `token` (it must match `net.token` on the UNO Q). Then check the connection:

```
python stepnet.py diag
```

`diag` checks that the UNO Q is reachable, that it accepts the right token and refuses a wrong one, and the round-trip time. It also runs the baseline detector on synthetic gait and checks the numpy forward pass against PyTorch on the local `model.npz`.

## Workflow

1. Put the UNO Q in **record** mode (the dashboard's Mode menu, or `POST /api/mode`).
2. Record the sessions in the protocol below, and enter each true count (dashboard or `truth`).
3. `python stepnet.py pull` downloads every finished session to `data/sessions/`. It resumes interrupted downloads and skips the session that's still recording.
4. `python stepnet.py label` auto-labels the strike times from the counts. Check `reports/label_<id>.html`.
5. `python stepnet.py train` trains the 5-member ensemble, tunes the threshold and calibrates the uncertainty. It writes `reports/index.html`.
6. `python stepnet.py export` writes `models/model.npz`, but only if numpy reproduces PyTorch to within 1e-4.
7. `python stepnet.py upload` sends the model to the UNO Q and prints its checks.
8. Switch the UNO Q back to **live** mode.

Live view: `python stepnet.py dash`, then open http://localhost:8080.

## Commands

| Command | What it does |
|---|---|
| `pull [--no-raw]` | Download new sessions. Sessions already pulled only get `meta.json` refreshed, so a truth entered later still arrives. |
| `truth <id> <steps> [--distance-in X] [--calib-walk] [--notes "..."]` | Send the ground truth to the UNO Q and update the local `meta.json`. |
| `label [--include-review]` | Count-constrained labelling (see below). |
| `train [--include-review] [--epochs 30] [--steps 50] [--seed 0]` | Train, calibrate and write the reports. |
| `export` | Write `models/model.npz` after the numpy vs PyTorch check. |
| `upload [--file X.npz]` | POST the model to `/api/model`. |
| `dash` | Run the dashboard on port 8080. |
| `diag` | Connectivity and self-checks. Exits with code 1 if any check fails. |

## How it works

**Labels from counts.** Hand-labelling strike times isn't practical, but tallying steps while walking is. `label` runs the baseline detector over a grid of its two thresholds (`gz_thr` 60–200 °/s, `hip_prom` 0.05–0.40 g). It keeps the setting whose right + left total is closest to your tally. If the best setting is still more than 2% (or 3 steps) off, the session is marked `needs_review` and left out of training. Sessions with a true count of 0 are pure negatives: the network learns that biking and fidgeting are not steps.

**Network.** StepTCN is a 1-D dilated convolutional network with dilations 1, 2, 4, 8 and 16. It has about 22,600 parameters and sees 1.25 s of context. It outputs `p_right` and `p_left` for every sample. A step is a local maximum ≥ `thr`, at least 0.30 s after the previous step of the same foot. `model.py` is the reference implementation, and `stepnet_np.py` is the numpy copy that matches the UNO Q.

**Honest evaluation.** The data is split **by session**, so no walk contributes windows to both training and testing. Each of the 5 ensemble members trains on a bootstrap resample of the training sessions. Augmentation covers mounting angle, gain, noise, clock skew between hip and ankle, a dropped sensor and cadence.

**Uncertainty.** σ_N² = κ²·(Σc(1−c) + member variance) + (r·N)². This adds three parts: the uncertainty of each detected event, the disagreement between ensemble members, and an error that grows with the count. κ is the smallest inflation that makes 95% of the held-out sessions fall inside the 95% interval. With only a handful of held-out sessions this is approximate, and the report says so.

**Distance.** Without calibration walks, distance = steps × `L0_m` (0.7303 m). With calibration walks, each step's length follows Weinberg: K·(hip bounce)^¼, with K fitted on those walks.

## Data-collection protocol

For every session, stand still for 5 s, do the activity, then stand still for 5 s. Count every footfall of **both** feet with a phone tally-counter app. Enter the count on the dashboard or with `truth`. Put the activity in the notes ("run" or "jog" for running), because the held-out split is stratified on it.

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

## Files

```
stepnet.py         CLI
model.py           PyTorch StepTCN (reference architecture)
stepnet_np.py      numpy forward pass + peak picker (mirrors the UNO Q); `python stepnet_np.py models/model.npz` checks a model
data.py            sessions, windows, augmentation, the 12-channel text + prep_hash
labeling.py        count-constrained auto-labelling
baseline.py        baseline detector (contract 8.5), pure numpy; `python baseline.py` runs its self-check
calibration.py     r, kappa, coverage, Weinberg distance
train.py           training, threshold tuning, reports, export
unoq.py            API client (token, retries, resumable downloads)
dashboard/         Flask server + offline HTML page
data/              pulled sessions + live logs (gitignored)
models/            ensemble.pt (gitignored), model.npz
reports/           label and training reports for the write-up
```

Live logs go to `data/live_logs/<date>/`:
- `events.csv`: a deletion is logged as an extra row with `deleted=1`.
- `status.csv`: one row per poll, including unreachable ones.
- `marks.csv`: one row per Mark button press.
