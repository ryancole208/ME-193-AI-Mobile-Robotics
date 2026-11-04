"""One-time guided setup -> config.json. Driven by the matrix and foot taps, resumable.

1. STAND STILL 10 s (worn normally). Stores the ankle gravity direction and the
   still noise of |a| and |g| (the gesture detector's "still before/after"
   test scales with it), and the hip's raw reading while worn (-> the starting
   gravity estimate for the hip vertical channel). If anything moves during
   the 10 s, the countdown restarts (at most 3 times).
2. HIP 6-FACE CALIBRATION (optional, recommended once). Take the UNO Q case off
   the belt and hold it still on each face as the matrix prompts: "+X" means
   the X axis points UP (it then reads +1 g; the others read 0 g). Each face is
   recorded for 3 s once the case is still and the right axis is up. A
   least-squares line per axis, counts = offset + scale * g over the 6 faces,
   gives the true zero-g offset and counts/g - this is what removes the
   ratiometric uncertainty of the GY-61 (its real supply may be 3.0-3.3 V).
   4 taps (with the ankle pod) skip it: nominal values are used.
3. GESTURE CHECK: do the 4-tap gesture up to 5 times within 90 s; each
   recognised one flashes the matrix. Near-misses are logged with the reason.

Progress is saved in setup_state.json after each part, so a restart (or a
dropped link) resumes where it stopped instead of starting over.
"""

from __future__ import annotations

import logging
import time

import numpy as np

import display as D
from hip_stream import wiring_problem
from settings import (
    CONFIG_JSON, CONFIG_SCHEMA, SETUP_STATE, nominal_hip_cal, read_json, write_json_atomic,
)

log = logging.getLogger("step.setup")

STAND_SAMPLES = 1000              # 10 s at 100 Hz
STAND_MAX_RESTARTS = 3
STILL_AMAG_STD = 0.05             # g; above this the "still" countdown restarts
FACES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
FACE_SAMPLES = 300                # 3 s
FACE_STILL_COUNTS = 50            # std of raw counts (~0.03 g) to call the hand-held case still
ANKLE_WAIT_S = 60                 # proceed hip-only if the ankle never shows up
GESTURE_TARGET = 5
GESTURE_TIMEOUT_S = 90


class SetupRunner:
    def __init__(self, settings: dict, display: D.Display, on_noise=None):
        self.settings = settings
        self.display = display
        self.on_noise = on_noise
        self.st = read_json(SETUP_STATE, {}) or {}
        self.start_t = time.monotonic()
        self.done = False
        self._reset_stand()
        self._face_buf: list = []
        self._face_recording = False
        self._still: list = []
        self._gest_t0 = None
        self._gest_found = 0
        self._gest_misses: list = []
        log.info("SETUP: " + ("resuming: " + ", ".join(k for k in self.st) if self.st else "starting from scratch"))
        if "stand" in self.st and self.on_noise:
            self.on_noise(self.st["stand"]["ankle"])

    # ------------------------------------------------------------ helpers
    def _save(self) -> None:
        write_json_atomic(SETUP_STATE, self.st)

    def _reset_stand(self) -> None:
        self._ank: list = []
        self._hip: list = []
        self._stand_restarts = getattr(self, "_stand_restarts", 0)

    @property
    def step(self) -> str:
        if "stand" not in self.st:
            return "stand"
        if "faces" not in self.st:
            return "faces"
        if "gestures" not in self.st:
            return "gestures"
        return "finish"

    # ------------------------------------------------------------ inputs
    def on_ankle(self, samples: np.ndarray) -> None:
        if self.step == "stand":
            self._ank.extend(samples.tolist())

    def on_hip(self, raw: np.ndarray) -> None:
        step = self.step
        if step == "stand":
            self._hip.extend(raw.tolist())
        elif step == "faces":
            self._face_input(raw)

    def on_gesture(self, ev) -> None:
        step = self.step
        if step == "faces" and ev.kind == "switch":
            log.info("SETUP: 4 taps -> hip 6-face calibration skipped, nominal ADXL335 values will be used")
            self.display.status(D.ST_FLASH1)
            self.st["faces"] = {"skipped": True}
            self._save()
        elif step == "gestures":
            if ev.kind == "switch":
                self._gest_found += 1
                self.display.status(D.ST_FLASH1)
                log.info(f"SETUP: gesture {self._gest_found}/{GESTURE_TARGET} recognised: {ev.describe()}")
            elif ev.kind == "near_miss":
                self._gest_misses.append(ev.reason)
                log.info(f"SETUP: gesture near-miss: {ev.describe()}")

    # ------------------------------------------------------------ steps
    def tick(self, now: float, ankle_ok: bool, hip_ok_data: bool) -> None:
        step = self.step
        if step == "stand":
            self._tick_stand(now, ankle_ok, hip_ok_data)
        elif step == "faces":
            self._tick_faces()
        elif step == "gestures":
            self._tick_gestures(now)
        elif step == "finish":
            self._finish()

    def _tick_stand(self, now: float, ankle_ok: bool, hip_ok_data: bool) -> None:
        waiting_for_ankle = not ankle_ok and now - self.start_t < ANKLE_WAIT_S
        if not hip_ok_data or waiting_for_ankle:
            self.display.status(D.ST_SEARCHING)
            self._ank, self._hip = [], []
            return
        n = len(self._hip)
        self.display.setup(D.STEP_STAND, max(0, int(np.ceil((STAND_SAMPLES - n) / 100))))
        if n < STAND_SAMPLES:
            return
        hip = np.asarray(self._hip[:STAND_SAMPLES], dtype=np.float64)
        problem = wiring_problem(hip.mean(axis=0))
        if problem:
            # never store floating-pin readings as the hip's standing gravity: wait for wiring
            log.error(f"SETUP: stand-still not recorded, hip sensor looks unwired: {problem}")
            self.display.status(D.ST_HIP_FAIL)
            self._ank, self._hip = [], []
            return
        ank =np.asarray(self._ank, dtype=np.float64) if len(self._ank) >= 500 else None
        amag_std = float(np.linalg.norm(ank[:, :3], axis=1).std()) if ank is not None else 0.0
        hip_std_g = float(np.linalg.norm(hip.std(axis=0)) / 1638.4)
        if (amag_std > STILL_AMAG_STD or hip_std_g > STILL_AMAG_STD) and self._stand_restarts < STAND_MAX_RESTARTS:
            self._stand_restarts += 1
            log.warning(f"SETUP: moved during stand-still (ankle |a| std {amag_std:.3f} g, hip {hip_std_g:.3f} g); "
                        f"restarting the countdown ({self._stand_restarts}/{STAND_MAX_RESTARTS})")
            self._ank, self._hip = [], []
            return
        if ank is not None:
            ankle = {"gravity_g": ank[:, :3].mean(axis=0).round(4).tolist(),
                     "amag_std_g": round(amag_std, 5),
                     "gmag_std_dps": round(float(np.linalg.norm(ank[:, 3:], axis=1).std()), 4)}
        else:
            log.warning("SETUP: no ankle data after 60 s - using default gesture noise levels; redo setup "
                        "later with the pod on for better gesture thresholds")
            ankle = {"gravity_g": None, "amag_std_g": 0.02, "gmag_std_dps": 2.0}
        self.st["stand"] = {"ankle": ankle, "hip_raw_mean": hip.mean(axis=0).round(2).tolist(),
                            "hip_raw_std": hip.std(axis=0).round(2).tolist()}
        self._save()
        if self.on_noise:
            self.on_noise(ankle)
        self.display.status(D.ST_FLASH3)
        log.info(f"SETUP 1/3 done: stand-still {self.st['stand']}")
        log.info("SETUP 2/3: hip 6-face calibration. Take the case off the belt and hold it still with the "
                 "shown axis pointing UP (+X, -X, +Y, -Y, +Z, -Z), 3 s each. 4 taps skip this step.")

    # faces --------------------------------------------------------
    def _face_input(self, raw: np.ndarray) -> None:
        done = self.st.setdefault("faces_partial", {})
        todo = [f for f in FACES if f not in done]
        if not todo:
            return
        face = todo[0]
        self._still.extend(raw.tolist())
        self._still = self._still[-100:]
        win = np.asarray(self._still, dtype=np.float64)
        if len(win) < 100:
            return
        still = float(win.std(axis=0).max()) < FACE_STILL_COUNTS
        g_nom = (win.mean(axis=0) - 8192.0) / 1638.4
        axis, sign = "XYZ".index(face[1]), (1 if face[0] == "+" else -1)
        right_way_up = int(np.argmax(np.abs(g_nom))) == axis and np.sign(g_nom[axis]) == sign \
            and abs(g_nom[axis]) > 0.6
        if not (still and right_way_up):
            if self._face_recording:
                log.info(f"SETUP: face {face} moved during recording - starting that face again")
            self._face_buf, self._face_recording = [], False
            return
        self._face_recording = True
        self._face_buf.extend(raw.tolist())
        if len(self._face_buf) >= FACE_SAMPLES:
            done[face] = np.asarray(self._face_buf[:FACE_SAMPLES]).mean(axis=0).round(2).tolist()
            log.info(f"SETUP: face {face} recorded: mean counts {done[face]}")
            self._face_buf, self._face_recording, self._still = [], False, []
            self.display.status(D.ST_FLASH1)
            self._save()

    def _tick_faces(self) -> None:
        done = self.st.get("faces_partial", {})
        todo = [f for f in FACES if f not in done]
        if todo:
            pct = int(100 * len(self._face_buf) / FACE_SAMPLES) if self._face_recording else -1
            self.display.setup(D.STEP_FACE_BASE + FACES.index(todo[0]), pct)
            return
        self.st["faces"] = solve_six_face(done)
        self._save()
        self.display.status(D.ST_FLASH3 if self.st["faces"].get("ok") else D.ST_HIP_FAIL)
        log.info(f"SETUP 2/3 done: hip calibration {self.st['faces']}")
        log.info(f"SETUP 3/3: gesture check - do the 4-tap gesture up to {GESTURE_TARGET} times "
                 f"(within {GESTURE_TIMEOUT_S} s); the matrix flashes for each one recognised")

    # gestures -----------------------------------------------------
    def _tick_gestures(self, now: float) -> None:
        if self._gest_t0 is None:
            self._gest_t0 = now
        self.display.setup(D.STEP_GESTURE_CHECK, self._gest_found)
        if self._gest_found >= GESTURE_TARGET or now - self._gest_t0 > GESTURE_TIMEOUT_S:
            self.st["gestures"] = {"recognised": self._gest_found, "target": GESTURE_TARGET,
                                   "near_misses": self._gest_misses[-10:]}
            self._save()
            if self._gest_found < 3:
                log.warning(f"SETUP: only {self._gest_found} gestures recognised. The defaults stay; tune "
                            "setup.yaml gestures (e.g. lower tap_g) using gesture_debug: true")

    def _finish(self) -> None:
        stand, faces = self.st["stand"], self.st["faces"]
        cal = faces["cal"] if faces.get("ok") else nominal_hip_cal()
        hip_g = ((np.asarray(stand["hip_raw_mean"]) - cal["offset"]) / np.asarray(cal["scale"])).round(4)
        cfg = {
            "schema": CONFIG_SCHEMA,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "leg": self.settings["leg"],
            "ankle": stand["ankle"],
            "hip": {"cal": cal, "gravity_g": hip_g.tolist(),
                    "noise_g": round(float(np.linalg.norm(stand["hip_raw_std"]) / np.mean(cal["scale"])), 5)},
            "gestures": self.settings["gestures"],
            "gesture_check": self.st["gestures"],
            "six_face": {k: v for k, v in faces.items() if k != "cal"},
        }
        write_json_atomic(CONFIG_JSON, cfg)
        SETUP_STATE.unlink(missing_ok=True)
        self.display.status(D.ST_CHECK)
        log.info(f"SETUP COMPLETE: wrote {CONFIG_JSON.name} (hip cal: {cal['method']}, hip gravity "
                 f"{hip_g.tolist()} g, |g| = {np.linalg.norm(hip_g):.3f})")
        self.done = True


def solve_six_face(faces: dict) -> dict:
    """counts_axis = offset + scale * g_axis, least squares over the 6 faces.
    For axis X the true g values on faces (+X, -X, +Y, -Y, +Z, -Z) are (1, -1, 0, 0, 0, 0)."""
    names = FACES
    R = np.asarray([faces[f] for f in names], dtype=np.float64)   # (6 faces, 3 axes)
    offset, scale, resid = [], [], []
    for ax in range(3):
        g = np.zeros(6)
        g[2 * ax], g[2 * ax + 1] = 1.0, -1.0
        A = np.column_stack([np.ones(6), g])
        coef, *_ = np.linalg.lstsq(A, R[:, ax], rcond=None)
        offset.append(float(coef[0]))
        scale.append(float(coef[1]))
        resid.append(float(np.sqrt(np.mean((A @ coef - R[:, ax]) ** 2))))
    ok = all(900 < s < 2500 for s in scale) and all(4000 < o < 12000 for o in offset)
    out = {"ok": ok, "faces": faces, "residual_counts": [round(r, 1) for r in resid],
           "cal": {"offset": [round(o, 2) for o in offset], "scale": [round(s, 2) for s in scale],
                   "method": "6face"}}
    if not ok:
        log.error(f"While solving the hip 6-face calibration: implausible result offset {offset}, scale {scale} "
                  "counts/g. Likely fix: the wrong face was up during a step, or the GY-61 is on 5 V. "
                  "Nominal values are used; redo setup to try again")
    return out
