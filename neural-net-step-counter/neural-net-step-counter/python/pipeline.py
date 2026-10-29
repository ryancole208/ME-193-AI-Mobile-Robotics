"""The engine: sensor packets in, counted steps / distance / matrix / API out.

Data flow (all in the main App Lab loop thread; callbacks only enqueue):

  ankle BLE packet -> AnkleLink (seq/t0 gaps) -> convert + leg mirror -+-> GestureDetector
                                                  ClockSync("ankle") <-+-> fusion.ankle buffer
  hip Bridge packet -> HipLink (seq/t0 gaps) -> calibrate to g --------+-> fusion.hip buffer
                                                  ClockSync("hip")   <-+
  every 0.5 s:
    Fusion.advance()  -> new 100 Hz grid rows (12 channels), 0.5 s behind real time
    StreamInfer.step  -> final p_right / p_left (+ per member) up to final_k
    NNCounter.update  -> committed NN events      BaselineCounter.update -> baseline events
    DistanceTracker   -> step lengths             Session (record mode) -> files
    Display + Shared  -> matrix, and the snapshots the API thread serves

Two sessions concepts are the same thing here: a SESSION is a counting period
(the count starts at 0); in record mode it is also written to sessions/<id>/.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

import display as D
from ble_client import AnkleLink, convert_ankle
from counter import BaselineCounter, EventLog, NNCounter
from diag import build_diag
from distance import DistanceTracker, format_distance, l0_from_walks
from fusion import Fusion, GridBuffer
from gestures import GestureDetector, GestureParams
from hip_stream import HipLink
from recorder import Session, rotate
from settings import (
    APP_DIR, CONFIG_JSON, DT, MODEL_NPZ, MODEL_PREV, SESSIONS_DIR, SETUP_STATE, StepError, backup_broken_config,
    nominal_hip_cal, read_json, validate_config, write_json_atomic,
)
from setup_mode import SetupRunner
from stepnet_np import StreamInfer, check_model
from timesync import ClockSync

log = logging.getLogger("step.engine")

STEP_S = 0.5
GESTURE_PAD_S = 0.5            # counted steps within 0.5 s of a tap gesture are deleted
REDO_SETUP_WINDOW_S = 30
CONFIRM_S = 5
STREAM_S = 10


class Engine:
    def __init__(self, settings: dict, display: D.Display, shared, evlog: EventLog, diag_mode: bool = False,
                 force_setup: bool = False):
        self.settings = settings
        self.display = display
        self.shared = shared
        self.evlog = evlog
        self.diag_mode = diag_mode
        self.leg = settings["leg"]
        self.units = settings["units"]
        self.start_t = time.monotonic()

        self.cfg = None if force_setup else self._load_config()
        self.in_setup = self.cfg is None
        cal = self.cfg["hip"]["cal"] if self.cfg else nominal_hip_cal()
        gravity0 = self.cfg["hip"]["gravity_g"] if self.cfg else [0.0, 0.0, 1.0]

        self.ankle = AnkleLink()
        self.hip = HipLink(cal)
        self.ank_sync = ClockSync("ankle")
        self.hip_sync = ClockSync("hip")
        self.fusion = Fusion(gravity0)
        self.grid = GridBuffer()
        self.gest = GestureDetector(GestureParams.from_settings(settings["gestures"],
                                                                self.cfg["ankle"] if self.cfg else None))
        self.gesture_log: list = []

        self.mode = settings["mode"]          # live | record (setup is self.in_setup)
        self.show = D.DM_STEPS
        self.session_id = None
        self.session_k0 = 0
        self.rec: Session | None = None

        self.model = None
        self.model_checks: dict = {}
        self.infer = None
        self.nn = None
        self.baseline = BaselineCounter(evlog)
        self.L0 = l0_from_walks(settings["calib_walks"])
        self.dist = {"nn": DistanceTracker(self.L0), "baseline": DistanceTracker(self.L0)}
        self._load_model_at_boot()

        self.exclusions: list = []            # (t0, t1) host times of tap gestures
        self.confirm = None                   # {"what": "setup"|"reset", "until": t}
        self.first_data_t = None
        self.setup = None
        self._last_step = 0.0
        self._last_save = 0.0
        self._last_rotate = 0.0
        self._last_rec_banner = 0.0
        self._last_ankle_x = 0.0
        self._last_hip_x = 0.0
        self._hip_was_ok = False
        self._last_diag_print = 0.0
        if self.in_setup:
            self._enter_setup(backup=False)
        else:
            self._check_leg()
            self.start_session()

    # ------------------------------------------------------------ config / model
    def _load_config(self):
        if not CONFIG_JSON.exists():
            log.info("no config.json yet: running the one-time setup")
            return None
        try:
            return validate_config(read_json(CONFIG_JSON))
        except (StepError, ValueError) as e:
            dst = backup_broken_config()
            log.error(f"{e} (kept as {dst.name}); running setup")
            self.display.status(D.ST_CFG_ERROR)
            return None

    def _check_leg(self) -> None:
        if self.cfg and self.cfg.get("leg") != self.leg:
            log.warning(f"setup.yaml says leg: {self.leg} but setup was done with leg: {self.cfg.get('leg')}. "
                        "Counting continues with setup.yaml's leg (where the pod physically is); redo setup "
                        "if you moved the pod")
            self.display.status(D.ST_LEG_MISMATCH)

    def _load_model_at_boot(self) -> None:
        model, self.model_checks = check_model(MODEL_NPZ)
        if model is None:
            log.warning("no usable model.npz: counting with the BASELINE detector (record sessions for training)")
            self.display.status(D.ST_NN_FAIL)
        else:
            self.set_model(model)

    def set_model(self, model) -> None:
        self.model = model
        self.infer = StreamInfer(model)
        self.grid.reset_streams(model.n_members)
        self.nn = NNCounter(model, self.evlog)
        self.dist["nn"] = DistanceTracker(self.L0, model.weinberg_K, model.r_L)
        if self.session_id:
            self.nn.new_session(self.session_id, max(self.session_k0, self.grid.end))
        if model.L0_m and abs(model.L0_m - self.L0) > 0.01:
            log.info(f"model.npz L0_m = {model.L0_m:.3f} m differs from setup.yaml calib_walks "
                     f"({self.L0:.3f} m); using setup.yaml")
        log.info(f"model {model.version} active: {model.n_members} members, thr {model.thr:.2f}, "
                 f"kappa {model.kappa:.2f}, r {model.r:.3f}, Weinberg K {model.weinberg_K}")

    def activate_uploaded(self, model, upload_path) -> None:
        """Called on the main thread after the API thread checked the upload."""
        if MODEL_NPZ.exists():
            os.replace(MODEL_NPZ, MODEL_PREV)
        os.replace(upload_path, MODEL_NPZ)
        self.set_model(model)
        if not self.in_setup:
            # The new network's count starts empty, so start a new session rather than
            # showing a count that jumps back to 0. It also keeps one model_version per
            # recorded session (meta.json), which the laptop's evaluation relies on.
            log.info(f"model {model.version} activated: starting a new session")
            self.display.banner("NEW NN")
            self.start_session()

    @property
    def using(self) -> str:
        return "nn" if self.nn is not None else "baseline"

    # ------------------------------------------------------------ sessions
    def _new_session_id(self) -> str:
        t = time.time()
        while True:
            sid = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
            if sid != self.session_id and not (SESSIONS_DIR / sid).exists():
                return sid
            t += 1

    def _grid_now(self) -> int:
        return self.grid.end if self.grid.base is not None else int((time.monotonic() - 0.5) / DT)

    def start_session(self) -> str:
        self.stop_recording()
        self.session_id = self._new_session_id()
        self.session_k0 = self._grid_now()
        if self.nn:
            self.nn.new_session(self.session_id, self.session_k0)
        self.baseline.new_session(self.session_id, self.session_k0)
        for d in self.dist.values():
            d.reset()
        if self.mode == "record":
            self.start_recording()
        log.info(f"session {self.session_id} started ({self.mode})")
        with self.shared.lock:
            self.shared.current_session = self.session_id if self.rec else None
        return self.session_id

    def start_recording(self) -> None:
        meta = {"session_id": self.session_id, "start": time.strftime("%Y-%m-%dT%H:%M:%S"), "stop": None,
                "leg": self.leg, "mode": self.mode, "model_version": self.model.version if self.model else None,
                "truth": None, "hip_cal": self.hip.cal,
                "gravity": {"hip_g": self.fusion.gravity.round(4).tolist(),
                            "ankle_g": (self.cfg or {}).get("ankle", {}).get("gravity_g")},
                "t0_mono": round(self.session_k0 * DT, 3)}
        self.rec = Session(self.session_id, meta, self.settings["log_raw"])
        with self.shared.lock:
            self.shared.current_session = self.session_id

    def stop_recording(self) -> None:
        if self.rec:
            self.rec.close()
            log.info(f"recording {self.rec.id} closed ({self.rec.fused.rows} fused rows)")
            self.rec = None
        with self.shared.lock:
            self.shared.current_session = None

    # ------------------------------------------------------------ setup
    def _enter_setup(self, backup: bool = True) -> None:
        if backup and CONFIG_JSON.exists():
            os.replace(CONFIG_JSON, APP_DIR / f"config.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
            SETUP_STATE.unlink(missing_ok=True)
        self.stop_recording()
        self.in_setup = True
        self.display.invalidate()

        def on_noise(ankle):
            self.gest.p.amag_noise_g, self.gest.p.gmag_noise_dps = ankle["amag_std_g"], ankle["gmag_std_dps"]

        self.setup = SetupRunner(self.settings, self.display, on_noise)

    def _finish_setup(self) -> None:
        self.cfg = validate_config(read_json(CONFIG_JSON))
        self.hip.set_cal(self.cfg["hip"]["cal"])
        self.fusion.set_gravity(self.cfg["hip"]["gravity_g"])
        self.fusion.hip.clear()                       # its values used the old calibration
        self.gest = GestureDetector(GestureParams.from_settings(self.settings["gestures"], self.cfg["ankle"]))
        self.in_setup = False
        self.setup = None
        self.display.invalidate()
        self.start_session()

    # ------------------------------------------------------------ inputs
    def on_ble(self, ev) -> None:
        if ev.kind == "imu" and self.rec:
            self.rec.add_raw("ankle", ev.t, ev.data)
        chunk = self.ankle.handle(ev)
        if chunk is None:
            return
        if chunk.reset:
            log.info("ankle stream restarted: resetting its clock sync")
            self.ank_sync.reset()
            self.fusion.ankle.clear()
        samples = convert_ankle(chunk.raw, self.leg)
        n = len(samples)
        self.ank_sync.add((chunk.start_index + n - 1) * DT, ev.t)
        self.fusion.ankle.add(chunk.start_index, samples)
        self.first_data_t = self.first_data_t or ev.t
        for gev in self.gest.push(chunk.start_index, samples, reset=chunk.reset):
            self.on_gesture(gev, ev.t)
        if self.in_setup:
            self.setup.on_ankle(samples)

    def on_hip_packet(self, data: bytes, t: float) -> None:
        if self.rec:
            self.rec.add_raw("hip", t, data)
        chunk = self.hip.handle_packet(data, t)
        if chunk is None:
            return
        if chunk.reset:
            log.info("hip stream restarted (sketch reset?): resetting its clock sync")
            self.hip_sync.reset()
            self.fusion.hip.clear()
        n = len(chunk.raw)
        self.hip_sync.add((chunk.start_index + n - 1) * DT, t)
        self.fusion.hip.add(chunk.start_index, self.hip.to_g(chunk.raw))
        self.first_data_t = self.first_data_t or t
        if self.in_setup:
            self.setup.on_hip(chunk.raw)

    def on_hip_status(self, args, t: float) -> None:
        self.hip.handle_status(*args, t=t)

    # ------------------------------------------------------------ gestures
    def excluded(self, t: float) -> bool:
        return any(a <= t <= b for a, b in self.exclusions)

    def on_gesture(self, gev, now: float) -> None:
        dbg = self.settings["gestures"]["gesture_debug"]
        desc = gev.describe()
        self.gesture_log = (self.gesture_log + [desc])[-5:]
        if gev.kind == "near_miss":
            if dbg:
                log.info(f"gesture rejected: {desc}")
            if self.in_setup:
                self.setup.on_gesture(gev)
            return
        log.info(f"gesture: {desc}")
        if self.in_setup:
            self.setup.on_gesture(gev)
            return
        t_first = float(self.ank_sync.to_host(gev.taps[0].index * DT))
        t_last = float(self.ank_sync.to_host(gev.taps[-1].index * DT))
        self._exclude(t_first - GESTURE_PAD_S, t_last + GESTURE_PAD_S, gev.kind)
        if self.confirm and gev.kind == "switch" and now <= self.confirm["until"]:
            what, self.confirm = self.confirm["what"], None
            self.display.invalidate()
            self.display.status(D.ST_FLASH3)
            if what == "setup":
                log.info("redo setup confirmed")
                self._enter_setup()
            else:
                log.info("reset confirmed: new session")
                self.display.banner("RESET")
                self.start_session()
            return
        if gev.locked_out:
            return
        if gev.kind == "switch":
            self.show = D.DM_DIST if self.show == D.DM_STEPS else D.DM_STEPS
            self.display.banner("DIST" if self.show == D.DM_DIST else "STEP")
            self.display.invalidate()
        elif gev.kind == "long":
            early = self.first_data_t is not None and now - self.first_data_t <= REDO_SETUP_WINDOW_S
            self.confirm = {"what": "setup" if early else "reset", "until": now + CONFIRM_S}
            log.info(f"8 taps: do 4 more within {CONFIRM_S} s to confirm {'REDO SETUP' if early else 'RESET'}")

    def _exclude(self, t0: float, t1: float, kind: str) -> None:
        self.exclusions = [(a, b) for a, b in self.exclusions if b > t0 - 120] + [(t0, t1)]
        if self.rec:
            self.rec.add_gesture(t0 + GESTURE_PAD_S, t1 - GESTURE_PAD_S, kind)
        removed = []
        for src, counter in (("nn", self.nn), ("baseline", self.baseline)):
            if counter is None:
                continue
            for ev in counter.delete_window(t0, t1):
                ev.deleted = True
                self.evlog.publish_deletion(ev)
                self.dist[src].remove(ev)
                if self.rec:
                    self.rec.add_event(ev)
                removed.append(f"{src}:{ev.foot}@{ev.t:.2f}")
        log.info(f"tap gesture {t0:.2f}-{t1:.2f} s: deleted {len(removed)} counted step(s) {removed}")

    # ------------------------------------------------------------ commands from the API
    def _commands(self) -> None:
        q = self.shared.commands
        while not q.empty():
            name, payload, reply = q.get_nowait()
            try:
                reply.put(self._command(name, payload))
            except Exception as e:
                log.error(f"While handling API command {name}: {type(e).__name__}: {e}")
                reply.put({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def _command(self, name: str, p) -> dict:
        if self.in_setup and name not in ("activate_model", "display"):
            return {"ok": False, "error": "setup is running; finish it first"}
        if name == "session_start":
            return {"ok": True, "session_id": self.start_session()}
        if name == "session_stop":
            sid = self.rec.id if self.rec else None
            self.stop_recording()
            return {"ok": True, "session_id": sid or self.session_id, "was_recording": sid is not None}
        if name == "reset":
            self.display.banner("RESET")
            return {"ok": True, "session_id": self.start_session()}
        if name == "mode":
            self.mode = p
            if p == "record" and not self.rec:
                self.start_session()
            elif p == "live":
                self.stop_recording()
            log.info(f"mode set to {p} over the API (setup.yaml still decides the mode at the next start)")
            return {"ok": True, "mode": p, "session_id": self.session_id}
        if name == "display":
            self.show = D.DM_DIST if p == "distance" else D.DM_STEPS
            self.display.invalidate()
            return {"ok": True, "show": p}
        if name == "truth":
            return self._truth(p)
        if name == "activate_model":
            model, path = p
            self.activate_uploaded(model, path)
            return {"ok": True, "active_version": model.version}
        return {"ok": False, "error": f"unknown command {name}"}

    def _truth(self, p: dict) -> dict:
        sid = p.get("session_id") or self.session_id
        truth = {"steps": p["steps"], "distance_in": p.get("distance_in"),
                 "is_calib_walk": bool(p.get("is_calib_walk", False)), "notes": str(p.get("notes", ""))}
        if self.rec and self.rec.id == sid:
            self.rec.meta["truth"] = truth
            self.rec.write_meta()
        else:
            path = SESSIONS_DIR / sid / "meta.json"
            meta = read_json(path) if (SESSIONS_DIR / sid).is_dir() else None
            if meta is None:
                return {"ok": False, "error": f"session {sid} was not recorded (use mode record to keep data)"}
            meta["truth"] = truth
            write_json_atomic(path, meta)
        log.info(f"truth for session {sid}: {truth}")
        return {"ok": True, "session_id": sid, "truth": truth}

    # ------------------------------------------------------------ the 0.5 s step
    def tick(self, now: float) -> None:
        self._commands()
        if self.confirm and now > self.confirm["until"]:
            log.info("confirmation timed out: nothing changed")
            self.confirm = None
            self.display.invalidate()
        if now - self._last_step < STEP_S:
            return
        self._last_step = now
        self._step(now)
        if now - self._last_save > 5:
            self._last_save = now
            self.evlog.save()
        if now - self._last_rotate > 60:
            self._last_rotate = now
            rotate(self.rec.id if self.rec else None)

    def _step(self, now: float) -> None:
        hip_ok = self.hip.ok(now)
        out = self.fusion.advance(now, self.ank_sync, self.hip_sync, hip_enabled=hip_ok)
        if out is not None:
            k0, ch = out
            self.grid.append(k0, ch)
            if self.rec:
                self.rec.add_rows(np.arange(k0, k0 + len(ch)) * DT, ch)

        if self.in_setup:
            self.setup.tick(now, self.ankle.link_state(now) == "ok",
                            self.hip.last_data_t is not None and now - self.hip.last_data_t < 1.0)
            if self.setup.done:
                self._finish_setup()
        else:
            new = []
            if self.infer is not None:
                self.infer.step(self.grid)
                new += self.nn.update(self.grid, self.infer.final_k, self.excluded)
            new += self.baseline.update(self.grid, self.excluded)
            for ev in sorted(new, key=lambda e: e.id):
                self.evlog.publish(ev)
                self.dist[ev.src].add(ev, self.grid)
                if self.rec:
                    self.rec.add_event(ev)
            self._update_display(now, hip_ok)
        self._publish(now, hip_ok)

    def summaries(self):
        if self.nn is not None:
            s = self.nn.summary()
        else:
            s = self.baseline.summary()
        d = self.dist[self.using].summary(s["sigma"])
        return s, d

    def _update_display(self, now: float, hip_ok: bool) -> None:
        dsp = self.display
        ankle = self.ankle.link_state(now)
        if hip_ok:
            self._hip_was_ok = True
        if self.confirm:
            dsp.setup(D.STEP_SET_CONFIRM if self.confirm["what"] == "setup" else D.STEP_RESET_CONFIRM, 0)
            return
        if self.first_data_t is None:
            dsp.status(D.ST_SEARCHING)
            return
        if ankle != "ok" and not hip_ok:
            dsp.status(D.ST_LINK_LOST)
            return
        s, d = self.summaries()
        text = str(s["N"]) if self.show == D.DM_STEPS else format_distance(d["dist_m"], self.units)
        dsp.value(text, self.show)
        # periodic reminders (status screens take priority over the number for a second)
        if self.rec and now - self._last_rec_banner > 10:
            self._last_rec_banner = now
            dsp.status(D.ST_REC)
        if ankle == "lost" and now - self._last_ankle_x > 10:
            self._last_ankle_x = now
            dsp.status(D.ST_ANKLE_LOST)
        if not hip_ok and self.hip.last_data_t is not None and now - self._last_hip_x > 30:
            self._last_hip_x = now
            dsp.status(D.ST_HIP_FAIL)

    # ------------------------------------------------------------ snapshots for the API
    def _publish(self, now: float, hip_ok: bool) -> None:
        # every contract field is always present (zeros during setup), so the laptop never KeyErrors
        s, d = self.summaries()
        k_now = self.grid.end
        recent = [e for e in (self.nn.events if self.nn else self.baseline.events) if e.k >= k_now - 1000]
        status = {"mode": "setup" if self.in_setup else self.mode,
                  "display": "steps" if self.show == D.DM_STEPS else "distance",
                  "steps": s["N"], "steps_sigma": round(s["sigma"], 3), "steps_ci": s["ci"],
                  "dist_m": round(d["dist_m"], 3), "dist_sigma_m": round(d["dist_sigma_m"], 3),
                  "units": self.units, "cadence_spm": round(len(recent) * 6.0, 1),
                  "var_split": {k: round(v, 3) for k, v in s["var_split"].items()}}
        if self.in_setup:
            status["setup_step"] = self.setup.step if self.setup else None
        status.update({
            "ankle_link": self.ankle.link_state(now), "ankle_batt_mv": self.ankle.battery_mV,
            "hip_ok": hip_ok, "model_version": self.model.version if self.model else None,
            "using": self.using, "infer_ms": round(self.infer.last_ms, 1) if self.infer else None,
            "uptime_s": round(now - self.start_t, 1), "session_id": self.session_id,
        })
        stream = self._stream()
        diag = build_diag(self, now)
        with self.shared.lock:
            self.shared.status = status
            self.shared.stream = stream
            self.shared.diag = diag["data"]
        if self.diag_mode and now - self._last_diag_print >= 2:
            self._last_diag_print = now
            log.info("DIAG\n" + diag["text"])

    def _stream(self) -> dict:
        if self.grid.base is None:
            return {"t": [], "ankle_gz": [], "hip_vert": [], "p_right": [], "p_left": []}
        end = self.grid.end
        k0, ch, p, _ = self.grid.rows(end - STREAM_S * 100, end)
        first = (-k0) % 4                     # 25 Hz: every 4th row, aligned to k % 4 == 0
        sl = slice(first, None, 4)
        ks = np.arange(k0, k0 + len(ch))[sl]
        return {"t": np.round(ks * DT, 2).tolist(),
                "ankle_gz": np.round(ch[sl, 5] * 500.0, 1).tolist(),
                "hip_vert": np.round(ch[sl, 7], 4).tolist(),
                "p_right": [None if not np.isfinite(v) else round(float(v), 4) for v in p[sl, 0]],
                "p_left": [None if not np.isfinite(v) else round(float(v), 4) for v in p[sl, 1]]}

    def shutdown(self) -> None:
        self.stop_recording()
        self.evlog.save()
