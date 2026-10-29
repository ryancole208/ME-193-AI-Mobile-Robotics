"""Files, settings and loud errors.

* setup.yaml  - human-edited settings (committed). Validated against DEFAULTS.
* config.json - written by the one-time setup (setup_mode.py): gravity, noise,
                hip calibration, gesture thresholds. Missing/invalid -> setup runs.
* state.json  - tiny runtime state that must survive restarts (the event-id
                counter, so ids are never reused; see counter.EventLog).

Every deliberate error says what it was doing, what went wrong and the most
likely fix, so a pasted log line is enough to diagnose it.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import yaml

APP_DIR = Path(os.environ.get("STEPCOUNT_APP_DIR", Path(__file__).resolve().parents[1]))
SETUP_YAML = APP_DIR / "setup.yaml"
CONFIG_JSON = APP_DIR / "config.json"
SETUP_STATE = APP_DIR / "setup_state.json"
STATE_JSON = APP_DIR / "state.json"
MODEL_NPZ = APP_DIR / "model.npz"
MODEL_PREV = APP_DIR / "model.prev.npz"
MODEL_UPLOAD = APP_DIR / "model.upload.npz"
RUN_MODE = APP_DIR / "RUN_MODE"          # one-shot: "diag" or "setup", deleted when read
SESSIONS_DIR = APP_DIR / "sessions"
LOGS_DIR = APP_DIR / "logs"
APP_YAML = APP_DIR / "app.yaml"

CONFIG_SCHEMA = 1
FS = 100                    # Hz, every stream and the fused grid
DT = 1.0 / FS


class StepError(Exception):
    def __init__(self, doing: str, problem: str, fix: str):
        self.doing, self.problem, self.fix = doing, problem, fix
        super().__init__(f"While {doing}: {problem}. Likely fix: {fix}")


# ---------------------------------------------------------------- setup.yaml
DEFAULTS = {
    "leg": "right",
    "mode": "live",
    "units": "imperial",
    "net": {"mode": "pull", "port": 8765, "token": "change-me", "laptop_url": "http://192.168.1.50:8080"},
    "calib_walks": [{"steps": 4, "distance_in": 115}],
    "gestures": {"tap_g": 1.6, "max_tap_ms": 150, "gap_min_ms": 120, "gap_max_ms": 600,
                 "still_noise_mult": 6, "gesture_debug": False},
    "log_raw": True,
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _fail(problem: str, fix: str):
    raise StepError("reading setup.yaml", problem, fix)


def validate_settings(s: dict) -> dict:
    unknown = set(s) - set(DEFAULTS)
    if unknown:
        _fail(f"unknown keys {sorted(unknown)}", "check the spelling against the comments in setup.yaml")
    if s["leg"] not in ("right", "left"):
        _fail(f"leg is {s['leg']!r}", "set leg: right or leg: left")
    if s["mode"] not in ("live", "record"):
        _fail(f"mode is {s['mode']!r}", "set mode: live or mode: record")
    if s["units"] not in ("imperial", "metric"):
        _fail(f"units is {s['units']!r}", "set units: imperial or units: metric")
    net = s["net"]
    if net.get("mode") not in ("pull", "push"):
        _fail(f"net.mode is {net.get('mode')!r}", "use pull (push is not needed: App Lab publishes the port)")
    if not isinstance(net.get("port"), int) or not 1 <= net["port"] <= 65535:
        _fail(f"net.port is {net.get('port')!r}", "use a port number such as 8765 (and list it under ports: in app.yaml)")
    if not isinstance(net.get("token"), str) or not net["token"]:
        _fail("net.token is empty", 'set a token string, e.g. token: "my-secret"')
    walks = s["calib_walks"]
    if not isinstance(walks, list) or not walks:
        _fail("calib_walks is empty", "keep at least the hallway walk: - {steps: 4, distance_in: 115}")
    for w in walks:
        if not (isinstance(w, dict) and isinstance(w.get("steps"), int) and w["steps"] > 0
                and isinstance(w.get("distance_in"), (int, float)) and w["distance_in"] > 0):
            _fail(f"calib_walks entry {w!r} is invalid", "each entry needs steps (> 0) and distance_in (> 0)")
    g = s["gestures"]
    for key in ("tap_g", "max_tap_ms", "gap_min_ms", "gap_max_ms", "still_noise_mult"):
        if not isinstance(g.get(key), (int, float)) or isinstance(g.get(key), bool) or g[key] <= 0:
            _fail(f"gestures.{key} is {g.get(key)!r}", f"set gestures.{key} to a positive number")
    if g["gap_min_ms"] >= g["gap_max_ms"]:
        _fail("gestures.gap_min_ms must be below gap_max_ms", "use e.g. 120 and 600")
    for key in ("log_raw",):
        if not isinstance(s[key], bool):
            _fail(f"{key} is not true/false", f"write {key}: true or {key}: false")
    if not isinstance(g.get("gesture_debug"), bool):
        _fail("gestures.gesture_debug is not true/false", "write gesture_debug: true or false")
    return s


def load_settings(path: Path = SETUP_YAML) -> dict:
    """setup.yaml merged over the defaults (a missing file means all defaults)."""
    if not Path(path).exists():
        return validate_settings(copy.deepcopy(DEFAULTS))
    try:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    except yaml.YAMLError as e:
        _fail(f"YAML syntax error: {e}", "fix the indentation/colons near the line shown")
    if not isinstance(raw, dict):
        _fail("the file is not a key: value mapping", "restore setup.yaml from git")
    return validate_settings(_merge(DEFAULTS, raw))


def app_yaml_ports() -> list:
    """Ports App Lab publishes (for the self-check: they must include net.port)."""
    try:
        return list((yaml.safe_load(APP_YAML.read_text()) or {}).get("ports") or [])
    except Exception:
        return []


# ---------------------------------------------------------------- JSON helpers
def write_json_atomic(path: Path, obj) -> None:
    """Write to a temp file and rename, so a crash never leaves half a file."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


# ---------------------------------------------------------------- config.json
def _vec(x, n=3) -> bool:
    return isinstance(x, list) and len(x) == n and all(isinstance(v, (int, float)) for v in x)


def validate_config(cfg: dict) -> dict:
    def bad(problem):
        raise StepError("loading config.json", problem, "redo setup (8 taps + 4 to confirm within 30 s of start, "
                        "or `echo setup > RUN_MODE` and restart)")

    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        bad(f"schema is {cfg.get('schema') if isinstance(cfg, dict) else None!r}, expected {CONFIG_SCHEMA}")
    for key in ("leg", "ankle", "hip", "gestures"):
        if key not in cfg:
            bad(f"key {key!r} is missing")
    a, h = cfg["ankle"], cfg["hip"]
    if not (_vec(a.get("gravity_g")) or a.get("gravity_g") is None):
        bad("ankle.gravity_g is not a 3-vector")
    for key in ("amag_std_g", "gmag_std_dps"):
        if not isinstance(a.get(key), (int, float)) or a[key] < 0:
            bad(f"ankle.{key} is invalid")
    cal = h.get("cal", {})
    if not (_vec(cal.get("offset")) and _vec(cal.get("scale")) and all(v > 0 for v in cal["scale"])):
        bad("hip.cal needs offset and positive scale 3-vectors")
    if not _vec(h.get("gravity_g")):
        bad("hip.gravity_g is not a 3-vector")
    return cfg


def nominal_hip_cal() -> dict:
    """ADXL335 nominal: zero-g = Vs/2, 0.330 V/g at Vs = 3.3 V, ratiometric.
    With the ADC referenced to 3.3 V and Vs = 3.3 V: offset 8192 counts,
    scale 0.330 / 3.3 * 16384 = 1638.4 counts/g. A GY-61 with its own regulator
    (Vs ~3.0 V) is off by ~10 %, which is why setup offers the 6-face calibration."""
    return {"offset": [8192.0] * 3, "scale": [1638.4] * 3, "method": "nominal"}


def backup_broken_config() -> Path:
    dst = APP_DIR / f"config.broken-{time.strftime('%Y%m%d-%H%M%S')}.json"
    os.replace(CONFIG_JSON, dst)
    return dst
