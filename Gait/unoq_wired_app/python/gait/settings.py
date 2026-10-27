"""setup.yaml: the human-edited options for setup (see the comments in that file)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .errors import ConfigError

DEFAULTS = {
    "leg": "right",
    "model_a": {"axis": "y", "use_abs": False},
    "model_b": {"top_n": 30},
    "record_s": 120,
    "countdown_s": 10,
    "stand_s": 10,
    "trim_s": 5,
    "gesture_check_count": 5,
    "gesture_check_timeout_s": 90,
    "holdout_record_s": 120,
    "cv_folds": 5,
    "max_windows_per_class": 200,
    "default_model": "A",
    "switch_mode": "gesture",
    "unsure_enabled": False,
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
    raise ConfigError("reading setup.yaml", problem, fix)


def validate_settings(s: dict) -> dict:
    unknown = set(s) - set(DEFAULTS)
    if unknown:
        _fail(f"unknown keys {sorted(unknown)}", "check the spelling against the comments in setup.yaml")
    if s["leg"] not in ("right", "left"):
        _fail(f"leg is {s['leg']!r}", "set leg: right or leg: left")
    if s["model_a"].get("axis") not in ("x", "y", "z"):
        _fail(f"model_a.axis is {s['model_a'].get('axis')!r}", "use x, y or z (y = up the shin)")
    if not isinstance(s["model_a"].get("use_abs"), bool):
        _fail("model_a.use_abs is not true/false", "write use_abs: false or use_abs: true")
    top_n = s["model_b"].get("top_n")
    if top_n is not None and (not isinstance(top_n, int) or top_n < 1):
        _fail(f"model_b.top_n is {top_n!r}", "use a positive whole number, or null to keep all features")
    for key in ("record_s", "countdown_s", "stand_s", "gesture_check_count", "gesture_check_timeout_s",
                "holdout_record_s", "cv_folds", "max_windows_per_class"):
        if not isinstance(s[key], (int, float)) or isinstance(s[key], bool) or s[key] <= 0:
            _fail(f"{key} is {s[key]!r}", f"set {key} to a positive number")
    if not isinstance(s["trim_s"], (int, float)) or s["trim_s"] < 0:
        _fail(f"trim_s is {s['trim_s']!r}", "set trim_s to 0 or more seconds")
    if s["record_s"] <= 2 * s["trim_s"] + 2.56:
        _fail(f"record_s={s['record_s']} leaves no data after trimming {s['trim_s']} s from each end",
              "make record_s much longer than 2 x trim_s (120 s is the default)")
    if s["cv_folds"] < 2:
        _fail("cv_folds must be at least 2", "set cv_folds: 5")
    if s["default_model"] not in ("A", "B"):
        _fail(f"default_model is {s['default_model']!r}", "use A or B")
    if s["switch_mode"] not in ("gesture", "auto_alternate"):
        _fail(f"switch_mode is {s['switch_mode']!r}", "use gesture or auto_alternate")
    for key in ("unsure_enabled", "log_raw"):
        if not isinstance(s[key], bool):
            _fail(f"{key} is not true/false", f"write {key}: true or {key}: false")
    return s


def load_settings(path: Path) -> dict:
    """setup.yaml merged over the defaults. A missing file means all defaults."""
    path = Path(path)
    if not path.exists():
        return validate_settings(copy.deepcopy(DEFAULTS))
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        _fail(f"YAML syntax error: {e}", "fix the indentation/colons near the line shown, or restore setup.yaml from git")
    if not isinstance(raw, dict):
        _fail("the file is not a key: value mapping", "restore setup.yaml from git")
    return validate_settings(_merge(DEFAULTS, raw))
