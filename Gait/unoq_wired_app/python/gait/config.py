"""config_wired.json: the ONLY thing live mode needs.

Setup writes it once; every later start loads and validates it. Validation is
strict on purpose: a config that loads but is subtly wrong would make the
models quietly misbehave, which is worse than a clear "CFG X" on the matrix.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

from .constants import CLASSES, SCHEMA_VERSION, SOURCE
from .errors import ConfigError
from .features import MODEL_B_FEATURES, _MODEL_A_RE

K_CHOICES = (1, 3, 5, 7, 9, 11)

DEFAULT_GESTURE = {
    "tap_g": 2.0,
    "max_tap_ms": 60,
    "min_gap_ms": 150,
    "max_gap_ms": 450,
    "taps_switch": 4,
    "taps_long": 8,
    "still_noise_mult": 3,
    "lockout_s": 3,
}

DOING = "validating config_wired.json"
REDO = "redo setup (delete or rename config_wired.json and restart the app)"


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validation_problems(cfg) -> list[str]:
    """Every problem found, as short sentences. Empty list = valid."""
    p: list[str] = []
    if not isinstance(cfg, dict):
        return ["top level is not a JSON object"]
    if cfg.get("schema_version") != SCHEMA_VERSION:
        p.append(f"schema_version is {cfg.get('schema_version')!r}, this app needs {SCHEMA_VERSION}")
    if cfg.get("source") != SOURCE:
        p.append(f"source is {cfg.get('source')!r}, expected {SOURCE!r} (not a config from a wired setup)")
    if cfg.get("classes") != CLASSES:
        p.append(f"classes is {cfg.get('classes')!r}, expected {CLASSES}")
    for block in ("signal", "model_a", "model_b", "display", "gesture"):
        if not isinstance(cfg.get(block), dict):
            p.append(f"missing block {block!r}")
    if p:
        return p

    s = cfg["signal"]
    for key in ("sample_rate_hz", "window_s", "hop_s", "max_missing_frac", "accel_lsb_per_g", "gyro_lsb_per_dps"):
        if not _num(s.get(key)) or s.get(key) <= 0:
            p.append(f"signal.{key} must be a positive number, got {s.get(key)!r}")
    if not p:
        win = s["window_s"] * s["sample_rate_hz"]
        hop = s["hop_s"] * s["sample_rate_hz"]
        if abs(win - round(win)) > 1e-6 or abs(hop - round(hop)) > 1e-6:
            p.append("signal.window_s and hop_s must be whole numbers of samples")
        if s["max_missing_frac"] >= 1:
            p.append("signal.max_missing_frac must be below 1")
    if s.get("leg") not in ("right", "left"):
        p.append(f"signal.leg is {s.get('leg')!r}")
    am = s.get("axis_map")
    if not isinstance(am, dict) or set(am) != {"x", "y", "z"}:
        p.append("signal.axis_map must have keys x, y, z")
    g = s.get("gravity_unit_vector")
    if not (isinstance(g, list) and len(g) == 3 and all(_num(v) for v in g)) or abs(math.hypot(*g) - 1) > 0.05:
        p.append(f"signal.gravity_unit_vector must be 3 numbers of length 1, got {g!r}")
    noise = s.get("noise", {})
    if not (isinstance(noise, dict) and _num(noise.get("amag_std_g")) and _num(noise.get("gmag_std_dps"))):
        p.append("signal.noise must hold amag_std_g and gmag_std_dps (from the stand-still step)")

    n_classes = len(CLASSES)
    for name in ("model_a", "model_b"):
        m = cfg[name]
        if m.get("type") != "knn":
            p.append(f"{name}.type must be 'knn'")
        if m.get("k") not in K_CHOICES:
            p.append(f"{name}.k must be one of {K_CHOICES}, got {m.get('k')!r}")
        if m.get("weights") != "distance":
            p.append(f"{name}.weights must be 'distance'")
        X, y = m.get("X"), m.get("y")
        if name == "model_a":
            if not (isinstance(m.get("feature"), str) and _MODEL_A_RE.match(m["feature"])):
                p.append(f"model_a.feature {m.get('feature')!r} is not like 'max_ay_g' or 'max_abs_ay_g'")
            dim = 1
        else:
            feats = m.get("features")
            if not (isinstance(feats, list) and feats and all(f in MODEL_B_FEATURES for f in feats)):
                p.append("model_b.features must be a non-empty list of known feature names")
                continue
            if len(set(feats)) != len(feats):
                p.append("model_b.features has duplicates")
            dim = len(feats)
        if not (isinstance(X, list) and isinstance(y, list) and len(X) == len(y) and len(X) > 0):
            p.append(f"{name}.X and {name}.y must be non-empty lists of equal length")
            continue
        if not all(isinstance(r, list) and len(r) == dim and all(_num(v) for v in r) for r in X):
            p.append(f"{name}.X rows must each hold {dim} finite numbers")
        if not all(isinstance(v, int) and 0 <= v < n_classes for v in y):
            p.append(f"{name}.y must hold class indices 0..{n_classes - 1}")
        else:
            missing = [CLASSES[c] for c in range(n_classes) if c not in y]
            if missing:
                p.append(f"{name} has no training examples for {missing}")
        sc = m.get("scaler", {})
        if not (isinstance(sc, dict) and isinstance(sc.get("mean"), list) and isinstance(sc.get("std"), list)
                and len(sc["mean"]) == dim and len(sc["std"]) == dim
                and all(_num(v) for v in sc["mean"]) and all(_num(v) and v > 0 for v in sc["std"])):
            p.append(f"{name}.scaler must hold {dim} means and {dim} positive stds")
        ud = m.get("unsure_distance")
        if ud is not None and not (_num(ud) and ud > 0):
            p.append(f"{name}.unsure_distance must be null or a positive number")

    d = cfg["display"]
    if d.get("default_model") not in ("A", "B"):
        p.append("display.default_model must be 'A' or 'B'")
    if d.get("switch_mode") not in ("gesture", "auto_alternate"):
        p.append("display.switch_mode must be 'gesture' or 'auto_alternate'")

    ge = cfg["gesture"]
    for key in DEFAULT_GESTURE:
        if not _num(ge.get(key)) or ge.get(key) <= 0:
            p.append(f"gesture.{key} must be a positive number, got {ge.get(key)!r}")
    if not p and ge["min_gap_ms"] >= ge["max_gap_ms"]:
        p.append("gesture.min_gap_ms must be below max_gap_ms")

    return p


def validate_config(cfg) -> dict:
    problems = validation_problems(cfg)
    if problems:
        raise ConfigError(DOING, "; ".join(problems), REDO)
    return cfg


def load_config(path: Path) -> dict:
    """Load and validate. Raises ConfigError (never returns a broken config)."""
    path = Path(path)
    if not path.exists():
        raise ConfigError("loading config_wired.json", f"{path} does not exist", "run the one-time setup (it starts automatically)")
    try:
        cfg = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError("loading config_wired.json", f"cannot parse {path.name}: {e}", REDO) from e
    return validate_config(cfg)


def backup_path(path: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    return path.with_name(f"config_wired.backup-{stamp}.json")


def backup_existing(path: Path) -> Path | None:
    """Rename an existing config_wired.json to config_wired.backup-<date>.json."""
    path = Path(path)
    if not path.exists():
        return None
    dest = backup_path(path)
    n = 1
    while dest.exists():
        dest = dest.with_name(dest.stem + f"-{n}.json")
        n += 1
    os.replace(path, dest)
    return dest


def save_config(cfg: dict, path: Path) -> Path | None:
    """Validate, back up any existing config, then write atomically. Returns the backup path."""
    validate_config(cfg)
    path = Path(path)
    backup = backup_existing(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=1))
    os.replace(tmp, path)
    return backup


# ------------------------------------------------------------ state.json (shown model)


def load_shown_model(path: Path, default: str) -> str:
    try:
        m = json.loads(Path(path).read_text()).get("shown_model")
        return m if m in ("A", "B") else default
    except (OSError, ValueError, AttributeError):
        return default


def save_shown_model(path: Path, model: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"shown_model": model}))
    os.replace(tmp, path)
