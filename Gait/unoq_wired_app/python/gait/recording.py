"""Recordings on disk: setup activities, held-out tests and raw live logs.

Format (CSV, one row per sample):

    # gait-recording v1 kind=activity class=walking leg=right created=2026-10-22T10:00:00
    index,ax,ay,az,gx,gy,gz
    0,12,2051,-33,4,-2,17
    ...

`index` is the stream sample index (100 Hz) from the packet tracker; a jump in
index is a gap (lost packets or an MPU-6050 FIFO overflow). The six values
are RAW MPU-6050 counts in sensor axes, exactly as received, so a recording can
be re-processed later with a different leg / axis map / unit conversion.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .errors import ConfigError

HEADER = ["index", "ax", "ay", "az", "gx", "gy", "gz"]
MAGIC = "# gait-recording v1"


@dataclass
class Recording:
    meta: dict
    index: np.ndarray            # (n,) int64 stream sample index
    raw: np.ndarray              # (n, 6) int16 counts
    path: Path | None = None

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def duration_s(self) -> float:
        return 0.0 if self.n == 0 else (int(self.index[-1]) - int(self.index[0]) + 1) / 100.0


@dataclass
class Recorder:
    """Accumulates SampleChunks (or (start, raw) pairs) into a Recording."""

    meta: dict = field(default_factory=dict)
    _idx: list = field(default_factory=list)
    _raw: list = field(default_factory=list)
    samples: int = 0

    def add(self, start_index: int, raw: np.ndarray) -> None:
        raw = np.asarray(raw, dtype=np.int16)
        self._idx.append(np.arange(start_index, start_index + len(raw), dtype=np.int64))
        self._raw.append(raw)
        self.samples += len(raw)

    def recording(self) -> Recording:
        if not self._idx:
            return Recording(dict(self.meta), np.empty(0, np.int64), np.empty((0, 6), np.int16))
        return Recording(dict(self.meta), np.concatenate(self._idx), np.vstack(self._raw))


def write_recording(path: Path, rec: Recording) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(rec.meta)
    meta.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        f.write(MAGIC + " " + " ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
        w = csv.writer(f)
        w.writerow(HEADER)
        for i, row in zip(rec.index.tolist(), rec.raw.tolist()):
            w.writerow([i, *row])
    tmp.replace(path)
    rec.path = path
    return path


class RecordingWriter:
    """Streams rows to a CSV as they arrive (live raw log), flushing every second."""

    def __init__(self, path: Path, meta: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", newline="")
        meta = dict(meta)
        meta.setdefault("created", datetime.now().isoformat(timespec="seconds"))
        self._f.write(MAGIC + " " + " ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
        self._w = csv.writer(self._f)
        self._w.writerow(HEADER)
        self._rows = 0

    def add(self, start_index: int, raw: np.ndarray) -> None:
        for k, row in enumerate(np.asarray(raw).tolist()):
            self._w.writerow([start_index + k, *row])
        self._rows += len(raw)
        if self._rows >= 100:
            self._f.flush()
            self._rows = 0

    def close(self) -> None:
        self._f.close()


def read_recording(path: Path) -> Recording:
    path = Path(path)
    doing = f"reading recording {path.name}"
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        raise ConfigError(doing, str(e), "check the path; recordings live in data/setup, data/holdout and logs/")
    meta = {}
    body = []
    for line in lines:
        if line.startswith("#"):
            if line.startswith(MAGIC):
                for tok in line[len(MAGIC):].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k] = v
            continue
        if line.strip():
            body.append(line)
    if not body or body[0].replace(" ", "") != ",".join(HEADER):
        raise ConfigError(doing, f"first data line is {body[0][:60] if body else '(empty)'!r}, expected {','.join(HEADER)}",
                          "the file is not a gait recording (or it is truncated); record it again")
    try:
        arr = np.loadtxt(body[1:], delimiter=",", dtype=np.int64, ndmin=2) if len(body) > 1 else np.empty((0, 7), np.int64)
    except ValueError as e:
        raise ConfigError(doing, f"bad number in the CSV: {e}", "the file was edited or truncated; record it again")
    if arr.shape[1] != 7:
        raise ConfigError(doing, f"rows have {arr.shape[1]} columns, expected 7", "record it again")
    return Recording(meta, arr[:, 0].astype(np.int64), arr[:, 1:].astype(np.int16), path)


def rotate_logs(folder: Path, keep: int = 20) -> None:
    """Delete the oldest files in logs/ so at most `keep` remain."""
    files = sorted((f for f in Path(folder).iterdir() if f.is_file()), key=lambda f: f.stat().st_mtime)
    for f in files[: max(0, len(files) - keep)]:
        try:
            f.unlink()
        except OSError:
            pass
