"""Session recording for training data (contract 9.2).

sessions/<YYYYMMDD-HHMMSS>/
  fused.npy      float32 (N, 13): column 0 = host time (s), 1-12 = the channels
                 of fusion.build_channels(), 100 Hz. Appended every 60 s; the
                 .npy header is rewritten in place with the new row count, so
                 the file is a valid .npy after every flush (and if the app dies).
  raw_ankle.bin, raw_hip.bin   records: float64 host_recv_time, uint16 len, packet bytes
  events.csv     id,t,foot,conf,src,deleted   (a deletion is a second row, same id, deleted=1)
  gestures.csv   t_first_tap,t_last_tap,kind
  meta.json      session_id, start, stop, leg, mode, model_version, truth, hip_cal, gravity
                 (+ t0_mono: host monotonic time at start, to line up with fused.npy)

Rotation: when sessions/ exceeds 2 GB, the oldest sessions are deleted first,
never the one being recorded.
"""

from __future__ import annotations

import ast
import logging
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from settings import SESSIONS_DIR, read_json, write_json_atomic

log = logging.getLogger("step.rec")

FLUSH_S = 60.0
ROTATE_BYTES = 2 * 1024 ** 3
HEADER_LEN = 128                 # fixed .npy header size, so it can be rewritten in place
RAW_REC = struct.Struct("<dH")


class NpyAppender:
    """An .npy file that grows: fixed-size header rewritten with the new shape."""

    def __init__(self, path: Path, ncols: int):
        self.path, self.ncols, self.rows = path, ncols, 0
        if path.exists():                                    # resume after a restart
            with open(path, "rb") as f:
                f.seek(10)
                hdr = ast.literal_eval(f.read(HEADER_LEN - 10).decode("latin1").strip())
                self.rows = int(hdr["shape"][0])
        else:
            with open(path, "wb") as f:
                f.write(self._header(0))

    def _header(self, rows: int) -> bytes:
        d = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (rows, self.ncols)
        body = d.encode("latin1")
        pad = HEADER_LEN - 10 - len(body) - 1
        return b"\x93NUMPY\x01\x00" + struct.pack("<H", HEADER_LEN - 10) + body + b" " * pad + b"\n"

    def append(self, arr: np.ndarray) -> None:
        if not len(arr):
            return
        arr = np.ascontiguousarray(arr, dtype="<f4")
        with open(self.path, "r+b") as f:
            f.seek(HEADER_LEN + self.rows * self.ncols * 4)
            f.write(arr.tobytes())
            f.truncate()
            self.rows += len(arr)
            f.seek(0)
            f.write(self._header(self.rows))


class Session:
    def __init__(self, session_id: str, meta: dict, log_raw: bool):
        self.id = session_id
        self.dir = SESSIONS_DIR / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.log_raw = log_raw
        self.fused = NpyAppender(self.dir / "fused.npy", 13)
        self._pending: list = []
        self._last_flush = time.monotonic()
        self._raw = {}
        for name in ("ankle", "hip"):
            self._raw[name] = open(self.dir / f"raw_{name}.bin", "ab") if log_raw else None
        self._events = open(self.dir / "events.csv", "a", buffering=1)
        if self._events.tell() == 0:
            self._events.write("id,t,foot,conf,src,deleted\n")
        self._gestures = open(self.dir / "gestures.csv", "a", buffering=1)
        if self._gestures.tell() == 0:
            self._gestures.write("t_first_tap,t_last_tap,kind\n")
        self.write_meta()

    def write_meta(self) -> None:
        write_json_atomic(self.dir / "meta.json", self.meta)

    def add_rows(self, t: np.ndarray, ch: np.ndarray) -> None:
        self._pending.append(np.column_stack([t, ch]).astype(np.float32))
        if time.monotonic() - self._last_flush >= FLUSH_S:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            self.fused.append(np.concatenate(self._pending))
            self._pending = []
        for f in self._raw.values():
            if f:
                f.flush()
        self._last_flush = time.monotonic()

    def add_raw(self, name: str, t: float, data: bytes) -> None:
        f = self._raw.get(name)
        if f:
            f.write(RAW_REC.pack(t, len(data)) + data)

    def add_event(self, ev) -> None:
        self._events.write(f"{ev.id},{ev.t:.3f},{ev.foot},{ev.conf:.4f},{ev.src},{int(ev.deleted)}\n")

    def add_gesture(self, t_first: float, t_last: float, kind: str) -> None:
        self._gestures.write(f"{t_first:.3f},{t_last:.3f},{kind}\n")

    def close(self) -> None:
        self.flush()
        for f in list(self._raw.values()) + [self._events, self._gestures]:
            if f:
                f.close()
        self.meta["stop"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.write_meta()


def session_dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file())


def list_sessions(current_id: str | None) -> list:
    out = []
    if not SESSIONS_DIR.exists():
        return out
    for d in sorted(SESSIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = read_json(d / "meta.json", {}) or {}
        out.append({"session_id": d.name, "start": meta.get("start"), "stop": meta.get("stop"),
                    "bytes": session_dir_bytes(d), "has_truth": bool(meta.get("truth")),
                    "recording": d.name == current_id})
    return out


def rotate(current_id: str | None) -> None:
    if not SESSIONS_DIR.exists():
        return
    dirs = sorted(d for d in SESSIONS_DIR.iterdir() if d.is_dir())    # ids sort by time
    sizes = {d: session_dir_bytes(d) for d in dirs}
    total = sum(sizes.values())
    for d in dirs:
        if total <= ROTATE_BYTES:
            break
        if d.name == current_id:
            continue
        log.warning(f"sessions/ is {total / 1e9:.2f} GB > 2 GB: deleting the oldest session {d.name}")
        shutil.rmtree(d, ignore_errors=True)
        total -= sizes[d]
