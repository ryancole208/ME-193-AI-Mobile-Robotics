"""Sliding windows over the sample-index stream.

Samples arrive as chunks tagged with their stream index. Missing indices (BLE
loss, FIFO overflow) are stored as NaN rows, so every window covers exactly
`window` consecutive sample indices and we can measure how much of it is
missing. Windows start on a fixed hop grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    start_index: int
    data: np.ndarray       # (window, n_channels), NaN where a sample is missing
    missing_frac: float
    valid: bool            # missing_frac <= max_missing_frac

    @property
    def end_index(self) -> int:
        return self.start_index + self.data.shape[0]


class Windower:
    def __init__(self, window: int = 256, hop: int = 128, max_missing_frac: float = 0.05, n_channels: int = 6):
        self.window = window
        self.hop = hop
        self.max_missing_frac = max_missing_frac
        self.n_channels = n_channels
        self._buf = np.empty((0, n_channels))
        self._buf_start: int | None = None   # stream index of _buf[0]
        self._next_start: int | None = None  # start index of the next window to emit

    def reset(self) -> None:
        self._buf = np.empty((0, self.n_channels))
        self._buf_start = None
        self._next_start = None

    def push(self, start_index: int, data: np.ndarray, reset: bool = False) -> list[Window]:
        data = np.asarray(data, dtype=float)
        if reset:
            self.reset()
        if self._buf_start is None:
            self._buf = data.copy()
            self._buf_start = start_index
            self._next_start = start_index
        else:
            buf_end = self._buf_start + len(self._buf)
            gap = start_index - buf_end
            if gap < 0:
                # overlapping/old samples: keep only the new part
                data = data[-gap:]
                start_index = buf_end
                gap = 0
            if gap > self.window:
                # a gap longer than a window can never be inside a valid window
                self.reset()
                return self.push(start_index, data)
            if gap > 0:
                self._buf = np.vstack([self._buf, np.full((gap, self.n_channels), np.nan)])
            self._buf = np.vstack([self._buf, data])

        out = []
        while self._next_start + self.window <= self._buf_start + len(self._buf):
            off = self._next_start - self._buf_start
            w = self._buf[off: off + self.window]
            missing = float(np.isnan(w[:, 0]).mean())
            out.append(Window(self._next_start, w.copy(), missing, missing <= self.max_missing_frac))
            self._next_start += self.hop
        # drop samples no future window needs
        drop = self._next_start - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start += drop
        return out


def window_array(data: np.ndarray, window: int, hop: int, max_missing_frac: float, start_index: int = 0) -> list[Window]:
    """Window a whole recording at once (setup, replay, evaluate)."""
    w = Windower(window, hop, max_missing_frac, n_channels=data.shape[1])
    return w.push(start_index, data)


def fill_missing(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN rows (edges take the nearest value)."""
    x = np.array(x, dtype=float, copy=True)
    bad = np.isnan(x[:, 0])
    if not bad.any():
        return x
    if bad.all():
        raise ValueError("window has no samples")
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        x[bad, c] = np.interp(idx[bad], idx[~bad], x[~bad, c])
    return x


def samples_to_dense(indices: np.ndarray, values: np.ndarray) -> tuple[int, np.ndarray]:
    """Recording rows (sample_index, values) -> (start_index, dense array with NaN gaps)."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return 0, np.empty((0, values.shape[1]))
    order = np.argsort(indices, kind="stable")
    indices, values = indices[order], values[order]
    start = int(indices[0])
    dense = np.full((int(indices[-1]) - start + 1, values.shape[1]), np.nan)
    dense[indices - start] = values
    return start, dense
