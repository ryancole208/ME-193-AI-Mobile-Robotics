"""Packet parsing and gap detection for BOTH sensor streams.

Ankle (Pico W "GaitNode", BLE; unchanged protocol from the handoff/unoq_app):
    uint16 seq; uint32 t0_us; uint8 n (1..10); int16 raw[n][6]   (ax ay az gx gy gz counts)
Hip (this app's sketch, Bridge "hip_packet"; deliberately the same header):
    uint16 seq; uint32 t0_us; uint8 n (1..10); uint16 raw[n][3]  (14-bit ADC counts)

StreamTracker (copied from unoq_app/python/gait/protocol.py) turns packets
into a continuous SAMPLE INDEX using seq AND t0_us, both modulo their wrap:
  * seq jump + t0 jump        -> packets lost in transit
  * seq continuous + t0 jump  -> samples lost at the source (Pico FIFO overflow,
                                 or the sketch skipping slots after a stall)
  * t0 backwards              -> the source restarted: resync (reset=True)
The index times 10 ms is the SENSOR clock of each sample; timesync.py maps
that to host time. Sensor time is never compared with host time anywhere else.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

SAMPLE_PERIOD_US = 10_000
SEQ_MODULO = 1 << 16
T0_MODULO = 1 << 32
MAX_N = 10
HEADER = struct.Struct("<HIB")
HEADER_BYTES = 7
RECENT_PACKETS = 64

# Pico W Info characteristic + expected values (same as unoq_app)
PROTOCOL_VERSION = 1
MIN_MTU_FOR_FULL_PACKET = HEADER_BYTES + 12 * MAX_N + 3


class PacketError(ValueError):
    pass


@dataclass
class Packet:
    seq: int
    t0_us: int
    n: int
    raw: np.ndarray  # (n, channels)


def parse_packet(data: bytes, channels: int, dtype: str) -> Packet:
    """channels=6, dtype='<i2' for the ankle; channels=3, dtype='<u2' for the hip."""
    if len(data) < HEADER_BYTES:
        raise PacketError(f"payload is {len(data)} bytes, shorter than the 7-byte header")
    seq, t0_us, n = HEADER.unpack_from(data, 0)
    if not 1 <= n <= MAX_N:
        raise PacketError(f"sample count n={n} is outside 1..{MAX_N}")
    need = HEADER_BYTES + 2 * channels * n
    if len(data) != need:
        raise PacketError(f"payload is {len(data)} bytes but n={n} needs exactly {need}")
    raw = np.frombuffer(data, dtype=dtype, count=channels * n, offset=HEADER_BYTES).reshape(n, channels)
    return Packet(seq, t0_us, n, raw.copy())


def parse_ankle(data: bytes) -> Packet:
    return parse_packet(data, 6, "<i2")


def parse_hip(data: bytes) -> Packet:
    return parse_packet(data, 3, "<u2")


def parse_battery(data: bytes) -> int:
    """uint16 vsys_mV; 0 means 'reading unavailable'."""
    if len(data) != 2:
        raise PacketError(f"battery payload is {len(data)} bytes, expected 2")
    return struct.unpack("<H", data)[0]


def parse_info(data: bytes) -> dict:
    if len(data) != 6:
        raise PacketError(f"Info payload is {len(data)} bytes, expected 6")
    version, accel_g, gyro_dps, rate = struct.unpack("<BBHH", data)
    return {"protocol_version": version, "accel_range_g": accel_g, "gyro_range_dps": gyro_dps,
            "sample_rate_hz": rate}


@dataclass
class Chunk:
    """Consecutive samples starting at stream sample `start_index`.
    reset=True: the source restarted; earlier sample indices are not comparable."""
    start_index: int
    raw: np.ndarray
    reset: bool = False


@dataclass
class StreamStats:
    packets: int = 0
    samples: int = 0
    packets_lost_seq: int = 0
    samples_missing_t0: int = 0
    source_gaps: int = 0        # t0 jumped although seq was continuous
    seq_gaps: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    resyncs: int = 0
    t0_misaligned: int = 0
    events: list = field(default_factory=list)

    @property
    def gaps(self) -> int:
        return self.seq_gaps + self.source_gaps + self.resyncs

    @property
    def loss_pct(self) -> float:
        expected = self.packets + self.packets_lost_seq
        return 100.0 * self.packets_lost_seq / expected if expected else 0.0

    def note(self, msg: str) -> None:
        self.events.append(msg)
        del self.events[:-50]


class StreamTracker:
    def __init__(self, max_gap_samples: int = 30_000):
        self.max_gap_samples = max_gap_samples
        self.stats = StreamStats()
        self._prev_seq = None
        self._prev_t0 = None
        self._prev_n = 0
        self._next_index = 0
        self._recent: dict = {}

    def push(self, pkt: Packet) -> Chunk | None:
        st = self.stats
        if self._prev_seq is None:
            return self._accept(pkt, 0, reset=self._prev_t0 is not None)
        dseq = (pkt.seq - self._prev_seq) % SEQ_MODULO
        expected_t0 = (self._prev_t0 + self._prev_n * SAMPLE_PERIOD_US) % T0_MODULO
        dt = (pkt.t0_us - expected_t0) % T0_MODULO
        if dt >= T0_MODULO // 2:
            dt -= T0_MODULO
        if self._recent.get(pkt.seq) == pkt.t0_us:
            st.duplicates += 1
            return None
        back = (self._prev_seq - pkt.seq) % SEQ_MODULO
        behind_us = (self._prev_t0 - pkt.t0_us) % T0_MODULO
        if 0 < back <= RECENT_PACKETS and dt < 0 and behind_us <= back * MAX_N * SAMPLE_PERIOD_US:
            st.out_of_order += 1
            return None
        if dt < 0 or dt // SAMPLE_PERIOD_US > self.max_gap_samples:
            st.resyncs += 1
            st.note(f"resync: seq {self._prev_seq}->{pkt.seq}, t0 jump {dt} us (source restart or very long gap)")
            self._prev_seq = None
            return self._accept(pkt, 0, reset=True)
        missing, rem = divmod(dt, SAMPLE_PERIOD_US)
        if rem:
            st.t0_misaligned += 1
            if rem >= SAMPLE_PERIOD_US // 2:
                missing += 1
        if dseq - 1 > 0:
            st.seq_gaps += 1
            st.packets_lost_seq += dseq - 1
            st.note(f"seq gap: {dseq - 1} packet(s) lost before seq {pkt.seq}, {missing} samples missing (t0)")
        elif missing > 0:
            st.source_gaps += 1
            st.note(f"source gap: seq continuous at {pkt.seq} but t0 jumped {missing} samples")
        st.samples_missing_t0 += missing
        return self._accept(pkt, missing, reset=False)

    def _accept(self, pkt: Packet, missing: int, reset: bool) -> Chunk:
        st = self.stats
        st.packets += 1
        st.samples += pkt.n
        start = self._next_index + missing
        if reset:
            start += 1
            self._recent.clear()
        self._prev_seq, self._prev_t0, self._prev_n = pkt.seq, pkt.t0_us, pkt.n
        self._recent[pkt.seq] = pkt.t0_us
        if len(self._recent) > RECENT_PACKETS:
            del self._recent[next(iter(self._recent))]
        self._next_index = start + pkt.n
        return Chunk(start, pkt.raw, reset)
