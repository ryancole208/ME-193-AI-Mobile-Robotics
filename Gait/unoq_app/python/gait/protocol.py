"""Parsing of the GaitNode BLE payloads and gap detection on the IMU stream.

IMU packet (little-endian, packed; see the shared protocol in README.md):

    uint16 seq      +1 per packet, wraps at 65535
    uint32 t0_us    sensor time of the first sample, wraps at 2^32
    uint8  n        samples in this packet (1..10)
    int16  raw[n][6] ax, ay, az, gx, gy, gz (raw MPU-6050 counts)

The tracker turns packets into a continuous *sample index* (sample 0 = the
first sample received after start). Sensor time is never compared with the
UNO Q clock: the index is derived only from seq and t0_us.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    IMU_HEADER_BYTES,
    MAX_SAMPLES_PER_PACKET,
    SAMPLE_BYTES,
    SAMPLE_PERIOD_US,
    SEQ_MODULO,
    T0_MODULO,
)
from .errors import PacketError

_HEADER = struct.Struct("<HIB")
RECENT_PACKETS = 64  # remembered (seq -> t0_us) to tell duplicates and late packets from a restart


@dataclass
class ImuPacket:
    seq: int
    t0_us: int
    n: int
    raw: np.ndarray  # shape (n, 6), int16


def parse_imu_packet(data: bytes) -> ImuPacket:
    """Decode one IMU notification. Raises PacketError on any mismatch."""
    if len(data) < IMU_HEADER_BYTES:
        raise PacketError(
            "parsing an IMU notification",
            f"payload is {len(data)} bytes, shorter than the 7-byte header",
            "check that the Pico W sends the packed <HIB header (no padding) on the IMU characteristic",
        )
    seq, t0_us, n = _HEADER.unpack_from(data, 0)
    if not 1 <= n <= MAX_SAMPLES_PER_PACKET:
        raise PacketError(
            "parsing an IMU notification",
            f"sample count n={n} is outside 1..{MAX_SAMPLES_PER_PACKET}",
            "check the Pico W packet builder; n must be 1..10",
        )
    expected = IMU_HEADER_BYTES + SAMPLE_BYTES * n
    if len(data) != expected:
        raise PacketError(
            "parsing an IMU notification",
            f"payload is {len(data)} bytes but n={n} needs exactly {expected}",
            "if the payload is shorter, the MTU is too small for n (the Pico W must lower n); "
            "if it is longer, the header layout differs from the shared protocol",
        )
    raw = np.frombuffer(data, dtype="<i2", count=6 * n, offset=IMU_HEADER_BYTES).reshape(n, 6)
    return ImuPacket(seq=seq, t0_us=t0_us, n=n, raw=raw.astype(np.int16))


def parse_battery(data: bytes) -> int:
    """Battery notification: uint16 vsys_mV. 0 means "reading unavailable"."""
    if len(data) != 2:
        raise PacketError(
            "parsing a battery notification",
            f"payload is {len(data)} bytes, expected 2 (uint16 vsys_mV)",
            "check the Pico W battery characteristic format",
        )
    return struct.unpack("<H", data)[0]


def parse_info(data: bytes) -> dict:
    """Info characteristic: uint8 version, uint8 accel_range_g, uint16 gyro_range_dps, uint16 sample_rate_hz."""
    if len(data) != 6:
        raise PacketError(
            "parsing the Info characteristic",
            f"payload is {len(data)} bytes, expected 6",
            "check the Pico W Info characteristic layout (<BBHH)",
        )
    version, accel_g, gyro_dps, rate = struct.unpack("<BBHH", data)
    return {
        "protocol_version": version,
        "accel_range_g": accel_g,
        "gyro_range_dps": gyro_dps,
        "sample_rate_hz": rate,
    }


def build_imu_packet(seq: int, t0_us: int, raw: np.ndarray) -> bytes:
    """Inverse of parse_imu_packet (used by the fake node and the tests)."""
    raw = np.asarray(raw, dtype="<i2")
    n = raw.shape[0]
    return _HEADER.pack(seq % SEQ_MODULO, t0_us % T0_MODULO, n) + raw.tobytes()


# ---------------------------------------------------------------- gap tracking


@dataclass
class SampleChunk:
    """A run of consecutive samples starting at a stream sample index.

    reset=True means the stream restarted (e.g. Pico W reboot or a very long
    gap) and anything buffered before this chunk must be discarded.
    """

    start_index: int
    raw: np.ndarray  # (n, 6) int16
    reset: bool = False


@dataclass
class StreamStats:
    packets: int = 0
    samples: int = 0
    packets_lost_seq: int = 0     # packets missing according to seq
    samples_missing_t0: int = 0   # samples missing according to t0_us
    fifo_gaps: int = 0            # t0_us jumped although seq was continuous
    seq_gaps: int = 0             # number of seq jumps (each may lose several packets)
    duplicates: int = 0
    out_of_order: int = 0
    resyncs: int = 0
    t0_misaligned: int = 0        # t0 jump not a multiple of 10 ms
    events: list = field(default_factory=list)  # human-readable gap log (last 50)

    @property
    def loss_pct(self) -> float:
        expected = self.packets + self.packets_lost_seq
        return 100.0 * self.packets_lost_seq / expected if expected else 0.0

    def note(self, msg: str) -> None:
        self.events.append(msg)
        del self.events[:-50]


class StreamTracker:
    """Detects gaps from BOTH seq jumps and t0_us discontinuities.

    For each packet we predict the next seq (prev+1 mod 2^16) and the next
    t0_us (prev_t0 + prev_n * 10000 mod 2^32). Both differences are taken
    modulo their wrap so wraparound is not a gap.

    * seq jump, t0 jump      -> packets lost in the air
    * seq continuous, t0 jump -> FIFO overflow on the Pico W (it added the lost
                                 samples to its count before continuing)
    * same seq and t0 as a recent packet -> duplicate (ignored)
    * up to 64 packets back, t0 consistent -> late packet (ignored)
    * t0 went backwards otherwise          -> stream restarted (Pico W reset), resync
    """

    def __init__(self, max_gap_samples: int = 30_000):
        # Gaps longer than this (default 5 min) are treated as a restart rather
        # than as missing samples, so a bogus t0 cannot produce a huge index jump.
        self.max_gap_samples = max_gap_samples
        self.stats = StreamStats()
        self._prev_seq: int | None = None
        self._prev_t0: int | None = None
        self._prev_n = 0
        # Stream index of the sample after the last one emitted. It is kept
        # across reconnects: sensor time keeps running while the link is down,
        # so the first packet after a reconnect tells us how much was missed.
        self._next_index = 0
        self._recent: dict[int, int] = {}

    def push(self, pkt: ImuPacket) -> SampleChunk | None:
        st = self.stats
        if self._prev_seq is None:
            return self._accept(pkt, missing=0, reset=False)

        dseq = (pkt.seq - self._prev_seq) % SEQ_MODULO
        expected_t0 = (self._prev_t0 + self._prev_n * SAMPLE_PERIOD_US) % T0_MODULO
        dt = (pkt.t0_us - expected_t0) % T0_MODULO
        if dt >= T0_MODULO // 2:
            dt -= T0_MODULO  # negative: time went backwards

        if self._recent.get(pkt.seq) == pkt.t0_us:
            st.duplicates += 1  # a packet we already have (e.g. a RESEND reply)
            return None
        back = (self._prev_seq - pkt.seq) % SEQ_MODULO
        behind_us = (self._prev_t0 - pkt.t0_us) % T0_MODULO
        if 0 < back <= RECENT_PACKETS and dt < 0 and behind_us <= back * MAX_SAMPLES_PER_PACKET * SAMPLE_PERIOD_US:
            st.out_of_order += 1  # a lost packet arriving late (RESEND): its samples are already windowed past
            return None
        if dt < 0 or dt // SAMPLE_PERIOD_US > self.max_gap_samples:
            st.resyncs += 1
            st.note(f"resync: seq {self._prev_seq}->{pkt.seq}, t0 jump {dt} us (Pico W restart or very long gap)")
            self._prev_seq = None
            return self._accept(pkt, missing=0, reset=True)

        missing, rem = divmod(dt, SAMPLE_PERIOD_US)
        if rem:
            # t0 should always move in whole 10 ms steps; round and remember it
            st.t0_misaligned += 1
            if rem >= SAMPLE_PERIOD_US // 2:
                missing += 1
        lost_packets = dseq - 1
        if lost_packets > 0:
            st.seq_gaps += 1
            st.packets_lost_seq += lost_packets
            st.note(f"seq gap: {lost_packets} packet(s) lost before seq {pkt.seq}, {missing} samples missing (t0)")
            if missing == 0:
                st.note(f"warning: seq jumped by {dseq} but t0_us is continuous at seq {pkt.seq}")
        elif missing > 0:
            st.fifo_gaps += 1
            st.note(f"FIFO overflow gap: seq continuous at {pkt.seq} but t0 jumped {missing} samples")
        st.samples_missing_t0 += missing
        return self._accept(pkt, missing=missing, reset=False)

    def _accept(self, pkt: ImuPacket, missing: int, reset: bool) -> SampleChunk:
        st = self.stats
        st.packets += 1
        st.samples += pkt.n
        start = self._next_index + missing
        if reset and self._prev_t0 is not None:
            start += 1  # leave a one-sample hole so the windower sees the break
        self._prev_seq, self._prev_t0, self._prev_n = pkt.seq, pkt.t0_us, pkt.n
        if reset:
            self._recent.clear()
        self._recent[pkt.seq] = pkt.t0_us
        if len(self._recent) > RECENT_PACKETS:
            del self._recent[next(iter(self._recent))]
        self._next_index = start + pkt.n
        return SampleChunk(start_index=start, raw=pkt.raw, reset=reset)
