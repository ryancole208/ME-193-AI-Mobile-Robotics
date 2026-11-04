"""Hip accelerometer (GY-61 / ADXL335 on A0-A2) - packets from the sketch.

The sketch samples at 100 Hz on its own micros() schedule and sends
    Bridge.notify("hip_packet", bytes)   uint16 seq; uint32 t0_us; uint8 n; uint16 raw[n][3]
    Bridge.notify("hip_status", late, skipped, adc_errors, adc_bits, samples)   once per second
The packets go through the same StreamTracker as the ankle, so a stall on the
MCU (t0 jump) or a lost Bridge message (seq jump) shows up as a gap.

Conversion: g = (counts - offset) / scale per axis. offset/scale come from
setup's 6-face calibration in config.json, or the nominal ADXL335 values
(settings.nominal_hip_cal) if that step was skipped. The hip frame is whatever
way the board sits in the case: the network only uses orientation-free hip
channels (vertical / horizontal / |a|), see fusion.build_channels().

Plausibility ("hip_ok"): an unplugged or dead GY-61 makes the analog pins float
or sit at a rail, so the hip is treated as ABSENT (its present flag goes to 0
and counting continues from the ankle) while
  * any axis is outside 5-95 % of the ADC range, or
  * the 2 s mean of |a| is outside 0.6-1.6 g (gravity alone is 1 g; walking
    and running average close to it).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from ble_client import RateMeter
from protocol import PacketError, StreamTracker, parse_hip

log = logging.getLogger("step.hip")

ADC_FULL = 16383.0           # 14-bit
RAIL_LO, RAIL_HI = 0.05 * ADC_FULL, 0.95 * ADC_FULL
RANGE_LIMIT_G = 3.0          # ADXL335 is specified to at least +-3 g; beyond this it may clip
AMAG_OK = (0.6, 1.6)
# Wiring test that needs no calibration: an ADXL335 on a 3.0-3.3 V supply reads within
# zero-g +- 1 g on every axis in any orientation (about 5960-9830 counts). Unwired A0-A2
# float elsewhere (about 4000 counts on this board), so mean counts outside zero-g +- 1.5 g
# (nominal) mean "not connected" rather than "not calibrated".
WIRED_RANGE = (8192 - 1.5 * 1638.4, 8192 + 1.5 * 1638.4)


def wiring_problem(raw_mean) -> str:
    """'' if the mean raw counts (3 axes) look like a connected GY-61, else the reason."""
    raw_mean = np.asarray(raw_mean, dtype=np.float64)
    if np.all((raw_mean > WIRED_RANGE[0]) & (raw_mean < WIRED_RANGE[1])):
        return ""
    return (f"mean counts {np.round(raw_mean).astype(int).tolist()} are outside {int(WIRED_RANGE[0])}-"
            f"{int(WIRED_RANGE[1])}, which no powered ADXL335 reads (GY-61 not wired, or VCC/GND/X/Y/Z "
            "swapped? VCC must be 3.3 V, X/Y/Z on A0/A1/A2)")


class HipLink:
    def __init__(self, cal: dict):
        self.tracker = StreamTracker()
        self.set_cal(cal)
        self.first_data_t = None
        self.last_data_t = None
        self.rate = RateMeter()
        self.packet_errors = 0
        self.late = self.skipped = self.adc_errors = 0
        self.adc_bits = None
        self.status_t = None
        self.samples_total = 0
        self.samples_clipped = 0
        self.samples_rail = 0
        self._amag = deque(maxlen=200)     # last 2 s of |a|, for plausibility
        self._raw_ok = deque(maxlen=200)
        self.last_raw = None
        self._ok = False
        self.reason = "no data yet"

    @property
    def stats(self):
        return self.tracker.stats

    def set_cal(self, cal: dict) -> None:
        self.cal = cal
        self.offset = np.asarray(cal["offset"], dtype=np.float64)
        self.scale = np.asarray(cal["scale"], dtype=np.float64)

    def to_g(self, raw: np.ndarray) -> np.ndarray:
        return (np.asarray(raw, dtype=np.float64) - self.offset) / self.scale

    def handle_packet(self, data: bytes, t: float):
        """Returns a Chunk (raw counts) or None."""
        try:
            pkt = parse_hip(data)
        except PacketError as e:
            self.packet_errors += 1
            if self.packet_errors % 100 == 1:
                log.error(f"While parsing a hip packet: {e} (bad packets: {self.packet_errors}). "
                          "Likely fix: sketch and python/protocol.py disagree; restart the app to re-flash")
            return None
        before = len(self.stats.events)
        chunk = self.tracker.push(pkt)
        for msg in self.stats.events[before:]:
            log.info(f"hip stream: {msg}")
        if chunk is None:
            return None
        if self.first_data_t is None:
            self.first_data_t = t
            log.info("hip samples arriving from the sketch")
        self.last_data_t = t
        self.rate.add(t, pkt.n)
        self._check(chunk.raw)
        return chunk

    def handle_status(self, late, skipped, adc_errors, adc_bits, samples, t: float) -> None:
        late, skipped, adc_errors = int(late), int(skipped), int(adc_errors)
        if adc_errors > self.adc_errors:
            log.warning(f"hip ADC read errors: {adc_errors} total. Likely fix: check A0-A2 wiring")
        if skipped > self.skipped:
            log.warning(f"the sketch skipped {skipped - self.skipped} hip samples (loop stalled > 100 ms)")
        self.late, self.skipped, self.adc_errors = late, skipped, adc_errors
        self.adc_bits = int(adc_bits)
        self.status_t = t

    def _check(self, raw: np.ndarray) -> None:
        g = self.to_g(raw)
        self.last_raw = raw[-1].copy()
        rail = (raw < RAIL_LO) | (raw > RAIL_HI)
        self.samples_total += len(raw)
        self.samples_rail += int(rail.any(axis=1).sum())
        self.samples_clipped += int((np.abs(g) >= RANGE_LIMIT_G).any(axis=1).sum())
        self._amag.extend(np.linalg.norm(g, axis=1).tolist())
        self._raw_ok.extend((~rail.any(axis=1)).tolist())
        ok, reason = self._judge()
        if ok != self._ok:
            (log.info if ok else log.warning)(f"hip sensor {'OK' if ok else 'IMPLAUSIBLE'}: {reason}")
        self._ok, self.reason = ok, reason

    def _judge(self):
        if len(self._amag) < 50:
            return False, "warming up"
        bad = 1.0 - np.mean(self._raw_ok)
        if bad > 0.2:
            return False, (f"{bad:.0%} of samples outside 5-95 % of the ADC range "
                           "(GY-61 unplugged, on 5 V, or VCC/GND swapped?)")
        m = float(np.mean(self._amag))
        if not AMAG_OK[0] <= m <= AMAG_OK[1]:
            return False, f"mean |a| over 2 s is {m:.2f} g, expected about 1 g (wiring or calibration)"
        return True, f"mean |a| {m:.2f} g"

    def ok(self, now: float) -> bool:
        return self._ok and self.last_data_t is not None and now - self.last_data_t < 1.0

    @property
    def clipped_pct(self) -> float:
        return 100.0 * self.samples_clipped / self.samples_total if self.samples_total else 0.0

    @property
    def rail_pct(self) -> float:
        return 100.0 * self.samples_rail / self.samples_total if self.samples_total else 0.0
