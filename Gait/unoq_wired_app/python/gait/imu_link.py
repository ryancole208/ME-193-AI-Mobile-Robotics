"""The wired "link": IMU packets and status that the sketch sends over Bridge.

The sketch calls (fire-and-forget Bridge.notify, registered in main.py):
  imu_packet(bytes)    one packet in the shared format: <HIB header + n x 6 int16
  imu_status(state, who_am_i, i2c_errors, reinits, fifo_overflows, bus_hz)   once per second
     state: 0 = I2C bus not ready, 1 = nothing answers at 0x68, 2 = wrong WHO_AM_I, 3 = streaming

Packets go through the same StreamTracker as in the Bluetooth app, so gaps
are detected from both seq and t0_us.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .errors import PacketError
from .protocol import SampleChunk, StreamTracker, parse_imu_packet

log = logging.getLogger("gait.imu")

STATE_BUS_ERROR, STATE_NO_DEVICE, STATE_WRONG_ID, STATE_OK = 0, 1, 2, 3
STATE_TEXT = {0: "I2C bus not ready", 1: "no device answers at 0x68", 2: "wrong WHO_AM_I", 3: "streaming"}


@dataclass
class ImuEvent:
    kind: str            # "packet" | "status"
    data: bytes = b""
    status: tuple = ()
    t: float = 0.0


class ImuLink:
    def __init__(self, logger=None):
        self.log = logger or log
        self.tracker = StreamTracker()
        self.first_data_t: float | None = None
        self.last_data_t: float | None = None
        self.status_t: float | None = None
        self.sensor_ok = False
        self.state: int | None = None
        self.who_am_i: int | None = None
        self.i2c_errors = 0
        self.reinits = 0
        self.fifo_overflows = 0
        self.bus_hz = 0
        self.packet_errors = 0
        self.n_values: set = set()
        self._last_error = ""

    @property
    def stats(self):
        return self.tracker.stats

    def handle(self, ev: ImuEvent) -> list[SampleChunk]:
        if ev.kind == "status":
            self._status(ev)
            return []
        try:
            pkt = parse_imu_packet(ev.data)
        except PacketError as e:
            self.packet_errors += 1
            if str(e) != self._last_error or self.packet_errors % 100 == 1:
                self.log.error(f"{e} (bad packets so far: {self.packet_errors})")
                self._last_error = str(e)
            return []
        if pkt.n not in self.n_values:
            self.n_values.add(pkt.n)
            self.log.info(f"IMU packets arriving: n = {pkt.n} samples/packet")
        before = len(self.stats.events)
        chunk = self.tracker.push(pkt)
        for msg in self.stats.events[before:]:
            self.log.info(f"stream: {msg}")
        if chunk is None:
            return []
        if self.first_data_t is None:
            self.first_data_t = ev.t
        self.last_data_t = ev.t
        return [chunk]

    def _status(self, ev: ImuEvent) -> None:
        try:
            state, who, errors, reinits, overflows, bus_hz = (int(v) for v in ev.status)
        except (TypeError, ValueError):
            self.log.error(f"While reading imu_status from the sketch: unexpected values {ev.status!r}. "
                           "Likely fix: the sketch and python/gait/imu_link.py disagree; restart the app")
            return
        if errors > self.i2c_errors:
            self.log.warning(f"I2C errors: {errors} total (+{errors - self.i2c_errors}); sensor re-inits {reinits}. "
                             "Likely fix: check the SDA/SCL/3.3V/GND wires and keep the cable short")
        ok = state == STATE_OK
        if ok and not self.sensor_ok:
            self.log.info(f"MPU-6050 ready: WHO_AM_I 0x{who:02X}, I2C {bus_hz // 1000} kHz")
        elif not ok and (self.sensor_ok or self.status_t is None):
            self.log.error(f"MPU-6050 not ready: {STATE_TEXT.get(state, state)} (WHO_AM_I 0x{who:02X}). Likely fix: "
                           "check the wiring (VCC to 3.3V, GND, SDA, SCL); the sketch retries every second")
        self.state = state
        self.sensor_ok, self.who_am_i, self.i2c_errors = ok, who, errors
        self.reinits, self.fifo_overflows, self.bus_hz = reinits, overflows, bus_hz
        self.status_t = ev.t
