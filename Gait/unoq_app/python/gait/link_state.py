"""Turns relay events (see bricks/ble_link) into sample chunks and link facts.

One LinkMonitor lives for the whole process, so the packet tracker keeps its
sample index across reconnects and mode changes (setup -> live).
"""

from __future__ import annotations

import logging

from .constants import MIN_MTU_FOR_FULL_PACKET, PROTOCOL_VERSION
from .errors import PacketError
from .protocol import SampleChunk, StreamTracker, parse_battery, parse_imu_packet, parse_info

log = logging.getLogger("gait.link")


class LinkMonitor:
    def __init__(self, logger=None):
        self.log = logger or log
        self.tracker = StreamTracker()
        self.relay_up = False
        self.relay_error = ""
        self.bluez: dict = {}
        self.state = "relay_down"        # relay_down | scanning | connecting | connected | disconnected | error
        self.connected = False
        self.connect_count = 0
        self.first_connect_t: float | None = None
        self.connect_t: float | None = None
        self.last_data_t: float | None = None
        self.battery_mV: int | None = None   # None = nothing received yet; 0 = reading unavailable
        self.battery_t: float | None = None
        self.rssi = None
        self.mtu = None
        self.mtu_source = ""
        self.info: dict = {}
        self.info_problems: list[str] = []
        self.n_this_connection: int | None = None
        self.n_values: set = set()
        self.packet_errors = 0
        self._last_packet_error = ""

    @property
    def stats(self):
        return self.tracker.stats

    def handle(self, ev) -> list[SampleChunk]:
        """Process one LinkEvent. Returns the sample chunks it produced (0 or 1)."""
        k = ev.kind
        if k == "imu":
            return self._imu(ev)
        if k == "bat":
            try:
                self.battery_mV = parse_battery(ev.data)
                self.battery_t = ev.t
            except PacketError as e:
                self.log.error(str(e))
        elif k == "relay_up":
            self.relay_up, self.relay_error = True, ""
            self.log.info(f"connected to the BLE relay at {ev.info.get('socket')}")
        elif k == "relay_down":
            if self.relay_up or self.relay_error != ev.info.get("error"):
                self.log.warning(
                    "BLE relay not reachable while connecting to its socket: "
                    f"{ev.info.get('error')}. Likely fix: the ble-relay container is still starting "
                    "(it waits for bleak to be installed on the first run) - if this persists, run "
                    "`docker ps` to check it exists and see the README, Troubleshooting.")
            self.relay_up, self.relay_error = False, ev.info.get("error", "")
            self.state = "relay_down"
            self._lost()
        elif k == "hello":
            self.bluez = ev.info.get("bluez", {})
            lvl = logging.INFO if self.bluez.get("ok") else logging.ERROR
            self.log.log(lvl, f"relay reports BlueZ: {self.bluez}")
        elif k == "status":
            self.state = ev.info.get("state", "?")
            if self.state != "connected":
                self._lost()
            self.log.info(f"BLE {self.state}: {ev.info.get('msg', '')}")
        elif k == "conn":
            self._conn(ev)
        elif k == "mtu":
            self.mtu = ev.info.get("mtu")
            self.log.info(f"MTU changed to {self.mtu}")
        elif k == "log":
            lvl = {"error": logging.ERROR, "warning": logging.WARNING}.get(ev.info.get("level"), logging.INFO)
            self.log.log(lvl, f"relay: {ev.info.get('msg')}")
        return []

    def _lost(self) -> None:
        if self.connected:
            self.log.warning("GaitNode disconnected")
        self.connected = False

    def _conn(self, ev) -> None:
        i = ev.info
        self.connected = True
        self.state = "connected"
        self.connect_count += 1
        self.connect_t = ev.t
        if self.first_connect_t is None:
            self.first_connect_t = ev.t
        self.rssi = i.get("rssi")
        self.mtu = i.get("mtu")
        self.mtu_source = i.get("mtu_source", "")
        self.n_this_connection = None
        self.info_problems = []
        try:
            self.info = parse_info(bytes.fromhex(i.get("info", "")))
            expect = {"protocol_version": PROTOCOL_VERSION, "accel_range_g": 16, "gyro_range_dps": 2000, "sample_rate_hz": 100}
            for key, want in expect.items():
                if self.info.get(key) != want:
                    self.info_problems.append(f"{key} is {self.info.get(key)}, expected {want}")
        except (PacketError, ValueError) as e:
            self.info = {}
            self.info_problems.append(f"Info characteristic unreadable: {e}")
        self.log.info(f"GaitNode connected (#{self.connect_count}) {i.get('address')} RSSI {self.rssi} dBm, "
                      f"MTU {self.mtu} ({self.mtu_source}), info {self.info}")
        if self.mtu is not None and self.mtu < MIN_MTU_FOR_FULL_PACKET and self.mtu_source != "default":
            self.log.warning(f"MTU {self.mtu} < {MIN_MTU_FOR_FULL_PACKET}: the Pico W will send fewer than 10 samples per packet")
        for p in self.info_problems:
            self.log.error(f"while checking the Info characteristic: {p}. Likely fix: the Pico W firmware does not "
                           "match the shared protocol; update one side so both agree")

    def _imu(self, ev) -> list[SampleChunk]:
        try:
            pkt = parse_imu_packet(ev.data)
        except PacketError as e:
            self.packet_errors += 1
            if str(e) != self._last_packet_error or self.packet_errors % 100 == 1:
                self.log.error(f"{e} (bad packets so far: {self.packet_errors})")
                self._last_packet_error = str(e)
            return []
        if not self.connected:  # data before the conn message (e.g. replayed by a fake relay)
            self.connected = True
            self.state = "connected"
            if self.first_connect_t is None:
                self.first_connect_t = ev.t
        if self.n_this_connection is None:
            self.n_this_connection = pkt.n
            self.log.info(f"first IMU packet on this connection: n = {pkt.n} samples/packet "
                          f"({len(ev.data)} bytes, MTU {self.mtu})")
        elif pkt.n != self.n_this_connection and pkt.n not in self.n_values:
            self.log.info(f"samples per packet changed to n = {pkt.n}")
        self.n_values.add(pkt.n)
        before = len(self.stats.events)
        chunk = self.tracker.push(pkt)
        for msg in self.stats.events[before:]:
            self.log.info(f"stream: {msg}")
        if chunk is not None:
            self.last_data_t = ev.t
            return [chunk]
        return []

    def battery_percent(self, empty_mV: float, full_mV: float) -> int:
        """-1 = unavailable (0 mV or nothing received)."""
        if not self.battery_mV:
            return -1
        pct = (self.battery_mV - empty_mV) / (full_mV - empty_mV) * 100
        return int(max(0, min(100, round(pct))))
