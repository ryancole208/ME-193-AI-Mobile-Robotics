"""Ankle node (Pico W "GaitNode") over BLE - copied/adapted from unoq_app.

Why a relay: App Lab runs this app in a Docker container that has no D-Bus and
cannot open Bluetooth sockets. bricks/ble_link/brick_compose.yaml (copied
unchanged from unoq_app) starts a second container, `ble-relay`, which runs
bleak against the host's BlueZ and forwards GaitNode notifications as JSON lines
over a Unix socket in .cache/ble_link/. This module is the app-side end:

  RelayClient  reconnects to the socket forever, one LinkEvent per JSON line
  AnkleLink    link state (connected / MTU / battery / Info) + StreamTracker
               (seq + t0_us gap detection) + unit conversion and leg mirroring

Only one BLE central can be connected to the Pico W, so unoq_app must not run
at the same time (App Lab only runs one app at a time anyway).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from protocol import (
    MIN_MTU_FOR_FULL_PACKET, PROTOCOL_VERSION, PacketError, StreamTracker, parse_ankle, parse_battery, parse_info,
)

log = logging.getLogger("step.ble")

DEFAULT_SOCKET = os.environ.get("BLE_RELAY_SOCKET", "/app/.cache/ble_link/relay.sock")

# MPU-6050 as configured by the Pico firmware: +-16 g, +-2000 deg/s
ACCEL_LSB_PER_G = 2048.0
GYRO_LSB_PER_DPS = 16.4

# Left-leg pod -> right-leg frame (derivation in unoq_app/python/gait/sensor.py):
# mirroring across the body midline flips accel x, and because angular velocity
# is a pseudovector the gyro picks up an extra minus: (gx, -gy, -gz).
LEFT_ACCEL_SIGNS = np.array([-1.0, 1.0, 1.0])
LEFT_GYRO_SIGNS = np.array([1.0, -1.0, -1.0])


def convert_ankle(raw: np.ndarray, leg: str) -> np.ndarray:
    """raw (n, 6) counts -> (n, 6) [ax ay az g][gx gy gz deg/s], right-leg frame."""
    raw = np.asarray(raw, dtype=np.float64)
    acc = raw[:, 0:3] / ACCEL_LSB_PER_G
    gyr = raw[:, 3:6] / GYRO_LSB_PER_DPS
    if leg == "left":
        acc = acc * LEFT_ACCEL_SIGNS
        gyr = gyr * LEFT_GYRO_SIGNS
    return np.hstack([acc, gyr])


@dataclass
class LinkEvent:
    """kind: relay_up | relay_down | hello | status | conn | imu | bat | mtu | log"""
    kind: str
    data: bytes = b""
    info: dict = field(default_factory=dict)
    t: float = 0.0


class RelayClient:
    """Reads the relay's Unix socket in a background thread (from bricks/ble_link)."""

    def __init__(self, on_event, socket_path: str = DEFAULT_SOCKET, retry_s: float = 1.0):
        self.on_event = on_event
        self.socket_path = socket_path
        self.retry_s = retry_s
        self._sock = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.last_error = ""

    def start(self) -> None:
        threading.Thread(target=self._run_forever, name="ble_client", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run_forever(self) -> None:
        while not self._stop.is_set():
            self._run_once()
            time.sleep(self.retry_s)

    def _run_once(self) -> None:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(self.socket_path)
        except OSError as e:
            msg = f"cannot open relay socket {self.socket_path}: {e}"
            if msg != self.last_error:
                self.last_error = msg
                self._emit(LinkEvent("relay_down", info={"error": msg}))
            return
        with self._lock:
            self._sock = s
        self.last_error = ""
        self._emit(LinkEvent("relay_up", info={"socket": self.socket_path}))
        buf = b""
        try:
            while not self._stop.is_set():
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        self._emit(self._decode(line))
        except OSError as e:
            self.last_error = f"relay socket error: {e}"
        finally:
            with self._lock:
                self._sock = None
            try:
                s.close()
            except OSError:
                pass
            self._emit(LinkEvent("relay_down", info={"error": self.last_error or "relay closed the socket"}))

    @staticmethod
    def _decode(line: bytes) -> LinkEvent:
        now = time.monotonic()
        try:
            msg = json.loads(line)
            kind = msg.get("t", "")
        except (json.JSONDecodeError, AttributeError):
            return LinkEvent("log", info={"level": "warning", "msg": f"malformed relay line {line[:60]!r}"}, t=now)
        if kind in ("imu", "bat"):
            try:
                return LinkEvent(kind, data=bytes.fromhex(msg.get("d", "")), t=now)
            except ValueError:
                return LinkEvent("log", info={"level": "warning", "msg": f"bad hex in relay {kind}"}, t=now)
        return LinkEvent(kind, info=msg, t=now)

    def _emit(self, ev: LinkEvent) -> None:
        if not ev.t:
            ev.t = time.monotonic()   # host receive time: the only host timestamp a packet gets
        self.on_event(ev)

    def write_control(self, text: str) -> bool:
        line = (json.dumps({"t": "write", "d": text}) + "\n").encode()
        with self._lock:
            if self._sock is None:
                return False
            try:
                self._sock.sendall(line)
                return True
            except OSError:
                return False


class AnkleLink:
    """Link facts + gap tracking for the ankle stream (adapted from unoq_app's LinkMonitor)."""

    def __init__(self):
        self.tracker = StreamTracker()
        self.relay_up = False
        self.bluez: dict = {}
        self.state = "relay_down"
        self.connected = False
        self.ever_connected = False
        self.connect_count = 0
        self.first_data_t = None
        self.last_data_t = None
        self.battery_mV = None
        self.mtu = None
        self.mtu_source = ""
        self.info: dict = {}
        self.info_problems: list = []
        self.n_values: set = set()
        self.packet_errors = 0
        self.rate = RateMeter()
        self._last_err = ""

    @property
    def stats(self):
        return self.tracker.stats

    def link_state(self, now: float) -> str:
        """For the API: ok | lost | searching."""
        if self.last_data_t is not None and now - self.last_data_t < 2.0:
            return "ok"
        return "lost" if self.ever_connected else "searching"

    def handle(self, ev: LinkEvent):
        """Returns (Chunk, packet bytes) for an IMU packet, else None."""
        k = ev.kind
        if k == "imu":
            return self._imu(ev)
        if k == "bat":
            try:
                self.battery_mV = parse_battery(ev.data)
            except PacketError as e:
                log.error(f"While parsing a battery notification: {e}. Likely fix: Pico firmware mismatch")
        elif k == "relay_up":
            self.relay_up = True
            log.info(f"connected to the BLE relay at {ev.info.get('socket')}")
        elif k == "relay_down":
            if self.relay_up:
                log.warning(f"BLE relay socket closed: {ev.info.get('error')}")
            else:
                log.info(f"BLE relay not reachable yet ({ev.info.get('error')}); the ble-relay container waits "
                         "for bleak to be installed on the first run (needs internet once)")
            self.relay_up = False
            self.state = "relay_down"
            self.connected = False
        elif k == "hello":
            self.bluez = ev.info.get("bluez", {})
            (log.info if self.bluez.get("ok") else log.error)(f"relay reports BlueZ: {self.bluez}")
        elif k == "status":
            self.state = ev.info.get("state", "?")
            if self.state != "connected":
                if self.connected:
                    log.warning("GaitNode disconnected")
                self.connected = False
            log.info(f"BLE {self.state}: {ev.info.get('msg', '')}")
        elif k == "conn":
            self._conn(ev)
        elif k == "mtu":
            self.mtu = ev.info.get("mtu")
        elif k == "log":
            lvl = {"error": logging.ERROR, "warning": logging.WARNING}.get(ev.info.get("level"), logging.INFO)
            log.log(lvl, f"relay: {ev.info.get('msg')}")
        return None

    def _conn(self, ev: LinkEvent) -> None:
        i = ev.info
        self.connected = self.ever_connected = True
        self.state = "connected"
        self.connect_count += 1
        self.mtu, self.mtu_source = i.get("mtu"), i.get("mtu_source", "")
        self.info_problems = []
        try:
            self.info = parse_info(bytes.fromhex(i.get("info", "")))
            expect = {"protocol_version": PROTOCOL_VERSION, "accel_range_g": 16, "gyro_range_dps": 2000,
                      "sample_rate_hz": 100}
            for key, want in expect.items():
                if self.info.get(key) != want:
                    self.info_problems.append(f"{key} is {self.info.get(key)}, expected {want}")
        except (PacketError, ValueError) as e:
            self.info_problems.append(f"Info characteristic unreadable: {e}")
        log.info(f"GaitNode connected (#{self.connect_count}) {i.get('address')} RSSI {i.get('rssi')} dBm, "
                 f"MTU {self.mtu} ({self.mtu_source}), info {self.info}")
        if self.mtu is not None and self.mtu < MIN_MTU_FOR_FULL_PACKET and self.mtu_source != "default":
            log.warning(f"MTU {self.mtu} < {MIN_MTU_FOR_FULL_PACKET}: fewer than 10 samples per packet")
        for p in self.info_problems:
            log.error(f"While checking the Info characteristic: {p}. Likely fix: the Pico W firmware does not "
                      "match the shared protocol (it must be unchanged from Gait/pico)")

    def _imu(self, ev: LinkEvent):
        try:
            pkt = parse_ankle(ev.data)
        except PacketError as e:
            self.packet_errors += 1
            if str(e) != self._last_err or self.packet_errors % 100 == 1:
                log.error(f"While parsing an ankle packet: {e} (bad packets: {self.packet_errors})")
                self._last_err = str(e)
            return None
        if not self.connected:
            self.connected = self.ever_connected = True
            self.state = "connected"
        if pkt.n not in self.n_values:
            self.n_values.add(pkt.n)
            log.info(f"ankle packets: n = {pkt.n} samples/packet")
        before = len(self.stats.events)
        chunk = self.tracker.push(pkt)
        for msg in self.stats.events[before:]:
            log.info(f"ankle stream: {msg}")
        if chunk is None:
            return None
        if self.first_data_t is None:
            self.first_data_t = ev.t
        self.last_data_t = ev.t
        self.rate.add(ev.t, pkt.n)
        return chunk


class RateMeter:
    """Samples per second over the last 5 s (for diag)."""

    def __init__(self, span_s: float = 5.0):
        self.span = span_s
        self._ev: list = []

    def add(self, t: float, n: int) -> None:
        self._ev.append((t, n))
        while self._ev and self._ev[0][0] < t - self.span:
            self._ev.pop(0)

    def rate(self, now: float) -> float:
        n = sum(k for t, k in self._ev if t >= now - self.span)
        return n / self.span
