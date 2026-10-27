"""ble_link brick: the app-side end of the BLE relay.

The relay container (relay.py, started by brick_compose.yaml) owns Bluetooth.
This brick connects to its Unix socket, turns each JSON line into a LinkEvent
and hands it to a callback. It reconnects to the socket forever, so the app
survives the relay restarting.

    from ble_link import BleLink
    link = BleLink(on_event=queue.put)
    link.write_control("PING")

`RelayClient` is the same thing without the App Lab @brick wrapper, so tests and
tools can use it outside App.run().
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field

DEFAULT_SOCKET = os.environ.get("BLE_RELAY_SOCKET", "/app/.cache/ble_link/relay.sock")


@dataclass
class LinkEvent:
    """One message from the relay.

    kind: "relay_up" | "relay_down" | "hello" | "status" | "conn" | "imu" | "bat" | "mtu" | "log"
    data: payload bytes for "imu" / "bat"
    info: the decoded JSON message for everything else
    t: time.monotonic() when it was received
    """

    kind: str
    data: bytes = b""
    info: dict = field(default_factory=dict)
    t: float = 0.0


class RelayClient:
    def __init__(self, on_event, socket_path: str = DEFAULT_SOCKET, retry_s: float = 1.0):
        self.on_event = on_event
        self.socket_path = socket_path
        self.retry_s = retry_s
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.last_error = ""

    # -------------------------------------------------- lifecycle
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_forever, name="ble_link", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            if not self._stop.is_set():
                time.sleep(self.retry_s)

    def run_once(self) -> None:
        """Connect, read until the socket closes, then return."""
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
        self.connected = True
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
                        self._emit(self.decode(line))
        except OSError as e:
            self.last_error = f"relay socket error: {e}"
        finally:
            self.connected = False
            with self._lock:
                self._sock = None
            try:
                s.close()
            except OSError:
                pass
            self._emit(LinkEvent("relay_down", info={"error": self.last_error or "relay closed the socket"}))

    # -------------------------------------------------- messages
    @staticmethod
    def decode(line: bytes) -> LinkEvent:
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
                return LinkEvent("log", info={"level": "warning", "msg": f"bad hex in relay {kind} message"}, t=now)
        return LinkEvent(kind, info=msg, t=now)

    def _emit(self, ev: LinkEvent) -> None:
        if not ev.t:
            ev.t = time.monotonic()
        self.on_event(ev)

    def write_control(self, text: str) -> bool:
        """Send an ASCII command to the Control characteristic (e.g. "PING")."""
        line = (json.dumps({"t": "write", "d": text}) + "\n").encode()
        with self._lock:
            if self._sock is None:
                return False
            try:
                self._sock.sendall(line)
                return True
            except OSError:
                return False


try:
    from arduino.app_utils import brick

    @brick
    class BleLink(RelayClient):
        """App Lab brick: started/stopped by App.run(); reads the relay in its own thread."""

        def __init__(self, on_event=None, socket_path: str = DEFAULT_SOCKET):
            super().__init__(on_event or (lambda ev: None), socket_path)

except ImportError:  # outside the App Lab image (e.g. a laptop running the tools)
    BleLink = RelayClient
