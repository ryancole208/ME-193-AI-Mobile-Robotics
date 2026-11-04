"""BLE relay: GaitNode (Pico W) <-> Unix socket. Runs in the `ble-relay` container.

Why this exists: see brick_compose.yaml. In short, the app's main container
cannot reach BlueZ, so this small process does all the Bluetooth work and
forwards everything over a Unix socket that both containers can see
(/app/.cache/ble_link/relay.sock by default).

It is deliberately dumb: it does not parse IMU packets. It only
  * scans for the advertised name, connects, reads the Info characteristic,
  * subscribes to IMU data + battery notifications after EVERY connection
    (the Pico W sends nothing until the CCCD is written),
  * reports the negotiated MTU, auto-reconnects forever,
  * forwards writes to the Control characteristic (PING, RESEND <seq>).

Wire format on the socket: one JSON object per line ("JSON lines").
relay -> app:
  {"t": "hello", "relay_version": 1, "bluez": {...}}       on accept
  {"t": "status", "state": "scanning"|"connecting"|"connected"|"disconnected"|"error", "msg": "..."}
  {"t": "conn", "address": "...", "rssi": -60, "mtu": 247, "mtu_source": "...", "info": "<hex>"}
  {"t": "imu", "d": "<hex payload>"}
  {"t": "bat", "d": "<hex payload>"}
  {"t": "mtu", "mtu": 247}                                    if it changes later
  {"t": "log", "level": "info"|"warning"|"error", "msg": "..."}
app -> relay:
  {"t": "write", "d": "PING"}                                  Control characteristic
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time

from bleak import BleakClient, BleakScanner

RELAY_VERSION = 1
SERVICE_UUID = "6e400000-7a1b-4c2d-9e10-a5c0f0e1d200"
IMU_UUID = "6e400001-7a1b-4c2d-9e10-a5c0f0e1d200"
BATTERY_UUID = "6e400002-7a1b-4c2d-9e10-a5c0f0e1d200"
CONTROL_UUID = "6e400003-7a1b-4c2d-9e10-a5c0f0e1d200"
INFO_UUID = "6e400004-7a1b-4c2d-9e10-a5c0f0e1d200"

DEVICE_NAME = os.environ.get("GAITNODE_NAME", "GaitNode")
SOCKET_PATH = os.environ.get("BLE_RELAY_SOCKET", "/app/.cache/ble_link/relay.sock")
SCAN_TIMEOUT_S = 10.0
PING_INTERVAL_S = 2.0   # the Pico W drops the connection after 6 s without one
ADDRESS_FILE = os.environ.get("BLE_RELAY_ADDRESS_FILE", os.path.join(os.path.dirname(SOCKET_PATH), "last_address"))


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} ble-relay: {msg}", flush=True)


class Relay:
    def __init__(self):
        self.writer: asyncio.StreamWriter | None = None
        self.status = {"t": "status", "state": "scanning", "msg": "starting"}
        self.conn_msg: dict | None = None
        self.bluez: dict = {"ok": False, "detail": "not checked yet"}
        self.control_queue: asyncio.Queue[bytes] = asyncio.Queue()

    # ------------------------------------------------------------ socket side
    def send(self, msg: dict) -> None:
        w = self.writer
        if w is None or w.is_closing():
            return
        try:
            w.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
        except Exception as e:  # app went away; it will reconnect
            log(f"dropping app connection after write error: {e}")
            self.writer = None

    def set_status(self, state: str, msg: str = "") -> None:
        self.status = {"t": "status", "state": state, "msg": msg}
        log(f"status {state} {msg}")
        self.send(self.status)

    def app_log(self, level: str, msg: str) -> None:
        log(f"{level}: {msg}")
        self.send({"t": "log", "level": level, "msg": msg})

    async def handle_app(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.writer is not None:
            log("a new app connection replaces the previous one")
            self.writer.close()
        self.writer = writer
        log("app connected to relay socket")
        self.send({"t": "hello", "relay_version": RELAY_VERSION, "device_name": DEVICE_NAME, "bluez": self.bluez})
        self.send(self.status)
        if self.conn_msg and self.status["state"] == "connected":
            self.send(self.conn_msg)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self.app_log("warning", f"ignored a malformed line from the app: {line[:80]!r}")
                    continue
                if msg.get("t") == "write":
                    await self.control_queue.put(str(msg.get("d", "")).encode("ascii"))
        finally:
            if self.writer is writer:
                self.writer = None
            writer.close()
            log("app disconnected from relay socket")

    # ------------------------------------------------------------ BlueZ check
    async def check_bluez(self) -> None:
        """Ask BlueZ (over D-Bus) which adapters exist and whether they are powered."""
        try:
            from dbus_fast import BusType
            from dbus_fast.aio import MessageBus

            bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
            intro = await bus.introspect("org.bluez", "/")
            om = bus.get_proxy_object("org.bluez", "/", intro).get_interface("org.freedesktop.DBus.ObjectManager")
            objs = await om.call_get_managed_objects()
            adapters = {
                path.rsplit("/", 1)[-1]: bool(ifaces["org.bluez.Adapter1"]["Powered"].value)
                for path, ifaces in objs.items()
                if "org.bluez.Adapter1" in ifaces
            }
            bus.disconnect()
            if not adapters:
                self.bluez = {"ok": False, "detail": "BlueZ answered but reports no Bluetooth adapter"}
            elif not any(adapters.values()):
                self.bluez = {"ok": False, "adapters": adapters, "detail": "adapter present but powered off"}
            else:
                self.bluez = {"ok": True, "adapters": adapters, "detail": "BlueZ reachable, adapter powered"}
        except FileNotFoundError:
            self.bluez = {"ok": False, "detail": "/run/dbus/system_bus_socket missing in the relay container"}
        except Exception as e:
            self.bluez = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        log(f"BlueZ check: {self.bluez}")
        self.send({"t": "hello", "relay_version": RELAY_VERSION, "device_name": DEVICE_NAME, "bluez": self.bluez})

    # ------------------------------------------------------------ remembering the address
    @staticmethod
    def save_address(address: str) -> None:
        """So the next start can tell BlueZ apart an address it already knows
        about from one it has never seen, without needing a scan first."""
        try:
            os.makedirs(os.path.dirname(ADDRESS_FILE), exist_ok=True)
            tmp = ADDRESS_FILE + ".tmp"
            with open(tmp, "w") as f:
                f.write(address)
            os.replace(tmp, ADDRESS_FILE)
            log(f"saved GaitNode address {address} to {ADDRESS_FILE}")
        except OSError as e:
            log(f"could not save GaitNode address {address}: {e}")

    @staticmethod
    def load_address() -> str | None:
        try:
            addr = open(ADDRESS_FILE).read().strip()
            return addr or None
        except FileNotFoundError:
            return None
        except OSError as e:
            log(f"could not read the saved GaitNode address: {e}")
            return None

    # ------------------------------------------------------------ stale-connection recovery
    @staticmethod
    async def _device_path(bus, address: str) -> str | None:
        """BlueZ's D-Bus object path for `address`, or None if it has never seen it."""
        intro = await bus.introspect("org.bluez", "/")
        om = bus.get_proxy_object("org.bluez", "/", intro).get_interface("org.freedesktop.DBus.ObjectManager")
        objs = await om.call_get_managed_objects()
        for path, ifaces in objs.items():
            dev = ifaces.get("org.bluez.Device1")
            if dev and dev["Address"].value.upper() == address.upper():
                return path
        return None

    async def _bluez_connected(self, address: str) -> bool | None:
        """True/False if BlueZ has a device record for `address`, None if it has none at all."""
        from dbus_fast import BusType
        from dbus_fast.aio import MessageBus

        bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        try:
            path = await self._device_path(bus, address)
            if path is None:
                return None
            intro = await bus.introspect("org.bluez", path)
            props = bus.get_proxy_object("org.bluez", path, intro).get_interface("org.freedesktop.DBus.Properties")
            connected = await props.call_get("org.bluez.Device1", "Connected")
            return bool(connected.value)
        finally:
            bus.disconnect()

    async def _bluez_disconnect(self, address: str) -> None:
        """Ask BlueZ directly to drop `address`, e.g. after a reused connection
        turned out to be stale - this is what lets the Pico W start advertising
        again so a scan can find it."""
        from dbus_fast import BusType
        from dbus_fast.aio import MessageBus

        bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        try:
            path = await self._device_path(bus, address)
            if path is None:
                log(f"startup check: BlueZ has no record of {address}; nothing to disconnect")
                return
            intro = await bus.introspect("org.bluez", path)
            dev_if = bus.get_proxy_object("org.bluez", path, intro).get_interface("org.bluez.Device1")
            await dev_if.call_disconnect()
            log(f"startup check: disconnected the stale BlueZ link to {address}")
        except Exception as e:
            log(f"startup check: could not disconnect {address} via BlueZ: {e}")
        finally:
            bus.disconnect()

    async def _should_reuse_address(self, address: str) -> bool:
        """True if BlueZ currently shows `address` (remembered from the last
        successful connection) as connected - meaning the previous relay
        process was most likely killed without disconnecting cleanly. The
        Pico W won't be advertising in that case, so a fresh scan would hang
        until SCAN_TIMEOUT_S and then retry forever. Connecting to the known
        address directly works instead, because BlueZ's Connect() on a device
        it already considers connected just succeeds immediately."""
        try:
            connected = await self._bluez_connected(address)
        except Exception as e:
            log(f"startup check: could not query BlueZ about {address} ({e}); scanning normally")
            return False
        if connected is None:
            log(f"startup check: BlueZ has no record of {address}; scanning normally")
            return False
        if not connected:
            log(f"startup check: BlueZ already shows {address} as disconnected; scanning normally")
            return False
        log(f"startup check: BlueZ still shows {address} as connected (the previous run likely didn't "
            "shut down cleanly) - connecting to it directly instead of scanning")
        return True

    # ------------------------------------------------------------ BLE side
    async def ble_forever(self) -> None:
        while True:
            try:
                await self.one_connection()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.set_status("error", f"{type(e).__name__}: {e}")
                if not self.bluez.get("ok"):
                    await self.check_bluez()
            self.conn_msg = None
            await asyncio.sleep(1.0)

    async def one_connection(self) -> None:
        saved = self.load_address()
        reused = bool(saved) and await self._should_reuse_address(saved)
        rssi = None
        if reused:
            dev = saved
        else:
            self.set_status("scanning", f"looking for '{DEVICE_NAME}'")
            seen = {}

            def match(d, adv) -> bool:
                if (adv.local_name or d.name) == DEVICE_NAME:
                    seen["rssi"] = adv.rssi
                    return True
                return False

            dev = await BleakScanner.find_device_by_filter(match, timeout=SCAN_TIMEOUT_S)
            if dev is None:
                return
            rssi = seen.get("rssi")

        address = dev if isinstance(dev, str) else dev.address
        self.set_status("connecting", f"{address} rssi {rssi}")
        lost = asyncio.Event()
        client = BleakClient(dev, disconnected_callback=lambda _c: lost.set())
        try:
            await client.connect()
        except Exception as e:
            # Connecting, specifically, is the one failure mode that means the
            # reused address was stale rather than genuinely reconnectable -
            # clear it from BlueZ so the next attempt (scan or retry) isn't
            # blocked by the same ghost connection.
            if reused:
                log(f"direct connect to the reused address {address} failed ({e}); clearing it via BlueZ")
                await self._bluez_disconnect(address)
            raise

        try:
            # The Pico W asks for a bigger MTU right after connecting; give the
            # exchange a moment, then ask BlueZ what it settled on.
            await asyncio.sleep(0.5)
            mtu, source = await self.read_mtu(client)
            info = bytes(await client.read_gatt_char(INFO_UUID))

            def on_imu(_char, data: bytearray) -> None:
                self.send({"t": "imu", "d": bytes(data).hex()})

            def on_bat(_char, data: bytearray) -> None:
                self.send({"t": "bat", "d": bytes(data).hex()})

            # Subscribe on every connection: the Pico W only streams after the
            # CCCD is written, and BlueZ forgets subscriptions on disconnect.
            await client.start_notify(IMU_UUID, on_imu)
            await client.start_notify(BATTERY_UUID, on_bat)
            try:
                on_bat(None, await client.read_gatt_char(BATTERY_UUID))
            except Exception as e:
                self.app_log("warning", f"initial battery read failed ({e}); waiting for the 10 s notification")

            self.conn_msg = {
                "t": "conn",
                "address": address,
                "rssi": rssi,
                "mtu": mtu,
                "mtu_source": source,
                "info": info.hex(),
            }
            self.set_status("connected", f"{address} MTU {mtu} ({source})" + (" (reused)" if reused else ""))
            self.send(self.conn_msg)
            self.save_address(address)

            last_mtu_check = time.monotonic()
            last_ping = time.monotonic()
            while not lost.is_set():
                try:
                    cmd = await asyncio.wait_for(self.control_queue.get(), timeout=1.0)
                    await client.write_gatt_char(CONTROL_UUID, cmd, response=True)
                    log(f"control write {cmd!r}")
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    self.app_log("warning", f"control write failed: {e}")
                now = time.monotonic()
                if now - last_ping >= PING_INTERVAL_S:
                    last_ping = now
                    try:
                        await client.write_gatt_char(CONTROL_UUID, b"PING", response=True)
                    except Exception as e:
                        self.app_log("warning", f"keepalive PING failed: {e}")
                if now - last_mtu_check > 5.0 and source != "default":
                    last_mtu_check = now
                    new_mtu = client.mtu_size
                    if new_mtu != mtu:
                        mtu = new_mtu
                        self.send({"t": "mtu", "mtu": mtu})
        finally:
            await self._clean_disconnect(client, "connection ending")
        self.set_status("disconnected", "GaitNode link lost; will scan again")

    async def _clean_disconnect(self, client: BleakClient, reason: str) -> None:
        """Best-effort: unsubscribe and disconnect so BlueZ actually drops the
        link and the Pico W can advertise again - used after every connection
        ends (not just on shutdown), so a stale link never has a chance to
        linger in the first place."""
        if not client.is_connected:
            log(f"clean disconnect ({reason}): already disconnected")
            return
        for uuid, name in ((IMU_UUID, "IMU"), (BATTERY_UUID, "battery")):
            try:
                await client.stop_notify(uuid)
                log(f"clean disconnect ({reason}): stopped {name} notifications")
            except Exception as e:
                log(f"clean disconnect ({reason}): stop_notify({name}) failed: {e}")
        try:
            await client.disconnect()
            log(f"clean disconnect ({reason}): disconnected from {client.address}")
        except Exception as e:
            log(f"clean disconnect ({reason}): disconnect() failed: {e}")

    @staticmethod
    async def read_mtu(client: BleakClient) -> tuple[int, str]:
        """MTU as BlueZ sees it. bleak's BlueZ backend needs _acquire_mtu() for a real value."""
        backend = getattr(client, "_backend", None)
        if backend is not None and hasattr(backend, "_acquire_mtu"):
            try:
                await backend._acquire_mtu()
                return int(client.mtu_size), "bluez"
            except Exception as e:
                log(f"could not acquire MTU from BlueZ ({e}); falling back to bleak's default")
        return int(client.mtu_size), "default"


async def main() -> None:
    relay = Relay()
    os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    server = await asyncio.start_unix_server(relay.handle_app, path=SOCKET_PATH)
    log(f"listening on {SOCKET_PATH}")
    await relay.check_bluez()

    # SIGTERM is what `docker stop` (and so App Lab's "stop app") sends; SIGINT
    # is Ctrl-C. Without a handler, asyncio has no default reaction to SIGTERM
    # at all - the process just dies mid-task, BleakClient never gets to
    # disconnect, and BlueZ is left thinking the Pico W is still connected (so
    # it stops advertising and the next start's scan never finds it).
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_shutdown(sig_name: str) -> None:
        log(f"received {sig_name}: shutting down cleanly")
        stop_event.set()

    for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGINT, "SIGINT")):
        loop.add_signal_handler(sig, _request_shutdown, name)

    async with server:
        forever = asyncio.create_task(relay.ble_forever())
        await stop_event.wait()
        log("shutdown: stopping the BLE loop")
        forever.cancel()
        try:
            await forever
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log(f"shutdown: BLE loop ended with {type(e).__name__}: {e}")
    log("shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Fallback for platforms/event loops where add_signal_handler isn't
        # available (main() normally handles SIGINT itself and this never fires).
        sys.exit(0)
