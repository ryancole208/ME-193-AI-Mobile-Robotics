# ble_node.py - the Bluetooth Low Energy side of the GaitNode.
#
# The Pico W is the BLE *peripheral*: it advertises as "GaitNode" and waits
# for the UNO Q (the *central*) to connect. It offers one GATT service with
# the four characteristics from the shared protocol (see README.md).
#
# Uses only MicroPython's built-in low-level `bluetooth` module (no aioble).
# Reference: https://docs.micropython.org/en/latest/library/bluetooth.html
#
# Upload this file to the Pico W as ble_node.py.

import bluetooth
import struct
import time
from micropython import const

import config

# Event codes passed to the IRQ handler (values from the MicroPython docs).
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_WRITE = const(3)
_IRQ_MTU_EXCHANGED = const(21)

# Characteristic flags (MicroPython docs).
_FLAG_READ = const(0x0002)
_FLAG_WRITE_NO_RESPONSE = const(0x0004)
_FLAG_WRITE = const(0x0008)
_FLAG_NOTIFY = const(0x0010)

# Advertising data field types (Bluetooth Core Spec Supplement, part A 1.x).
_ADV_TYPE_FLAGS = const(0x01)
_ADV_TYPE_UUID128_COMPLETE = const(0x07)
_ADV_TYPE_NAME = const(0x09)
_ADV_FLAGS_GENERAL_NO_BREDR = const(0x06)   # LE general discoverable, no classic BT

_DEFAULT_MTU = const(23)     # every BLE link starts at 23 until an exchange
_PREFERRED_MTU = const(247)  # what we ask for; the result is min(ours, theirs)
_ADV_INTERVAL_US = const(100_000)


def _log(*args):
    if config.DEBUG:
        print("[ble]", *args)


def _adv_field(adv_type, value):
    return bytes((len(value) + 1, adv_type)) + value


class BLENode:
    """GATT server for the GaitNode protocol.

    The IRQ handler only records what happened. Anything slow (MTU exchange,
    answering commands) is done by main.py calling service() and
    pop_control() from its loop.
    """

    def __init__(self):
        self.ble = bluetooth.BLE()
        self.ble.active(True)
        self.ble.config(gap_name=config.BLE_NAME, mtu=_PREFERRED_MTU)
        self.ble.irq(self._irq)

        service = (
            bluetooth.UUID(config.SERVICE_UUID),
            (
                (bluetooth.UUID(config.IMU_CHAR_UUID), _FLAG_NOTIFY),
                (bluetooth.UUID(config.BATTERY_CHAR_UUID), _FLAG_READ | _FLAG_NOTIFY),
                (bluetooth.UUID(config.CONTROL_CHAR_UUID), _FLAG_WRITE | _FLAG_WRITE_NO_RESPONSE),
                (bluetooth.UUID(config.INFO_CHAR_UUID), _FLAG_READ),
            ),
        )
        ((self.h_imu, self.h_battery, self.h_control, self.h_info),) = \
            self.ble.gatts_register_services((service,))

        # "RESEND 65535" is 12 bytes; 32 leaves room for typos without truncation.
        self.ble.gatts_set_buffer(self.h_control, 32)
        self.ble.gatts_write(self.h_info, struct.pack(
            "<BBHH", config.PROTOCOL_VERSION, config.ACCEL_RANGE_G,
            config.GYRO_RANGE_DPS, config.SAMPLE_RATE_HZ))
        self.ble.gatts_write(self.h_battery, struct.pack("<H", 0))

        # Name (2 + 8) + flags (3) + 128-bit UUID (2 + 16) = 31 bytes, exactly
        # the legacy advertising limit, so everything fits in one packet.
        # bytes(UUID) is already in the little-endian order BLE sends.
        self._adv = (_adv_field(_ADV_TYPE_FLAGS, bytes((_ADV_FLAGS_GENERAL_NO_BREDR,)))
                     + _adv_field(_ADV_TYPE_NAME, config.BLE_NAME.encode())
                     + _adv_field(_ADV_TYPE_UUID128_COMPLETE,
                                  bytes(bluetooth.UUID(config.SERVICE_UUID))))
        if len(self._adv) > 31:
            raise RuntimeError("advertising payload is %d bytes, max 31; shorten BLE_NAME"
                               % len(self._adv))

        self.conn = None          # connection handle, None when nobody is connected
        self.mtu = _DEFAULT_MTU
        self._mtu_pending = False
        self._control = None      # last command written by the central (bytes)
        self._just_connected = False
        self._adv_pending = False  # restart advertising from service()
        self.last_ping_ms = 0      # when the last PING (or the connect) happened
        self.pings = 0
        self.heartbeat_drops = 0   # connections dropped for missing PINGs
        self.connects = 0
        self.notify_fails = 0
        self._ccc_ok = True       # False if this firmware won't let us read the CCCD
        self.advertise()

    # --- events ----------------------------------------------------------------

    def _irq(self, event, data):
        if event == _IRQ_CENTRAL_CONNECT:
            conn, _addr_type, _addr = data
            self.conn = conn
            self.mtu = _DEFAULT_MTU
            self._mtu_pending = True       # ask for a bigger MTU from service()
            self._just_connected = True
            self._adv_pending = False      # advertising stops on connect anyway
            self.last_ping_ms = time.ticks_ms()   # heartbeat timer starts now
            self.connects += 1
            self._clear_subscriptions()    # a new connection starts unsubscribed
            _log("central connected, handle", conn)
        elif event == _IRQ_CENTRAL_DISCONNECT:
            conn, _addr_type, _addr = data
            if conn != self.conn:
                # A connection drop_if_silent() already forgot. Its state is
                # cleared and advertising restarted, and a new central may
                # be connected by now, so leave everything alone.
                _log("old connection", conn, "closed")
                return
            _log("central disconnected, handle", conn)
            self.conn = None
            self.mtu = _DEFAULT_MTU
            self._mtu_pending = False
            self._clear_subscriptions()
            self.advertise()               # be findable again straight away
        elif event == _IRQ_GATTS_WRITE:
            _conn, attr = data
            if attr == self.h_control:
                self._control = bytes(self.ble.gatts_read(self.h_control))
                # Stamp the heartbeat here rather than in main.py, so a PING
                # can't be missed if another command overwrites _control
                # before the main loop picks it up.
                if self._control.strip().upper() == b"PING":
                    self.last_ping_ms = time.ticks_ms()
                    self.pings += 1
        elif event == _IRQ_MTU_EXCHANGED:
            _conn, mtu = data
            self.mtu = mtu
            self._mtu_pending = False
            _log("MTU exchanged:", mtu, "-> up to", self.samples_per_packet(), "samples/packet")

    # --- called from main loop -------------------------------------------------

    def advertise(self):
        self.ble.gap_advertise(_ADV_INTERVAL_US, adv_data=self._adv)
        _log("advertising as", config.BLE_NAME)

    def service(self):
        """Do deferred work. Call once per loop."""
        if self._adv_pending and self.conn is None:
            try:
                self.advertise()
                self._adv_pending = False
            except OSError as e:
                _log("restarting advertising failed (%s); retrying next loop" % e)
        if self._mtu_pending and self.conn is not None:
            self._mtu_pending = False
            try:
                self.ble.gattc_exchange_mtu(self.conn)
            except OSError as e:
                # Some centrals start the exchange themselves; that's fine.
                _log("MTU exchange request failed:", e)

    def drop_if_silent(self, timeout_ms):
        """Disconnect a central that has sent no PING for timeout_ms.

        Fixes the "zombie" link: when the UNO Q app restarts, BlueZ can keep
        the connection open with nobody behind it, so no disconnect event
        ever arrives and the node would never advertise again. The node's
        state is cleared straight away instead of waiting for the disconnect
        event, which may come late or not at all. Returns True if it dropped
        the connection.
        """
        conn = self.conn
        if conn is None or time.ticks_diff(time.ticks_ms(), self.last_ping_ms) < timeout_ms:
            return False
        self.heartbeat_drops += 1
        _log("no PING for %d ms: disconnecting handle %d and advertising again (drop #%d)"
             % (timeout_ms, conn, self.heartbeat_drops))
        try:
            self.ble.gap_disconnect(conn)
        except OSError as e:
            _log("gap_disconnect failed (%s); forgetting the connection anyway" % e)
        self.conn = None
        self.mtu = _DEFAULT_MTU
        self._mtu_pending = False
        self._clear_subscriptions()
        self._adv_pending = True        # service() restarts advertising
        return True

    def take_connect_event(self):
        """True once after each new connection (used for the LED blink)."""
        if self._just_connected:
            self._just_connected = False
            return True
        return False

    def pop_control(self):
        """The last command written to the control characteristic, or None."""
        cmd = self._control
        self._control = None
        return cmd

    def samples_per_packet(self):
        """Largest n with 7 + 12n <= MTU - 3, between 1 and MAX_SAMPLES_PER_PACKET."""
        n = (self.mtu - 3 - config.PACKET_HEADER_BYTES) // config.BYTES_PER_SAMPLE
        if n < 1:
            return 1
        if n > config.MAX_SAMPLES_PER_PACKET:
            return config.MAX_SAMPLES_PER_PACKET
        return n

    def subscribed(self, value_handle):
        """True if the central has turned on notifications for this characteristic.

        MicroPython does not report CCCD writes as events, and gatts_notify
        sends whether or not the central subscribed. MicroPython places each
        characteristic's CCCD at value_handle + 1, and its value can be read
        back, so we check the "notify" bit (bit 0) there.
        """
        if self.conn is None:
            return False
        if not self._ccc_ok:
            return True
        try:
            return bool(self.ble.gatts_read(value_handle + 1)[0] & 1)
        except (OSError, IndexError) as e:
            # Fall back to "always subscribed" rather than never sending.
            self._ccc_ok = False
            _log("can't read CCCD (%s); notifying without checking subscription" % e)
            return True

    def notify(self, value_handle, data):
        """Send one notification. Returns False (and carries on) if it fails."""
        if self.conn is None:
            return False
        try:
            self.ble.gatts_notify(self.conn, value_handle, data)
            return True
        except OSError:
            # Usually the BLE stack's TX queue is full. Skip this packet.
            self.notify_fails += 1
            return False

    def set_battery(self, mv):
        """Store vsys_mV so a read gets it, and notify it if subscribed."""
        self.ble.gatts_write(self.h_battery, struct.pack("<H", mv))
        if self.subscribed(self.h_battery):
            self.notify(self.h_battery, None)   # None = send the stored value

    def _clear_subscriptions(self):
        for h in (self.h_imu, self.h_battery):
            try:
                self.ble.gatts_write(h + 1, b"\x00\x00")
            except OSError:
                pass
