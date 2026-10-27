# main.py - GaitNode ankle sensor firmware for the Raspberry Pi Pico W.
#
# Runs automatically at every power-up. Reads the MPU-6050's FIFO about
# every 100 ms, packs the samples into protocol packets (README "Shared
# wireless protocol") and notifies them to the UNO Q over BLE.
#
# Upload LAST, after config.py, mpu6050_fifo.py and ble_node.py, as main.py.
# To stop it: Ctrl+C in Thonny's shell (or the Stop button). It waits
# STARTUP_DELAY_S seconds at power-up so this always works.

import gc
import struct
import sys
import time
import micropython
from machine import ADC, I2C, Pin, WDT, disable_irq, enable_irq, reset

import config

print("starting in %d s, press Ctrl+C to stop" % config.STARTUP_DELAY_S)
time.sleep(config.STARTUP_DELAY_S)

from ble_node import BLENode                               # noqa: E402
from mpu6050_fifo import MPU6050, BYTES_PER_SAMPLE, FIFO_SIZE  # noqa: E402

micropython.alloc_emergency_exception_buf(100)

HEADER = config.PACKET_HEADER_BYTES
NMAX = config.MAX_SAMPLES_PER_PACKET
RING = config.RESEND_BUFFER_PACKETS
PERIOD_US = config.SAMPLE_PERIOD_US


def log(*args):
    if config.DEBUG:
        print(*args)


def swap_copy(dst, off, src, nbytes):
    """Copy nbytes from src to dst[off:], swapping each byte pair.

    The MPU-6050 FIFO is big-endian; the protocol is little-endian, so
    swapping the two bytes of every int16 converts it.
    """
    i = 0
    while i < nbytes:
        dst[off + i] = src[i + 1]
        dst[off + i + 1] = src[i]
        i += 2


def read_vsys_mv():
    """Battery voltage at VSYS in millivolts, or 0 if disabled.

    Pico W method (Pico W datasheet, "Using the ADC"): VSYS/3 is on ADC3
    (GPIO29), but GPIO29 is also the wireless chip's SPI clock, and GPIO25
    (its chip select) must be high so the chip ignores the line while we
    measure. Interrupts are off for the ~20 us read so the BLE driver cannot
    start an SPI transfer mid-read. Afterwards GPIO29 is put back to its
    wireless function (the driver also re-applies it before each transfer).
    """
    if not config.BATTERY_READ_ENABLED:
        return 0
    state = disable_irq()
    try:
        Pin(25, Pin.OUT, value=1)
        Pin(29, Pin.IN)
        raw = ADC(29).read_u16()
        Pin(29, Pin.ALT, pull=Pin.PULL_DOWN, alt=7)
    finally:
        enable_irq(state)
    return raw * 9900 // 65535       # x3 divider x 3300 mV reference


class Node:
    def __init__(self):
        self.ble = BLENode()
        self.i2c = I2C(config.I2C_ID, sda=Pin(config.I2C_SDA_PIN),
                       scl=Pin(config.I2C_SCL_PIN), freq=config.I2C_FREQ)
        self.imu = MPU6050(self.i2c, config.MPU_ADDR, NMAX, config.WHO_AM_I_OK)
        self.led = Pin("LED", Pin.OUT)
        self.led_state = -1
        self.blink_ticks = 0

        # Resend ring: the last RING packets, built in place. views[slot][n]
        # is a ready-made memoryview of the first 7 + 12n bytes, so sending
        # never has to slice (slicing allocates).
        size = HEADER + BYTES_PER_SAMPLE * NMAX
        self.ring = [bytearray(size) for _ in range(RING)]
        self.views = []
        for b in self.ring:
            mv = memoryview(b)
            self.views.append([None] + [mv[:HEADER + BYTES_PER_SAMPLE * n]
                                        for n in range(1, NMAX + 1)])
        self.ring_seq = [-1] * RING
        self.ring_n = [0] * RING
        self.slot = 0

        # Packet header state.
        self.seq = 0
        # t0_us = samples x 10000 mod 2^32. MicroPython on the RP2040 only
        # stores integers below 2^30 without allocating memory, so the
        # 32-bit value is kept as two 16-bit halves. Packing them as "<HH"
        # gives exactly the same bytes as one little-endian uint32.
        self.t0_lo = 0
        self.t0_hi = 0

        # FIFO bookkeeping for overflow / error recovery.
        self.imu_ok = False
        self.imu_started = False   # False until the first successful start
        self.last_good_ms = 0      # when the FIFO was last emptied
        self.backlog = 0           # whole samples left in the FIFO then
        self.next_imu_retry = time.ticks_ms()

        self.streaming = False
        self.vsys_mv = 0

        # Counters (printed every 5 s with DEBUG, useful for step 5 tests).
        self.packets = 0
        self.sent = 0
        self.overflows = 0
        self.i2c_errors = 0
        self.reinits = 0
        self.lost = 0

    # --- time stamp ---------------------------------------------------------

    def advance(self, samples):
        """Add samples x 10000 us to t0, wrapping at 2^32."""
        while samples > 0:
            step = samples if samples < 10000 else 10000   # keeps ints < 2^30
            lo = self.t0_lo + step * PERIOD_US
            self.t0_hi = (self.t0_hi + (lo >> 16)) & 0xFFFF
            self.t0_lo = lo & 0xFFFF
            samples -= step

    # --- IMU ------------------------------------------------------------------

    def start_imu(self):
        """(Re)initialise the sensor and FIFO; account for samples lost meanwhile."""
        try:
            self.imu.configure(config.DLPF_CFG, config.SMPLRT_DIV,
                               config.ACCEL_RANGE_G, config.GYRO_RANGE_DPS)
            self.imu.fifo_start()
        except (OSError, RuntimeError) as e:
            self.i2c_errors += 1
            log("IMU init failed (%s); check the 4 sensor wires. Retrying in 1 s." % e)
            self.next_imu_retry = time.ticks_add(time.ticks_ms(), 1000)
            return
        self.resync(restarted=True)
        self.imu_ok = True
        log("IMU running")

    def resync(self, restarted=False):
        """Count samples lost since the last good read and move t0 past them.

        Called after an overflow or a re-initialisation. The FIFO is emptied,
        so the samples left in it at the last read (backlog) plus everything
        sampled since then are gone. They're estimated from the Pico W's
        clock. ticks_ms is used rather than ticks_us because ticks_us wraps
        after ~18 minutes, and 1 ms resolution is plenty against a 10 ms
        sample period.
        """
        if not restarted:
            self.imu.fifo_start()
        now = time.ticks_ms()
        if self.imu_started:
            gap = time.ticks_diff(now, self.last_good_ms)
            lost = self.backlog + (gap * 1000 + PERIOD_US // 2) // PERIOD_US
            self.advance(lost)
            self.lost += lost
            self.reinits += 1 if restarted else 0
            log("resync: about %d samples lost, t0 moved forward" % lost)
        self.imu_started = True
        self.last_good_ms = now
        self.backlog = 0

    def read_fifo(self):
        """Empty the FIFO into packets of n samples; leftovers wait for next time."""
        now = time.ticks_ms()
        if self.imu.fifo_overflowed():
            self.overflows += 1
            log("FIFO overflow #%d" % self.overflows)
            self.resync()
            return
        count = self.imu.fifo_count()
        if count >= FIFO_SIZE:
            self.overflows += 1
            log("FIFO full #%d" % self.overflows)
            self.resync()
            return
        k = count // BYTES_PER_SAMPLE
        n = self.ble.samples_per_packet()
        while k >= n:
            self.emit(self.imu.fifo_read(n), n)
            k -= n
        self.backlog = k
        self.last_good_ms = now

    # --- packets ----------------------------------------------------------------

    def emit(self, fifo_view, n):
        """Build one packet in the next ring slot and send it if streaming."""
        slot = self.slot
        buf = self.ring[slot]
        struct.pack_into("<HHHB", buf, 0, self.seq, self.t0_lo, self.t0_hi, n)
        swap_copy(buf, HEADER, fifo_view, n * BYTES_PER_SAMPLE)
        self.ring_seq[slot] = self.seq
        self.ring_n[slot] = n
        self.packets += 1
        if self.streaming and self.ble.notify(self.ble.h_imu, self.views[slot][n]):
            self.sent += 1
        self.slot = (slot + 1) % RING
        self.seq = (self.seq + 1) & 0xFFFF
        self.advance(n)

    def handle_control(self, cmd):
        try:
            text = cmd.decode().strip().upper()
        except UnicodeError:
            text = ""
        if text == "PING":
            # PING doubles as the heartbeat (ble_node stamps it), so it
            # arrives every 2 s. Reply with the reading from the 10 s timer
            # rather than reading VSYS again: the VSYS pin is shared with
            # the wireless chip, so read it no more often than needed. Not
            # logged, since it would flood the shell; see pings= in status.
            self.ble.set_battery(self.vsys_mv)
        elif text.startswith("RESEND"):
            try:
                want = int(text[6:])
            except ValueError:
                log("bad RESEND command: %r" % cmd)
                return
            for slot in range(RING):
                if self.ring_seq[slot] == want:
                    self.ble.notify(self.ble.h_imu, self.views[slot][self.ring_n[slot]])
                    log("RESEND %d: sent" % want)
                    return
            log("RESEND %d: no longer in the buffer" % want)
        else:
            log("unknown control command: %r" % cmd)

    # --- LED --------------------------------------------------------------------

    def update_led(self, loop):
        if self.ble.take_connect_event():
            self.blink_ticks = 6                     # on/off x3 at 100 ms
        if self.blink_ticks > 0:
            on = 1 if self.blink_ticks % 2 == 0 else 0
            self.blink_ticks -= 1
        elif self.ble.conn is None:
            on = (loop // 5) % 2                     # 0.5 s on, 0.5 s off
        else:
            on = 0                                   # streaming: off saves power
        if on != self.led_state:                     # LED is on the wireless chip,
            self.led.value(on)                       # so only write on change
            self.led_state = on

    # --- main loop ----------------------------------------------------------------

    def run(self):
        wdt = None
        if config.WATCHDOG_ENABLED:
            wdt = WDT(timeout=config.WATCHDOG_TIMEOUT_MS)
            log("watchdog ON (%d ms)" % config.WATCHDOG_TIMEOUT_MS)
        if config.PING_TIMEOUT_MS:
            log("heartbeat ON: dropping a central after %d ms without PING"
                % config.PING_TIMEOUT_MS)
        else:
            log("heartbeat OFF (PING_TIMEOUT_MS = 0)")

        self.start_imu()
        self.vsys_mv = read_vsys_mv()
        self.ble.set_battery(self.vsys_mv)

        loop = 0
        t_next = time.ticks_ms()
        t_battery = time.ticks_add(t_next, config.BATTERY_PERIOD_MS)
        t_health = time.ticks_add(t_next, 1000)
        t_status = time.ticks_add(t_next, 5000)
        was_streaming = False

        while True:
            if wdt:
                wdt.feed()
            self.ble.service()
            now = time.ticks_ms()

            # Subscription: check every loop until subscribed, then once a
            # second (reading the CCCD makes a tiny object, so not too often).
            if self.ble.conn is None:
                self.streaming = False
            elif not self.streaming or loop % 10 == 0:
                self.streaming = self.ble.subscribed(self.ble.h_imu)
            if self.streaming != was_streaming:
                log("streaming %s (MTU %d, n = %d)" % (
                    "STARTED" if self.streaming else "stopped",
                    self.ble.mtu, self.ble.samples_per_packet()))
                was_streaming = self.streaming

            # IMU: keep emptying the FIFO even when nobody is connected, so it
            # never overflows; packets then only go into the resend ring.
            if self.imu_ok:
                try:
                    self.read_fifo()
                    if time.ticks_diff(now, t_health) >= 0:
                        t_health = time.ticks_add(now, 1000)
                        if not self.imu.healthy():
                            raise RuntimeError("sensor lost its settings")
                except (OSError, RuntimeError) as e:
                    self.i2c_errors += 1
                    self.imu_ok = False
                    log("IMU error #%d: %s -> reinitialising" % (self.i2c_errors, e))
                    self.next_imu_retry = now
            if not self.imu_ok and time.ticks_diff(now, self.next_imu_retry) >= 0:
                self.start_imu()

            cmd = self.ble.pop_control()
            if cmd is not None:
                self.handle_control(cmd)

            # Heartbeat: drop a connected central that has stopped sending
            # PING (dead link) so the node advertises again. Checked after
            # the control command so a PING that just arrived counts.
            if config.PING_TIMEOUT_MS:
                self.ble.drop_if_silent(config.PING_TIMEOUT_MS)

            if time.ticks_diff(now, t_battery) >= 0:
                t_battery = time.ticks_add(now, config.BATTERY_PERIOD_MS)
                self.vsys_mv = read_vsys_mv()
                self.ble.set_battery(self.vsys_mv)

            self.update_led(loop)

            if config.DEBUG and time.ticks_diff(now, t_status) >= 0:
                t_status = time.ticks_add(now, 5000)
                print("status: conn=%s stream=%s mtu=%d n=%d pkts=%d sent=%d notify_fail=%d "
                      "pings=%d hb_drops=%d ovf=%d i2c_err=%d reinit=%d lost=%d vsys=%dmV "
                      "mem_free=%d"
                      % (self.ble.conn is not None, self.streaming, self.ble.mtu,
                         self.ble.samples_per_packet(), self.packets, self.sent,
                         self.ble.notify_fails, self.ble.pings, self.ble.heartbeat_drops,
                         self.overflows, self.i2c_errors,
                         self.reinits, self.lost, self.vsys_mv, gc.mem_free()))

            gc.collect()        # quiet point: a short collection every loop
            loop += 1
            t_next = time.ticks_add(t_next, config.LOOP_PERIOD_MS)
            wait = time.ticks_diff(t_next, time.ticks_ms())
            if wait > 0:
                time.sleep_ms(wait)
            else:
                t_next = time.ticks_ms()   # fell behind; don't try to catch up


try:
    Node().run()
except KeyboardInterrupt:
    print("main.py stopped by Ctrl+C")
except Exception as e:
    print("main.py crashed:")
    sys.print_exception(e)
    if config.REBOOT_ON_CRASH:
        print("rebooting in 5 s (set REBOOT_ON_CRASH = False in config.py to stay stopped)")
        time.sleep(5)
        reset()
