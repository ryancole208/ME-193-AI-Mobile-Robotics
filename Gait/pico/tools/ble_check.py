"""tools/ble_check.py - laptop-side checker for the GaitNode (runs on the PC, NOT the Pico).

Setup (once):   pip install bleak
Step 3 test:    python ble_check.py --counter          (with tests/t3_ble_counter.py running)
Step 4 test:    python ble_check.py                    (with main.py running; 5 minutes)
                python ble_check.py --seconds 60       (shorter run)

Full mode decodes every IMU packet, converts to g and deg/s, and at the end
reports packet loss, sample loss (from seq gaps AND t0_us jumps, as the
protocol requires) and the effective sample rate.

Close nRF Connect (or anything else connected to the GaitNode) first: the
node accepts only one connection at a time and stops advertising while
connected.
"""

import argparse
import asyncio
import struct
import sys
import time

from bleak import BleakClient, BleakScanner

NAME = "GaitNode"
IMU_UUID = "6e400001-7a1b-4c2d-9e10-a5c0f0e1d200"
BATTERY_UUID = "6e400002-7a1b-4c2d-9e10-a5c0f0e1d200"
CONTROL_UUID = "6e400003-7a1b-4c2d-9e10-a5c0f0e1d200"
INFO_UUID = "6e400004-7a1b-4c2d-9e10-a5c0f0e1d200"

HEADER = struct.Struct("<HIB")    # seq, t0_us, n  (7 bytes)
PERIOD_US = 10_000
ACCEL_LSB_PER_G = 2048.0
GYRO_LSB_PER_DPS = 16.4


class Stats:
    def __init__(self):
        self.packets = 0
        self.samples = 0
        self.bad_len = 0
        self.seq_gaps = 0          # packets missing according to seq
        self.t0_gap_samples = 0    # samples missing according to t0_us
        self.n_seen = {}
        self.prev = None           # (seq, t0, n) of the previous packet
        self.first_wall = None
        self.last_wall = None
        self.first_t0_total = 0    # sensor time covered, unwrapped, in us
        self.sensor_us = 0
        self.last_sample = None
        self.battery_mv = None

    def on_imu(self, data):
        now = time.monotonic()
        if len(data) < HEADER.size:
            self.bad_len += 1
            return
        seq, t0, n = HEADER.unpack_from(data, 0)
        if not 1 <= n <= 10 or len(data) != HEADER.size + 12 * n:
            self.bad_len += 1
            print("bad packet: %d bytes, n=%d" % (len(data), n))
            return
        raw = struct.unpack_from("<%dh" % (6 * n), data, HEADER.size)

        if self.prev is not None:
            pseq, pt0, pn = self.prev
            missing = (seq - pseq - 1) & 0xFFFF
            expected_t0 = (pt0 + pn * PERIOD_US) & 0xFFFFFFFF
            jump_us = (t0 - expected_t0) & 0xFFFFFFFF
            if jump_us >= 1 << 31:          # t0 went backwards: treat as no gap
                jump_us = 0
            if missing:
                self.seq_gaps += missing
                print("  seq gap: %d packet(s) missing before seq %d" % (missing, seq))
            if jump_us:
                self.t0_gap_samples += jump_us // PERIOD_US
                print("  t0 jump: about %d sample(s) missing before seq %d"
                      % (jump_us // PERIOD_US, seq))
            self.sensor_us += (t0 - pt0) & 0xFFFFFFFF
        else:
            self.first_wall = now

        self.prev = (seq, t0, n)
        self.last_wall = now
        self.packets += 1
        self.samples += n
        self.n_seen[n] = self.n_seen.get(n, 0) + 1
        self.last_sample = raw[-6:]

    def report(self, final=False):
        dur = (self.last_wall - self.first_wall) if self.packets > 1 else 0
        expected = self.samples + self.t0_gap_samples
        loss = 100.0 * self.t0_gap_samples / expected if expected else 0.0
        ploss = 100.0 * self.seq_gaps / (self.packets + self.seq_gaps) if self.packets else 0.0
        rate = self.samples / dur if dur > 0 else 0.0
        if not final:
            s = self.last_sample
            conv = ""
            if s:
                conv = ("  acc g %+5.2f %+5.2f %+5.2f  gyro dps %+7.1f %+7.1f %+7.1f"
                        % (s[0] / ACCEL_LSB_PER_G, s[1] / ACCEL_LSB_PER_G, s[2] / ACCEL_LSB_PER_G,
                           s[3] / GYRO_LSB_PER_DPS, s[4] / GYRO_LSB_PER_DPS, s[5] / GYRO_LSB_PER_DPS))
            print("%4.0f s  pkts %6d  rate %6.2f Hz  pkt loss %5.2f%%%s"
                  % (dur, self.packets, rate, ploss, conv))
            return
        print()
        print("=== GaitNode summary ===")
        print("duration            : %.1f s" % dur)
        print("packets received    : %d   (n values seen: %s)" % (self.packets, self.n_seen))
        print("bad packets         : %d" % self.bad_len)
        print("packets lost (seq)  : %d  = %.2f %%" % (self.seq_gaps, ploss))
        print("samples received    : %d" % self.samples)
        print("samples lost (t0)   : %d  = %.2f %%  (target < 2 %%)" % (self.t0_gap_samples, loss))
        print("effective rate      : %.2f Hz received  (100 Hz x (1 - loss))" % rate)
        if dur > 10 and self.sensor_us:
            ratio = self.sensor_us / 1e6 / dur
            print("sensor clock        : %.4f x laptop clock  (MPU-6050 spec is within +-1 %%)" % ratio)
        if self.battery_mv is not None:
            print("last battery        : %d mV%s" % (self.battery_mv,
                  "  (0 = battery reading disabled)" if self.battery_mv == 0 else ""))


async def find(name, address):
    if address:
        return address
    print("scanning for '%s' (15 s)..." % name)
    dev = await BleakScanner.find_device_by_name(name, timeout=15.0)
    if dev is None:
        print("NOT FOUND. Is the Pico W running main.py / t3_ble_counter.py and the LED blinking "
              "slowly? Is nRF Connect still connected to it?")
        sys.exit(1)
    print("found %s at %s" % (name, dev.address))
    return dev


async def run_counter(target, seconds):
    last = [None]
    count = [0]
    gaps = [0]

    def on_notify(_char, data):
        if len(data) != 4:
            print("unexpected %d-byte notification: %s" % (len(data), data.hex()))
            return
        v = struct.unpack("<I", data)[0]
        if last[0] is not None and v != last[0] + 1:
            gaps[0] += v - last[0] - 1
        last[0] = v
        count[0] += 1
        if count[0] % 10 == 0:
            print("counter %d  (received %d, missed %d)" % (v, count[0], gaps[0]))

    async with BleakClient(target) as client:
        print("connected, MTU %d" % client.mtu_size)
        await client.start_notify(IMU_UUID, on_notify)
        await asyncio.sleep(seconds)
        await client.stop_notify(IMU_UUID)
    print("done: received %d, missed %d" % (count[0], gaps[0]))


async def run_full(target, seconds):
    st = Stats()

    def on_battery(_char, data):
        mv = struct.unpack("<H", bytes(data[:2]))[0]
        if mv != st.battery_mv:          # every PING gets a reply; print changes only
            print("  battery: %d mV%s" % (mv, " (reading disabled)" if mv == 0 else ""))
        st.battery_mv = mv

    async with BleakClient(target) as client:
        print("connected, MTU %d (need >= 130 for 10 samples/packet)" % client.mtu_size)
        info = await client.read_gatt_char(INFO_UUID)
        ver, acc_g, gyro_dps, rate = struct.unpack("<BBHH", bytes(info[:6]))
        print("info: protocol v%d, +-%d g, +-%d deg/s, %d Hz" % (ver, acc_g, gyro_dps, rate))
        await client.start_notify(IMU_UUID, lambda _c, d: st.on_imu(bytes(d)))
        await client.start_notify(BATTERY_UUID, on_battery)
        await client.write_gatt_char(CONTROL_UUID, b"PING", response=True)
        print("subscribed; sending PING every 2 s as the heartbeat (the node drops the link "
              "after 6 s without one). Running %d s, Ctrl+C to stop early." % seconds)
        t_end = time.monotonic() + seconds
        tick = 0
        try:
            while time.monotonic() < t_end and client.is_connected:
                await asyncio.sleep(1.0)
                tick += 1
                if tick % 2 == 0:
                    try:
                        await client.write_gatt_char(CONTROL_UUID, b"PING", response=True)
                    except Exception as e:          # link lost; the loop condition ends it
                        print("  PING failed: %s" % e)
                st.report()
        except asyncio.CancelledError:
            pass
        if not client.is_connected:
            print("DISCONNECTED by the node or out of range")
    st.report(final=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counter", action="store_true", help="step 3: expect the t3 counter")
    ap.add_argument("--seconds", type=float, default=300, help="test length (default 300)")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--address", help="skip scanning and connect to this address")
    args = ap.parse_args()

    async def go():
        target = await find(args.name, args.address)
        if args.counter:
            await run_counter(target, args.seconds)
        else:
            await run_full(target, args.seconds)

    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
