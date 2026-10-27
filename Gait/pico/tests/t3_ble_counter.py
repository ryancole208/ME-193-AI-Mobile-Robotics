# tests/t3_ble_counter.py - build step 3: does Bluetooth work?
#
# Before running: upload config.py and ble_node.py to the Pico W.
# (The IMU is not used here; this works with nothing wired up.)
# To run: open this file in Thonny and press F5. Stop with Ctrl+C.
#
# Advertises as "GaitNode" with the real GaitNode service. Once a central
# subscribes to the IMU characteristic, it notifies a 4-byte little-endian
# counter (uint32) 10 times a second. Check it with the nRF Connect phone app
# or with tools/ble_check.py --counter on the laptop.

import struct
import time

import config
from ble_node import BLENode

print("t3_ble_counter: BLE advertising + counter test")
node = BLENode()
print("Advertising as '%s'. Connect with nRF Connect or tools/ble_check.py --counter."
      % config.BLE_NAME)

buf = bytearray(4)
counter = 0
was_connected = False
was_subscribed = False
t_next = time.ticks_ms()

while True:
    node.service()

    connected = node.conn is not None
    if connected != was_connected:
        print("CONNECTED" if connected else "DISCONNECTED (advertising again)")
        was_connected = connected

    sub = node.subscribed(node.h_imu)
    if sub != was_subscribed:
        print("central %s IMU notifications (MTU %d)"
              % ("SUBSCRIBED to" if sub else "unsubscribed from", node.mtu))
        was_subscribed = sub

    cmd = node.pop_control()
    if cmd is not None:
        print("control write:", cmd)

    if sub:
        struct.pack_into("<I", buf, 0, counter)
        if node.notify(node.h_imu, buf):
            counter += 1
            if counter % 50 == 0:
                print("sent counter", counter, " notify failures:", node.notify_fails)

    t_next = time.ticks_add(t_next, 100)
    wait = time.ticks_diff(t_next, time.ticks_ms())
    if wait > 0:
        time.sleep_ms(wait)
    else:
        t_next = time.ticks_ms()
