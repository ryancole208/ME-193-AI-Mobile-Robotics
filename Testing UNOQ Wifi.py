"""Advertise an Arduino UNO Q over mDNS, equivalent to:
dns-sd -P "RQ-D2" _arduino._tcp local 80 RQ-D2.local 10.5.12.199 board=unoq ...
Requires: pip install zeroconf
"""
import socket
import time

from zeroconf import ServiceInfo, Zeroconf

info = ServiceInfo(
    type_="_arduino._tcp.local.",
    name="RQ-D2._arduino._tcp.local.",
    addresses=[socket.inet_aton("172.20.10.2")],
    port=80,
    server="RQ-D2.local.",
    properties={
        "board": "unoq",
        "vid": "0x2341",
        "pid": "0x0078",
        "vid.0": "0x2341",
        "pid.0": "0x0078",
    },
)

zc = Zeroconf()
zc.register_service(info)
print("Advertising RQ-D2 (172.20.10.2). Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    zc.unregister_service(info)
    zc.close()