"""Print (and optionally save) the sketch's Monitor output by reading the router's monitor port directly.

Same output as `arduino-app-cli monitor`, but also works from scripts and non-interactive shells.
Usage:  python3 tools/monitor.py [seconds] [logfile]      (no seconds = run until Ctrl+C)
"""
import socket
import sys
import time

secs = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
log = open(sys.argv[2], "ab") if len(sys.argv) > 2 else None
s = socket.create_connection(("127.0.0.1", 7500))   # arduino-router's monitor port
s.settimeout(0.5)
t0 = time.time()
try:
    while time.time() - t0 < secs:
        try:
            d = s.recv(4096)
        except socket.timeout:
            continue
        if not d:
            break
        sys.stdout.write(d.decode(errors="replace"))
        sys.stdout.flush()
        if log:
            log.write(d)
            log.flush()
except KeyboardInterrupt:
    pass
