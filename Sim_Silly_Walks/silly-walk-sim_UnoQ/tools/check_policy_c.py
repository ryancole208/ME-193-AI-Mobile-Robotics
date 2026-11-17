"""Compare the STM32's policy output (step 5 test sketch) with the Python reference.

Run from the app folder while the step 5 test sketch is running:
    python3 tools/check_policy_c.py
"""
import json
import re
import socket
import time
from pathlib import Path

TOL = 1e-4
expected = json.loads((Path(__file__).parent / "policy_test_vectors.json").read_text())

s = socket.create_connection(("127.0.0.1", 7500))   # arduino-router monitor port
s.settimeout(0.5)
buf, t0 = "", time.time()
PATTERN = re.compile(r"# POLICY CHECK BEGIN(.*?)# POLICY CHECK END", re.S)   # one full printout
while not PATTERN.search(buf):
    if time.time() - t0 > 15:
        raise SystemExit("no policy check output from the sketch within 15 s - is the step 5 sketch running?")
    try:
        buf += s.recv(4096).decode(errors="replace")
    except socket.timeout:
        pass
block = PATTERN.search(buf).group(1)

worst = 0.0
for m in re.finditer(r"^V (\d+) (\S+) (\S+)", block, re.M):
    i, got = int(m.group(1)), (float(m.group(2)), float(m.group(3)))
    want = expected[i]["action"]
    err = max(abs(g - w) for g, w in zip(got, want))
    worst = max(worst, err)
    print(f"test {i:2d}: STM32 ({got[0]:+.5f}, {got[1]:+.5f})  python ({want[0]:+.5f}, {want[1]:+.5f})  err {err:.1e}")
print(re.search(r"# TIME.*", block).group(0))
print(f"max error {worst:.2e} -> {'PASS' if worst < TOL else 'FAIL'} (tolerance {TOL})")
