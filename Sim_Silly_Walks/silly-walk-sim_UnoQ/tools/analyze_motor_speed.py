"""Estimate each arm's rotation speed from the IMU recording of the step 2 motor-speed sketch.

An arm that is not perfectly balanced shakes the chassis once per revolution (a symmetric arm can show
twice per revolution instead), so the strongest vibration frequency between 0.3 and 12 Hz tells us the
rotation speed. Your hand count of revolutions decides between "once" and "twice per revolution".

Usage (standard library only):
    python3 tools/analyze_motor_speed.py logfile.txt      # a saved monitor log (tools/monitor.py ... logfile)
"""
import math
import re
import sys

FS = 100.0                     # sample rate of the sketch (Hz)
F_MIN, F_MAX, F_STEP = 0.3, 12.0, 0.01


def spectrum_peak(x):
    """Frequency (Hz) with the largest DFT magnitude in [F_MIN, F_MAX], plus that magnitude."""
    n = len(x)
    m = sum(x) / n
    x = [v - m for v in x]
    w = [0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]   # Hann window
    xw = [a * b for a, b in zip(x, w)]
    best = (0.0, 0.0)
    f = F_MIN
    while f <= F_MAX:
        c = s = 0.0
        k = 2 * math.pi * f / FS
        for i, v in enumerate(xw):
            c += v * math.cos(k * i)
            s += v * math.sin(k * i)
        mag = math.hypot(c, s)
        if mag > best[1]:
            best = (f, mag)
        f += F_STEP
    return best


def main():
    text = open(sys.argv[1], errors="replace").read()
    for name in ("LEFT", "RIGHT"):
        m = re.search(rf"# BEGIN {name}\n(.*?)# END {name}", text, re.S)
        if not m:
            print(f"{name}: no recording found")
            continue
        rows = [list(map(float, l.split()[2:])) for l in m.group(1).splitlines() if l.startswith("D ")]
        rows = rows[100:]   # skip the first second (motor spinning up)
        print(f"{name}: {len(rows)} samples")
        cols = ["gx", "gy", "gz", "ax", "ay", "az"]
        peaks = []
        for j, c in enumerate(cols):
            f, mag = spectrum_peak([r[j] for r in rows])
            peaks.append((mag / (max(abs(r[j]) for r in rows) + 1e-9), f, c))
            print(f"  {c}: peak {f:5.2f} Hz")
        _, f, c = max(peaks)
        print(f"  strongest ({c}): {f:.2f} Hz -> if once per rev: {2 * math.pi * f:.1f} rad/s ({60 * f:.0f} rpm); "
              f"if twice per rev: {math.pi * f:.1f} rad/s ({30 * f:.0f} rpm)")


if __name__ == "__main__":
    main()
