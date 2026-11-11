"""Find new_angle1: servo 2's angle with the arm resting above the safe door.

    1. from ArduinoApps/:     python tools\\deploy.py servo-calibrate
    2. from Midterm2_Safe/:   ..\\my_env\\Scripts\\python.exe servo_calibrate.py   (in a terminal)

The locked position is (servo 1, servo 2) = (0, new_angle1). The rest of the
sequence (0/30 for servo 1, 67/97 for servo 2) is already set in the sketches.

Servos stay LIMP until you press E, so nothing moves on its own. The first angle
sent is the one shown on screen; change it with the arrows BEFORE pressing E.

Keys
  1 / 2          pick servo 1 (D9) or servo 2 (D10)
  Left / Right   -1 / +1 degree        (also A / D)
  Down / Up      -5 / +5 degrees       (also S / W)
  E              power the servo at the shown angle      X  make it limp again
  N              save servo 2's angle as new_angle1 (locked, resting on the door)
  T              servo 2: jump between new_angle1 and 97 (open) to check it
  P              write new_angle1 into the Finger Safe + test sketches and quit
  Q              quit without writing
"""
import msvcrt
import re
import socket
import subprocess
import sys
from pathlib import Path

from stream_to_unoq import find_adb

PORT = 8771
APPS = Path(__file__).resolve().parent.parent / "ArduinoApps"
SKETCHES = [APPS / "finger-safe/sketch/sketch.ino", APPS / "servo-button-test/sketch/sketch.ino"]
REST_RE = re.compile(r"(const int LOCK_REST_DEG = )(\d+)(;)")
OPEN_S2 = 97     # servo 2 in the open position


def read_rest():
    return int(REST_RE.search(SKETCHES[0].read_text(encoding="utf-8")).group(2))


def write_rest(deg):
    for path in SKETCHES:
        text = path.read_text(encoding="utf-8")
        text, n = REST_RE.subn(lambda m: f"{m.group(1)}{deg}{m.group(3)}", text)
        path.write_text(text, encoding="utf-8")
        print(f"  {path.relative_to(APPS.parent)}: LOCK_REST_DEG = {deg} ({n} line)")


class Board:
    def __init__(self):
        subprocess.run([find_adb(), "forward", f"tcp:{PORT}", f"tcp:{PORT}"], check=True,
                       timeout=15, capture_output=True)
        try:
            self.sock = socket.create_connection(("127.0.0.1", PORT), timeout=5)
        except OSError:
            sys.exit("Can't reach the servo-calibrate app. Deploy it first:\n"
                     "  cd ..\\ArduinoApps && python tools\\deploy.py servo-calibrate")
        self.f = self.sock.makefile("rwb")

    def set(self, i, deg):
        self.f.write(f"{i} {deg}\n".encode())
        self.f.flush()
        reply = self.f.readline().decode().strip()
        if reply != "ok":
            print(f"\n  board said: {reply or 'nothing (app stopped?)'}")

    def close(self):
        for i in range(2):
            try:
                self.set(i, -1)
            except OSError:
                pass
        self.sock.close()


def read_key():
    k = msvcrt.getwch()
    if k in ("\x00", "\xe0"):
        return {"K": "left", "M": "right", "H": "up", "P": "down"}.get(msvcrt.getwch(), "")
    return k.lower()


def main():
    print(__doc__)
    rest = read_rest()
    board = Board()
    angle = [0, OPEN_S2]      # start at known-good angles: servo 1 at 0, servo 2 at open (97)
    powered = [False, False]
    sel = 0
    step = {"left": -1, "a": -1, "right": 1, "d": 1, "down": -5, "s": -5, "up": 5, "w": 5}

    def status():
        p = "ON " if powered[sel] else "limp"
        print(f"\r servo {sel + 1} ({'D9' if sel == 0 else 'D10'}) [{p}] angle {angle[sel]:3d}   "
              f"new_angle1 = {rest}        ", end="", flush=True)

    try:
        status()
        while True:
            k = read_key()
            if k in ("1", "2"):
                sel = int(k) - 1
            elif k in step:
                angle[sel] = max(0, min(180, angle[sel] + step[k]))
                if powered[sel]:
                    board.set(sel, angle[sel])
            elif k == "e":
                powered[sel] = True
                board.set(sel, angle[sel])
            elif k == "x":
                powered[sel] = False
                board.set(sel, -1)
            elif k == "n":
                if sel != 1:
                    print("\n  N saves SERVO 2's angle: press 2 first.")
                else:
                    rest = angle[1]
            elif k == "t":
                angle[1] = OPEN_S2 if angle[1] == rest else rest
                powered[1] = True
                sel = 1
                board.set(1, angle[1])
            elif k == "p":
                print(f"\n\nSaving new_angle1 = {rest}:")
                write_rest(rest)
                print("Done. Redeploy finger-safe (and servo-button-test) to use it.")
                break
            elif k in ("q", "\x1b"):
                print("\nQuit without writing.")
                break
            status()
    except KeyboardInterrupt:
        print("\nQuit without writing.")
    finally:
        board.close()   # both servos limp


if __name__ == "__main__":
    main()
