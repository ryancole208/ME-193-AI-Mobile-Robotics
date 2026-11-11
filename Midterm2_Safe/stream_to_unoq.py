"""Send the DICCIK camera from the laptop to the Finger Safe app on the UNO Q.

    ..\\my_env\\Scripts\\python.exe stream_to_unoq.py

The UNO Q stays on the laptop's USB-C port; the camera is on another laptop USB
port. This script:
  1. runs `adb forward tcp:8770 tcp:8770` (laptop localhost:8770 -> board port 8770),
  2. opens the camera BY NAME (config.CAMERA_NAME), crops + letterboxes each frame
     to the 224x224 BGR image the model was trained on, and sends it raw (no JPEG),
  3. shows a window with the board's answer (count, state, message).
The model, lock logic, LED matrix and servos all run on the UNO Q.

Keys in the window:  B = push button (wake / restart),  Q or Esc = quit.
Close app.py first: only one program can use the camera.
"""
import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import config
from camera import Camera
from preprocess import frame_to_raw

PORT = 8770
HEADER = struct.Struct("<4sBHHI")   # keep in sync with ArduinoApps/finger-safe/python/main.py
MAGIC = b"FSF1"
FLAG_BUTTON = 1
APP_LAB_ADB = Path(os.environ.get("LOCALAPPDATA", "")) / "Arduino15/packages/arduino/tools/adb/32.0.0/adb.exe"


def find_adb():
    adb = shutil.which("adb") or (str(APP_LAB_ADB) if APP_LAB_ADB.exists() else None)
    if adb is None:
        raise SystemExit("adb not found (install Arduino App Lab, or put adb on PATH)")
    return adb


def forward(adb, port):
    out = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=15).stdout
    if not any(l.strip().endswith("device") for l in out.splitlines()[1:]):
        raise SystemExit("UNO Q not found over USB (adb devices is empty). Is the USB-C cable in?")
    subprocess.run([adb, "forward", f"tcp:{port}", f"tcp:{port}"], check=True, timeout=15)
    print(f"adb forward tcp:{port} -> UNO Q port {port}")


def connect(port):
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(5)
            return s, s.makefile("rb")
        except OSError:
            print("waiting for the Finger Safe app on the UNO Q ...")
            time.sleep(2)


def draw(frame, raw, info, fps):
    view = cv2.resize(frame, (854, 480))
    side = cv2.resize(raw, (480, 480), interpolation=cv2.INTER_NEAREST)
    out = np.hstack([view, side])
    lines = ["no answer from the UNO Q yet"] if info is None else [
        f"count {info['count']}  ({info['confidence']:.0%})   raw {info['raw_count']}",
        f"{info['state'].upper()}  code {info['entered']}",
        info["message"],
        f"board model {info['model_ms']} ms   {fps:.1f} fps",
    ]
    for i, t in enumerate(lines):
        y = 32 + i * 34
        cv2.putText(out, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if info and info.get("result"):
        color = (0, 200, 0) if info["result"] == "unlocked" else (0, 0, 220)
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, 12)
    cv2.putText(out, "B = button   Q = quit", (12, out.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-name", default=config.CAMERA_NAME)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-window", action="store_true", help="print results instead of a window")
    args = ap.parse_args()

    forward(find_adb(), args.port)
    cam = Camera(name=args.camera_name, width=config.CAPTURE_WIDTH, height=config.CAPTURE_HEIGHT)
    sock, rfile = connect(args.port)
    print("connected to the UNO Q")

    last_id, info, fps, t_last, press = -1, None, 0.0, None, False
    try:
        while True:
            last_id, _ = cam.wait_frame(last_id, timeout=1.0)
            _, frame = cam.get_raw_frame()
            if frame is None:
                print("camera not ready ...")
                continue
            raw = frame_to_raw(frame)   # 224x224 letterboxed BGR, same as the saved training PNGs
            flags = FLAG_BUTTON if press else 0
            press = False
            try:
                sock.sendall(HEADER.pack(MAGIC, flags, raw.shape[0], raw.shape[1], raw.nbytes))
                sock.sendall(raw.tobytes())
                line = rfile.readline()
                if not line:
                    raise ConnectionError("board closed the connection")
                info = json.loads(line)
            except (OSError, ConnectionError, ValueError) as e:
                print(f"lost the UNO Q ({e}); reconnecting")
                sock.close()
                info = None
                sock, rfile = connect(args.port)
                continue

            now = time.monotonic()
            if t_last is not None:
                inst = 1 / max(now - t_last, 1e-6)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            t_last = now

            if args.no_window:
                print(f"count {info['count']} ({info['confidence']:.2f}) [{info['state']}] "
                      f"{info['message']}  {fps:.1f} fps")
                continue
            cv2.imshow("Finger Safe -> UNO Q", draw(frame, raw, info, fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                press = True
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
