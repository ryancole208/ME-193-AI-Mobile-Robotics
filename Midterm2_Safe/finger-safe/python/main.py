"""Finger-Count Safe on the UNO Q (Linux side).

frames -> finger-count CNN (ONNX Runtime) -> 5-frame smoothing -> lock logic
-> Bridge.call("update_state", mode, digits, count, hold%) -> sketch (LED matrix + servos).
The sketch calls Bridge "button_pressed" when the push button is pressed (wake / restart).

Two frame sources (SOURCE below):
  "laptop"  The camera is on the laptop. Midterm2_Safe/stream_to_unoq.py sends
            224x224 letterboxed BGR frames over the USB-C cable (adb forward to
            STREAM_PORT). One reply line of JSON goes back per frame. The model
            still runs here on the board.
  "usb"     The camera is plugged into the board (through a powered USB-C hub).

Model files: python/model/finger_cnn.onnx + model_config.json (from `python -m ml.export`).
Without them the app still runs: the matrix blinks "!" and it re-checks every 5 s.
infer_standalone.py is a copy of Midterm2_Safe/infer_standalone.py; keep them identical.
"""
import json
import os
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from arduino.app_utils import App, Bridge
    ON_BOARD = True
except ImportError:   # laptop dry run
    ON_BOARD = False

import infer_standalone as fs

# ===================== SETTINGS =====================
SOURCE = os.environ.get("SOURCE", "auto")       # "auto", "usb" or "laptop"
                            # auto = camera plugged into the board if one is found,
                            # otherwise wait for the laptop stream (re-checked every 5 s)
STREAM_PORT = 8770          # must match app.yaml `ports` and stream_to_unoq.py
MODEL_DIR = Path(__file__).resolve().parent / "model"
# "usb" only. None = find the webcam automatically. On the UNO Q, /dev/video0
# and /dev/video1 are the Qualcomm "venus" hardware encoder, NOT a camera.
CAMERA_INDEX = int(os.environ["CAMERA_INDEX"]) if "CAMERA_INDEX" in os.environ else None
SECRET_CODE = None          # None = use model_config.json's code (3, 1, 4); or e.g. (2, 5, 1)
AUTO_WAKE = False           # True = wake by itself whenever asleep (use if no button is wired).
                            # The button on D2 is wired (2026-11-11).
ONNX_THREADS = 2
RETRY_SECONDS = 5
HEARTBEAT_SECONDS = 1.0     # resend the state at least this often
# ====================================================

# Keep in sync with MODE_* in sketch/sketch.ino.
MODE_ASLEEP, MODE_AWAKE, MODE_UNLOCKED, MODE_LOCKED, MODE_ERROR = range(5)

# Laptop -> board frame header (keep in sync with stream_to_unoq.py):
# magic b"FSF1", flags (bit 0 = button press), height, width, payload bytes.
HEADER = struct.Struct("<4sBHHI")
MAGIC = b"FSF1"
FLAG_BUTTON = 1

if not ON_BOARD:
    class Bridge:   # noqa: D101 - stand-in that prints
        @staticmethod
        def provide(name, fn):
            pass

        @staticmethod
        def call(name, *args):
            print(f"[bridge] {name}{args}")


def find_usb_camera():
    """Index of the first V4L2 capture device that isn't the SoC encoder/decoder, or None."""
    sysfs = Path("/sys/class/video4linux")
    if not sysfs.exists():
        return 0   # not Linux (laptop dry run)
    for dev in sorted(sysfs.glob("video*"), key=lambda d: int(d.name[5:])):
        name = (dev / "name").read_text().strip() if (dev / "name").exists() else ""
        if "venus" in name.lower():
            continue
        index = (dev / "index").read_text().strip() if (dev / "index").exists() else "0"
        if index != "0":   # UVC cameras add a 2nd metadata-only node (index 1)
            continue
        print(f"Found camera /dev/{dev.name}: {name}")
        return int(dev.name[5:])
    return None


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("laptop disconnected")
        buf += chunk
    return bytes(buf)


class LaptopSource:
    """TCP server: one laptop client at a time, lock-step frame -> reply."""

    def __init__(self, port):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("0.0.0.0", port))
        self.srv.listen(1)
        self.srv.settimeout(1.0)
        self.conn = None
        print(f"Waiting for the laptop stream on port {port}")

    def read(self):
        """(frame, button_pressed) or (None, False) if no laptop / it left."""
        if self.conn is None:
            try:
                self.conn, addr = self.srv.accept()
            except socket.timeout:
                return None, False
            self.conn.settimeout(5.0)
            self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Laptop connected from {addr}")
        try:
            magic, flags, h, w, n = HEADER.unpack(recv_exact(self.conn, HEADER.size))
            if magic != MAGIC or n != h * w * 3:
                raise ConnectionError(f"bad header {magic!r} {h}x{w} {n}")
            frame = np.frombuffer(recv_exact(self.conn, n), np.uint8).reshape(h, w, 3)
            return frame, bool(flags & FLAG_BUTTON)
        except (OSError, ConnectionError) as e:
            print(f"Laptop stream ended: {e}")
            self.drop()
            return None, False

    def reply(self, info):
        if self.conn is None:
            return
        try:
            self.conn.sendall((json.dumps(info) + "\n").encode())
        except OSError:
            self.drop()

    def drop(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        self.conn = None

    def describe(self):
        return "laptop stream"


class UsbSource:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.cap = None

    def read(self):
        if self.cap is None:
            index = CAMERA_INDEX if CAMERA_INDEX is not None else find_usb_camera()
            if index is None:
                print("No USB camera found (only the SoC encoder). Plug the webcam in via a USB-C hub.")
                time.sleep(RETRY_SECONDS)
                return None, False
            cap = fs.open_camera(index, self.width, self.height)
            if not cap.isOpened():
                print(f"Camera /dev/video{index} did not open")
                cap.release()
                time.sleep(RETRY_SECONDS)
                return None, False
            w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera /dev/video{index} open at {w:.0f}x{h:.0f}")
            self.cap = cap
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print("Camera read failed; reopening")
            self.cap.release()
            self.cap = None
            return None, False
        return frame, False

    def reply(self, info):
        pass


class AutoSource:
    """Board camera when one is plugged in; otherwise the laptop stream.
    A connected laptop keeps priority until it disconnects."""

    def __init__(self, width, height, port):
        self.usb = UsbSource(width, height)
        self.port = port
        self.laptop = None
        self.next_check = 0.0
        self.using = None

    def _switch(self, name):
        if name != self.using:
            print(f"Frame source: {name}")
            self.using = name

    def read(self):
        if self.usb.cap is not None:
            frame, pressed = self.usb.read()
            if frame is not None:
                self._switch("board camera")
            return frame, pressed
        laptop_busy = self.laptop is not None and self.laptop.conn is not None
        now = time.monotonic()
        if not laptop_busy and now >= self.next_check:
            self.next_check = now + RETRY_SECONDS
            if CAMERA_INDEX is not None or find_usb_camera() is not None:
                return self.usb.read()          # opens it (sleeps + retries if it fails)
        if self.laptop is None:
            self.laptop = LaptopSource(self.port)
        frame, pressed = self.laptop.read()     # waits up to 1 s for a laptop
        if frame is not None:
            self._switch("laptop stream")
        return frame, pressed

    def reply(self, info):
        if self.using == "laptop stream" and self.laptop is not None:
            self.laptop.reply(info)


class Safe:
    def __init__(self):
        self.mutex = threading.Lock()
        self.model = None
        self.lock = None
        self.smoother = None
        self.source = None
        self.next_retry = 0.0
        self.last_sent = None
        self.last_sent_t = 0.0
        self.last_print = None
        self.frames = 0
        self.t_fps = time.monotonic()

    def ensure_model(self):
        if self.model is not None:
            return True
        try:
            self.model = fs.FingerModel(MODEL_DIR, threads=ONNX_THREADS)
        except FileNotFoundError:
            print(f"No model yet: put finger_cnn.onnx + model_config.json in {MODEL_DIR}")
            return False
        except Exception as e:  # bad/old files: say so, keep retrying
            print(f"Model failed to load: {e!r}")
            return False
        cfg = self.model.cfg
        with self.mutex:
            self.lock = fs.LockLogic.from_config(cfg, SECRET_CODE)
        self.smoother = fs.Smoother(self.model.counts, cfg["smooth_frames"])
        print(f"Loaded model {cfg['run_id']} (val acc {cfg.get('val_accuracy')}), "
              f"code {self.lock.code}, source {SOURCE}")
        w, h = cfg["capture"]["width"], cfg["capture"]["height"]
        if SOURCE == "laptop":
            self.source = LaptopSource(STREAM_PORT)
        elif SOURCE == "usb":
            self.source = UsbSource(w, h)
        else:
            self.source = AutoSource(w, h, STREAM_PORT)
        return True

    # ---- Bridge ----
    def on_button(self):
        with self.mutex:
            if self.lock is not None:
                self.lock.press(time.monotonic())
                print("Button: awake, show your first number")
        return True

    def send(self, mode, digits, count, progress):
        state = (mode, digits, count, progress)
        now = time.monotonic()
        if state != self.last_sent or now - self.last_sent_t > HEARTBEAT_SECONDS:
            try:
                Bridge.call("update_state", *state)
            except Exception as e:
                print(f"Bridge call failed: {e!r}")
            self.last_sent, self.last_sent_t = state, now

    # ---- one iteration ----
    def step(self):
        now = time.monotonic()
        if now < self.next_retry:
            time.sleep(0.05)
            return
        if not self.ensure_model():
            self.send(MODE_ERROR, 0, -1, 0)
            self.next_retry = now + RETRY_SECONDS
            return

        frame, pressed = self.source.read()
        if frame is None:
            self.smoother.clear()
            self.send(MODE_ERROR, 0, -1, 0)
            return
        if pressed:
            self.on_button()

        t0 = time.perf_counter()
        p = self.model.probabilities(frame)
        ms = (time.perf_counter() - t0) * 1000
        count, conf = self.smoother.add(p)
        now = time.monotonic()
        with self.mutex:
            if AUTO_WAKE and self.lock.state == "asleep":
                self.lock.press(now)
            digit = self.lock.update(count, conf, now)
            st, entered, progress = self.lock.state, list(self.lock.entered), self.lock.progress
            result, message = self.lock.result, self.lock.message
        if digit is not None:
            print(f"Digit {digit} registered ({len(entered)}/3)")

        if st == "asleep":
            mode = MODE_ASLEEP
        elif st == "awake":
            mode = MODE_AWAKE
        else:
            mode = MODE_UNLOCKED if result == "unlocked" else MODE_LOCKED
        self.send(mode, len(entered), count, int(progress * 100))
        self.source.reply({
            "count": count, "confidence": round(conf, 3),
            "raw_count": self.model.counts[int(p.argmax())],
            "state": st, "entered": entered, "progress": round(progress, 2),
            "message": message, "result": result, "model_ms": round(ms, 1),
        })

        if (st, message) != self.last_print:
            print(f"[{st}] {message}")
            self.last_print = (st, message)
        self.frames += 1
        if now - self.t_fps > 10:
            print(f"{self.frames / (now - self.t_fps):.1f} fps, model {ms:.1f} ms, "
                  f"last count {count} ({conf:.2f})")
            self.frames, self.t_fps = 0, now


safe = Safe()
Bridge.provide("button_pressed", safe.on_button)


def loop():
    safe.step()


if ON_BOARD:
    App.run(user_loop=loop)
elif __name__ == "__main__":
    try:
        while True:
            loop()
    except KeyboardInterrupt:
        pass
