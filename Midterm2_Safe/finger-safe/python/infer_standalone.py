"""Finger-count inference with NO project code: only cv2, numpy, onnxruntime.

The UNO Q app (ArduinoApps/finger-safe/python/) uses a copy of this file. Keep the
two copies identical (Midterm2_Safe/ is the original).

Needs, in one folder (default: models/ next to this file):
    finger_cnn.onnx  +  model_config.json      (made by `python -m ml.export`)

Command line (laptop checks):
    python infer_standalone.py --image data/raw/three/val1/<file>.png      # a saved 224x224 PNG
    python infer_standalone.py --image frame.jpg                           # a full camera frame
    python infer_standalone.py --dir data/raw --session val1               # accuracy on saved data
    python infer_standalone.py --camera 0                                  # live, prints counts

As a module:
    m = FingerModel("models")
    count, conf, probs = m.predict(frame)          # raw BGR camera frame
    lock = LockLogic.from_config(m.cfg)            # same rules as lock.py
"""
import argparse
import collections
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent


class FingerModel:
    def __init__(self, model_dir=HERE / "models", threads=2):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
        c = self.cfg
        self.classes = c["classes"]
        self.counts = [c["class_to_count"][k] for k in self.classes]
        self.raw_size = c["letterbox"]["size"]
        self.fill = c["letterbox"]["fill"]
        self.in_size = c["input_resize"]["size"]
        self.mean = np.float32(c["mean"])
        self.std = np.float32(c["std"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / c["onnx_file"]), so,
                                            providers=["CPUExecutionProvider"])
        self.input_name = c["input_name"]

    # ---- preprocessing (must match Midterm2_Safe/preprocess.py exactly) ----
    def crop(self, frame):
        h, w = frame.shape[:2]
        k = self.cfg["crop"]
        x0 = min(max(int(round(k["x0"] * w)), 0), w - 1)
        y0 = min(max(int(round(k["y0"] * h)), 0), h - 1)
        x1 = min(max(int(round(k["x1"] * w)), x0 + 1), w)
        y1 = min(max(int(round(k["y1"] * h)), y0 + 1), h)
        return frame[y0:y1, x0:x1]

    def letterbox(self, img):
        size = self.raw_size
        h, w = img.shape[:2]
        scale = size / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        out = np.full((size, size) + img.shape[2:], self.fill, img.dtype)
        x0, y0 = (size - nw) // 2, (size - nh) // 2
        out[y0:y0 + nh, x0:x0 + nw] = small
        return out

    def to_raw(self, frame):
        """Full camera frame -> 224x224 letterboxed BGR (the saved-PNG form).
        A frame that is already raw_size x raw_size is treated as a saved PNG."""
        if frame.shape[0] == frame.shape[1] == self.raw_size:
            return frame
        return self.letterbox(self.crop(frame))

    def tensor(self, frame):
        """BGR frame (or saved PNG) -> float32 (1, 3, 96, 96). BGR order, no RGB swap."""
        raw = self.to_raw(frame)
        img = cv2.resize(raw, (self.in_size, self.in_size), interpolation=cv2.INTER_AREA)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1))[None].astype(np.float32)

    # ---- model ----
    def probabilities(self, frame):
        logits = self.session.run(None, {self.input_name: self.tensor(frame)})[0][0]
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def predict(self, frame):
        p = self.probabilities(frame)
        i = int(p.argmax())
        return self.counts[i], float(p[i]), p


class Smoother:
    """Average the last n probability vectors (same as the laptop app)."""

    def __init__(self, counts, n):
        self.counts = counts
        self.buf = collections.deque(maxlen=n)

    def clear(self):
        self.buf.clear()

    def add(self, p):
        self.buf.append(p)
        avg = np.mean(self.buf, axis=0)
        i = int(avg.argmax())
        return self.counts[i], float(avg[i])


class LockLogic:
    """Port of Midterm2_Safe/lock.py (same rules, settings from model_config.json).

    asleep --press--> awake --3 digits--> result (unlocked / locked) --> asleep
    A digit registers when the same count 1-5 stays on top with smoothed
    confidence >= conf_threshold for hold_seconds; each digit must differ from
    the previous one; 0 never registers.
    """
    CODE_COUNTS = (1, 2, 3, 4, 5)

    def __init__(self, code, hold_seconds, conf_threshold, result_seconds, awake_timeout):
        code = tuple(int(d) for d in code)
        if len(code) != 3 or any(d not in self.CODE_COUNTS for d in code) or len(set(code)) != 3:
            raise ValueError(f"secret code must be 3 different numbers 1-5, got {code}")
        self.code = code
        self.hold_seconds = hold_seconds
        self.conf_threshold = conf_threshold
        self.result_seconds = result_seconds
        self.awake_timeout = awake_timeout
        self._sleep()

    @classmethod
    def from_config(cls, cfg, code=None):
        return cls(code or cfg["secret_code"], cfg["hold_seconds"], cfg["conf_threshold"],
                   cfg["result_seconds"], cfg["awake_timeout_seconds"])

    def _sleep(self):
        self.state = "asleep"
        self.entered = []
        self.hold_count = None
        self.hold_start = None
        self.progress = 0.0
        self.message = "Push the button to wake the safe"
        self.result = None
        self._until = None
        self._last_activity = None

    def press(self, now):
        self._sleep()
        self.state = "awake"
        self.message = "Show your first number"
        self._last_activity = now

    def reset(self):
        self._sleep()

    def update(self, count, confidence, now):
        """Feed the latest smoothed prediction. Returns the new digit if one registered."""
        if self.state == "result":
            if now >= self._until:
                self._sleep()
            return None
        if self.state != "awake":
            return None
        if now - self._last_activity > self.awake_timeout:
            self._sleep()
            self.message = "Timed out. Push the button to try again"
            return None
        if count not in self.CODE_COUNTS or confidence < self.conf_threshold:
            self._hold(None, now)
            self.message = ("Show a number (1-5)" if count not in self.CODE_COUNTS
                            else f"Hold {count} steady (confidence {confidence:.0%})")
            return None
        if self.entered and count == self.entered[-1]:
            self._hold(None, now)
            self.message = "Change your number"
            return None
        self._hold(count, now)
        self._last_activity = now
        self.progress = min(1.0, (now - self.hold_start) / self.hold_seconds)
        self.message = f"Holding {count}..."
        if self.progress < 1.0:
            return None
        self.entered.append(count)
        self._hold(None, now)
        if len(self.entered) < len(self.code):
            self.message = f"Got {count}. Next number"
            return count
        self.state = "result"
        self.result = "unlocked" if tuple(self.entered) == self.code else "locked"
        self.message = "UNLOCKED" if self.result == "unlocked" else "STAYS LOCKED"
        self._until = now + self.result_seconds
        return count

    def _hold(self, count, now):
        if count is None:
            self.hold_count, self.hold_start, self.progress = None, None, 0.0
        elif count != self.hold_count:
            self.hold_count, self.hold_start, self.progress = count, now, 0.0


def open_camera(index, width, height):
    """Open a V4L2/DirectShow camera at width x height with MJPG (set AFTER the size)."""
    import sys
    api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, api)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _eval_dir(m, root, session):
    root = Path(root)
    tot = ok = 0
    for cls, cnt in zip(m.classes, m.counts):
        paths = sorted((root / cls / session).glob("*.png"))
        right = sum(m.predict(cv2.imread(str(p)))[0] == cnt for p in paths)
        tot += len(paths)
        ok += right
        print(f"{cls:6s} {right}/{len(paths)}")
    print(f"accuracy {ok}/{tot} = {100 * ok / max(tot, 1):.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(HERE / "models"), help="folder with the .onnx + model_config.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image")
    g.add_argument("--dir", help="data/raw folder; evaluates <dir>/<class>/<session>/*.png")
    g.add_argument("--camera", type=int, help="camera index (0 = /dev/video0 on Linux)")
    ap.add_argument("--session", default="val1")
    args = ap.parse_args()

    m = FingerModel(args.models)
    print(f"model {m.cfg['run_id']}, classes {m.classes}")
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
        count, conf, p = m.predict(img)
        print(f"count {count}  confidence {conf:.3f}  probs {np.round(p, 3).tolist()}")
    elif args.dir:
        _eval_dir(m, args.dir, args.session)
    else:
        cap = open_camera(args.camera, m.cfg["capture"]["width"], m.cfg["capture"]["height"])
        if not cap.isOpened():
            raise SystemExit(f"camera {args.camera} did not open")
        sm = Smoother(m.counts, m.cfg["smooth_frames"])
        try:
            while True:
                okf, frame = cap.read()
                if not okf:
                    print("no frame")
                    time.sleep(0.5)
                    continue
                t0 = time.perf_counter()
                count, conf, p = m.predict(frame)
                ms = (time.perf_counter() - t0) * 1000
                sc, sconf = sm.add(p)
                print(f"count {count} ({conf:.2f})  smoothed {sc} ({sconf:.2f})  {ms:.1f} ms")
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()


if __name__ == "__main__":
    main()
