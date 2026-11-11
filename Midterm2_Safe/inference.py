"""Finger-count inference on the laptop (PyTorch).

    from inference import predict_count
    count, confidence, probabilities = predict_count(frame)   # frame = raw BGR camera frame

`LiveWorker` runs the model in its own thread on the newest camera frame,
smooths the probabilities over the last SMOOTH_FRAMES frames, and drives the
lock demo, so the video stream never waits for the network.
"""
import collections
import json
import logging
import threading
import time

import numpy as np
import torch

import config
from gestures import CLASS_TO_COUNT, CLASSES
from lock import LockDemo
from ml.model import FingerCNN
from preprocess import preprocess_frame

log = logging.getLogger(__name__)


class FingerCounter:
    def __init__(self, weights=None, meta=None, threads=2):
        weights = weights or config.MODELS_DIR / "finger_cnn.pt"
        meta = meta or config.MODELS_DIR / "train_meta.json"
        self.meta = json.loads(open(meta, encoding="utf-8").read())
        if self.meta["classes"] != CLASSES:
            raise RuntimeError(f"model classes {self.meta['classes']} != gestures.CLASSES {CLASSES}")
        # The model must see exactly what it was trained on.
        pp = self.meta["preprocess"]
        if pp["grayscale"] != config.GRAYSCALE or pp["input_size"] != config.INPUT_SIZE:
            raise RuntimeError(
                f"model was trained with grayscale={pp['grayscale']}, input={pp['input_size']}, "
                f"but config.py has GRAYSCALE={config.GRAYSCALE}, INPUT_SIZE={config.INPUT_SIZE}")
        self.mean, self.std = self.meta["mean"], self.meta["std"]
        torch.set_num_threads(threads)
        self.model = FingerCNN(n_classes=len(CLASSES), in_ch=self.meta["input_shape"][1])
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()
        self.run_id = self.meta.get("run_id")
        log.info("Loaded model %s (val acc %.1f%%)", self.run_id,
                 100 * self.meta.get("val", {}).get("accuracy", float("nan")))

    @torch.no_grad()
    def probabilities(self, frame):
        x = torch.from_numpy(preprocess_frame(frame, self.mean, self.std))
        return torch.softmax(self.model(x), 1)[0].numpy()

    def predict_count(self, frame):
        """Raw BGR frame -> (finger count 0-5, confidence, probabilities per class)."""
        p = self.probabilities(frame)
        i = int(p.argmax())
        return CLASS_TO_COUNT[CLASSES[i]], float(p[i]), p


_default = None


def predict_count(frame):
    """Module-level convenience: (count, confidence, probabilities) for one frame."""
    global _default
    if _default is None:
        _default = FingerCounter()
    return _default.predict_count(frame)


class LiveWorker:
    """Background thread: newest frame -> model -> smoothing -> lock demo."""

    def __init__(self, camera, counter, lock=None):
        self.camera = camera
        self.counter = counter
        self.lock_demo = lock or LockDemo()
        self._probs = collections.deque(maxlen=config.SMOOTH_FRAMES)
        self._mutex = threading.Lock()
        self._state = {"ready": False}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def press(self):
        with self._mutex:
            self.lock_demo.press(time.monotonic())

    def reset(self):
        with self._mutex:
            self.lock_demo.reset()

    def state(self):
        with self._mutex:
            return dict(self._state, lock=self.lock_demo.snapshot())

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        last_id, fps, last_t = -1, 0.0, None
        while not self._stop.is_set():
            frame_id, _ = self.camera.wait_frame(last_id, timeout=0.5)
            if frame_id == last_id:
                with self._mutex:
                    self._state["ready"] = False
                    self._probs.clear()
                continue
            last_id, raw = self.camera.get_raw_frame()
            if raw is None:
                continue
            t0 = time.perf_counter()
            try:
                p = self.counter.probabilities(raw)
            except Exception:
                log.exception("inference failed")
                self._stop.wait(1.0)
                continue
            ms = (time.perf_counter() - t0) * 1000
            now = time.monotonic()
            if last_t is not None:
                inst = 1.0 / max(now - last_t, 1e-6)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            last_t = now

            with self._mutex:
                self._probs.append(p)
                avg = np.mean(self._probs, axis=0)
                i = int(avg.argmax())
                count, conf = CLASS_TO_COUNT[CLASSES[i]], float(avg[i])
                self.lock_demo.update(count, conf, now)
                self._state = {
                    "ready": True, "count": count, "confidence": round(conf, 4),
                    "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, avg)},
                    "raw_count": CLASS_TO_COUNT[CLASSES[int(p.argmax())]],
                    "inference_ms": round(ms, 1), "fps": round(fps, 1),
                    "model": self.counter.run_id,
                }
