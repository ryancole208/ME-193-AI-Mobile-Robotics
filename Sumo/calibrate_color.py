"""
Color calibration: teach the KNN classifier what "floor" and "blue tape" look like.

    python calibrate_color.py            (or: Run and Debug > "Sumo: calibrate color")

How to use it
  1. Mount the Color Sensor on the robot as it will be in the match, and put the
     ROBOT on the surface (so the sensor is at its real height above the floor;
     height changes the readings a lot).
  2. Press  f  with the sensor over the FLOOR. While it records (~2 s), slowly slide
     the robot around so it sees a few different spots. Repeat at 3-4 places,
     including the floor OUTSIDE the ring.
  3. Press  b  with the sensor fully over the BLUE TAPE. Slide it along the tape
     while it records. Repeat at 3-4 places around the ring.
  4. Press  t  to test live: move the robot on and off the tape. Each reading
     is shown as the KNN sees it AND as the sensor's own color name sees it
     ("lego" method), so you can choose COLOR_METHOD in config.py. Works before
     recording too (KNN column shows '-'). Any key stops the test.
  5. Press  s  to save to color_samples.json (sumo.py loads it at startup).
     u = undo the last recording, q = quit without saving.

What gets saved: every reading's (rawRed, rawGreen, rawBlue, reflection) with its
label. color_classifier.KnnColorClassifier does the rest (see that file).
Only used when config.COLOR_METHOD = "knn"; with "lego" no calibration is needed.
"""

import math
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import config
from color_classifier import BLUE, FLOOR, KnnColorClassifier, LegoColorClassifier, features_from_sensor

SAMPLES_PER_RECORDING = 60       # at ~30 ms per reading this takes ~2 s
SAMPLES_PATH = Path(__file__).with_name(config.COLOR_SAMPLES_FILE)


# ---------------------------------------------------------------------------
# Pure helpers (no hardware; tested in tests/test_calibrate.py)
# ---------------------------------------------------------------------------

def leave_one_out_accuracy(samples, k=config.KNN_K):
    """Fraction of samples classified correctly by a KNN trained on all the OTHER samples.

    This estimates how well the classifier will do on readings it has never seen.
    Below ~98 % the two surfaces are too similar (or a recording was mislabeled).
    Returns (accuracy, list of (index, true label, predicted label) mistakes).
    """
    mistakes = []
    for i, (label, feats) in enumerate(samples):
        rest = samples[:i] + samples[i + 1:]
        try:
            clf = KnnColorClassifier(rest, k=k, max_distance=None)
        except ValueError:          # removing this sample left only one class
            continue
        pred = clf.classify(feats)
        if pred != label:
            mistakes.append((i, label, pred))
    return 1.0 - len(mistakes) / len(samples), mistakes


def class_summary(samples):
    """{label: (count, mean feature vector)} for printing."""
    out = {}
    for label in sorted({lab for lab, _ in samples}):
        rows = [f for lab, f in samples if lab == label]
        means = tuple(sum(col) / len(rows) for col in zip(*rows))
        out[label] = (len(rows), means)
    return out


def separation(samples):
    """Distance between the floor and blue averages, in units of the typical spread.

    Like a signal-to-noise ratio: above ~5 the two surfaces are easy to tell apart;
    below ~3 expect occasional mistakes (debouncing helps, but recalibrate if you can).
    """
    by = {FLOOR: [], BLUE: []}
    for lab, f in samples:
        if lab in by:
            by[lab].append(f)
    if not by[FLOOR] or not by[BLUE]:
        return None
    n = len(by[FLOOR][0])
    means = {lab: [sum(r[i] for r in rows) / len(rows) for i in range(n)] for lab, rows in by.items()}
    spread = []
    for i in range(n):
        var = 0.0
        for lab, rows in by.items():
            var += sum((r[i] - means[lab][i]) ** 2 for r in rows) / len(rows)
        spread.append(max(1.0, math.sqrt(var / 2)))
    return math.sqrt(sum(((means[FLOOR][i] - means[BLUE][i]) / spread[i]) ** 2 for i in range(n)))


def load_existing(path=SAMPLES_PATH):
    if not path.exists():
        return []
    return [(lab, tuple(f)) for lab, f in KnnColorClassifier.load_samples(path)]


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

class LiveColorSensor:
    """Connects to the Color Sensor and collects every notification in a list."""

    def __init__(self):
        import legoeducation as le
        self.cs = le.ColorSensor()
        self._lock = threading.Lock()
        self._readings = []
        self.cs.set_notification_callback(self._on_notification)

    def _on_notification(self, _n):
        # Runs on the library's thread: only record. cs.sensor was just replaced
        # with a fresh object, so keeping a reference to it is safe.
        s = self.cs.sensor
        feats = features_from_sensor(s)
        if feats is not None:
            with self._lock:
                self._readings.append((feats, s))

    def connect(self):
        while not self.cs.connected:
            print(f"Connecting to the Color Sensor on card {config.CARD_SERIAL} ...")
            try:
                self.cs.connect(card_color=config.CARD_COLOR, card_serial=config.CARD_SERIAL,
                                device_notification_delay=config.NOTIFICATION_MS)
            except Exception as e:
                if type(e).__name__ == "VersionMismatchError":
                    raise
                print(f"  error: {e}")
            if not self.cs.connected:
                print("  not found. Is it on, charged, and on the right card? Retrying (Ctrl+C to quit).")
                time.sleep(2)
        print("Color Sensor connected.\n")

    def take(self, n, timeout_s=10.0):
        """Wait for the next n readings and return them."""
        with self._lock:
            self._readings.clear()
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            with self._lock:
                if len(self._readings) >= n:
                    return [f for f, _ in self._readings[:n]]
            time.sleep(0.01)
        with self._lock:
            got = [f for f, _ in self._readings]
        print(f"  only got {len(got)} of {n} readings in {timeout_s:.0f} s (connection weak?)")
        return got

    def latest(self):
        """[(features, raw sensor reading), ...] since the last call."""
        with self._lock:
            out, self._readings = self._readings, []
        return out

    def close(self):
        if self.cs.connected:
            self.cs.disconnect()


def get_key(prompt):
    """Wait for one key press (Windows), or a line of input elsewhere."""
    print(prompt, end="", flush=True)
    try:
        import msvcrt
        ch = msvcrt.getwch()
        print(ch)
        return ch.lower()
    except ImportError:
        return (input() or " ")[0].lower()


def key_pressed():
    try:
        import msvcrt
        if msvcrt.kbhit():
            msvcrt.getwch()
            return True
    except ImportError:
        pass
    return False


def report(samples):
    counts = Counter(lab for lab, _ in samples)
    print(f"\n  Samples so far: {dict(counts) or 'none'}")
    for label, (n, means) in class_summary(samples).items():
        r, g, b, refl = means
        print(f"    {label:6s} avg  R={r:6.1f} G={g:6.1f} B={b:6.1f} reflection={refl:5.1f}")
    sep = separation(samples)
    if sep is not None:
        acc, mistakes = leave_one_out_accuracy(samples)
        verdict = "good" if sep >= 5 and acc >= 0.98 else ("OK" if sep >= 3 else "POOR: recalibrate")
        print(f"  Separation {sep:.1f} (want > 5)   leave-one-out accuracy {acc:.1%}   -> {verdict}")
    print()


def lego_color_name(c):
    """The sensor's own color id -> name, e.g. 3 -> 'BLUE', 10 -> 'AZURE'."""
    import legoeducation as le
    try:
        c = int(c)
    except (TypeError, ValueError):
        return "?"
    for name in dir(le):
        if name.startswith("LEGO_COLOR_") and getattr(le, name) == c:
            return name.removeprefix("LEGO_COLOR_")
    return str(c)


def live_test(sensor, samples):
    """Shows BOTH methods side by side, so you can choose COLOR_METHOD in config.py."""
    knn = KnnColorClassifier(samples) if {FLOOR, BLUE} <= {lab for lab, _ in samples} else None
    lego = LegoColorClassifier()
    print("  Live test: move the robot on/off the tape. Press any key to stop.")
    print(f"  {'knn':8s} {'lego':8s} sensor's color   raw values")
    last = None
    while not key_pressed():
        for feats, s in sensor.latest():
            k = knn.classify(feats) if knn else "-"
            g = lego.classify_sensor(s)
            if (k, g) != last:
                flag = "   <-- methods disagree" if knn and k != g else ""
                print(f"  {k.upper():8s} {g.upper():8s} {lego_color_name(s.color):15s}  "
                      f"R={feats[0]:.0f} G={feats[1]:.0f} B={feats[2]:.0f} refl={feats[3]:.0f}{flag}")
                last = (k, g)
        time.sleep(0.02)
    print("  Live test stopped.\n")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    samples = load_existing()
    if samples:
        print(f"Found {SAMPLES_PATH.name} with {len(samples)} samples.")
        if get_key("Keep them and add more (k), or start fresh (n)? ") == "n":
            samples = []
    batches = []                 # sizes of recordings made this session, for undo
    sensor = LiveColorSensor()
    try:
        sensor.connect()
        print(__doc__.split("How to use it")[1].split("What gets saved")[0])
        while True:
            report(samples)
            key = get_key("[f]loor  [b]lue  [t]est  [u]ndo  [s]ave+quit  [q]uit > ")
            if key in ("f", "b"):
                label = FLOOR if key == "f" else BLUE
                print(f"  Recording {SAMPLES_PER_RECORDING} {label.upper()} readings: slide the robot slowly ...")
                got = sensor.take(SAMPLES_PER_RECORDING)
                samples += [(label, f) for f in got]
                batches.append(len(got))
                print(f"  recorded {len(got)}.")
            elif key == "u":
                if batches:
                    n = batches.pop()
                    del samples[len(samples) - n:]
                    print(f"  removed the last {n} samples.")
                else:
                    print("  nothing to undo in this session.")
            elif key == "t":
                live_test(sensor, samples)     # KNN column shows '-' until both are recorded
            elif key == "s":
                if not {FLOOR, BLUE} <= {lab for lab, _ in samples}:
                    print("  need both floor and blue samples before saving.")
                    continue
                note = f"recorded {time.strftime('%Y-%m-%d %H:%M')} with calibrate_color.py"
                KnnColorClassifier.save_samples(SAMPLES_PATH, samples, note=note)
                print(f"  saved {len(samples)} samples to {SAMPLES_PATH}")
                return 0
            elif key == "q":
                print("  quit without saving.")
                return 0
    except KeyboardInterrupt:
        print("\nInterrupted; nothing saved.")
        return 130
    finally:
        sensor.close()


if __name__ == "__main__":
    sys.exit(main())
