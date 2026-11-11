"""Probe camera indexes 0-5 and report which ones open and at what resolution.

Each camera is asked for 1920x1080; the DICCIK Full HD webcam should honour that,
while most built-in laptop cameras top out at 1280x720 or lower.
"""
import os
import platform
import time

# Hide OpenCV's noisy "can't open index" warnings for missing cameras.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402

MAX_INDEX = 5
REQUEST_W, REQUEST_H = 1920, 1080


def backend():
    # DirectShow opens much faster and more reliably than MSMF on Windows.
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


def probe(index):
    cap = cv2.VideoCapture(index, backend())
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_H)
        # First frames can be empty while the sensor warms up.
        frame = None
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        else:
            return {"frame_ok": False, "res": "-"}
        h, w = frame.shape[:2]
        return {"frame_ok": True, "res": f"{w}x{h}"}
    finally:
        cap.release()


def main():
    print(f"OS: {platform.system()}  |  OpenCV {cv2.__version__}")
    print(f"Requesting {REQUEST_W}x{REQUEST_H} from each camera...\n")
    print(f"{'Index':<7}{'Opens':<8}{'Frame':<8}{'Resolution':<12}Note")
    print("-" * 52)
    for i in range(MAX_INDEX + 1):
        r = probe(i)
        if r is None:
            print(f"{i:<7}{'no':<8}{'-':<8}{'-':<12}")
            continue
        note = "<- likely DICCIK (Full HD)" if r["res"] == f"{REQUEST_W}x{REQUEST_H}" else ""
        print(f"{i:<7}{'yes':<8}{'yes' if r['frame_ok'] else 'no':<8}{r['res']:<12}{note}")
    print("\nNames for each index: python camera_names.py")
    print("The app picks the camera by name (config.CAMERA_NAME); --camera <index> overrides.")


if __name__ == "__main__":
    main()
