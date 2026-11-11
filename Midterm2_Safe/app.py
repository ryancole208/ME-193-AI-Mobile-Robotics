"""Finger-Count Safe webpage: live webcam view (/) and data collection (/collect).

Run:  python app.py            (camera chosen by name: config.CAMERA_NAME)
"""
import argparse
import logging

import cv2
from flask import Flask, Response, jsonify, render_template, request

import config
from camera import Camera
from collector import Collector, draw_collect_overlay
from gestures import CLASS_TO_COUNT, CLASSES, DISPLAY_NAMES, HOW_TO
from processing import process_frame

try:  # torch + a trained model are optional: the app still streams and collects without them
    from inference import FingerCounter, LiveWorker
except ImportError:  # pragma: no cover
    FingerCounter = LiveWorker = None

log = logging.getLogger("safe")

JPEG_QUALITY = 80
BOUNDARY = b"frame"


def mjpeg_stream(camera, overlay=None):
    """Yield multipart JPEG parts until the camera drops out.

    Ending the response on disconnect (rather than hanging) lets the browser
    notice the break; the page then reloads the stream once /status recovers.
    """
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    last_id = -1
    misses = 0
    while True:
        frame_id, frame = camera.wait_frame(last_id, timeout=1.0)
        if frame is None:
            misses += 1
            if misses >= 3 or not camera.connected:
                return
            continue
        misses = 0
        last_id = frame_id
        if overlay is not None:
            frame = overlay(frame)
        ok, jpg = cv2.imencode(".jpg", frame, params)
        if not ok:
            continue
        yield (b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
               + jpg.tobytes() + b"\r\n")


def create_app(camera, live=None):
    app = Flask(__name__)
    collector = Collector(camera)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/video_feed")
    def video_feed():
        # ?overlay=collect draws the crop box (collect page only).
        overlay = draw_collect_overlay if request.args.get("overlay") == "collect" else None
        resp = Response(mjpeg_stream(camera, overlay),
                        mimetype="multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/status")
    def status():
        return jsonify(camera.status())

    # ---- live finger count + lock demo ------------------------------------------

    @app.get("/live")
    def live_state():
        if live is None:
            return jsonify({"ready": False, "model_loaded": False})
        return jsonify(dict(live.state(), model_loaded=True))

    @app.post("/lock/press")
    def lock_press():
        if live is None:
            return jsonify({"ok": False, "message": "no model loaded"}), 400
        live.press()
        return jsonify({"ok": True})

    @app.post("/lock/reset")
    def lock_reset():
        if live is not None:
            live.reset()
        return jsonify({"ok": True})

    # ---- data collection ----------------------------------------------------

    @app.get("/collect")
    def collect():
        page_cfg = {
            "classes": CLASSES, "display": DISPLAY_NAMES, "howto": HOW_TO,
            "burst_fps": config.BURST_FPS, "burst_seconds": config.BURST_SECONDS,
            "countdown": config.COUNTDOWN_SECONDS, "target": config.TARGET_PER_CLASS,
            "counts": CLASS_TO_COUNT, "val_prefix": config.VAL_SESSION_PREFIX,
        }
        return render_template("collect.html", page_cfg=page_cfg)

    @app.post("/collect/burst")
    def collect_burst():
        d = request.get_json(force=True, silent=True) or {}
        try:
            ok, msg = collector.start_burst(d.get("class_id"), d.get("session_id"),
                                            d.get("seconds", config.BURST_SECONDS),
                                            d.get("fps", config.BURST_FPS))
        except (TypeError, ValueError):
            ok, msg = False, "bad duration or rate"
        return jsonify({"ok": ok, "message": msg}), (200 if ok else 400)

    @app.get("/collect/status")
    def collect_status():
        return jsonify(collector.status())

    @app.post("/collect/undo")
    def collect_undo():
        ok, msg = collector.undo_last()
        return jsonify({"ok": ok, "message": msg}), (200 if ok else 400)

    @app.get("/collect/counts")
    def collect_counts():
        return jsonify(collector.counts())

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Finger-Count Safe webcam viewer and data collection")
    p.add_argument("--camera-name", default=config.CAMERA_NAME,
                   help="camera name, or part of it (see camera_names.py)")
    p.add_argument("--camera", type=int, default=None,
                   help="fixed camera index; overrides --camera-name")
    p.add_argument("--width", type=int, default=config.CAPTURE_WIDTH)
    p.add_argument("--height", type=int, default=config.CAPTURE_HEIGHT)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; use 0.0.0.0 to allow other devices on the LAN")
    p.add_argument("--port", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    # Werkzeug logs every /status poll; keep the console readable.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    name = None if args.camera is not None else args.camera_name
    camera = Camera(args.camera, args.width, args.height, process=process_frame, name=name)
    live = None
    if FingerCounter is None:
        log.warning("torch not available: live finger count disabled")
    elif not (config.MODELS_DIR / "finger_cnn.pt").exists():
        log.warning("No models/finger_cnn.pt yet: live finger count disabled (train first)")
    else:
        live = LiveWorker(camera, FingerCounter())
    app = create_app(camera, live)
    log.info("Open http://%s:%d  (data collection: /collect)  Ctrl+C to quit",
             args.host, args.port)
    try:
        # threaded: each MJPEG client holds a connection open.
        # no reloader: it would spawn a second process that opens the camera again.
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down")
        camera.release()


if __name__ == "__main__":
    main()
