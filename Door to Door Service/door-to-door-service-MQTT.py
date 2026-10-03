"""Door to Door Service: find the green Lego minifig with the laptop camera and
publish its position over MQTT so the UnoQ can show it and center the car.

Each frame, the most confident "Green-Lego" detection is published to MQTT_TOPIC as JSON:

    {"found": true, "x": 412, "y": 230, "w": 640, "h": 480, "dx": 92, "dy": -10, "conf": 0.87}

x, y   -- center of the minifig's bounding box, in pixels (origin is the top-left of the frame)
w, h   -- frame size in pixels
dx, dy -- offset of the minifig from the center of the frame (dx > 0: right of center,
          dy > 0: below center). Drive dx toward 0 to center the car.
conf   -- detection confidence, 0 to 1

When no minifig is seen: {"found": false, "w": 640, "h": 480}

Usage:
    python door-to-door-service-MQTT.py            # run the camera and publish
    python door-to-door-service-MQTT.py --train    # (re)train the model from the dataset zip
Press q in the camera window to quit.
"""

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mqttlib import BROKER, MQTTClient  # noqa: E402  (lives in the repo root)

MQTT_TOPIC = "ME193/RQ-D2"   # change this to publish somewhere else
PUBLISH_HZ = 10              # max MQTT messages per second, so the broker isn't flooded
CAMERA_INDEX = 0             # 0 is the built-in laptop camera
CONF_THRESHOLD = 0.5         # ignore detections less confident than this

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "green_lego.pt"
DATASET_ZIP = HERE / "Green-Lego.v1-training-testv1.yolov8.zip"
DATASET_DIR = HERE / "green_lego_dataset"


def train(epochs, imgsz):
    """Unzip the Roboflow dataset, train yolov8n on it, and save the weights to MODEL_PATH."""
    if not DATASET_DIR.exists():
        with zipfile.ZipFile(DATASET_ZIP) as z:
            z.extractall(DATASET_DIR)
    # Roboflow's data.yaml uses "../train/images" paths; point them at the unzipped folder.
    data_yaml = DATASET_DIR / "data_local.yaml"
    data_yaml.write_text(
        f"path: {DATASET_DIR.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['Green-Lego']\n"
    )

    model = YOLO("yolov8n.pt")   # downloads the pretrained starting weights the first time
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                project=str(HERE / "runs"), name="green_lego", exist_ok=True)
    best = Path(model.trainer.best)
    MODEL_PATH.write_bytes(best.read_bytes())
    print(f"Saved trained model to {MODEL_PATH}")


def best_detection(result):
    """Return (x, y, conf) of the most confident box's center, or None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    i = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
    return (x1 + x2) / 2, (y1 + y2) / 2, float(boxes.conf[i])


def run(show):
    if not MODEL_PATH.exists():
        sys.exit(f"No model at {MODEL_PATH}. Run with --train first.")
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {CAMERA_INDEX}.")

    print(f"Connecting to MQTT broker {BROKER}...")
    with MQTTClient() as mqtt:
        print(f"Publishing minifig position to {MQTT_TOPIC}. Press q in the window to quit.")
        last_publish = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame grab failed.")
                    break
                h, w = frame.shape[:2]

                result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
                det = best_detection(result)
                if det:
                    x, y, conf = det
                    msg = {"found": True, "x": round(x), "y": round(y), "w": w, "h": h,
                           "dx": round(x - w / 2), "dy": round(y - h / 2), "conf": round(conf, 2)}
                else:
                    msg = {"found": False, "w": w, "h": h}

                now = time.time()
                if now - last_publish >= 1 / PUBLISH_HZ:
                    payload = json.dumps(msg)
                    mqtt.publish(MQTT_TOPIC, payload)
                    print(f"{MQTT_TOPIC} <- {payload}")
                    last_publish = now

                if show:
                    view = result.plot()
                    cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
                    if det:
                        cv2.line(view, (w // 2, h // 2), (msg["x"], msg["y"]), (0, 255, 0), 2)
                        label = f"x={msg['x']} y={msg['y']}  dx={msg['dx']} dy={msg['dy']}"
                    else:
                        label = "no minifig"
                    cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("Door to Door Service", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="train the model from the dataset zip, then exit")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs (default 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="training image size (default 640)")
    parser.add_argument("--no-window", action="store_true", help="don't show the camera preview window")
    args = parser.parse_args()

    if args.train:
        train(args.epochs, args.imgsz)
    else:
        run(show=not args.no_window)
