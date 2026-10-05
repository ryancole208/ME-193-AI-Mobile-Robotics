"""Door to Door Service: find the green and blue Lego minifigs with the laptop camera and
publish their positions over MQTT so each minifig's car (each with its own UnoQ) can center on it.

Each frame, the most confident detection of each color is published as JSON to that color's
topic, MQTT_TOPIC + "/green" and MQTT_TOPIC + "/blue":

    ME193/RQ-D2/green  {"found": true, "x": 170, "y": 230, "w": 640, "h": 480, "target_x": 160, "dx": 10, "conf": 0.87}
    ME193/RQ-D2/blue   {"found": true, "x": 470, "y": 250, "w": 640, "h": 480, "target_x": 480, "dx": -10, "conf": 0.91}

x, y     -- center of the minifig's bounding box, in pixels (origin is the top-left of the frame)
w, h     -- frame size in pixels
target_x -- where that minifig should sit horizontally. If only one minifig is seen, it is the
            middle of the frame (50%). If both are seen, green's is 25% and blue's is 75%.
dx       -- x - target_x (dx > 0: right of its target). Drive dx toward 0 to center the car.
            Only x is centered; y depends on the camera angle and is just reported.
conf     -- detection confidence, 0 to 1

A minifig that isn't seen is sent as {"found": false, "w": 640, "h": 480}.

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
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mqttlib import BROKER, MQTTClient  # noqa: E402  (lives in the repo root)

MQTT_TOPIC = "ME193/RQ-D2"   # base topic; each color publishes to MQTT_TOPIC/green and MQTT_TOPIC/blue
PUBLISH_HZ = 10              # max MQTT messages per second, so the broker isn't flooded
CAMERA_INDEX = 0             # 0 is the built-in laptop camera
CONF_THRESHOLD = 0.5         # ignore detections less confident than this

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "lego_minifigs.pt"
DATASET_ZIP = HERE / "Green-Lego.v3-training-testv3.yolov8.zip"
DATASET_DIR = HERE / "lego_dataset"

CLASSES = {"green": "Green-Lego", "blue": "Blue-Lego"}   # message key -> class name in the dataset
SOLO_TARGET = 0.50                                       # target x (fraction of width) with one minifig
PAIR_TARGETS = {"green": 0.25, "blue": 0.75}             # target x when both are on screen
COLORS = {"green": (0, 255, 0), "blue": (255, 0, 0)}     # BGR, for the preview window


def train(epochs, imgsz):
    """Unzip the Roboflow dataset, train yolov8n on it, and save the weights to MODEL_PATH."""
    if not DATASET_DIR.exists():
        with zipfile.ZipFile(DATASET_ZIP) as z:
            z.extractall(DATASET_DIR)
    # Roboflow's data.yaml uses "../train/images" paths; point them at the unzipped folder.
    names = yaml.safe_load((DATASET_DIR / "data.yaml").read_text())["names"]
    data_yaml = DATASET_DIR / "data_local.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": DATASET_DIR.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }))

    model = YOLO("yolov8n.pt")   # downloads the pretrained starting weights the first time
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                project=str(HERE / "runs"), name="lego_minifigs", exist_ok=True)
    best = Path(model.trainer.best)
    MODEL_PATH.write_bytes(best.read_bytes())
    print(f"Saved trained model to {MODEL_PATH}")


def best_detections(result):
    """Return {"green": (x, y, conf), "blue": ...} for the most confident box of each class seen."""
    found = {}
    boxes = result.boxes
    if boxes is None:
        return found
    for key, class_name in CLASSES.items():
        best = None
        for i in range(len(boxes)):
            if result.names[int(boxes.cls[i])] != class_name:
                continue
            conf = float(boxes.conf[i])
            if best is None or conf > best[2]:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                best = ((x1 + x2) / 2, (y1 + y2) / 2, conf)
        if best:
            found[key] = best
    return found


def build_messages(dets, w, h):
    """Turn detections into one MQTT message per color, with that minifig's x target and dx."""
    both = len(dets) == len(CLASSES)
    msgs = {}
    for key in CLASSES:
        if key not in dets:
            msgs[key] = {"found": False, "w": w, "h": h}
            continue
        x, y, conf = dets[key]
        target_x = round(w * (PAIR_TARGETS[key] if both else SOLO_TARGET))
        msgs[key] = {"found": True, "x": round(x), "y": round(y), "w": w, "h": h,
                     "target_x": target_x, "dx": round(x) - target_x, "conf": round(conf, 2)}
    return msgs


def run(show):
    if not MODEL_PATH.exists():
        sys.exit(f"No model at {MODEL_PATH}. Run with --train first.")
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {CAMERA_INDEX}.")

    print(f"Connecting to MQTT broker {BROKER}...")
    with MQTTClient() as mqtt:
        topics = {key: f"{MQTT_TOPIC}/{key}" for key in CLASSES}
        print(f"Publishing minifig positions to {', '.join(topics.values())}. Press q in the window to quit.")
        last_publish = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame grab failed.")
                    break
                h, w = frame.shape[:2]

                result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
                msgs = build_messages(best_detections(result), w, h)

                now = time.time()
                if now - last_publish >= 1 / PUBLISH_HZ:
                    for key, topic in topics.items():
                        payload = json.dumps(msgs[key])
                        mqtt.publish(topic, payload)
                        print(f"{topic} <- {payload}")
                    last_publish = now

                if show:
                    view = result.plot()
                    for row, key in enumerate(CLASSES):
                        m, color = msgs[key], COLORS[key]
                        if m["found"]:
                            # vertical line at this minifig's x target, and a line from it to the minifig
                            cv2.line(view, (m["target_x"], 0), (m["target_x"], h), color, 1)
                            cv2.line(view, (m["target_x"], m["y"]), (m["x"], m["y"]), color, 2)
                            label = f"{key}: x={m['x']} target={m['target_x']} dx={m['dx']}"
                        else:
                            label = f"{key}: not seen"
                        cv2.putText(view, label, (10, 30 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
