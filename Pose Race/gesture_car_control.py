"""
Gesture-controlled LEGO Education race car.

Uses MediaPipe (Hand Landmarker Task) to read hand gestures from the webcam
and the `legoeducation` Python API to drive a Double Motor car over Bluetooth.

Gestures:
    Point up    -> drive forward
    Point down  -> drive backward
    Point left  -> turn left
    Point right -> turn right
    Closed fist -> stop
    No hand seen -> stop (safety)

Setup:
    pip install legoeducation mediapipe opencv-python
    Power on the car's Double Motor and put it in Bluetooth broadcast mode
    before running this script.
"""

import math
import os
import time
import urllib.request

import cv2
import legoeducation as le
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

# ---------------------------------------------------------------------------
# Configuration - adjust to match your hardware / preferences
# ---------------------------------------------------------------------------
CARD_COLOR = le.LEGO_COLOR_BLUE
CARD_SERIAL = "3685"

CAMERA_INDEX = 0   # preferred index; tried first, then auto-detected if it fails
MAX_CAMERA_INDEX = 4  # highest index to probe when the preferred one doesn't work
MIRROR_PREVIEW = True  # flip the camera image so it behaves like a mirror

DRIVE_SPEED = 60   # forward/backward max speed, 0-100
TURN_SPEED = 10    # left/right turn max speed, 0-100

# Number of consecutive frames a new direction must be seen before it is
# sent to the car. Filters out single-frame misreads. A fist (stop) always
# takes effect immediately since it's the safety gesture.
CONFIRM_FRAMES = 4

# Motors ramp up rather than jumping straight to full speed: a direction
# starts at RAMP_START_FRACTION of its max speed and reaches full speed
# after being held continuously for RAMP_UP_SECONDS.
RAMP_START_FRACTION = 0.3
RAMP_UP_SECONDS = 1.5

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Hand landmark indices (MediaPipe hand model)
WRIST = 0
FINGER_JOINTS = {
    "index": (5, 6, 8),   # (mcp, pip, tip)
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

FORWARD, BACKWARD, LEFT, RIGHT, STOP = "FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand landmark model (one-time download)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _finger_curled(landmarks, mcp_idx, pip_idx, tip_idx):
    """A finger is curled if its tip sits closer to the wrist than its
    knuckle (pip) does. Works regardless of how the hand is rotated."""
    wrist = landmarks[WRIST]
    return _dist(landmarks[tip_idx], wrist) < _dist(landmarks[pip_idx], wrist) * 0.9


def classify_gesture(landmarks):
    curled = [
        _finger_curled(landmarks, mcp, pip, tip)
        for (mcp, pip, tip) in FINGER_JOINTS.values()
    ]
    if all(curled):
        return STOP

    wrist = landmarks[WRIST]
    index_tip = landmarks[FINGER_JOINTS["index"][2]]
    dx = index_tip.x - wrist.x
    dy = index_tip.y - wrist.y
    angle = math.degrees(math.atan2(-dy, dx))  # 0=right, 90=up, +-180=left, -90=down

    if -45 <= angle < 45:
        return RIGHT
    if 45 <= angle < 135:
        return FORWARD
    if angle >= 135 or angle < -135:
        return LEFT
    return BACKWARD


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for connection in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def open_camera(preferred_index=CAMERA_INDEX, max_index=MAX_CAMERA_INDEX):
    """Open a webcam, preferring `preferred_index` but falling back to
    scanning other indices. Some indices report as "opened" even though
    they can't actually deliver frames (e.g. a phantom Continuity Camera
    entry on macOS), so a successful frame read is required too."""
    candidates = [preferred_index] + [i for i in range(max_index + 1) if i != preferred_index]
    for index in candidates:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                if index != preferred_index:
                    print(f"Camera index {preferred_index} unavailable; using index {index} instead.")
                return cap
        cap.release()
    return None


def max_speed_for(command):
    if command in (FORWARD, BACKWARD):
        return DRIVE_SPEED
    if command in (LEFT, RIGHT):
        return TURN_SPEED
    return 0


def ramped_speed(command, held_since):
    """Speed for `command`, ramping from RAMP_START_FRACTION of its max
    speed up to full speed the longer it has been continuously held."""
    max_speed = max_speed_for(command)
    if held_since is None:
        return max_speed
    progress = min(1.0, (time.time() - held_since) / RAMP_UP_SECONDS)
    start_speed = max_speed * RAMP_START_FRACTION
    return round(start_speed + (max_speed - start_speed) * progress)


def send_command(car, command, speed=0):
    print(f"-> {command} (speed={speed})")
    if command == FORWARD:
        car.movement_move(direction=le.MOVEMENT_DIRECTION_FORWARD, speed=speed)
    elif command == BACKWARD:
        car.movement_move(direction=le.MOVEMENT_DIRECTION_BACKWARD, speed=speed)
    elif command == LEFT:
        car.movement_move(direction=le.MOVEMENT_DIRECTION_LEFT, speed=speed)
    elif command == RIGHT:
        car.movement_move(direction=le.MOVEMENT_DIRECTION_RIGHT, speed=speed)
    elif command == STOP:
        car.movement_stop()


def main():
    ensure_model_downloaded()

    print(f"Connecting to Double Motor (blue card, serial {CARD_SERIAL})...")
    car = le.DoubleMotor()
    car.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not car.connected:
        print("Error connecting to the Double Motor. Check that it is powered "
              "on and broadcasting, then try again.")
        return

    print("Connected. Starting camera...")
    cap = open_camera()
    if cap is None:
        print("Error: could not open a webcam. Check camera permissions for "
              "your terminal/IDE in System Settings > Privacy & Security > Camera.")
        car.disconnect()
        return

    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    current_command = STOP
    pending_command = STOP
    pending_count = 0
    command_start_time = None  # when current_command last changed to a direction
    last_sent_speed = 0
    start_time = time.time()

    try:
        with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Warning: failed to read camera frame.")
                    break

                if MIRROR_PREVIEW:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    draw_landmarks(frame, landmarks)
                    raw_gesture = classify_gesture(landmarks)
                else:
                    raw_gesture = STOP

                # Fist / no-hand always stops immediately (safety).
                # Direction changes require a few consistent frames first.
                if raw_gesture == STOP:
                    pending_command = STOP
                    pending_count = CONFIRM_FRAMES
                elif raw_gesture == pending_command:
                    pending_count += 1
                else:
                    pending_command = raw_gesture
                    pending_count = 1

                if pending_count >= CONFIRM_FRAMES and pending_command != current_command:
                    current_command = pending_command
                    command_start_time = time.time() if current_command != STOP else None
                    last_sent_speed = None  # force a fresh send below

                if current_command == STOP:
                    if last_sent_speed != 0:
                        send_command(car, STOP)
                        last_sent_speed = 0
                else:
                    speed = ramped_speed(current_command, command_start_time)
                    if speed != last_sent_speed:
                        send_command(car, current_command, speed)
                        last_sent_speed = speed

                cv2.putText(frame, f"Command: {current_command}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow("Gesture Car Control (press q to quit)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        print("Stopping car and disconnecting...")
        car.movement_stop()
        car.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
