"""The one preprocessing pipeline: training, evaluation, live inference, UNO Q.

Pixel path (identical everywhere):
    raw BGR frame --crop_rect (green pad)--> --letterbox--> 224x224 BGR "raw" image (saved PNG)
    raw image --gray, resize 96x96, /255--> float [0,1] --normalize(mean, std)--> model input

Letterbox = scale the pad rectangle to fit the square, then pad the leftover
top/bottom with LETTERBOX_FILL, so nothing is cut off and nothing is stretched.
Live frames go through the same 224 step as saved images, so a live frame and a
saved PNG of the same scene give the same tensor. Augmentation (training only)
happens between `raw_to_unit` and `normalize`.

Only OpenCV + NumPy, so the UNO Q can import this file without torch.
"""
import cv2
import numpy as np

import config


def settings():
    """Everything that defines the pipeline; saved next to the model."""
    return {
        "crop": {"x0": config.CROP_X0, "y0": config.CROP_Y0,
                 "x1": config.CROP_X1, "y1": config.CROP_Y1,
                 "units": "fraction_of_frame"},
        "letterbox": {"to": "square", "fill": config.LETTERBOX_FILL, "align": "center"},
        "raw_size": config.RAW_SAVE_SIZE,
        "input_size": config.INPUT_SIZE,
        "grayscale": config.GRAYSCALE,
        "resize_interpolation": "INTER_AREA",
        "scale": "divide_by_255",
    }


def crop_box(width, height):
    """(x0, y0, w, h) of the crop rectangle in a width x height frame, clamped inside it."""
    x0 = min(max(int(round(config.CROP_X0 * width)), 0), width - 1)
    y0 = min(max(int(round(config.CROP_Y0 * height)), 0), height - 1)
    x1 = min(max(int(round(config.CROP_X1 * width)), x0 + 1), width)
    y1 = min(max(int(round(config.CROP_Y1 * height)), y0 + 1), height)
    return x0, y0, x1 - x0, y1 - y0


def crop_rect(frame):
    h, w = frame.shape[:2]
    x0, y0, cw, ch = crop_box(w, h)
    return frame[y0:y0 + ch, x0:x0 + cw]


def letterbox(img, size, fill=None):
    """Fit `img` inside a size x size square without cropping or stretching."""
    fill = config.LETTERBOX_FILL if fill is None else fill
    h, w = img.shape[:2]
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size) + img.shape[2:], fill, img.dtype)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def frame_to_raw(frame):
    """Raw camera frame -> letterboxed 224x224 BGR image (what gets saved)."""
    return letterbox(crop_rect(frame), config.RAW_SAVE_SIZE)


def raw_to_unit(raw):
    """Saved/raw image (BGR or gray) -> float32 96x96 (or 96x96x3) in [0,1]."""
    img = raw
    if config.GRAYSCALE and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (config.INPUT_SIZE, config.INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def normalize(unit, mean, std):
    """[0,1] image -> normalized float32 array shaped (C, H, W)."""
    x = (unit - np.float32(mean)) / np.float32(std)
    if x.ndim == 2:
        x = x[None]
    else:
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
    return x.astype(np.float32)


def preprocess_frame(frame, mean, std):
    """Raw BGR frame from OpenCV -> model-ready array shaped (1, C, 96, 96).

    Returns a NumPy array (works for PyTorch via torch.from_numpy and for ONNX
    Runtime directly, so the UNO Q does not need torch)."""
    return normalize(raw_to_unit(frame_to_raw(frame)), mean, std)[None]
