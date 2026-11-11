"""Export the trained model for the UNO Q (Step 7).

    python -m ml.export

Reads   models/finger_cnn.pt + models/train_meta.json
Writes  models/finger_cnn.onnx      fixed input 1x3x96x96 named "input", output "logits"
        models/model_config.json    everything infer_standalone.py needs (no project code)

Then checks ONNX against PyTorch on validation images (max |diff| and argmax
agreement) and times ONNX Runtime on the CPU.

The spec asked for a 1x1x96x96 grayscale input. The model takes COLOR (BGR, 3
channels) because grayscale topped out around 83%: finger shadows look like
fingers in gray (handoff Session 11).
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from gestures import CLASS_TO_COUNT, CLASSES  # noqa: E402
from ml.dataset import split_sessions  # noqa: E402
from ml.model import FingerCNN  # noqa: E402
from preprocess import normalize, raw_to_unit  # noqa: E402

ONNX_NAME = "finger_cnn.onnx"


def main():
    meta = json.loads((config.MODELS_DIR / "train_meta.json").read_text(encoding="utf-8"))
    shape = meta["input_shape"]
    model = FingerCNN(n_classes=len(CLASSES), in_ch=shape[1])
    model.load_state_dict(torch.load(config.MODELS_DIR / "finger_cnn.pt", map_location="cpu"))
    model.eval()

    onnx_path = config.MODELS_DIR / ONNX_NAME
    torch.onnx.export(model, torch.zeros(*shape), str(onnx_path), input_names=["input"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")

    # Parity: 50 val images spread over the classes, PyTorch vs ONNX Runtime.
    import onnxruntime as ort
    _, val = split_sessions()
    rng = np.random.default_rng(0)
    pick = [val[i] for i in sorted(rng.choice(len(val), size=min(50, len(val)), replace=False))]
    x = np.stack([normalize(raw_to_unit(cv2.imread(str(p))), meta["mean"], meta["std"])
                  for p, _ in pick])
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(None, {"input": x[i:i + 1]})[0] for i in range(len(x))])
    max_diff = float(np.abs(ref - got).max())
    agree = int((ref.argmax(1) == got.argmax(1)).sum())
    ok = max_diff < 1e-4 and agree == len(x)
    print(f"parity on {len(x)} val images: max |diff| {max_diff:.2e}, "
          f"argmax agree {agree}/{len(x)} -> {'PASS' if ok else 'FAIL'}")

    # Speed (model only), 1 thread and the default thread count.
    bench = {}
    for threads in (1, 0):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        one = x[:1]
        for _ in range(20):
            s.run(None, {"input": one})
        t0 = time.perf_counter()
        n = 300
        for _ in range(n):
            s.run(None, {"input": one})
        bench["1_thread" if threads == 1 else "default_threads"] = round(
            (time.perf_counter() - t0) / n * 1000, 3)
    print(f"ONNX Runtime CPU ms/frame (model only, laptop): {bench}")

    val_m = meta.get("val", {})
    cfg = {
        "onnx_file": ONNX_NAME,
        "run_id": meta["run_id"],
        "classes": CLASSES,
        "class_to_count": CLASS_TO_COUNT,
        "input_name": "input",
        "output_name": "logits",
        "input_shape": shape,
        "channel_order": "BGR",          # OpenCV order; do NOT convert to RGB
        "grayscale": False,
        "capture": {"width": config.CAPTURE_WIDTH, "height": config.CAPTURE_HEIGHT},
        "crop": {"x0": config.CROP_X0, "y0": config.CROP_Y0, "x1": config.CROP_X1,
                 "y1": config.CROP_Y1, "units": "fraction_of_frame",
                 "rounding": "round(frac * size), clamp inside the frame"},
        "letterbox": {"size": config.RAW_SAVE_SIZE, "fill": config.LETTERBOX_FILL,
                      "align": "center", "interpolation": "INTER_AREA"},
        "input_resize": {"size": shape[2], "interpolation": "INTER_AREA"},
        "scale": 1 / 255,
        "mean": meta["mean"],
        "std": meta["std"],
        "smooth_frames": config.SMOOTH_FRAMES,
        "hold_seconds": config.HOLD_SECONDS,
        "conf_threshold": config.CONF_THRESHOLD,
        "secret_code": list(config.SECRET_CODE),
        "result_seconds": config.RESULT_SECONDS,
        "awake_timeout_seconds": config.AWAKE_TIMEOUT_SECONDS,
        "val_accuracy": val_m.get("accuracy"),
        "val_recall": val_m.get("recall"),
        "parity": {"images": len(x), "max_abs_diff": max_diff, "argmax_agree": agree},
        "laptop_onnx_ms_per_frame": bench,
    }
    out = config.MODELS_DIR / "model_config.json"
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")
    if not ok:
        sys.exit("parity check FAILED")


if __name__ == "__main__":
    main()
