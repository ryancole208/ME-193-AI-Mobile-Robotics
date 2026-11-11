"""Load the collected PNGs, split them by burst, and augment on the GPU.

Frames inside one burst are nearly identical, so the split is done by BURST,
never by frame: a validation burst is never seen in training. Otherwise
validation accuracy would mostly measure memorisation.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from gestures import CLASSES, CLASS_TO_INDEX  # noqa: E402
from ml.data_report import read, scan  # noqa: E402
from preprocess import raw_to_unit  # noqa: E402


def split_bursts(val_fraction=0.2, seed=0):
    """Return (train, val) lists of (path, label). Per class, about `val_fraction`
    of the bursts (at least 1) go to validation, chosen with a fixed seed."""
    tree = scan()
    rng = random.Random(seed)
    train, val = [], []
    for cls in CLASSES:
        bursts = [paths for s in sorted(tree.get(cls, {})) for _, paths in
                  sorted(tree[cls][s].items())]
        if len(bursts) < 2:
            raise SystemExit(f"class {cls!r} needs at least 2 bursts (has {len(bursts)})")
        rng.shuffle(bursts)
        n_val = max(1, round(len(bursts) * val_fraction))
        label = CLASS_TO_INDEX[cls]
        for i, paths in enumerate(bursts):
            (val if i < n_val else train).extend((p, label) for p in paths)
    return train, val


def load(items):
    """List of (path, label) -> (N,1,H,W) float32 tensor in [0,1], (N,) int64 labels."""
    x = np.stack([raw_to_unit(read(p)) for p, _ in items])[:, None]
    y = np.array([lbl for _, lbl in items], np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def augment(x):
    """Random, label-safe changes to a batch (N,1,H,W) in [0,1], on its device.

    NO horizontal/vertical flips and only small rotations: left/right/forward IS
    the label, so flipping or turning the image would change the true class.
    """
    n, _, h, w = x.shape
    dev = x.device
    # Small rotation (+-10 deg), zoom (0.9-1.1) and shift (+-8%).
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(10)
    scale = 1 + (torch.rand(n, device=dev) * 2 - 1) * 0.1
    tx = (torch.rand(n, device=dev) * 2 - 1) * 0.08 * 2
    ty = (torch.rand(n, device=dev) * 2 - 1) * 0.08 * 2
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    aspect = w / h  # keep rotation true in pixels on a non-square image
    theta = torch.stack([
        torch.stack([cos, -sin / aspect, tx], 1),
        torch.stack([sin * aspect, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # Brightness, contrast and gamma, for lighting changes between sessions.
    b = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.07
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.3
    g = torch.exp((torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.2)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - mean) * c + mean + b).clamp(0, 1) ** g
    # Sensor noise.
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1)
