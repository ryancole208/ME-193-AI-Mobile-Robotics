"""Load the collected PNGs, split them by SESSION, and augment on the GPU.

Burst frames are near-duplicates, so a random frame split would make accuracy
look far better than it is. Validation = every session whose ID starts with
config.VAL_SESSION_PREFIX ("val"); everything else is training.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gestures import CLASS_TO_INDEX, CLASSES  # noqa: E402
from ml.data_report import is_val, read, scan  # noqa: E402
from preprocess import raw_to_unit  # noqa: E402


def split_sessions():
    """Return (train, val) lists of (path, label)."""
    tree = scan()
    train, val = [], []
    for cls in CLASSES:
        label = CLASS_TO_INDEX[cls]
        for session, bursts in sorted(tree.get(cls, {}).items()):
            items = [(p, label) for _, paths in sorted(bursts.items()) for p in paths]
            (val if is_val(session) else train).extend(items)
    if not val:
        raise SystemExit("No validation session found. Record one whose session ID starts "
                         "with 'val' (e.g. val1) on the /collect page.")
    missing = [c for c in CLASSES if not any(lbl == CLASS_TO_INDEX[c] for _, lbl in val)]
    if missing:
        raise SystemExit(f"Validation session has no images for: {', '.join(missing)}")
    return train, val


def load(items):
    """List of (path, label) -> (N,C,S,S) float32 tensor in [0,1], (N,) int64 labels."""
    x = np.stack([raw_to_unit(read(p)) for p, _ in items])
    x = x[:, None] if x.ndim == 3 else x.transpose(0, 3, 1, 2)   # gray (N,1,H,W) / BGR (N,3,H,W)
    y = np.array([lbl for _, lbl in items], np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def augment(x, rot_deg=180.0, vflip=True, color_jitter=0.0):
    """Random, label-safe changes to a batch (N,1,H,W) in [0,1], on its device.

    Finger count doesn't depend on direction, so rotation and flips are label-safe.
    Training uses +-25 deg and horizontal flips only (the hand always enters from
    the user's side). Areas rotated in from outside are filled with black, the
    same as the letterbox bands.
    """
    n = x.shape[0]
    dev = x.device
    u = lambda *s: torch.rand(*s, device=dev) * 2 - 1  # uniform in [-1, 1]

    # Flips (as sign changes in the affine matrix), rotation, scale 0.9-1.1, shift +-10%.
    fx = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    fy = (torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0) if vflip
          else torch.ones(n, device=dev))
    ang = u(n) * math.radians(rot_deg)
    scale = 1 + u(n) * 0.1
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([
        torch.stack([cos * fx, -sin * fy, u(n) * 0.2], 1),
        torch.stack([sin * fx, cos * fy, u(n) * 0.2], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Brightness / contrast / gamma jitter.
    b = u(n, 1, 1, 1) * 0.06
    c = 1 + u(n, 1, 1, 1) * 0.3
    g = torch.exp(u(n, 1, 1, 1) * 0.25)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - mean) * c + mean + b).clamp(0, 1) ** g

    # White-balance / saturation jitter (color input only): an independent gain
    # per channel in [e^-cj, e^cj], then saturation 1 +- cj, so the same hand
    # under warm, cool or green-tinted light looks like training data.
    if color_jitter > 0 and x.shape[1] == 3:
        gains = torch.exp(u(n, 3, 1, 1) * color_jitter)
        x = (x * gains).clamp(0, 1)
        gray = x.mean(dim=1, keepdim=True)
        sat = 1 + u(n, 1, 1, 1) * color_jitter
        x = (gray + (x - gray) * sat).clamp(0, 1)

    # Slight blur on about a third of the images (3x3 box), then sensor noise.
    blur = torch.rand(n, 1, 1, 1, device=dev) < 0.33
    x = torch.where(blur, F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False), x)
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1)
