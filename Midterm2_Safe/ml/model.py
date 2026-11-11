"""Small from-scratch CNN that counts fingers (6 classes: 0-5).

Input (N, 1, 96, 96) normalized grayscale -> logits (N, 6). Only Conv, BatchNorm,
ReLU, MaxPool, global average pool, Dropout and Linear, so it exports cleanly to
ONNX and stays small for the UNO Q. Built from random init; no pretrained parts.
"""
import torch
from torch import nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FingerCNN(nn.Module):
    def __init__(self, n_classes=6, in_ch=1, widths=(16, 32, 64, 96), dropout=0.3):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers.append(block(c, w))
            c = w
        self.features = nn.Sequential(*layers)   # 96x96 -> 6x6 after 4 pools
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import time
    m = FingerCNN().eval()
    x = torch.zeros(1, 1, 96, 96)
    print(m(x).shape, f"{count_params(m):,} parameters")
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(5):
            m(x)
        t = time.perf_counter()
        for _ in range(50):
            m(x)
    print(f"CPU, 1 thread: {(time.perf_counter() - t) / 50 * 1000:.1f} ms/frame")
