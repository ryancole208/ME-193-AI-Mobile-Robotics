"""Small from-scratch CNN for the 8 gestures.

Input (N, 1, 88, 160) normalized grayscale -> logits (N, 8). Only Conv, BatchNorm,
ReLU, MaxPool, global average pool and Linear, so it exports cleanly to ONNX and
stays small enough for the UNO Q (~100k parameters).
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


class GestureCNN(nn.Module):
    def __init__(self, n_classes=8, in_ch=1, widths=(16, 32, 48, 64), dropout=0.3):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers.append(block(c, w))
            c = w
        self.features = nn.Sequential(*layers)   # 88x160 -> 5x10 after 4 pools
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = GestureCNN()
    print(m(torch.zeros(2, 1, 88, 160)).shape, f"{count_params(m):,} parameters")
