"""StepTCN in PyTorch: the single source of truth for the architecture (contract 8.4).

A temporal convolutional network (TCN): 1-D convolutions along time, with the
dilation doubling each residual block so the network sees 1.25 s of context
(receptive field 125 samples) with only ~22.6k parameters. That is small enough
to run in numpy on the UNO Q in a few ms, and big enough to see a full stride.

    Conv1d(12->32, k=5, d=1) + ReLU
    4 x [ x + ReLU(Conv1d(32->32, k=5, d)) ]   d = 2, 4, 8, 16
    Conv1d(32->2, k=1)  -> logits (right, left)

"same" zero padding keeps the output aligned sample-for-sample with the input.
No batch-norm (it would need running statistics on the UNO Q and behaves badly
with tiny batches); dropout only during training.
"""
import numpy as np
import torch
import torch.nn as nn

N_IN = 12
WIDTH = 32
KERNEL = 5
DILATIONS = (2, 4, 8, 16)


class StepTCN(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.c0 = nn.Conv1d(N_IN, WIDTH, KERNEL, padding=KERNEL // 2)
        self.res = nn.ModuleList(
            nn.Conv1d(WIDTH, WIDTH, KERNEL, dilation=d, padding=d * (KERNEL // 2)) for d in DILATIONS
        )
        self.out = nn.Conv1d(WIDTH, 2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, 12, T) normalised input -> (batch, 2, T) logits."""
        h = torch.relu(self.c0(x))
        for conv in self.res:
            h = h + self.drop(torch.relu(conv(h)))
        return self.out(h)


def receptive_field():
    return 1 + (KERNEL - 1) * (1 + sum(DILATIONS))


def param_count(model=None):
    model = model or StepTCN()
    return sum(p.numel() for p in model.parameters())


def to_npz_dict(model, i):
    """Member i's weights under the contract key names, float32, PyTorch (out, in, k) layout."""
    g = lambda t: t.detach().cpu().numpy().astype(np.float32)
    d = {f"m{i}_c0_w": g(model.c0.weight), f"m{i}_c0_b": g(model.c0.bias),
         f"m{i}_out_w": g(model.out.weight), f"m{i}_out_b": g(model.out.bias)}
    for j, conv in enumerate(model.res):
        d[f"m{i}_r{j}_w"] = g(conv.weight)
        d[f"m{i}_r{j}_b"] = g(conv.bias)
    return d


def from_npz_dict(npz, i):
    """Rebuild member i as a PyTorch model (used by `diag` to cross-check a model.npz)."""
    m = StepTCN()
    t = lambda k: torch.from_numpy(np.asarray(npz[k], np.float32))
    with torch.no_grad():
        m.c0.weight.copy_(t(f"m{i}_c0_w")); m.c0.bias.copy_(t(f"m{i}_c0_b"))
        m.out.weight.copy_(t(f"m{i}_out_w")); m.out.bias.copy_(t(f"m{i}_out_b"))
        for j, conv in enumerate(m.res):
            conv.weight.copy_(t(f"m{i}_r{j}_w")); conv.bias.copy_(t(f"m{i}_r{j}_b"))
    return m.eval()


def probs(models, x):
    """Sigmoid outputs of every member for one (12, T) array -> (M, 2, T) numpy."""
    xt = torch.from_numpy(np.asarray(x, np.float32))[None]
    with torch.no_grad():
        return np.stack([torch.sigmoid(m.eval()(xt))[0].numpy() for m in models])


if __name__ == "__main__":
    print(f"StepTCN: {param_count():,} parameters, receptive field {receptive_field()} samples")
