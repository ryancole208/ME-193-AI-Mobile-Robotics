"""Train the gesture CNN.

Run from Midterm2_Safe/:  ..\\my_env\\Scripts\\python.exe -m ml.train
Options: --epochs 60  --batch 64  --lr 3e-3  --val-fraction 0.2  --seed 0

Saves the FINAL epoch. With only ~1 validation burst per class, the "best" val
epoch is mostly luck (epoch 11 once hit 95.5% while still unstable); OneCycle ends
at a low learning rate, so the last epoch is the stable one.

Saves:
    models/gesture_cnn.pt      weights
    models/model_config.json   classes, preprocessing, mean/std, input size, results
    reports/training_curves.png
    reports/confusion_val.png  and the matrix printed in the console
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import preprocess  # noqa: E402
from gestures import CLASSES  # noqa: E402
from ml.dataset import augment, load, split_bursts  # noqa: E402
from ml.model import GestureCNN, count_params  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train the gesture CNN")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="train on CPU even if a GPU exists")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, x, y, mean, std, batch=256):
    model.eval()
    preds = []
    for i in range(0, len(x), batch):
        preds.append(model((x[i:i + batch] - mean) / std).argmax(1))
    pred = torch.cat(preds)
    cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for t, p in zip(y.cpu(), pred.cpu()):
        cm[t, p] += 1
    return (pred == y).float().mean().item(), cm


def print_confusion(cm):
    short = [c.replace("pinch_", "p_").replace("forward", "fwd")[:8] for c in CLASSES]
    print(f"{'true \\ pred':<14}" + "".join(f"{s:>9}" for s in short) + "   recall")
    for i, row in enumerate(cm.tolist()):
        rec = row[i] / max(sum(row), 1)
        print(f"{CLASSES[i]:<14}" + "".join(f"{v:>9}" for v in row) + f"   {rec:6.1%}")


def save_plots(history, cm, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="val")
    a1.set_title("loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(ep, [h["train_acc"] for h in history], label="train (augmented)")
    a2.plot(ep, [h["val_acc"] for h in history], label="val")
    a2.set_title("accuracy"); a2.set_xlabel("epoch"); a2.set_ylim(0, 1.01); a2.legend()
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm.numpy(), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = int(cm[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black")
    ax.set_title("validation confusion matrix (final epoch)")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_val.png", dpi=110); plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    train_items, val_items = split_bursts(args.val_fraction, args.seed)
    print(f"loading {len(train_items)} train + {len(val_items)} val images (split by burst)...")
    t0 = time.time()
    xtr, ytr = load(train_items)
    xva, yva = load(val_items)
    print(f"loaded in {time.time() - t0:.1f}s, input {tuple(xtr.shape[1:])}")

    # Normalization stats from the TRAINING set only.
    mean, std = xtr.mean().item(), xtr.std().item()
    xtr, ytr, xva, yva = xtr.to(dev), ytr.to(dev), xva.to(dev), yva.to(dev)

    model = GestureCNN(n_classes=len(CLASSES)).to(dev)
    print(f"model: GestureCNN, {count_params(model):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * ((len(xtr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = config.MODELS_DIR / "gesture_cnn.pt"
    best_acc, best_epoch, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        tot_loss, tot_ok = 0.0, 0
        for i in range(0, len(xtr), args.batch):
            idx = perm[i:i + args.batch]
            xb = (augment(xtr[idx]) - mean) / std
            out = model(xb)
            loss = F.cross_entropy(out, ytr[idx], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_ok += (out.argmax(1) == ytr[idx]).sum().item()

        model.eval()
        with torch.no_grad():
            val_loss = sum(F.cross_entropy(model((xva[j:j + 256] - mean) / std), yva[j:j + 256],
                                           reduction="sum").item()
                           for j in range(0, len(xva), 256)) / len(xva)
        val_acc, cm = evaluate(model, xva, yva, mean, std)
        h = {"epoch": epoch, "train_loss": tot_loss / len(xtr), "train_acc": tot_ok / len(xtr),
             "val_loss": val_loss, "val_acc": val_acc}
        history.append(h)
        mark = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            mark = "  (best so far)"
        print(f"epoch {epoch:3d}/{args.epochs}  loss {h['train_loss']:.3f}  "
              f"train {h['train_acc']:6.1%}  val loss {val_loss:.3f}  val {val_acc:6.1%}{mark}")

    torch.save(model.state_dict(), weights_path)
    final_acc, final_cm = val_acc, cm
    print(f"\nfinal (saved) val accuracy {final_acc:.1%}   "
          f"(highest was {best_acc:.1%} at epoch {best_epoch}, for reference only)")
    print_confusion(final_cm)

    cfg = {
        "classes": CLASSES,
        "model": "GestureCNN",
        "weights": weights_path.name,
        "input_shape": [1, 1, config.INPUT_SIZE[1], config.INPUT_SIZE[0]],
        "mean": mean,
        "std": std,
        "preprocess": preprocess.settings(),
        "training": {
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "weight_decay": args.weight_decay, "val_fraction": args.val_fraction,
            "seed": args.seed, "n_train": len(train_items), "n_val": len(val_items),
            "saved_epoch": args.epochs, "final_val_acc": final_acc,
            "best_epoch": best_epoch, "best_val_acc": best_acc,
            "val_bursts": sorted({p.stem.rsplit("_", 1)[0] for p, _ in val_items}),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    (config.MODELS_DIR / "model_config.json").write_text(json.dumps(cfg, indent=2))
    save_plots(history, final_cm, config.REPORTS_DIR)
    print(f"\nsaved {weights_path.relative_to(config.ROOT)}, models/model_config.json, "
          "reports/training_curves.png, reports/confusion_val.png")


if __name__ == "__main__":
    main()
