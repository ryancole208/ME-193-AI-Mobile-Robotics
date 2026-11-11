"""Train and evaluate the finger-count CNN.

Run from Midterm2_Safe/:  ..\\my_env\\Scripts\\python.exe -m ml.train
Options: --epochs 40 --batch 64 --lr 1e-3 --weight-decay 1e-4 --seed 0 --rot 25 [--vflip] [--gray] [--input-size N]

Train = all sessions not starting with "val"; validation = the "val" sessions.
The checkpoint with the best validation macro-F1 is kept. Each run gets an ID
(r<YYYYMMDD-HHMMSS>) and writes:
    models/finger_cnn.pt                        best weights (the latest run)
    models/train_meta.json                      classes, preprocessing, mean/std, results
    reports/<run_id>/metrics.json               accuracy, per-class recall/F1, confusion
    reports/<run_id>/confusion.png
    reports/<run_id>/curves.png
    reports/<run_id>/misclassified.png          grid of wrong validation images
    reports/<run_id>/finger_cnn.pt              copy of this run's best weights
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import preprocess  # noqa: E402
from gestures import CLASS_TO_COUNT, CLASSES  # noqa: E402
from ml.data_report import read  # noqa: E402
from ml.dataset import augment, load, split_sessions  # noqa: E402
from ml.model import FingerCNN, count_params  # noqa: E402

K = len(CLASSES)


def parse_args():
    p = argparse.ArgumentParser(description="Train the finger-count CNN")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    # The hand always enters the pad from the user's side, so full rotation and
    # vertical flips only made the task harder (see handoff Session 11).
    p.add_argument("--rot", type=float, default=25.0, help="max augmentation rotation, degrees")
    p.add_argument("--vflip", action="store_true", help="also flip vertically")
    p.add_argument("--color-jitter", type=float, default=0.3,
                   help="white-balance/saturation jitter strength (0 = off), for lighting changes")
    p.add_argument("--input-size", type=int, default=config.INPUT_SIZE,
                   help="model input side in pixels (also set config.INPUT_SIZE to deploy it)")
    p.add_argument("--gray", action="store_true",
                   help="grayscale input for an experiment (deploy uses config.GRAYSCALE)")
    p.add_argument("--cpu", action="store_true", help="train on CPU even if a GPU exists")
    return p.parse_args()


def pick_device(force_cpu):
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    if not force_cpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, x, mean, std, batch=256):
    model.eval()
    out = [model((x[i:i + batch] - mean) / std) for i in range(0, len(x), batch)]
    return torch.cat(out)


def metrics(y, pred):
    cm = np.zeros((K, K), int)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[t, p] += 1
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    precision = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    return {
        "accuracy": float(cm.diagonal().sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "recall": {c: float(r) for c, r in zip(CLASSES, recall)},
        "f1": {c: float(v) for c, v in zip(CLASSES, f1)},
        "most_confused": {"true": CLASSES[i], "pred": CLASSES[j], "count": int(off[i, j])},
        "confusion": cm.tolist(),
    }


def print_confusion(cm):
    print(f"{'true \\ pred':<12}" + "".join(f"{c:>7}" for c in CLASSES) + "   recall")
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:<12}" + "".join(f"{v:>7}" for v in row)
              + f"   {row[i] / max(sum(row), 1):6.1%}")


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
    a2.plot(ep, [h["val_acc"] for h in history], label="val accuracy")
    a2.plot(ep, [h["val_f1"] for h in history], label="val macro-F1")
    a2.set_title("validation"); a2.set_xlabel("epoch"); a2.set_ylim(0, 1.01); a2.legend()
    fig.tight_layout(); fig.savefig(out_dir / "curves.png", dpi=110); plt.close(fig)

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    labels = [f"{c} ({CLASS_TO_COUNT[c]})" for c in CLASSES]
    ax.set_xticks(range(K), labels, rotation=45, ha="right")
    ax.set_yticks(range(K), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(K):
        for j in range(K):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("validation confusion (best checkpoint)")
    fig.tight_layout(); fig.savefig(out_dir / "confusion.png", dpi=110); plt.close(fig)


def save_misclassified(items, y, pred, conf, out_dir, max_n=48, cols=8):
    wrong = [i for i in range(len(y)) if y[i] != pred[i]]
    if not wrong:
        return 0
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(wrong, min(max_n, len(wrong)), replace=False).tolist())
    s = 160
    tiles = []
    for i in pick:
        img = cv2.resize(read(items[i][0]), (s, s), interpolation=cv2.INTER_AREA)
        txt = f"{CLASS_TO_COUNT[CLASSES[y[i]]]}->{CLASS_TO_COUNT[CLASSES[pred[i]]]} {conf[i]:.2f}"
        cv2.putText(img, txt, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, txt, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 230, 255), 1)
        tiles.append(img)
    tiles += [np.zeros((s, s, 3), np.uint8)] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    ok, buf = cv2.imencode(".png", grid)
    (out_dir / "misclassified.png").write_bytes(buf.tobytes())
    return len(wrong)


def main():
    args = parse_args()
    config.INPUT_SIZE = args.input_size   # read by preprocess.raw_to_unit
    if args.gray:
        config.GRAYSCALE = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = pick_device(args.cpu)
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    run_id = "r" + time.strftime("%Y%m%d-%H%M%S")
    run_dir = config.REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_items, val_items = split_sessions()
    xtr, ytr = load(train_items)
    xva, yva = load(val_items)
    n_per = np.bincount(ytr.numpy(), minlength=K)
    n_val = np.bincount(yva.numpy(), minlength=K)
    print(f"run {run_id}: {len(xtr)} train / {len(xva)} val images (split by session)")
    print("train per class: " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, n_per)))
    print("val   per class: " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, n_val)))

    # Normalization stats from the TRAINING set only.
    mean, std = xtr.mean().item(), xtr.std().item()
    # Class-balanced loss when counts are uneven (weights average to 1).
    w = torch.tensor(len(ytr) / (K * np.maximum(n_per, 1)), dtype=torch.float32)
    uneven = n_per.max() > 1.1 * n_per.min()
    xtr, ytr, xva, yva = xtr.to(dev), ytr.to(dev), xva.to(dev), yva.to(dev)
    w = w.to(dev) if uneven else None
    print(f"class-balanced loss: {'on' if uneven else 'off'}")

    in_ch = 1 if config.GRAYSCALE else 3
    model = FingerCNN(n_classes=K, in_ch=in_ch).to(dev)
    print(f"model: FingerCNN, {count_params(model):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(xtr) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    weights_path = config.MODELS_DIR / "finger_cnn.pt"
    best = {"macro_f1": -1.0}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        tot_loss, tot_ok = 0.0, 0
        for i in range(0, len(xtr), args.batch):
            idx = perm[i:i + args.batch]
            out = model((augment(xtr[idx], args.rot, args.vflip, args.color_jitter) - mean) / std)
            loss = F.cross_entropy(out, ytr[idx], weight=w, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_ok += (out.argmax(1) == ytr[idx]).sum().item()

        logits = predict(model, xva, mean, std)
        val_loss = F.cross_entropy(logits, yva).item()
        m = metrics(yva.cpu(), logits.argmax(1).cpu())
        history.append({"epoch": epoch, "train_loss": tot_loss / len(xtr),
                        "train_acc": tot_ok / len(xtr), "val_loss": val_loss,
                        "val_acc": m["accuracy"], "val_f1": m["macro_f1"]})
        mark = ""
        if m["macro_f1"] > best["macro_f1"]:
            best = dict(m, epoch=epoch)
            torch.save(model.state_dict(), weights_path)
            mark = "  * saved"
        print(f"epoch {epoch:3d}/{args.epochs}  loss {tot_loss / len(xtr):.3f}  "
              f"train {tot_ok / len(xtr):6.1%}  val loss {val_loss:.3f}  "
              f"val acc {m['accuracy']:6.1%}  F1 {m['macro_f1']:.3f}{mark}")

    # Final evaluation of the saved best checkpoint.
    model.load_state_dict(torch.load(weights_path, map_location=dev))
    probs = torch.softmax(predict(model, xva, mean, std), 1).cpu()
    conf, pred = probs.max(1)
    final = metrics(yva.cpu(), pred)
    final["epoch"] = best["epoch"]
    n_wrong = save_misclassified(val_items, yva.cpu().tolist(), pred.tolist(), conf.tolist(), run_dir)
    save_plots(history, final["confusion"], run_dir)
    shutil.copy2(weights_path, run_dir / "finger_cnn.pt")

    gate = final["accuracy"] >= 0.90 and min(final["recall"].values()) >= 0.85
    print(f"\nrun {run_id}: best epoch {final['epoch']}, val accuracy {final['accuracy']:.1%}, "
          f"macro-F1 {final['macro_f1']:.3f}")
    print_confusion(final["confusion"])
    mc = final["most_confused"]
    print(f"most confused: true {mc['true']} -> predicted {mc['pred']} ({mc['count']} images)")
    print(f"accuracy gate (acc >= 90%, every recall >= 85%): {'PASS' if gate else 'FAIL'}")

    result = {
        "run_id": run_id, "classes": CLASSES, "class_to_count": CLASS_TO_COUNT,
        "model": "FingerCNN", "params": count_params(model),
        "input_shape": [1, in_ch, config.INPUT_SIZE, config.INPUT_SIZE],
        "mean": mean, "std": std, "preprocess": preprocess.settings(),
        "train_per_class": dict(zip(CLASSES, n_per.tolist())),
        "val_per_class": dict(zip(CLASSES, n_val.tolist())),
        "train_sessions": sorted({p.parent.name for p, _ in train_items}),
        "val_sessions": sorted({p.parent.name for p, _ in val_items}),
        "args": vars(args), "class_balanced_loss": bool(uneven),
        "val": final, "gate_pass": gate, "n_misclassified": n_wrong, "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    meta = {k: v for k, v in result.items() if k != "history"}
    (config.MODELS_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved models/finger_cnn.pt, models/train_meta.json, reports/{run_id}/")


if __name__ == "__main__":
    main()
