"""Classification metrics and their Markdown tables (no scikit-learn needed)."""

from __future__ import annotations

import numpy as np

from .constants import CLASSES


def confusion_matrix(y_true, y_pred, n_classes: int = len(CLASSES)) -> np.ndarray:
    """Rows = true class, columns = predicted class. Predictions < 0 ("unsure") are dropped."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        if 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def per_class(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), np.nan)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support.astype(int)}


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean recall over the classes that are present (an "unsure" counts as wrong)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls)) if recalls else float("nan")


def _fmt(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def confusion_markdown(cm: np.ndarray, classes=CLASSES) -> str:
    head = "| true \\ predicted | " + " | ".join(classes) + " |"
    sep = "|---|" + "---:|" * len(classes)
    rows = [f"| **{c}** | " + " | ".join(str(v) for v in cm[i]) + " |" for i, c in enumerate(classes)]
    return "\n".join([head, sep, *rows])


def per_class_markdown(cm: np.ndarray, classes=CLASSES) -> str:
    pc = per_class(cm)
    lines = ["| class | precision | recall | F1 | windows |", "|---|---:|---:|---:|---:|"]
    for i, c in enumerate(classes):
        lines.append(f"| {c} | {_fmt(pc['precision'][i])} | {_fmt(pc['recall'][i])} | {_fmt(pc['f1'][i])} | {pc['support'][i]} |")
    return "\n".join(lines)
