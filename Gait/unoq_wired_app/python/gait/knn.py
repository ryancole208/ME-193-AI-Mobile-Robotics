"""k-nearest-neighbours in plain numpy.

A prediction depends only on what is stored in config_wired.json: the training
feature matrix X (unscaled), labels y, the z-score scaler and k. Distance is
Euclidean on z-scored features and votes are weighted by 1/distance.

Tie and zero-distance rules copy scikit-learn's KNeighborsClassifier
(weights="distance") so tests can check that both give the same answers:
if a query has neighbours at distance exactly 0, only those vote (weight 1 each).
"""

from __future__ import annotations

import numpy as np


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)
        self.std = np.where(std < 1e-12, 1.0, std)  # constant feature: leave it unscaled

    @classmethod
    def fit(cls, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        return cls(X.mean(axis=0), X.std(axis=0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class KNNClassifier:
    def __init__(self, X, y, k: int, scaler: StandardScaler, n_classes: int):
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X[:, None]
        self.y = np.asarray(y, dtype=int)
        self.k = int(k)
        self.scaler = scaler
        self.n_classes = n_classes
        self._Xs = scaler.transform(self.X)

    @classmethod
    def fit(cls, X, y, k: int, n_classes: int) -> "KNNClassifier":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return cls(X, y, k, StandardScaler.fit(X), n_classes)

    @classmethod
    def from_config(cls, m: dict, n_classes: int) -> "KNNClassifier":
        return cls(m["X"], m["y"], m["k"], StandardScaler(m["scaler"]["mean"], m["scaler"]["std"]), n_classes)

    def kneighbors(self, Xq) -> tuple[np.ndarray, np.ndarray]:
        Q = self.scaler.transform(np.atleast_2d(np.asarray(Xq, dtype=float)))
        # squared Euclidean distances, (n_query, n_train)
        d2 = (Q ** 2).sum(1)[:, None] - 2 * Q @ self._Xs.T + (self._Xs ** 2).sum(1)[None, :]
        d = np.sqrt(np.maximum(d2, 0.0))
        k = min(self.k, len(self.y))
        idx = np.argsort(d, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(d, idx, axis=1), idx

    def vote(self, dist: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Distance-weighted class scores, shape (n_query, n_classes)."""
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        zero = dist == 0
        rows = zero.any(axis=1)
        w[rows] = zero[rows].astype(float)
        scores = np.zeros((len(dist), self.n_classes))
        labels = self.y[idx]
        for c in range(self.n_classes):
            scores[:, c] = (w * (labels == c)).sum(axis=1)
        return scores

    def predict(self, Xq) -> tuple[np.ndarray, np.ndarray]:
        """Returns (labels, distance to the nearest neighbour)."""
        dist, idx = self.kneighbors(Xq)
        scores = self.vote(dist, idx)
        return scores.argmax(axis=1), dist[:, 0]

    def to_config(self) -> dict:
        return {
            "k": self.k,
            "weights": "distance",
            "scaler": self.scaler.to_dict(),
            "X": self.X.tolist(),
            "y": self.y.tolist(),
        }
