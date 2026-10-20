"""A hand-checkable three-layer, one-neuron gradient-descent example.

This is deliberately separate from :mod:`chessmovenet.model`.  The chess
model is sized for a real multiclass task; this file is the compact example
for the course requirement where every layer has exactly one neuron and every
quantity can be printed or calculated by hand.

Architecture (for a batch of scalar inputs)::

    x -> Dense(1) -> ReLU -> Dense(1) -> ReLU -> Dense(1) -> prediction

The loss is mean squared error.  ``train`` uses mini-batches, so the same
forward/backward/update pattern used by the chess model is visible without
large matrices or thousands of output classes.
"""

from __future__ import annotations

import argparse

import numpy as np


class ThreeLayerOneNeuronNet:
    """Three scalar dense layers with ReLU in the first two layers."""

    def __init__(self) -> None:
        # Positive, non-zero values keep the tiny ReLU demonstration away from
        # a dead-unit start and make its first calculation easy to inspect.
        self.w1 = np.array([[0.5]], dtype=np.float64)
        self.b1 = np.array([0.1], dtype=np.float64)
        self.w2 = np.array([[0.5]], dtype=np.float64)
        self.b2 = np.array([0.1], dtype=np.float64)
        self.w3 = np.array([[0.5]], dtype=np.float64)
        self.b3 = np.array([0.1], dtype=np.float64)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return predictions and intermediates needed for backpropagation."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 1)
        z1 = X @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        prediction = a2 @ self.w3 + self.b3
        return prediction, {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

    def backward(self, cache: dict[str, np.ndarray], y: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
        """Differentiate mean squared error through all three dense layers."""
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = y.shape[0]
        # MSE = mean((prediction - y)^2), hence dL/dprediction below.
        d_prediction = 2.0 * (prediction - y) / n
        d_w3 = cache["a2"].T @ d_prediction
        d_b3 = d_prediction.sum(axis=0)
        d_a2 = d_prediction @ self.w3.T
        d_z2 = d_a2 * (cache["z2"] > 0.0)
        d_w2 = cache["a1"].T @ d_z2
        d_b2 = d_z2.sum(axis=0)
        d_a1 = d_z2 @ self.w2.T
        d_z1 = d_a1 * (cache["z1"] > 0.0)
        d_w1 = cache["X"].T @ d_z1
        d_b1 = d_z1.sum(axis=0)
        return {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2, "w3": d_w3, "b3": d_b3}

    def step(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        """Perform one gradient-descent update: parameter -= lr * gradient."""
        for name, gradient in gradients.items():
            setattr(self, name, getattr(self, name) - learning_rate * gradient)


def mean_squared_error(prediction: np.ndarray, y: np.ndarray) -> float:
    """Return the batch mean squared error."""
    return float(np.mean((prediction - np.asarray(y).reshape(-1, 1)) ** 2))


def train(
    net: ThreeLayerOneNeuronNet,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 500,
    batch_size: int = 4,
    learning_rate: float = 0.05,
) -> list[float]:
    """Train the scalar network in batches and return one loss per epoch."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if X.shape[0] != y.shape[0] or X.shape[0] == 0:
        raise ValueError("X and y must have the same non-zero number of rows")
    if batch_size < 1 or learning_rate <= 0.0 or epochs < 1:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    history = []
    for _ in range(epochs):
        for start in range(0, X.shape[0], batch_size):
            X_batch = X[start : start + batch_size]
            y_batch = y[start : start + batch_size]
            prediction, cache = net.forward(X_batch)
            net.step(net.backward(cache, y_batch, prediction), learning_rate)
        prediction, _ = net.forward(X)
        history.append(mean_squared_error(prediction, y))
    return history


def main() -> None:
    """Fit y = 2x + 1 and print the before/after loss."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    X = np.linspace(0.0, 1.0, 16).reshape(-1, 1)
    y = 2.0 * X + 1.0
    net = ThreeLayerOneNeuronNet()
    before, _ = net.forward(X)
    history = train(net, X, y, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    after, _ = net.forward(X)
    print(f"initial MSE: {mean_squared_error(before, y):.6f}")
    print(f"final MSE:   {history[-1]:.6f}")
    print("predictions:", np.round(after[:, 0], 3).tolist())


if __name__ == "__main__":
    main()
