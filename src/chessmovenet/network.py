"""A tiny two-layer NumPy neural network for ChessMoveNet.

Only NumPy is used. No backpropagation yet -- this module handles parameter
initialization, the forward pass, and the cross-entropy loss.
"""

from __future__ import annotations

import numpy as np


def init_params(input_size=773, hidden_size=256, output_size=4096, seed=0):
    """Initialize network parameters.

    Weights are drawn from a standard normal and scaled by
    sqrt(2 / fan_in) (He initialization), which keeps the variance of
    activations stable across layers. Biases start at zero. All arrays are
    float32.
    """
    rng = np.random.default_rng(seed)
    params = {
        "W1": (
            rng.standard_normal((input_size, hidden_size))
            * np.sqrt(2.0 / input_size)
        ).astype(np.float32),
        "b1": np.zeros(hidden_size, dtype=np.float32),
        "W2": (
            rng.standard_normal((hidden_size, output_size))
            * np.sqrt(2.0 / hidden_size)
        ).astype(np.float32),
        "b2": np.zeros(output_size, dtype=np.float32),
    }
    return params


def relu(z):
    """Rectified linear unit: max(0, z), applied element-wise."""
    return np.maximum(0.0, z)


def softmax(z):
    """Row-wise softmax with numerical-stability shift.

    Subtracting the row max before exponentiating prevents overflow when
    logits are large, without changing the result (adding a constant to every
    logit in a row leaves softmax unchanged).
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(params, X):
    """Forward pass through the two-layer network.

    Returns (P, cache) where P is the softmax probability matrix and cache
    holds the intermediates needed later for backpropagation:
        X   input (N, input_size)
        Z1  pre-activation of hidden layer (N, hidden_size)
        A1  ReLU activation of hidden layer (N, hidden_size)
        Z2  pre-activation of output layer (N, output_size)
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # Hidden layer pre-activation: N x input_size @ input_size x hidden_size.
    Z1 = X @ W1 + b1
    # Hidden layer activation: ReLU element-wise.
    A1 = relu(Z1)
    # Output layer pre-activation: logits before softmax.
    Z2 = A1 @ W2 + b2
    # Convert logits to class probabilities.
    P = softmax(Z2)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}
    return P, cache


def cross_entropy(P, y):
    """Mean cross-entropy loss for integer class targets.

    P is (N, output_size) of softmax probabilities; y is an int array of
    shape (N,) with the true class index for each row.
    """
    N = P.shape[0]
    # Pick the predicted probability of the true class for each row.
    correct_probs = P[np.arange(N), y] + 1e-12
    return -np.mean(np.log(correct_probs))
