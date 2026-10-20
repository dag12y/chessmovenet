"""Tests for the small, hand-checkable course gradient-descent example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "three_layer_gradient_descent.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("three_layer_demo", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_one_neuron_layers_produce_scalar_batch_outputs_and_gradients():
    demo = _load_module()
    net = demo.ThreeLayerOneNeuronNet()
    prediction, cache = net.forward([[0.0], [1.0]])
    gradients = net.backward(cache, [[1.0], [3.0]], prediction)

    assert prediction.shape == (2, 1)
    assert set(gradients) == {"w1", "b1", "w2", "b2", "w3", "b3"}
    assert all(gradient.shape == (1, 1) or gradient.shape == (1,) for gradient in gradients.values())


def test_batched_gradient_descent_reduces_toy_regression_loss():
    demo = _load_module()
    X = np.linspace(0.0, 1.0, 16).reshape(-1, 1)
    y = 2.0 * X + 1.0
    net = demo.ThreeLayerOneNeuronNet()
    before, _ = net.forward(X)
    initial_loss = demo.mean_squared_error(before, y)

    history = demo.train(net, X, y, epochs=500, batch_size=4, learning_rate=0.05)
    assert history[-1] < initial_loss * 0.1
