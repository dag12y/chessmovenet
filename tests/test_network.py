"""Tests for chessmovenet.network."""

import numpy as np

from chessmovenet.encode import encode_fens
from chessmovenet.moves import encode_moves
from chessmovenet.network import (
    cross_entropy,
    forward,
    init_params,
    relu,
    softmax,
)

INPUT_SIZE = 773
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4096


def _load_training_batch(n=1000):
    data = np.load("data/processed/train.npz", allow_pickle=True)
    X = encode_fens(data["fen"][:n])
    y = encode_moves(data["uci"][:n])
    return X, y


def test_init_params_shapes_and_dtypes():
    params = init_params()
    assert params["W1"].shape == (INPUT_SIZE, HIDDEN_SIZE)
    assert params["b1"].shape == (HIDDEN_SIZE,)
    assert params["W2"].shape == (HIDDEN_SIZE, OUTPUT_SIZE)
    assert params["b2"].shape == (OUTPUT_SIZE,)
    for arr in params.values():
        assert arr.dtype == np.float32


def test_init_params_zero_biases():
    params = init_params()
    assert np.all(params["b1"] == 0.0)
    assert np.all(params["b2"] == 0.0)


def test_forward_output_shapes():
    params = init_params()
    X = np.random.default_rng(1).standard_normal((5, INPUT_SIZE)).astype(np.float32)
    P, cache = forward(params, X)
    assert P.shape == (5, OUTPUT_SIZE)
    assert cache["X"].shape == (5, INPUT_SIZE)
    assert cache["Z1"].shape == (5, HIDDEN_SIZE)
    assert cache["A1"].shape == (5, HIDDEN_SIZE)
    assert cache["Z2"].shape == (5, OUTPUT_SIZE)


def test_probability_rows_sum_to_one():
    params = init_params()
    X = np.random.default_rng(2).standard_normal((10, INPUT_SIZE)).astype(np.float32)
    P, _ = forward(params, X)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_softmax_overflow_invariance():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((4, 6)).astype(np.float32)
    assert np.allclose(softmax(z), softmax(z + 1000.0), rtol=1e-3, atol=1e-4)


def test_relu_zeroes_negatives():
    z = np.array([[-3.0, -1.0, 0.0, 2.0, 5.0]], dtype=np.float32)
    assert np.allclose(relu(z), [[0.0, 0.0, 0.0, 2.0, 5.0]])


def test_cross_entropy_zero_on_certain_prediction():
    P = np.zeros((3, 4), dtype=np.float32)
    P[0, 1] = 1.0
    P[1, 2] = 1.0
    P[2, 0] = 1.0
    y = np.array([1, 2, 0])
    loss = cross_entropy(P, y)
    assert abs(loss) < 1e-6


def test_initial_loss_close_to_log_output_size():
    X, y = _load_training_batch(1000)
    params = init_params(seed=0)
    P, _ = forward(params, X)
    loss = cross_entropy(P, y)
    expected = np.log(OUTPUT_SIZE)
    assert abs(loss - expected) < 0.5, f"loss {loss:.4f} not within 0.5 of {expected:.4f}"
