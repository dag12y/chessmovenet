"""Tests for :mod:`chessmovenet.model` (NumPy move-prediction network).

All inputs are synthetic: random ``float32`` vectors in the same style
as :func:`chessmovenet.encode.encode_board`.  No personal game data is
read.  Several tests zero out (or fill) the public weight arrays so the
expected logits and probabilities can be computed by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chessmovenet.encode import BOARD_FEATURE_SIZE
from chessmovenet.loss import cross_entropy_loss
from chessmovenet.model import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    ChessMoveNet,
    relu,
    sigmoid,
    softmax,
    tanh,
)
from chessmovenet.moves import OUTPUT_SIZE

# A fixed batch of synthetic "board vectors": 4 rows of 781 features.
# A seeded generator makes the data reproducible across runs.
RNG = np.random.default_rng(123)
BATCH = RNG.normal(size=(4, BOARD_FEATURE_SIZE)).astype(np.float32)
SINGLE = BATCH[0].copy()


# ---------------------------------------------------------------------------
# Weight initialization: shapes, dtypes, and reproducibility
# ---------------------------------------------------------------------------


def test_default_sizes_match_project_constants() -> None:
    """The network's defaults are exactly the project's 781/256/4272."""
    assert INPUT_SIZE == BOARD_FEATURE_SIZE == 781
    assert HIDDEN_SIZE == 256
    assert OUTPUT_SIZE == 4272

    net = ChessMoveNet()
    assert net.input_size == 781
    assert net.hidden_size == 256
    assert net.output_size == 4272


def test_parameter_shapes_are_documented_ones() -> None:
    """W1 (781, 256), b1 (256,), W2 (256, 4272), b2 (4272,)."""
    net = ChessMoveNet(seed=0)

    assert net.W1.shape == (INPUT_SIZE, HIDDEN_SIZE)
    assert net.b1.shape == (HIDDEN_SIZE,)
    assert net.W2.shape == (HIDDEN_SIZE, OUTPUT_SIZE)
    assert net.b2.shape == (OUTPUT_SIZE,)


def test_parameter_dtypes_are_float32() -> None:
    """All four parameter arrays use float32, like the board encoding."""
    net = ChessMoveNet(seed=0)

    for parameter in (net.W1, net.b1, net.W2, net.b2):
        assert parameter.dtype == np.float32


def test_biases_start_at_zero_weights_do_not() -> None:
    """Biases are a neutral zero start; weights carry random noise."""
    net = ChessMoveNet(seed=0)

    assert (net.b1 == 0.0).all()
    assert (net.b2 == 0.0).all()
    # Weights must actually be random (not all zero) and finite.
    assert (net.W1 != 0.0).any()
    assert (net.W2 != 0.0).any()
    assert np.isfinite(net.W1).all() and np.isfinite(net.W2).all()


def test_same_seed_gives_identical_weights() -> None:
    """Two networks with seed=7 are bit-identical in all parameters."""
    first = ChessMoveNet(seed=7)
    second = ChessMoveNet(seed=7)

    np.testing.assert_array_equal(first.W1, second.W1)
    np.testing.assert_array_equal(first.b1, second.b1)
    np.testing.assert_array_equal(first.W2, second.W2)
    np.testing.assert_array_equal(first.b2, second.b2)


def test_different_seeds_give_different_weights() -> None:
    """Different seeds produce different initializations."""
    first = ChessMoveNet(seed=7)
    second = ChessMoveNet(seed=8)

    assert not np.array_equal(first.W1, second.W1)
    assert not np.array_equal(first.W2, second.W2)


def test_invalid_layer_sizes_are_rejected() -> None:
    """Non-integer sizes raise TypeError; zero/negative sizes ValueError."""
    with pytest.raises(TypeError, match="input_size"):
        ChessMoveNet(input_size=781.5)
    with pytest.raises(ValueError, match="hidden_size"):
        ChessMoveNet(hidden_size=0)
    with pytest.raises(ValueError, match="output_size"):
        ChessMoveNet(output_size=-5)


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------


def test_relu_zeroes_negatives_and_keeps_the_rest() -> None:
    """relu([-2, -0.0, 0, 0.5, 3]) == [0, 0, 0, 0.5, 3]."""
    z = np.array([-2.0, -0.0, 0.0, 0.5, 3.0], dtype=np.float32)

    result = relu(z)

    np.testing.assert_array_equal(
        result, np.array([0.0, 0.0, 0.0, 0.5, 3.0], dtype=np.float32)
    )


def test_relu_preserves_shape_on_batches() -> None:
    """ReLU works element-wise: same shape, negatives zeroed, rest kept."""
    z = RNG.normal(size=(3, 5)).astype(np.float32)

    result = relu(z)

    assert result.shape == z.shape
    assert (result[z < 0] == 0.0).all()
    np.testing.assert_array_equal(result[z > 0], z[z > 0])


def test_sigmoid_is_stable_and_has_expected_values() -> None:
    values = np.array([-1e4, -1.0, 0.0, 1.0, 1e4], dtype=np.float32)

    result = sigmoid(values)

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, [0.0, 0.2689414, 0.5, 0.7310586, 1.0])


def test_tanh_is_bounded_odd_and_zero_at_zero() -> None:
    values = np.array([-1e4, -1.0, 0.0, 1.0, 1e4], dtype=np.float32)

    result = tanh(values)

    assert np.isfinite(result).all()
    assert (result >= -1.0).all() and (result <= 1.0).all()
    assert result[2] == 0.0
    np.testing.assert_allclose(result[0], -result[-1])
    np.testing.assert_allclose(result[1], -result[3])


def test_hidden_activation_is_configurable_and_backpropagates() -> None:
    X = RNG.normal(size=(3, 4)).astype(np.float32)
    y = np.array([0, 1, 2], dtype=np.int64)
    for activation in ("relu", "sigmoid", "tanh"):
        net = ChessMoveNet(4, 5, 3, seed=0, activation=activation)
        logits = net.forward(X)
        gradients = net.backward(X, y)
        assert logits.shape == (3, 3)
        assert all(np.isfinite(gradient).all() for gradient in gradients.values())


def test_invalid_hidden_activation_is_rejected() -> None:
    with pytest.raises(ValueError, match="activation"):
        ChessMoveNet(activation="gelu")
    with pytest.raises(TypeError, match="activation"):
        ChessMoveNet(activation=3)


# ---------------------------------------------------------------------------
# Softmax: valid probabilities, and numerical stability
# ---------------------------------------------------------------------------


def test_softmax_rows_are_valid_probability_distributions() -> None:
    """Finite values in [0, 1], one row of shape (C,), rows sum to 1."""
    logits = (RNG.normal(size=(6, OUTPUT_SIZE)) * 3.0).astype(np.float32)

    probs = softmax(logits)

    assert probs.shape == (6, OUTPUT_SIZE)
    assert np.isfinite(probs).all()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_softmax_works_on_a_single_1d_row() -> None:
    """A 1-D input comes back as a 1-D distribution summing to 1."""
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    probs = softmax(logits)

    assert probs.shape == (3,)
    np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-6)


def test_softmax_is_invariant_to_a_constant_shift() -> None:
    """softmax(z + c) == softmax(z) for any constant c (exactly the
    identity the stable implementation exploits)."""
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    np.testing.assert_allclose(
        softmax(logits + 1000.0), softmax(logits), rtol=1e-5
    )


def test_softmax_of_large_logits_stays_finite_and_correct() -> None:
    """Naive exp(1000) overflows float32 to inf; the stable version
    must return the same finite probabilities as the small equivalent."""
    big = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)

    probs = softmax(big)

    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs, softmax(big - 1000.0), rtol=1e-5)


def test_softmax_of_extreme_logits_stays_finite() -> None:
    """Even +/-1e4 logits produce a finite distribution summing to 1."""
    logits = np.array([1e4, -1e4, 0.0], dtype=np.float32)

    probs = softmax(logits)

    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# Forward pass: shapes, dtypes, single vs batch
# ---------------------------------------------------------------------------


def test_forward_single_input_returns_4272_logits() -> None:
    """One (781,) vector in -> one (4272,) logit vector out."""
    net = ChessMoveNet(seed=0)

    logits = net.forward(SINGLE)

    assert logits.shape == (OUTPUT_SIZE,)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()


def test_forward_batch_returns_one_row_per_position() -> None:
    """An (N, 781) batch -> (N, 4272) logits, one row per position."""
    net = ChessMoveNet(seed=0)

    logits = net.forward(BATCH)

    assert logits.shape == (BATCH.shape[0], OUTPUT_SIZE)
    assert logits.dtype == np.float32


def test_forward_with_softmax_returns_probability_rows() -> None:
    """With apply_softmax=True each row is a valid distribution."""
    net = ChessMoveNet(seed=0)

    probs = net.forward(BATCH, apply_softmax=True)

    assert probs.shape == (BATCH.shape[0], OUTPUT_SIZE)
    assert np.isfinite(probs).all()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_forward_batch_rows_match_single_forwards() -> None:
    """Feeding row i alone gives (about) the same result as row i of
    the batch — batching is only an efficiency trick, never a semantic
    change.  (Tiny float differences are allowed because BLAS may use
    different summation orders for different batch sizes; float32
    results can differ in the last few ULPs.)"""
    net = ChessMoveNet(seed=0)

    batch_logits = net.forward(BATCH)
    for i, position in enumerate(BATCH):
        np.testing.assert_allclose(
            batch_logits[i], net.forward(position), rtol=1e-5, atol=1e-5
        )


# ---------------------------------------------------------------------------
# ReLU inside the network, verified with hand-set weights
# ---------------------------------------------------------------------------


def test_negative_hidden_activations_collapse_logits_to_b2() -> None:
    """With W1 = 0 every neuron sees z = b1 = -1; ReLU turns all of
    them into 0, so the output layer contributes nothing but b2."""
    net = ChessMoveNet(seed=0)
    net.W1[:] = 0.0
    net.W2[:] = 0.0
    net.b1[:] = -1.0
    net.b2[:] = np.linspace(0.0, 1.0, net.output_size, dtype=np.float32)

    logits = net.forward(SINGLE)

    np.testing.assert_array_equal(logits, net.b2)


def test_positive_hidden_activations_flow_through_to_logits() -> None:
    """With W1 = 0 and b1 = 1, relu(1) = 1 for all 256 neurons, so
    logits = (row of ones) @ W2 + b2 = column sums of W2 + b2.  With
    W2 uniformly 0.01 every logit must be 256 * 0.01 = 2.56."""
    net = ChessMoveNet(seed=0)
    net.W1[:] = 0.0
    net.b1[:] = 1.0
    net.W2[:] = 0.01
    net.b2[:] = 0.0

    logits = net.forward(SINGLE)

    np.testing.assert_allclose(logits, 2.56, rtol=1e-5)


def test_fully_zero_network_is_exactly_uniform() -> None:
    """All logits 0 -> all exp(0) equal -> uniform 1/4272 each.  This
    also proves the zero biases really are a neutral start."""
    net = ChessMoveNet(seed=0)
    for parameter in (net.W1, net.b1, net.W2, net.b2):
        parameter[:] = 0.0

    probs = net.forward(SINGLE, apply_softmax=True)

    np.testing.assert_allclose(probs, 1.0 / net.output_size, rtol=1e-6)


# ---------------------------------------------------------------------------
# predict: the best class and its probability
# ---------------------------------------------------------------------------


def test_predict_single_returns_python_int_and_float() -> None:
    """Single input in -> plain (int label, float probability) out."""
    net = ChessMoveNet(seed=0)

    label, probability = net.predict(SINGLE)

    assert isinstance(label, int)
    assert isinstance(probability, float)
    assert 0 <= label < OUTPUT_SIZE
    assert 0.0 <= probability <= 1.0


def test_predict_batch_returns_one_label_and_probability_per_row() -> None:
    """Batch in -> two (N,) arrays out, aligned with the input rows."""
    net = ChessMoveNet(seed=0)

    labels, probabilities = net.predict(BATCH)

    assert labels.shape == (BATCH.shape[0],)
    assert probabilities.shape == (BATCH.shape[0],)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_predict_matches_argmax_of_forward_probabilities() -> None:
    """predict is literally argmax/argmax-value of the softmax output."""
    net = ChessMoveNet(seed=0)

    probs = net.forward(BATCH, apply_softmax=True)
    labels, confidences = net.predict(BATCH)

    np.testing.assert_array_equal(labels, np.argmax(probs, axis=-1))
    np.testing.assert_allclose(
        confidences, np.max(probs, axis=-1), rtol=1e-6
    )


def test_controlled_biases_drive_a_hand_computable_prediction() -> None:
    """Hand-built logits: b2 = 8.0 on class 41, 0 on all others, so
    softmax gives class 41 exactly e^8 / (e^8 + 4271)."""
    net = ChessMoveNet(seed=0)
    net.W1[:] = 0.0
    net.W2[:] = 0.0
    net.b2[:] = 0.0
    net.b2[41] = 8.0

    label, probability = net.predict(SINGLE)
    expected = math.exp(8.0) / (math.exp(8.0) + (OUTPUT_SIZE - 1))

    assert label == 41
    assert probability == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# predict_top_k: the k best classes
# ---------------------------------------------------------------------------


def test_top_k_single_returns_k_sorted_labels_and_probs() -> None:
    """Single input -> two (k,) arrays, probabilities descending."""
    net = ChessMoveNet(seed=0)

    labels, probabilities = net.predict_top_k(SINGLE, k=5)

    assert labels.shape == (5,)
    assert probabilities.shape == (5,)
    assert np.all(probabilities[:-1] >= probabilities[1:])  # descending
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert ((labels >= 0) & (labels < OUTPUT_SIZE)).all()


def test_top_k_batch_returns_n_by_k_arrays() -> None:
    """Batch input -> (N, k) labels and (N, k) probabilities."""
    net = ChessMoveNet(seed=0)

    labels, probabilities = net.predict_top_k(BATCH, k=3)

    assert labels.shape == (BATCH.shape[0], 3)
    assert probabilities.shape == (BATCH.shape[0], 3)


def test_top_k_matches_the_sorted_forward_probabilities() -> None:
    """The k returned probabilities are exactly the k largest values
    of the full softmax output, in descending order."""
    net = ChessMoveNet(seed=0)

    probs = net.forward(SINGLE, apply_softmax=True)
    _, top_probs = net.predict_top_k(SINGLE, k=10)

    np.testing.assert_allclose(
        top_probs, np.sort(probs)[-10:][::-1], rtol=1e-6
    )


def test_top_k_of_one_agrees_with_predict() -> None:
    """k=1 must return exactly the class that predict returns."""
    net = ChessMoveNet(seed=0)

    best_labels, _ = net.predict(BATCH)
    labels, _ = net.predict_top_k(BATCH, k=1)

    np.testing.assert_array_equal(labels[:, 0], best_labels)


def test_top_k_accepts_the_full_output_size_as_boundary() -> None:
    """k = output_size is the largest valid value and must work."""
    net = ChessMoveNet(seed=0)

    labels, probabilities = net.predict_top_k(SINGLE, k=OUTPUT_SIZE)

    assert labels.shape == (OUTPUT_SIZE,)
    assert probabilities.shape == (OUTPUT_SIZE,)


def test_top_k_rejects_k_outside_the_valid_range() -> None:
    """k = 0, negative k, or k > 4272 raise a clear ValueError."""
    net = ChessMoveNet(seed=0)

    with pytest.raises(ValueError, match="between 1 and"):
        net.predict_top_k(SINGLE, k=0)
    with pytest.raises(ValueError, match="between 1 and"):
        net.predict_top_k(SINGLE, k=-3)
    with pytest.raises(ValueError, match="between 1 and"):
        net.predict_top_k(SINGLE, k=OUTPUT_SIZE + 1)


def test_top_k_rejects_non_integer_k() -> None:
    """Floats, strings, and booleans are not acceptable k values."""
    net = ChessMoveNet(seed=0)

    with pytest.raises(TypeError, match="integer k"):
        net.predict_top_k(SINGLE, k=2.5)
    with pytest.raises(TypeError, match="integer k"):
        net.predict_top_k(SINGLE, k="3")
    with pytest.raises(TypeError, match="integer k"):
        net.predict_top_k(SINGLE, k=True)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_forward_rejects_wrong_feature_count() -> None:
    """780 (or 782) features is not a valid board vector."""
    net = ChessMoveNet(seed=0)

    with pytest.raises(ValueError, match="781"):
        net.forward(np.zeros(780, dtype=np.float32))
    with pytest.raises(ValueError, match="781"):
        net.forward(np.zeros((2, 782), dtype=np.float32))


def test_forward_rejects_arrays_with_wrong_dimensions() -> None:
    """3-D arrays and scalars are not boards and not batches."""
    net = ChessMoveNet(seed=0)

    with pytest.raises(ValueError, match="dimension"):
        net.forward(np.zeros((2, 3, BOARD_FEATURE_SIZE), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        net.forward(np.float32(1.0))


def test_forward_rejects_non_numeric_input() -> None:
    """A list of strings cannot be interpreted as board features."""
    net = ChessMoveNet(seed=0)

    with pytest.raises(TypeError, match="float"):
        net.forward(["not", "numbers"])


def test_forward_rejects_empty_batch() -> None:
    """A (0, 781) batch has no positions to score — likely an
    off-by-one slice in future training code, so fail loudly here."""
    net = ChessMoveNet(seed=0)

    with pytest.raises(ValueError, match="at least one position"):
        net.forward(np.zeros((0, BOARD_FEATURE_SIZE), dtype=np.float32))


def test_forward_rejects_non_finite_features() -> None:
    """NaN or infinite board features would silently poison every
    logit and gradient; the error must point at the input."""
    net = ChessMoveNet(seed=0)
    x_nan = SINGLE.copy()
    x_nan[5] = np.nan
    x_inf = SINGLE.copy()
    x_inf[7] = np.inf

    with pytest.raises(ValueError, match="finite"):
        net.forward(x_nan)
    with pytest.raises(ValueError, match="finite"):
        net.forward(x_inf)
    # backward goes through the same validation, so bad inputs cannot
    # silently produce NaN gradients either.
    with pytest.raises(ValueError, match="finite"):
        net.backward(x_nan, 3)


def test_predict_and_top_k_validate_their_inputs_too() -> None:
    """The convenience methods share forward's validation."""
    net = ChessMoveNet(seed=0)
    bad = np.zeros(10, dtype=np.float32)

    with pytest.raises(ValueError, match="781"):
        net.predict(bad)
    with pytest.raises(ValueError, match="781"):
        net.predict_top_k(bad, k=2)


# ---------------------------------------------------------------------------
# Backpropagation: gradients of the mean cross-entropy loss
# ---------------------------------------------------------------------------


def test_backward_gradients_match_parameter_shapes_and_are_finite() -> None:
    """Every gradient array mirrors its parameter's shape, is finite,
    and the dict keys are exactly the parameter names."""
    net = ChessMoveNet(seed=0)
    labels = np.array([10, 20, 30, 40])

    grads = net.backward(BATCH, labels)

    assert set(grads) == {"W1", "b1", "W2", "b2"}
    for name in ("W1", "b1", "W2", "b2"):
        assert grads[name].shape == getattr(net, name).shape
        assert np.isfinite(grads[name]).all()


def test_backward_works_for_a_single_example_with_scalar_label() -> None:
    """One (781,) position and a bare integer label produce the same
    gradient shapes as a full batch."""
    net = ChessMoveNet(seed=0)

    grads = net.backward(SINGLE, 7)

    for name in ("W1", "b1", "W2", "b2"):
        assert grads[name].shape == getattr(net, name).shape
        assert np.isfinite(grads[name]).all()


def test_backward_leaves_the_parameters_untouched() -> None:
    """backward only *reports* gradients; it must not update weights."""
    net = ChessMoveNet(seed=0)
    before = {
        name: getattr(net, name).copy() for name in ("W1", "b1", "W2", "b2")
    }

    net.backward(BATCH, np.array([0, 1, 2, 3]))

    for name in ("W1", "b1", "W2", "b2"):
        np.testing.assert_array_equal(getattr(net, name), before[name])


def test_backward_gradients_are_averaged_over_the_batch() -> None:
    """The gradient of the *mean* loss over a batch equals the mean of
    the per-example gradients (each example's forward pass is
    independent, so only the final /N couples them)."""
    net = ChessMoveNet(input_size=6, hidden_size=5, output_size=4, seed=3)
    X = np.random.default_rng(9).normal(size=(3, 6)).astype(np.float32)
    labels = np.array([0, 1, 3])

    batch_grads = net.backward(X, labels)
    per_example = [net.backward(X[i], int(labels[i])) for i in range(3)]

    for name in ("W1", "b1", "W2", "b2"):
        mean_of_singles = np.mean([g[name] for g in per_example], axis=0)
        np.testing.assert_allclose(
            batch_grads[name], mean_of_singles, rtol=1e-4, atol=1e-6
        )


def test_backward_rejects_invalid_labels() -> None:
    """Wrong counts, wrong shapes, float labels, and out-of-range
    labels all fail with clear errors."""
    net = ChessMoveNet(seed=0)

    with pytest.raises(ValueError, match="4 label"):  # too few
        net.backward(BATCH, [0, 1])
    with pytest.raises(ValueError, match="4 label"):  # scalar vs batch
        net.backward(BATCH, 0)
    with pytest.raises(ValueError, match="shape"):  # 2-D label block
        net.backward(BATCH, np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(TypeError, match="integers"):  # floats
        net.backward(BATCH, [0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="0 \\.\\. 4271"):  # out of range
        net.backward(BATCH, [0, 1, 2, OUTPUT_SIZE])


def test_backward_rejects_invalid_inputs() -> None:
    """The input validation of forward applies unchanged."""
    net = ChessMoveNet(seed=0)

    with pytest.raises(ValueError, match="781"):
        net.backward(np.zeros(10, dtype=np.float32), 0)
    with pytest.raises(ValueError, match="dimension"):
        net.backward(np.zeros((2, 3, BOARD_FEATURE_SIZE)), [0, 1])


def test_numerical_gradient_check_on_a_tiny_network() -> None:
    """Compare every analytical gradient entry against a central
    finite difference: ``(L(w+eps) - L(w-eps)) / (2 * eps)``.

    The network is deliberately tiny (5 -> 4 -> 3, so only 39 parameter
    entries) to keep the ~78 extra forward passes instant.  Parameters
    are promoted to float64 — finite differences in float32 would drown
    the comparison in rounding noise.
    """
    net = ChessMoveNet(input_size=5, hidden_size=4, output_size=3, seed=11)
    net.W1 = net.W1.astype(np.float64)
    net.b1 = net.b1.astype(np.float64)
    net.W2 = net.W2.astype(np.float64)
    net.b2 = net.b2.astype(np.float64)

    X = np.random.default_rng(0).normal(size=(2, 5)).astype(np.float32)
    labels = np.array([1, 2])

    # Guard against ReLU kinks: the finite-difference formula assumes a
    # smooth region, so every pre-activation must sit safely away from 0
    # (otherwise a +/-eps step could flip a unit's on/off state).
    Z1 = X @ net.W1 + net.b1
    assert np.min(np.abs(Z1)) > 1e-3

    analytic = net.backward(X, labels)

    eps = 1e-5
    for name in ("W1", "b1", "W2", "b2"):
        parameter = getattr(net, name)
        numeric = np.empty_like(np.asarray(parameter))
        for index in np.ndindex(parameter.shape):
            original = parameter[index]

            parameter[index] = original + eps
            loss_plus = cross_entropy_loss(net.forward(X), labels)
            parameter[index] = original - eps
            loss_minus = cross_entropy_loss(net.forward(X), labels)
            parameter[index] = original  # restore before continuing

            numeric[index] = (loss_plus - loss_minus) / (2.0 * eps)

        np.testing.assert_allclose(
            analytic[name], numeric, rtol=1e-4, atol=1e-7, err_msg=name
        )

