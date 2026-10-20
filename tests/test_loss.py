"""Tests for :mod:`chessmovenet.loss` (cross-entropy loss).

Everything here is synthetic: logits come from seeded generators or
hand-written vectors, labels are plain integers.  No model or personal
game data is involved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chessmovenet.loss import cross_entropy_loss
from chessmovenet.moves import OUTPUT_SIZE

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Basic behavior: scalar, finite, correct formula
# ---------------------------------------------------------------------------


def test_loss_is_a_finite_non_negative_python_float() -> None:
    """A batch of random logits yields one finite float >= 0."""
    logits = RNG.normal(size=(8, OUTPUT_SIZE)).astype(np.float32)
    labels = RNG.integers(0, OUTPUT_SIZE, size=8)

    loss = cross_entropy_loss(logits, labels)

    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss >= 0.0


def test_loss_matches_the_naive_neg_log_softmax_definition() -> None:
    """For well-conditioned logits the stable implementation equals
    the textbook -mean(log(softmax(logits)[true class]))."""
    logits = RNG.normal(size=(5, 10)) * 2.0  # float64, no extremes
    labels = np.array([3, 0, 9, 1, 4])

    # Naive definition, computed carefully in float64.
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    naive = -np.log(probs[np.arange(5), labels]).mean()

    assert cross_entropy_loss(logits, labels) == pytest.approx(
        naive, rel=1e-9
    )


def test_loss_is_small_for_correct_and_big_for_wrong_label() -> None:
    """Row where class 0 dominates: predicting 0 costs almost nothing,
    predicting 1 costs a lot."""
    logits = np.zeros((1, 6))
    logits[0, 0] = 8.0  # strongly favors class 0 (p ~= 0.998)

    correct = cross_entropy_loss(logits, [0])
    wrong = cross_entropy_loss(logits, [1])

    assert correct < 0.01
    assert wrong > 5.0
    assert correct < wrong


def test_loss_decreases_when_the_true_class_logit_rises() -> None:
    """Raising the true class's logit from 0 to 3 must lower the loss
    (the core monotonicity requirement)."""
    labels = np.array([2])
    before = np.array([[0.0, 0.0, 0.0, 0.0]])
    after = np.array([[0.0, 0.0, 3.0, 0.0]])

    assert cross_entropy_loss(after, labels) < cross_entropy_loss(
        before, labels
    )


# ---------------------------------------------------------------------------
# Batches and batches of one
# ---------------------------------------------------------------------------


def test_batch_loss_is_the_mean_of_per_row_losses() -> None:
    """loss(batch, labels) == mean over rows of loss(row, label)."""
    logits = RNG.normal(size=(4, 32)) * 1.5
    labels = np.array([0, 7, 31, 16])

    batch_loss = cross_entropy_loss(logits, labels)
    row_losses = [
        cross_entropy_loss(logits[i], int(labels[i])) for i in range(4)
    ]

    assert batch_loss == pytest.approx(float(np.mean(row_losses)), rel=1e-9)


def test_batch_of_one_matches_the_single_row_form() -> None:
    """(1, C) with a (1,) label and (C,) with a scalar label agree."""
    row = RNG.normal(size=OUTPUT_SIZE)

    as_batch = cross_entropy_loss(row[np.newaxis, :], np.array([5]))
    as_single = cross_entropy_loss(row, 5)

    assert as_batch == pytest.approx(as_single, rel=1e-12)
    assert isinstance(as_batch, float)


# ---------------------------------------------------------------------------
# Extreme logits: the whole point of the stable implementation
# ---------------------------------------------------------------------------


def test_extreme_logits_stay_finite_and_correct() -> None:
    """logits of +/-1e4 produce finite losses; naive log(softmax)
    would give -inf/NaN here."""
    logits = np.array([[1e4, -1e4, 0.0]])

    win = cross_entropy_loss(logits, [0])  # true class dominates
    loss = cross_entropy_loss(logits, [1])  # true class annihilated

    assert math.isfinite(win)
    assert math.isfinite(loss)
    assert win >= 0.0 and loss >= 0.0
    # log-sum-exp of [1e4, -1e4, 0] minus logit 1e4 is ~log(e^-2e4) ...
    # i.e. the winning loss is essentially zero.
    assert win < 1e-6
    # The losing loss is approximately z_max - z_true = 1e4 - (-1e4).
    assert loss == pytest.approx(2e4, rel=1e-3)


def test_uniform_logits_give_loss_equal_to_log_num_classes() -> None:
    """All-equal logits -> uniform probabilities -> loss = ln(C),
    exactly like a freshly initialized balanced network."""
    logits = np.zeros((2, OUTPUT_SIZE))
    labels = np.array([0, OUTPUT_SIZE - 1])

    assert cross_entropy_loss(logits, labels) == pytest.approx(
        math.log(OUTPUT_SIZE), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_rejects_logits_with_wrong_dimensionality() -> None:
    """Scalars and 3-D arrays are not batches of logit rows."""
    with pytest.raises(ValueError, match="dimension"):
        cross_entropy_loss(np.float32(1.0), [0])
    with pytest.raises(ValueError, match="dimension"):
        cross_entropy_loss(np.zeros((2, 3, 4)), [0, 1])


def test_rejects_empty_batch() -> None:
    """Zero examples means there is nothing to average."""
    with pytest.raises(ValueError, match="at least one"):
        cross_entropy_loss(np.zeros((0, OUTPUT_SIZE)), [])


def test_rejects_label_count_mismatch() -> None:
    """Three logit rows need exactly three labels."""
    logits = np.zeros((3, OUTPUT_SIZE))

    with pytest.raises(ValueError, match="3 label"):
        cross_entropy_loss(logits, [0, 1])
    with pytest.raises(ValueError, match="3 label"):
        cross_entropy_loss(logits, [0, 1, 2, 3])


def test_rejects_labels_with_wrong_shape() -> None:
    """A (2, 2) label block is not one label per example."""
    logits = np.zeros((4, OUTPUT_SIZE))

    with pytest.raises(ValueError, match="shape"):
        cross_entropy_loss(logits, np.zeros((2, 2), dtype=np.int64))


def test_rejects_non_integer_labels() -> None:
    """Float labels are a bug, not a rounding opportunity."""
    logits = np.zeros((2, OUTPUT_SIZE))

    with pytest.raises(TypeError, match="integers"):
        cross_entropy_loss(logits, [0.0, 1.0])


def test_rejects_out_of_range_labels() -> None:
    """Labels must index a real class: -1 and n_classes are invalid."""
    logits = np.zeros((2, 10))

    with pytest.raises(ValueError, match="0 \\.\\. 9"):
        cross_entropy_loss(logits, [0, 10])
    with pytest.raises(ValueError, match="0 \\.\\. 9"):
        cross_entropy_loss(logits, [-1, 5])


def test_rejects_non_numeric_logits() -> None:
    """Strings cannot become a logit array."""
    with pytest.raises(TypeError, match="numbers"):
        cross_entropy_loss(["not", "numbers"], [0, 1])


def test_rejects_logits_with_zero_classes() -> None:
    """(N, 0) logits describe a world with no moves to choose from;
    the error must say that — not the confusing "0 .. -1" a bare
    range check would produce."""
    with pytest.raises(ValueError, match="at least one class"):
        cross_entropy_loss(np.zeros((2, 0)), [0, 1])


def test_rejects_non_finite_logits() -> None:
    """NaN or +/-inf logits (e.g. from an exploded training step)
    raise immediately instead of silently returning a NaN loss that
    would then corrupt every weight."""
    with pytest.raises(ValueError, match="finite"):
        cross_entropy_loss([[np.nan, 0.0, 1.0]], [0])
    with pytest.raises(ValueError, match="finite"):
        cross_entropy_loss([[np.inf, 0.0, 1.0]], [2])
    with pytest.raises(ValueError, match="finite"):
        cross_entropy_loss([[-np.inf, 0.0, 1.0]], [0])
    # A single bad entry hidden in a large batch is caught too, and
    # the message counts how many were found.
    logits = RNG.normal(size=(4, OUTPUT_SIZE))
    logits[2, 17] = np.nan
    with pytest.raises(ValueError, match="1 of them"):
        cross_entropy_loss(logits, np.zeros(4, dtype=np.int64))

