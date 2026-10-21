"""Cross-entropy loss for move-class prediction.

The loss is the network's report card: one number saying how badly it
predicted the moves of a batch.  Training (a later phase) will try to
make this number small.

The formula
-----------
For one position with logits ``z`` (4,272 raw scores) and true label
``y`` (the move actually played), cross-entropy is

.. math::

    L = -\\log p_y

where ``p = softmax(z)`` is the predicted probability distribution.
Over a batch of ``N`` positions we take the mean:

.. math::

    L = -\\frac{1}{N} \\sum_{i=1}^{N} \\log p^{(i)}_{y_i}

In plain language: look at the probability the network gave the move
that was *actually played*; take ``-log`` of it.  Confident and right
→ probability near 1 → ``-log(1) = 0`` loss.  Confident and wrong →
probability near 0 → ``-log(0)`` blows up toward infinity.  The mean
spreads the score evenly over the batch.

Why not just compute softmax, then ``-log``?
--------------------------------------------
Because softmax can *underflow*: if one logit is far larger than the
true class's logit, the true class's probability rounds to exactly
``0.0`` in float32, and ``log(0) = -inf`` → a ruined (NaN) loss.  So
this module never forms those tiny probabilities.  Instead it uses the
**log-sum-exp** identity, which is algebraically the same expression:

.. math::

    -\\log p_y = \\log \\sum_j e^{z_j} - z_y

Two properties make it safe:

1. Each row of logits is first shifted by its own maximum.  This is a
   mathematical no-op (the shift cancels between numerator and
   denominator of the softmax), but it guarantees every ``exp()``
   argument is ``<= 0`` — ``exp`` can then only *underflow* gently to
   0, never overflow to ``inf``.
2. The largest term of the sum becomes ``exp(0) = 1``, so the ``log``
   of the sum is always ``>= 0`` and well-defined.

The result is the same loss a naive implementation would give whenever
the naive one works at all — but it keeps working with logits of
``+/-1e4`` (see the tests).
"""

from __future__ import annotations

import numpy as np


def _check(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate loss inputs and put them into canonical shape.

    Canonical form: ``logits`` a 2-D ``(N, C)`` float64 array and
    ``labels`` a 1-D ``(N,)`` int64 array.  A 1-D logit row and a
    scalar label are accepted as shorthand for a batch of one.

    Args:
        logits: Raw scores, shape ``(C,)`` or ``(N, C)``.
        labels: Integer class labels, scalar or shape ``(N,)``.

    Returns:
        The normalized ``(logits, labels)`` pair.

    Raises:
        TypeError: If ``logits`` contain non-numbers or the labels are
            not integers.
        ValueError: If any shape is wrong, the batch is empty, there are
            no classes, a logit is NaN or infinite, or a label falls
            outside ``0 .. C - 1``.
    """
    try:
        logits = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"logits must contain numbers, got: {exc}") from exc

    labels = np.asarray(labels)
    if labels.ndim == 0:  # scalar label -> batch of one
        labels = labels.reshape(1)

    if logits.ndim == 1:  # a single row -> batch of one
        logits = logits[np.newaxis, :]
    if logits.ndim != 2:
        raise ValueError(
            f"expected logits of shape (N, n_classes), but got "
            f"{logits.ndim} dimension(s) with shape {logits.shape}"
        )
    n, n_classes = logits.shape
    if n == 0:
        raise ValueError("logits must contain at least one example")
    if n_classes == 0:
        # Without at least one class there is no move to score, and the
        # label range below would degenerate to the nonsense "0 .. -1".
        raise ValueError(
            "logits must contain at least one class column, but the "
            "last axis has width 0"
        )

    # Non-finite logits (NaN or +/-inf) usually mean something upstream
    # exploded — an overflow during training, or NaN features in the
    # input.  Caught here, the error names the real problem; left alone,
    # they would silently turn the loss (and later every weight) into
    # NaN with no warning at all.
    n_bad = int(logits.size - np.count_nonzero(np.isfinite(logits)))
    if n_bad:
        raise ValueError(
            f"logits must be finite (no NaN or infinity), but "
            f"{n_bad} of them are not"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"expected labels of shape ({n},), got shape {labels.shape}"
        )
    if labels.shape[0] != n:
        raise ValueError(
            f"expected {n} label(s) to match the logits batch, "
            f"got {labels.shape[0]}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"labels must be integers, got dtype {labels.dtype}")

    labels = labels.astype(np.int64)
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError(
            f"labels must be in 0 .. {n_classes - 1}, "
            f"got a value outside this range"
        )
    return logits, labels


def cross_entropy_loss(logits: np.ndarray, labels) -> float:
    """Mean multiclass cross-entropy between logits and true labels.

    Computed with the numerically stable log-sum-exp form (see the
    module docstring) — softmax probabilities are never logged.

    Args:
        logits: Raw scores from the network's forward pass, shape
            ``(N, n_classes)`` — e.g. ``net.forward(X)`` with the
            default ``apply_softmax=False``.  A single row of shape
            ``(n_classes,)`` is accepted as a batch of one.
        labels: The correct integer class per example: shape ``(N,)``,
            or a bare integer for a single row.  Each value must be in
            ``0 .. n_classes - 1``.

    Returns:
        The **mean** loss over the batch as a Python ``float`` — a
        scalar that is finite and ``>= 0`` for any finite logits.

    Raises:
        TypeError: If ``logits`` are not numeric or ``labels`` are not
            integers.
        ValueError: If shapes disagree, the batch is empty, there are no
            classes, a logit is NaN or infinite, or a label is out of
            range.
    """
    logits, labels = _check(logits, labels)
    n = logits.shape[0]

    # Stable log-sum-exp.
    row_max = np.max(logits, axis=1, keepdims=True)  # (N, 1)
    shifted = logits - row_max  # (N, C), all <= 0
    log_sum_exp = row_max[:, 0] + np.log(np.sum(np.exp(shifted), axis=1))

    # Negative log-probability of the correct class.
    true_logits = logits[np.arange(n), labels]  # (N,)
    per_example = log_sum_exp - true_logits  # (N,), each >= 0

    return float(np.mean(per_example))

