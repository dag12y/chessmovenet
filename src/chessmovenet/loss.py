"""Stable mean cross-entropy loss for multiclass logits."""

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
        # A zero-width class axis has no valid label range.
        raise ValueError(
            "logits must contain at least one class column, but the "
            "last axis has width 0"
        )

    # Reject invalid scores before they propagate NaNs through training.
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

