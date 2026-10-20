"""Mini-batch training: the loop that turns gradients into learning.

Phase 3, Step 10.  Everything the network needs in order to learn is
now in place: ``forward`` (predictions), ``cross_entropy_loss`` (one
number saying how wrong the predictions are), and ``backward``
(gradients for every weight).  This module is the *recipe* that
combines them:

1. **Shuffle** the training examples with a seeded RNG and cut them
   into mini-batches (see :func:`iterate_minibatches`).
2. For each batch do the four-beat cycle::

       logits = net.forward(X_batch)
       loss   = cross_entropy_loss(logits, y_batch)
       grads  = net.backward(X_batch, y_batch)
       update_parameters(net, grads, lr)   # W -= lr * dW, for all four

   That update is the *entire* optimizer -- plain gradient descent,
   no library.
3. After one full pass over the data (an **epoch**), record the
   average training loss, then measure validation loss **and the
   top-1 / top-5 move-class accuracies** (see
   :func:`evaluate_accuracy`) on both splits.  All of these
   measurements are forward-only: they never update a weight, because
   a validation set the model learns from stops measuring
   generalization and starts measuring memorization.

The test set is deliberately *absent* from this module: it will be
evaluated once, at the very end, after every choice is frozen.

Memory profile (16 GB machine): the training matrix (118006 x 781
float32 = ~369 MB) is loaded once and never copied wholesale; each
mini-batch is a ~0.4 MB gather; gradients are the size of the
parameters (~14 MB total).  Peak overhead beyond the dataset itself is
far below 100 MB.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from chessmovenet.checkpoint import save_checkpoint, save_history
from chessmovenet.loss import cross_entropy_loss
from chessmovenet.model import PARAMETER_NAMES, ChessMoveNet

# PARAMETER_NAMES now lives in chessmovenet.model (next to the
# parameters themselves); importing it here keeps
# ``from chessmovenet.train import PARAMETER_NAMES`` working.


def _check_positive_int(value, name: str) -> int:
    """Validate that ``value`` is an integer >= 1; return it as ``int``.

    Bools are rejected explicitly because ``isinstance(True, int)`` is
    True in Python -- passing ``batch_size=True`` should fail, not
    silently mean 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"expected {name} to be an integer, got {type(value).__name__}"
        )
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _check_learning_rate(lr) -> float:
    """Validate that ``lr`` is a finite positive number; return as float."""
    if isinstance(lr, bool) or not isinstance(
        lr, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"expected lr to be a number, got {type(lr).__name__}")
    lr = float(lr)
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"lr must be a finite positive number, got {lr}")
    return lr


def _check_unit_interval(value, name: str) -> float:
    """Validate an Adam decay coefficient in the half-open interval [0, 1)."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"expected {name} to be a number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1), got {value}")
    return value


def _check_epsilon(value) -> float:
    """Validate Adam's positive denominator stabilizer."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"expected adam_epsilon to be a number, got {type(value).__name__}"
        )
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"adam_epsilon must be finite and positive, got {value}")
    return value


def _check_seed(seed) -> int:
    """Validate that ``seed`` is a non-negative integer; return as int."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(
            f"expected seed to be an integer, got {type(seed).__name__}"
        )
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return seed


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive hidden-layer widths for the CLI."""
    try:
        widths = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden sizes must be comma-separated integers, e.g. 256,128"
        ) from exc
    if not widths or any(width < 1 for width in widths):
        raise argparse.ArgumentTypeError(
            "hidden sizes must be positive integers, e.g. 256,128"
        )
    return widths



def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int = 128, seed: int = 0
):
    """Yield reproducibly shuffled mini-batches of ``(X_batch, y_batch)``.

    The rows of ``X`` and the labels of ``y`` always stay paired: row
    ``k`` of every yielded ``X_batch`` is the position whose move is
    label ``k`` of the matching ``y_batch``.

    Args:
        X: Feature matrix, shape ``(N, features)`` (e.g. ``X_train``,
            118006 x 781 float32).  Never modified.
        y: Label vector, shape ``(N,)`` (e.g. ``y_train``).  Never
            modified.
        batch_size: Examples per batch; any integer >= 1.  The final
            batch is shorter when ``batch_size`` does not divide ``N``.
        seed: Seed for the shuffle RNG.  The same (X, batch_size, seed)
            always produces the same sequence of batches.

    Yields:
        ``(X_batch, y_batch)`` pairs: float32 ``(b, features)`` and
        int64 ``(b,)``, where the last batch may have ``b < batch_size``.

    Raises:
        TypeError: If ``batch_size`` is not an integer.
        ValueError: If ``X`` is not 2-D, ``y`` is not 1-D, their example
            counts differ, the dataset is empty, or ``batch_size < 1``.
    """
    # --- Validation, done eagerly (not inside the generator) ----------
    # A generator body does not run until the first next() call, so a
    # mistake in the arguments would otherwise stay hidden until the
    # training loop is already underway.  Validating here means a bad
    # call fails at the call site, where the bug actually is.
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(
            f"expected X to be 2-D (N, features), but got "
            f"{X.ndim} dimension(s) with shape {X.shape}"
        )
    if y.ndim != 1:
        raise ValueError(f"expected y to be 1-D (N,), but got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of examples, got "
            f"{X.shape[0]} and {y.shape[0]}"
        )
    if X.shape[0] == 0:
        raise ValueError("cannot iterate over an empty dataset")
    batch_size = _check_positive_int(batch_size, "batch_size")
    seed = _check_seed(seed)

    n = X.shape[0]

    # One shuffle per call.  permutation(n) is a random ordering of the
    # indices 0 .. n-1 -- roughly 0.9 MB of int64 for 118006 examples,
    # instead of the 369 MB a full shuffled copy of X would cost.
    order = np.random.default_rng(seed).permutation(n)  # (n,)

    def batches():
        """Inner generator: validation is done, now just feed batches."""
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]  # (b,)
            # Fancy indexing gathers exactly these rows as a fresh
            # copy (~b x 781 floats).  The parent arrays are read-only
            # here: shuffling never permutes X or y in place.
            yield X[idx], y[idx]

    return batches()


def _validated_gradients(
    net: ChessMoveNet, grads: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Validate gradients and return them in the model parameter order."""
    if not isinstance(grads, dict):
        raise TypeError(
            f"expected grads to be a dict, got {type(grads).__name__}"
        )
    parameter_names = net.parameter_names
    if set(grads) != set(parameter_names):
        raise ValueError(
            f"grads must have exactly the keys {parameter_names}, "
            f"got {tuple(sorted(grads))}"
        )

    checked: dict[str, np.ndarray] = {}
    for name in parameter_names:
        gradient = np.asarray(grads[name])
        parameter = getattr(net, name)
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"gradient {name} has shape {gradient.shape}, but the "
                f"parameter has shape {parameter.shape}"
            )
        if not np.issubdtype(gradient.dtype, np.number):
            raise TypeError(
                f"gradient {name} must contain numbers, got dtype "
                f"{gradient.dtype}"
            )
        if not np.isfinite(gradient).all():
            raise ValueError(f"gradient {name} must be finite")
        checked[name] = gradient.astype(parameter.dtype, copy=False)
    return checked


class AdamOptimizer:
    """Adam optimizer with per-parameter first and second moments.

    Adam combines momentum (an exponentially weighted mean of gradients) with
    an adaptive step size (an exponentially weighted mean of squared
    gradients).  The bias corrections make the early steps unbiased even
    though both moment estimates begin at zero.
    """

    def __init__(
        self,
        net: ChessMoveNet,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.lr = _check_learning_rate(lr)
        self.beta1 = _check_unit_interval(beta1, "adam_beta1")
        self.beta2 = _check_unit_interval(beta2, "adam_beta2")
        self.epsilon = _check_epsilon(epsilon)
        self.steps = 0
        self.m = {
            name: np.zeros_like(getattr(net, name)) for name in net.parameter_names
        }
        self.v = {
            name: np.zeros_like(getattr(net, name)) for name in net.parameter_names
        }

    def step(self, net: ChessMoveNet, grads: dict[str, np.ndarray]) -> None:
        """Update ``net`` once using Adam's bias-corrected moments."""
        grads = _validated_gradients(net, grads)
        self.steps += 1
        bias1 = 1.0 - self.beta1**self.steps
        bias2 = 1.0 - self.beta2**self.steps

        for name in net.parameter_names:
            gradient = grads[name]
            self.m[name] *= self.beta1
            self.m[name] += (1.0 - self.beta1) * gradient
            self.v[name] *= self.beta2
            self.v[name] += (1.0 - self.beta2) * np.square(gradient)

            m_hat = self.m[name] / bias1
            v_hat = self.v[name] / bias2
            getattr(net, name)[:] -= self.lr * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )


def update_parameters(
    net: ChessMoveNet, grads: dict[str, np.ndarray], lr
) -> None:
    """Apply one gradient-descent step to the network's weights, in place.

    The whole update, for each of the four parameters::

        parameter -= lr * gradient

    For example ``W1 -= lr * dW1``: every weight slides *against* its
    gradient by a step proportional to ``lr``.  Why against?  The
    gradient points in the direction the loss increases fastest, so
    subtracting it moves the weight in the direction the loss
    *decreases* -- downhill.  ``lr`` (the learning rate) scales how
    far down we step in one go: too small and training crawls, too
    large and each step overshoots and the loss can diverge.

    The parameters are modified **in place** (the arrays the network
    already owns); nothing new is allocated beyond one temporary per
    parameter, and dtypes stay float32 because ``float * float32
    array`` is float32 in NumPy's promotion rules.

    Args:
        net: A :class:`~chessmovenet.model.ChessMoveNet` whose ``W1``,
            ``b1``, ``W2``, ``b2`` will be updated.
        grads: The dict returned by :meth:`ChessMoveNet.backward` --
            exactly the keys ``"W1", "b1", "W2", "b2"``, each gradient
            the same shape as its parameter.
        lr: Learning rate; a finite number > 0 (e.g. ``0.001``).

    Returns:
        ``None`` -- the network itself is the result.

    Raises:
        TypeError: If ``lr`` is not a number or a gradient is not
            numeric.
        ValueError: If ``lr`` is not finite and positive, the keys of
            ``grads`` are not exactly the four parameter names, or a
            gradient's shape does not match its parameter.
    """
    lr = _check_learning_rate(lr)

    grads = _validated_gradients(net, grads)
    for name in net.parameter_names:
        gradient = grads[name]
        parameter = getattr(net, name)
        # The update itself: parameter = parameter - lr * gradient.
        # "-=" computes into the existing array (no new allocation)
        # and keeps float32, so memory stays flat epoch after epoch.
        parameter -= lr * gradient


def evaluate_loss(
    net: ChessMoveNet,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
) -> float:
    """Average loss of ``net`` on ``(X, y)`` -- **forward pass only**.

    The network is read, never written: no ``backward``, no
    ``update_parameters``.  That is the whole point of evaluation --
    measuring must not change what is being measured.  (The weights
    after this call are bit-for-bit identical to before it.)

    Mini-batches keep memory bounded: a 16246-row validation set never
    sits in memory as one big forward pass, only ``batch_size`` rows at
    a time (~0.4 MB).

    The per-batch losses are combined as a *weighted* mean (weight =
    batch size), so a short final batch cannot skew the average --
    the result equals the loss computed over all rows at once.

    Args:
        net: The network to evaluate (unmodified).
        X: Feature matrix ``(N, features)`` -- e.g. ``X_val``.
        y: Labels ``(N,)`` -- e.g. ``y_val``.
        batch_size: Rows per forward pass; same default as training.

    Returns:
        The mean cross-entropy as a Python ``float``.

    Raises:
        TypeError: If ``batch_size`` is not an integer.
        ValueError: If shapes are invalid, the data is empty, or
            ``batch_size < 1`` (raised by :func:`iterate_minibatches`).
    """
    total_loss = 0.0  # sum of (batch loss * batch size)
    total_rows = 0  # sum of batch sizes
    # seed only controls the *order* rows are visited; a weighted mean
    # is order-independent, so any fixed seed gives the same answer.
    for X_batch, y_batch in iterate_minibatches(
        X, y, batch_size=batch_size, seed=0
    ):
        loss = cross_entropy_loss(net.forward(X_batch), y_batch)
        total_loss += loss * X_batch.shape[0]
        total_rows += X_batch.shape[0]

    return total_loss / total_rows


def evaluate_accuracy(
    net: ChessMoveNet,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
) -> dict[str, float]:
    """Top-1 and top-5 accuracy of ``net`` on ``(X, y)`` -- read-only.

    What this measures (and what it does not)
    ------------------------------------------
    The model scores all 4,272 move classes for every position; this
    metric compares those scores with the move the player *actually
    played*:

    - **top-1**: the single highest-scoring class is the right move;
    - **top-5**: the true move appears among the five highest-scoring
      classes.

    Both are **raw move-class accuracies**: no move is filtered for
    legality, so a "hit" is a hit on the recorded label across all
    classes -- not necessarily on a *playable* move.  Restricting the
    candidate set to legal moves would be a different metric
    ("legal-move accuracy"); it is deliberately not mixed in here.
    The distinction is documented in the README.

    Properties:

    - **Read-only**: only ``forward`` is called; parameters are never
      touched (same contract as :func:`evaluate_loss`).
    - **Batched**: memory stays at ``batch_size`` rows, and a short
      final batch counts each of its rows exactly once -- the
      percentage is over *rows*, never over batches.
    - **Exact**: hits are integer comparisons on raw logits.  ``softmax``
      is unnecessary because it never reorders logits, and no logs or
      divisions are involved.
    - ``top5`` uses ``min(5, output_size)`` classes, so a toy network
      with <= 5 outputs is trivially 100% top-5 (every class fits).
    - Non-finite logits (a diverged network) raise instead of silently
      corrupting the counts.

    Args:
        net: Any object with ``forward(X)`` returning ``(N, C)`` scores
            and an ``output_size`` attribute (in practice a
            :class:`ChessMoveNet`).
        X: Feature matrix ``(N, features)``; validated as usual.
        y: Integer labels ``(N,)``, each in ``0 .. output_size - 1``.
        batch_size: Rows per forward pass (>= 1).

    Returns:
        ``{"top1": p1, "top5": p5}`` -- percentages in ``[0, 100]``
        as Python floats.

    Raises:
        TypeError: If ``batch_size`` is not an integer or the labels
            are not integers.
        ValueError: If shapes are invalid, the data is empty, a label
            is out of range, or the produced logits are non-finite.
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError(f"y must contain integer labels, got dtype {y.dtype}")
    n_classes = int(net.output_size)
    if n_classes < 1:
        raise ValueError(
            f"net.output_size must be positive, got {n_classes}"
        )
    if y.size and (int(y.min()) < 0 or int(y.max()) >= n_classes):
        raise ValueError(
            f"labels must be in 0 .. {n_classes - 1}, "
            f"got a value outside this range"
        )
    k5 = min(5, n_classes)

    rows = 0  # examples scored
    top1_hits = 0
    top5_hits = 0

    for X_batch, y_batch in iterate_minibatches(
        X, y, batch_size=batch_size, seed=0
    ):
        logits = net.forward(X_batch)  # (b, C) -- forward validates X
        if not np.isfinite(logits).all():
            raise ValueError(
                "logits contain NaN or infinite values; the network's "
                "weights may have diverged"
            )

        # Top-1: argmax over all classes (ties -> lowest index, stable).
        best = np.argmax(logits, axis=-1)  # (b,)
        top1_hits += int(np.count_nonzero(best == y_batch))

        # Top-5: argsort ascending puts the largest scores last, so the
        # final k5 columns are the top-k classes (unordered is fine --
        # we only test membership).
        top5_idx = np.argsort(logits, axis=-1)[:, -k5:]  # (b, k5)
        in_top5 = np.any(top5_idx == y_batch[:, np.newaxis], axis=-1)  # (b,)
        top5_hits += int(np.count_nonzero(in_top5))

        rows += y_batch.shape[0]

    return {"top1": 100.0 * top1_hits / rows, "top5": 100.0 * top5_hits / rows}



def train(
    net: ChessMoveNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 0.001,
    seed: int = 0,
    optimizer: str = "sgd",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    verbose: bool = True,
) -> dict[str, list]:
    """Train ``net`` with mini-batch gradient descent; return history.

    One call = ``epochs`` passes over the training data.  Each epoch:

    1. shuffle the training rows (seed ``seed + epoch``, so every epoch
       shuffles differently but the *whole run* is reproducible from
       one seed), split into mini-batches;
    2. for each batch: forward -> loss -> backward -> optimizer update;
    3. record the average training loss over all batches (weighted by
       batch size), then -- forward only, **no weight updates** --
       measure validation loss with :func:`evaluate_loss` and raw
       top-1 / top-5 move-class accuracy on *both* splits with
       :func:`evaluate_accuracy`;
    4. record how long the epoch took (including those measurements).

    Accuracy is measured *after* the epoch's updates with the final
    epoch weights -- unlike the loss, which is averaged while the
    weights are still moving -- so train and val accuracy for an epoch
    are computed with identical weights and are directly comparable.

    The test set is never accepted as an argument here on purpose:
    nothing in this function can see it.

    Args:
        net: The network to train -- updated **in place**.
        X_train, y_train: Training features ``(N, 781)`` and labels
            ``(N,)``; only these are ever learned from.
        X_val, y_val: Validation features and labels; measured only.
        epochs: Full passes over the training data (>= 1).
        batch_size: Examples per gradient update (>= 1).
        lr: Learning rate, finite and > 0.
        seed: Base seed for the shuffle RNG (>= 0).
        optimizer: ``"sgd"`` for plain gradient descent (the default), or
            ``"adam"`` for Adam's adaptive, momentum-based update.
        adam_beta1, adam_beta2, adam_epsilon: Adam hyperparameters. They are
            used only when ``optimizer="adam"``.
        verbose: When True, print one line per epoch.

    Returns:
        A dict with seven equal-length lists (one entry per epoch):
        ``"train_loss"``, ``"val_loss"`` (floats), ``"seconds"``
        (float, wall-clock epoch duration), and the raw move-class
        accuracies ``"train_acc_top1"``, ``"train_acc_top5"``,
        ``"val_acc_top1"``, ``"val_acc_top5"`` (floats in ``[0, 100]``).

    Raises:
        TypeError: If an integer-like argument is not an integer or
            ``lr`` is not a number.
        ValueError: If any setting is out of range, or a data pair has
            mismatched counts / is empty.
    """
    # Fail fast on bad settings *before* the first epoch starts --
    # a typo in --epochs 0 should not cost a minute of compute to
    # discover.
    epochs = _check_positive_int(epochs, "epochs")
    batch_size = _check_positive_int(batch_size, "batch_size")
    lr = _check_learning_rate(lr)
    seed = _check_seed(seed)
    if not isinstance(optimizer, str):
        raise TypeError(
            f"optimizer must be a string, got {type(optimizer).__name__}"
        )
    optimizer = optimizer.lower()
    if optimizer not in {"sgd", "adam"}:
        raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer!r}")
    adam = (
        AdamOptimizer(net, lr, adam_beta1, adam_beta2, adam_epsilon)
        if optimizer == "adam"
        else None
    )

    for tag, X, y in (
        ("X_train", X_train, y_train),
        ("X_val", X_val, y_val),
    ):
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(
                f"{tag}: expected 2-D X and 1-D y, got "
                f"{X.ndim}-D and {y.ndim}-D"
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"{tag}: X has {X.shape[0]} rows but y has {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError(f"{tag}: cannot train or evaluate on 0 rows")
    # Keep the normalized arrays (asarray is a no-op for ndarrays, so
    # no copies of the 369 MB matrix are made here).
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "seconds": [],
        "train_acc_top1": [],
        "train_acc_top5": [],
        "val_acc_top1": [],
        "val_acc_top5": [],
    }

    for epoch in range(epochs):
        started = time.perf_counter()

        # --- Pass 1: learn from the training set ---------------------
        total_loss = 0.0
        total_rows = 0
        for X_batch, y_batch in iterate_minibatches(
            X_train, y_train, batch_size=batch_size, seed=seed + epoch
        ):
            logits = net.forward(X_batch)  # (b, 4272)
            loss = cross_entropy_loss(logits, y_batch)  # Python float
            grads = net.backward(X_batch, y_batch)  # W1,b1,W2,b2 grads
            if adam is None:
                update_parameters(net, grads, lr)  # W -= lr * dW
            else:
                adam.step(net, grads)

            total_loss += loss * X_batch.shape[0]
            total_rows += X_batch.shape[0]

        train_loss = total_loss / total_rows

        # --- Pass 2: measure (never learn from) train and val ---------
        # All four calls are forward-only; the weights the accuracy
        # numbers are computed with are the epoch's final weights.
        val_loss = evaluate_loss(net, X_val, y_val, batch_size=batch_size)
        train_acc = evaluate_accuracy(
            net, X_train, y_train, batch_size=batch_size
        )
        val_acc = evaluate_accuracy(net, X_val, y_val, batch_size=batch_size)

        seconds = time.perf_counter() - started
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["seconds"].append(seconds)
        history["train_acc_top1"].append(train_acc["top1"])
        history["train_acc_top5"].append(train_acc["top5"])
        history["val_acc_top1"].append(val_acc["top1"])
        history["val_acc_top5"].append(val_acc["top5"])

        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs}  "
                f"train loss {train_loss:.4f} "
                f"top1 {train_acc['top1']:.1f}% top5 {train_acc['top5']:.1f}%  "
                f"val loss {val_loss:.4f} "
                f"top1 {val_acc['top1']:.1f}% top5 {val_acc['top5']:.1f}%  "
                f"{seconds:.1f}s",
                flush=True,
            )

    return history


def main() -> None:
    """Command-line entry point: train on ``split.npz`` (train + val only).

    Run from the project root::

        PYTHONPATH=src python -m chessmovenet.train --epochs 3

    Only ``X_train``/``y_train``/``X_val``/``y_val`` are read out of the
    archive -- ``X_test``/``y_test`` stay on disk, untouched.  After
    training, the weights are written to ``--checkpoint`` and the
    per-epoch records to ``--history`` (both refuse to overwrite
    existing files unless ``--overwrite`` is given).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train ChessMoveNet with mini-batch gradient descent on the "
            "prepared dataset (training + validation only; the test set "
            "is never loaded)."
        )
    )
    parser.add_argument(
        "--data",
        default="data/processed/split.npz",
        help="path to the prepared split archive (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="full passes over the training set (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="examples per gradient update (default: %(default)s)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="base seed for weight init and shuffling (default: %(default)s)",
    )
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam"), default="sgd",
        help="weight-update rule (default: %(default)s)",
    )
    parser.add_argument(
        "--activation", choices=("relu", "sigmoid", "tanh"), default="relu",
        help="hidden-layer activation (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-sizes", type=_parse_hidden_sizes, default=(256,),
        help="comma-separated hidden layer widths (default: 256)",
    )
    parser.add_argument(
        "--adam-beta1", type=float, default=0.9,
        help="Adam first-moment decay (default: %(default)s)",
    )
    parser.add_argument(
        "--adam-beta2", type=float, default=0.999,
        help="Adam second-moment decay (default: %(default)s)",
    )
    parser.add_argument(
        "--adam-epsilon", type=float, default=1e-8,
        help="Adam numerical stabilizer (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint", default="artifacts/checkpoint.npz",
        help="where to save trained weights (default: %(default)s)",
    )
    parser.add_argument(
        "--history", default="artifacts/history.npz",
        help="where to save the per-epoch history (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace the checkpoint/history files if they already exist",
    )
    args = parser.parse_args()

    # --- Fail fast, before loading 471 MB or training for minutes ----
    # Output paths are validated first so a refusal (existing file
    # without --overwrite, missing .npz suffix) costs zero compute.
    for flag, raw in (
        ("--checkpoint", args.checkpoint),
        ("--history", args.history),
    ):
        out = Path(raw)
        if out.suffix != ".npz":
            parser.error(f"{flag} must end in .npz, got {raw!r}")
        if out.exists() and not args.overwrite:
            parser.error(
                f"{flag} file {out} already exists; re-run with "
                f"--overwrite or choose another path"
            )

    # Load the archive once and touch only the four arrays we need.
    # An NpzFile reads each member lazily on demand (data[key]), so the
    # test rows are never even read from disk.
    data = np.load(args.data)
    try:
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
    finally:
        # Each member is fully decompressed into memory on access, so
        # the archive can be closed immediately afterwards.
        data.close()
    print(
        f"loaded {args.data}: "
        f"train {X_train.shape}, val {X_val.shape} "
        f"(test set untouched)",
        flush=True,
    )

    net = ChessMoveNet(
        seed=args.seed,
        activation=args.activation,
        hidden_sizes=args.hidden_sizes,
    )
    print(
        f"training: epochs={args.epochs} batch_size={args.batch_size} "
        f"lr={args.lr} optimizer={args.optimizer} activation={args.activation} "
        f"hidden_sizes={args.hidden_sizes} seed={args.seed}",
        flush=True,
    )
    history = train(
        net,
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        optimizer=args.optimizer,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )

    total = sum(history["seconds"])
    print(
        f"finished {len(history['seconds'])} epoch(s) in {total:.1f}s  |  "
        f"final train loss {history['train_loss'][-1]:.4f} "
        f"top1 {history['train_acc_top1'][-1]:.1f}%, "
        f"val loss {history['val_loss'][-1]:.4f} "
        f"top1 {history['val_acc_top1'][-1]:.1f}% "
        f"top5 {history['val_acc_top5'][-1]:.1f}%",
        flush=True,
    )

    # Persist the results (paths were validated before training; the
    # overwrite=False guard inside each saver still applies).
    save_checkpoint(args.checkpoint, net, overwrite=args.overwrite)
    save_history(args.history, history, overwrite=args.overwrite)
    print(f"saved checkpoint -> {args.checkpoint}", flush=True)
    print(f"saved history    -> {args.history}", flush=True)


if __name__ == "__main__":
    main()




