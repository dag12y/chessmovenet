"""A minimal fully connected neural network that predicts chess moves.

The network learns to imitate a player: given the 781-feature encoding of
a board position (see :mod:`chessmovenet.encode`), it produces one score
per move class (see :mod:`chessmovenet.moves`).  Only NumPy is used —
every line below is the honest arithmetic that ML frameworks normally
hide behind their own abstractions.

Architecture
------------

::

    781 inputs -> 256 hidden units (ReLU) -> 4272 logits -> softmax?

Layer by layer:

1. **Hidden layer** — ``Z1 = X @ W1 + b1`` then ``A1 = relu(Z1)``.
   781 numbers go in, 256 come out.  Each hidden neuron computes a
   weighted sum of the inputs; after training, neurons tend to respond
   to useful *patterns* of pieces and features.  ReLU (``max(0, z)``)
   keeps only positive evidence and zeroes the rest.  This is what
   makes the network *non-linear*: without it, stacking layers would
   collapse algebraically into one big matrix multiply, no matter how
   many layers we add.

2. **Output layer** — ``logits = A1 @ W2 + b2``.  256 numbers go in,
   4272 come out — one **logit** (raw score) per possible move class.
   Logits are unbounded real numbers: a bigger logit means "the network
   leans toward this class", but logits are **not** probabilities —
   they can be negative, exceed 1, and do not sum to 1.

3. **Softmax** (applied only on request) — turns the 4272 logits of
   each row into 4272 numbers in ``(0, 1)`` that sum to 1: a
   probability distribution over the move classes.  See :func:`softmax`
   for the numerically stable implementation.

Weight initialization (He / Kaiming)
------------------------------------
Weights are drawn from ``N(0, sqrt(2 / fan_in))``, where ``fan_in`` is
the number of inputs a unit receives (781 for the hidden layer, 256 for
the output layer).  ReLU zeroes roughly half of its inputs, so a plain
``1/fan_in`` scale would shrink signals as they pass through; the
factor ``2`` compensates for exactly that and keeps the size of
activations roughly stable from layer to layer.  Biases start at zero:
zero is neutral, and the symmetric start means the fresh network is
"undecided" rather than accidentally favoring some classes.

Reproducibility
---------------
Passing a ``seed`` creates the weights from a local
``np.random.default_rng(seed)`` stream, so two networks built with the
same seed get bit-identical parameters and building a network never
disturbs NumPy's global random state.

Legality
--------
The network scores **all** 4272 classes for every position, including
moves that are illegal in that position (a rook sliding like a queen,
a pawn jumping two squares from the third rank, ...).  Nothing here
filters by legality: the highest-probability move is the network's raw
opinion, and a caller that needs playable moves must combine it with a
legality check.  This module deliberately stays chess-agnostic — it
only ever sees vectors of numbers.

Not in this module (future phases): backpropagation, loss functions, a
training loop, and saving/loading weights.
"""

from __future__ import annotations

import numpy as np

from chessmovenet.encode import BOARD_FEATURE_SIZE
from chessmovenet.moves import OUTPUT_SIZE

# The network consumes exactly what chessmovenet.encode produces and
# emits exactly one score per class of chessmovenet.moves, so the sizes
# are imported (single source of truth) rather than re-typed here.
INPUT_SIZE = BOARD_FEATURE_SIZE  # 781 features per encoded position
HIDDEN_SIZE = 256  # neurons in the single hidden layer


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified linear unit: ``max(0, z)``, applied element-wise.

    The activation function of the hidden layer.  Negative pre-activations
    become ``0.0``; zero and positive values pass through unchanged.

    Args:
        z: Any array of pre-activation values, any shape.

    Returns:
        An array of the **same shape** as ``z`` with every negative
        value replaced by ``0.0``.
    """
    return np.maximum(z, 0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis.

    Converts raw scores (logits) into probabilities: every output lies
    in ``(0, 1)`` and the values along the last axis sum to 1.

    Why "stable"?  A naive ``exp(z) / sum(exp(z))`` overflows float32
    for logits much larger than ~88: ``exp(1000)`` is ``inf`` and
    ``inf / inf`` is ``NaN``.  Subtracting the maximum logit of each row
    first changes nothing mathematically — for any constant ``c``,
    ``exp(z - c) / sum(exp(z - c))`` equals ``exp(z) / sum(exp(z))`` —
    but the largest exponent in each row becomes ``exp(0) = 1`` instead
    of something enormous.

    Args:
        logits: A 1-D array of shape ``(C,)`` or a 2-D batch of shape
            ``(N, C)`` of raw scores.

    Returns:
        An array of the same shape, holding probabilities along the
        last axis (each row sums to ``1.0``), dtype float32 for
        float32 input.
    """
    # keepdims=True keeps the max in shape (N, 1) so NumPy broadcasting
    # subtracts one row-max from every element of the corresponding row.
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


class ChessMoveNet:
    """Two-layer fully connected network: 781 -> 256 (ReLU) -> 4272.

    The four parameter arrays are plain, public NumPy attributes so a
    future training phase (backpropagation and gradient updates —
    deliberately *not* implemented here) can read and adjust them
    directly without any accessor machinery.

    Attributes:
        input_size: Features per position (default 781).
        hidden_size: Hidden neurons (default 256).
        output_size: Move classes (default 4272).
        W1: Hidden-layer weights, shape ``(input_size, hidden_size)`` =
            ``(781, 256)``, float32, He-initialized.
        b1: Hidden-layer biases, shape ``(hidden_size,)`` = ``(256,)``,
            float32, zeros.
        W2: Output-layer weights, shape ``(hidden_size, output_size)``
            = ``(256, 4272)``, float32, He-initialized.
        b2: Output-layer biases, shape ``(output_size,)`` = ``(4272,)``,
            float32, zeros.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = OUTPUT_SIZE,
        seed: int | None = None,
    ) -> None:
        """Create the network and initialize its parameters.

        Args:
            input_size: Number of input features per position.
            hidden_size: Number of neurons in the hidden layer.
            output_size: Number of output move classes.
            seed: Optional integer seed for reproducible weights.  The
                same seed always produces the same initial weights;
                ``None`` gives fresh random weights each time.

        Raises:
            TypeError: If a size is not an integer.
            ValueError: If a size is less than 1.
        """
        # --- Sizes: small guards so mistakes surface immediately ---
        for name, value in (
            ("input_size", input_size),
            ("hidden_size", hidden_size),
            ("output_size", output_size),
        ):
            is_int = isinstance(value, (int, np.integer))
            if isinstance(value, bool) or not is_int:
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)

        # --- Reproducible random initialization -------------------------
        # default_rng(seed) is a local, isolated random stream: the same
        # seed always yields bit-identical weights, and using it never
        # touches np.random's global state.
        rng = np.random.default_rng(seed)

        # He (Kaiming) initialization: normal noise with standard
        # deviation sqrt(2 / fan_in), scaled so that ReLU keeps the
        # typical size of activations stable from layer to layer (see
        # the module docstring).  astype(np.float32) matches the dtype
        # of encode_board's output, keeping all arithmetic in float32.
        self.W1 = rng.normal(
            0.0,
            np.sqrt(2.0 / self.input_size),
            size=(self.input_size, self.hidden_size),
        ).astype(np.float32)  # (781, 256)

        # Zero biases: a neutral, symmetric start — no neuron and no
        # output class is favored before training.
        self.b1 = np.zeros(self.hidden_size, dtype=np.float32)  # (256,)

        self.W2 = rng.normal(
            0.0,
            np.sqrt(2.0 / self.hidden_size),
            size=(self.hidden_size, self.output_size),
        ).astype(np.float32)  # (256, 4272)

        self.b2 = np.zeros(self.output_size, dtype=np.float32)  # (4272,)

    def _check_input(self, X: np.ndarray) -> np.ndarray:
        """Convert ``X`` to float32 and validate its shape.

        Accepts a single board vector of shape ``(input_size,)`` or a
        batch of shape ``(N, input_size)``; nested Python lists of the
        same shapes work too (they are converted to arrays).

        Args:
            X: The candidate input.

        Returns:
            ``X`` as a float32 NumPy array, 1-D or 2-D.

        Raises:
            TypeError: If ``X`` cannot be converted to a numeric array.
            ValueError: If ``X`` has the wrong number of dimensions or
                the wrong feature count in its last axis.
        """
        try:
            X = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"X must contain numbers convertible to float32, got: {exc}"
            ) from exc

        if X.ndim not in (1, 2):
            raise ValueError(
                f"expected a 1-D board vector of shape ({self.input_size},) "
                f"or a 2-D batch of shape (N, {self.input_size}), "
                f"but got an array with {X.ndim} dimension(s) "
                f"and shape {X.shape}"
            )
        if X.shape[-1] != self.input_size:
            raise ValueError(
                f"each position must have exactly {self.input_size} "
                f"features, but the last axis of the input has "
                f"{X.shape[-1]}"
            )
        return X

    def forward(
        self, X: np.ndarray, apply_softmax: bool = False
    ) -> np.ndarray:
        """Run the network on one board vector or a batch of them.

        The full computation, with shapes for a batch of size ``N``
        (a single input is treated as ``N = 1`` internally):

        1. ``Z1 = X @ W1 + b1``      -> ``(N, 256)`` — hidden linear layer.
        2. ``A1 = relu(Z1)``         -> ``(N, 256)`` — negative evidence off.
        3. ``logits = A1 @ W2 + b2`` -> ``(N, 4272)`` — one raw score per
           move class.
        4. Optionally ``softmax(logits)`` -> probabilities.

        Args:
            X: Board features of shape ``(781,)`` (single position) or
                ``(N, 781)`` (batch), e.g. rows of ``encode_board`` or
                ``DatasetSplit.X_train``.
            apply_softmax: When True, return probabilities instead of
                raw logits (see the module docstring for the difference).

        Returns:
            A float32 array of shape ``(4272,)`` for a single input or
            ``(N, 4272)`` for a batch — logits by default, probabilities
            when ``apply_softmax=True``.  **No legality filtering is
            done**: every class gets a score, including moves that
            cannot be played in this position.
        """
        X = self._check_input(X)
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]  # (781,) -> (1, 781): treat as batch of 1

        # The heart of the network: two matrix multiplications with a
        # ReLU in between.  ``@`` is NumPy's matrix multiply, and the
        # (256,) / (4272,) bias vectors are automatically added to
        # every one of the N rows via broadcasting.
        Z1 = X @ self.W1 + self.b1  # (N,781)@(781,256)+(256,) = (N,256)
        A1 = relu(Z1)  # (N, 256)
        # (N,256)@(256,4272)+(4272,) = (N, 4272)
        logits = A1 @ self.W2 + self.b2

        output = softmax(logits) if apply_softmax else logits

        # "One vector in -> one vector out": drop the batch axis we added.
        return output[0] if single else output

    def predict(self, X: np.ndarray):
        """Return the most probable move class and its probability.

        The predicted class is the index of the largest probability
        (``argmax``); since softmax never changes the ordering of the
        logits, it is also the index of the largest logit.  Note this
        is the network's *opinion*, not a legal-move check: the argmax
        may well be an illegal move in the given position.

        Args:
            X: Board features of shape ``(781,)`` or ``(N, 781)``.

        Returns:
            For a single input: ``(label, probability)`` as a Python
            ``int`` and ``float``.  For a batch: ``(labels, probs)``,
            two float/int NumPy arrays of shape ``(N,)`` — the best
            class and its probability for each row.
        """
        probabilities = self.forward(X, apply_softmax=True)  # (N, 4272)
        labels = np.argmax(probabilities, axis=-1)
        confidences = np.max(probabilities, axis=-1)
        if probabilities.ndim == 1:
            return int(labels), float(confidences)
        return labels, confidences

    def predict_top_k(self, X: np.ndarray, k: int = 5):
        """Return the ``k`` most probable move classes, best first.

        Like :meth:`predict`, this performs **no legality filtering**:
        the top classes are simply the largest probabilities over all
        ``output_size`` classes.

        Args:
            X: Board features of shape ``(781,)`` or ``(N, 781)``.
            k: How many top classes to return; must satisfy
                ``1 <= k <= output_size``.

        Returns:
            A ``(labels, probs)`` pair.  For a single input both arrays
            have shape ``(k,)``; for a batch, shape ``(N, k)``.  Each
            row is sorted by probability, highest first.

        Raises:
            TypeError: If ``k`` is not an integer.
            ValueError: If ``k`` is outside ``1 .. output_size``.
        """
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError(f"expected an integer k, got {type(k).__name__}")
        k = int(k)
        if k < 1 or k > self.output_size:
            raise ValueError(
                f"k must be between 1 and {self.output_size} (the number "
                f"of move classes), got {k}"
            )

        probabilities = self.forward(X, apply_softmax=True)  # (N, 4272)
        single = probabilities.ndim == 1
        if single:
            probabilities = probabilities[np.newaxis, :]  # -> (1, 4272)

        # argsort sorts ascending, so the k largest entries sit at the
        # end; slicing them off and flipping gives descending order.
        # (With only 4272 classes a full sort is cheap — no need for
        # the more complex np.argpartition.)
        order = np.argsort(probabilities, axis=-1)  # (N, 4272)
        top_labels = order[:, -k:][:, ::-1]  # (N, k), best class first
        top_probs = np.take_along_axis(
            probabilities, top_labels, axis=-1
        )  # (N, k), matching probabilities

        if single:
            return top_labels[0], top_probs[0]  # (k,) each
        return top_labels, top_probs


