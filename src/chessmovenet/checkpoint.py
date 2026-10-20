"""Checkpointing: save and reload trained weights and training history.

Two file formats, both plain NumPy ``.npz`` archives:

- **Checkpoint** -- the four trained parameter arrays ``W1, b1, W2,
  b2``.  Loading one rebuilds a :class:`~chessmovenet.model.ChessMoveNet`
  with *bit-identical* weights, ready for evaluation or inference.
- **History** -- one entry per epoch: ``epoch``, ``train_loss``,
  ``val_loss``, ``seconds``.  Small arrays you can reload later to plot
  or compare runs.

Why no pickle
-------------
``np.savez`` stores each array as a raw ``.npy`` payload.  Because
every value here is plain numeric data (float32 weights, float64
losses, int64 epoch numbers), the files load with NumPy's *default*
``allow_pickle=False``.  Pickled ``.npz`` files can execute arbitrary
code when opened, which makes them unsafe to load from untrusted
sources -- this module never writes them, and :func:`load_checkpoint`
explicitly opens with ``allow_pickle=False`` to prove it.

Overwrite policy
----------------
Neither saver ever replaces an existing file unless ``overwrite=True``
is passed: a finished training run must not be destroyed by an
accidental re-run.  Existing files raise ``FileExistsError`` with the
exact flag needed to proceed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chessmovenet.model import PARAMETER_NAMES, ChessMoveNet

# What a history file must contain (train() returns the last three;
# "epoch" is derived as 1..N when saving if the caller omits it).
HISTORY_KEYS = ("epoch", "train_loss", "val_loss", "seconds")

# Accuracy columns written when the caller provides them (Step 11).
# They are optional so history files from older runs -- like the
# 3-epoch baseline, which has none -- keep loading unchanged.
OPTIONAL_HISTORY_KEYS = (
    "train_acc_top1",
    "train_acc_top5",
    "val_acc_top1",
    "val_acc_top5",
)


def _check_npz_path(path, role: str) -> Path:
    """Normalize ``path`` to a ``Path`` and require an ``.npz`` suffix.

    ``np.savez`` silently appends ``.npz`` to names that lack it,
    which would make the path you checked differ from the file actually
    written.  Refusing up front keeps "the path I passed" and "the file
    on disk" the same thing.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"{role} path must end in .npz, got {str(path)!r}")
    return path


def _check_writable(path: Path, overwrite: bool) -> None:
    """Create parent directories; refuse to clobber unless allowed."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it "
            f"or choose a different path"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    path, net: ChessMoveNet, *, overwrite: bool = False
) -> None:
    """Save the network's four parameters to ``path`` (a ``.npz`` file).

    Args:
        path: Destination file; must end in ``.npz``.  Missing parent
            directories (e.g. ``artifacts/``) are created.
        net: The network whose ``W1, b1, W2, b2`` are stored.  Written
            as float32, exactly the dtype the model trains in.
        overwrite: When False (the default) an existing file raises
            ``FileExistsError`` instead of being replaced.

    Raises:
        ValueError: If ``path`` does not end in ``.npz``.
        FileExistsError: If ``path`` exists and ``overwrite`` is False.
    """
    path = _check_npz_path(path, "checkpoint")
    _check_writable(path, overwrite)

    arrays = {
        name: np.asarray(getattr(net, name), dtype=np.float32)
        for name in PARAMETER_NAMES
    }
    # A Unicode scalar is safe with allow_pickle=False and lets a loaded model
    # retain whether its hidden layer uses ReLU, sigmoid, or tanh.  Older
    # checkpoints have no such field and are treated as ReLU below.
    arrays["activation"] = np.asarray(net.activation)
    # np.savez writes numeric arrays with pickle disabled on read;
    # no object arrays ever enter the file.
    np.savez(path, **arrays)


def load_checkpoint(path) -> ChessMoveNet:
    """Rebuild a :class:`ChessMoveNet` from a checkpoint saved earlier.

    The architecture itself is *derived from the stored shapes*
    (``W1`` is ``input x hidden``, ``W2`` is ``hidden x output``), then
    every array is cross-checked so a corrupt or mismatched file fails
    with a specific message instead of a mysterious downstream error.

    Args:
        path: A ``.npz`` file written by :func:`save_checkpoint`.

    Returns:
        A **fresh** network with parameters bit-identical to the saved
        one (``seed`` is irrelevant -- all weights are overwritten).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a parameter array is missing or the arrays are
            not mutually consistent (wrong dimensionality, shapes that
            do not chain ``input -> hidden -> output``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint found at {path}")

    # allow_pickle=False is the default and is kept explicit here: a
    # file that *needs* pickle to load is a file this module never
    # produced, and loading it would be a code-execution risk.
    with np.load(path, allow_pickle=False) as stored:
        missing = [k for k in PARAMETER_NAMES if k not in stored.files]
        if missing:
            raise ValueError(
                f"checkpoint {path} is missing parameter array(s): {missing}"
            )
        arrays = {
            name: np.asarray(stored[name], dtype=np.float32)
            for name in PARAMETER_NAMES
        }
        activation = (
            str(np.asarray(stored["activation"]).item())
            if "activation" in stored.files
            else "relu"
        )

    W1, b1, W2, b2 = (arrays[k] for k in PARAMETER_NAMES)

    # --- Cross-consistency checks: the shapes must chain -------------
    if W1.ndim != 2 or W2.ndim != 2:
        raise ValueError(
            f"W1 and W2 must be 2-D weight matrices, got "
            f"{W1.ndim}-D and {W2.ndim}-D"
        )
    if b1.shape != (W1.shape[1],):
        raise ValueError(
            f"b1 has shape {b1.shape}, but W1 implies a hidden layer "
            f"of width {W1.shape[1]}"
        )
    if W2.shape[0] != W1.shape[1]:
        raise ValueError(
            f"W2 expects {W2.shape[0]} hidden inputs, but W1 produces "
            f"{W1.shape[1]}"
        )
    if b2.shape != (W2.shape[1],):
        raise ValueError(
            f"b2 has shape {b2.shape}, but W2 implies an output layer "
            f"of width {W2.shape[1]}"
        )

    # Constructor validates that every size is a positive integer; a
    # zero-width layer in a hand-crafted file is rejected right there.
    net = ChessMoveNet(
        input_size=W1.shape[0],
        hidden_size=W1.shape[1],
        output_size=W2.shape[1],
        seed=0,  # overwritten immediately; only shapes matter here
        activation=activation,
    )
    net.W1, net.b1, net.W2, net.b2 = W1, b1, W2, b2
    return net


def save_history(path, history: dict, *, overwrite: bool = False) -> None:
    """Save per-epoch training records to ``path`` (a ``.npz`` file).

    Accepts exactly what :func:`chessmovenet.train.train` returns --
    ``{"train_loss": [...], "val_loss": [...], "seconds": [...]}`` --
    and stores them alongside a derived ``epoch`` counter (``1..N``).
    A pre-existing ``"epoch"`` key is kept as-is.  The Step-11 accuracy
    columns (:data:`OPTIONAL_HISTORY_KEYS`), when present, are saved
    too; when absent (older runs, like the 3-epoch baseline), the file
    simply has no accuracy columns and still loads fine.

    Validation happens *before* the file is touched, so a history that
    is malformed (missing keys, ragged lengths, NaN from a diverged
    run) produces a ``ValueError`` and leaves no half-written file
    behind.

    Args:
        path: Destination file; must end in ``.npz``.  Missing parent
            directories are created.
        history: Mapping whose ``train_loss``, ``val_loss`` and
            ``seconds`` are equal-length, 1-D, finite sequences with at
            least one epoch.
        overwrite: When False (default), an existing file raises
            ``FileExistsError``.

    Raises:
        TypeError: If ``history`` is not a dict.
        ValueError: If a required key is missing, lengths differ, a
            sequence is empty or not 1-D, a value is not finite, or
            ``path`` does not end in ``.npz``.
        FileExistsError: If ``path`` exists and ``overwrite`` is False.
    """
    path = _check_npz_path(path, "history")
    if not isinstance(history, dict):
        raise TypeError(
            f"history must be a dict, got {type(history).__name__}"
        )

    missing = [k for k in HISTORY_KEYS[1:] if k not in history]
    if missing:
        raise ValueError(
            f"history is missing required key(s): {missing} "
            f"(need {list(HISTORY_KEYS)})"
        )

    # Required columns first, then any accuracy columns the caller has.
    keys = list(HISTORY_KEYS[1:]) + [
        k for k in OPTIONAL_HISTORY_KEYS if k in history
    ]

    columns: dict[str, np.ndarray] = {}
    for key in keys:
        values = np.asarray(history[key])
        if values.ndim != 1:
            raise ValueError(
                f"history[{key!r}] must be 1-D, got shape {values.shape}"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(
                f"history[{key!r}] must contain numbers, got dtype "
                f"{values.dtype}"
            )
        columns[key] = values.astype(np.float64)

    lengths = {k: v.shape[0] for k, v in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"history entries must have equal lengths, got {lengths}"
        )
    n = columns["train_loss"].shape[0]
    if n == 0:
        raise ValueError("history must contain at least one epoch")
    for key, values in columns.items():
        bad = int(values.size - np.count_nonzero(np.isfinite(values)))
        if bad:
            raise ValueError(
                f"history[{key!r}] contains {bad} non-finite value(s) "
                f"(NaN or infinity) -- did training diverge?"
            )
        if key in OPTIONAL_HISTORY_KEYS and (
            values.min() < 0.0 or values.max() > 100.0
        ):
            raise ValueError(
                f"history[{key!r}] must be percentages in 0..100, got "
                f"range {values.min():.4g}..{values.max():.4g}"
            )

    if "epoch" in history:
        epoch = np.asarray(history["epoch"])
        if epoch.ndim != 1 or epoch.shape[0] != n:
            raise ValueError(
                f"history['epoch'] must be 1-D of length {n}, got "
                f"shape {epoch.shape}"
            )
        if not np.issubdtype(epoch.dtype, np.integer):
            raise TypeError(
                f"history['epoch'] must contain integers, got dtype "
                f"{epoch.dtype}"
            )
        epoch = epoch.astype(np.int64)
    else:
        epoch = np.arange(1, n + 1, dtype=np.int64)

    _check_writable(path, overwrite)
    # **columns writes exactly what was validated: epoch + required
    # columns + whichever accuracy columns were present in the dict.
    np.savez(path, epoch=epoch, **columns)


def load_history(path) -> dict[str, list]:
    """Load a history saved by :func:`save_history`.

    Args:
        path: A ``.npz`` file written by :func:`save_history`.

    Returns:
        A dict with the four keys of :data`HISTORY_KEYS`, each a Python
        list (``epoch`` ints; losses and durations floats) -- ready for
        plotting or printing.  Any accuracy columns the file contains
        (:data:`OPTIONAL_HISTORY_KEYS`) are included as well; files
        saved before Step 11 simply lack them.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a key is missing, a column is not 1-D, lengths
            disagree, or values are not finite.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no history file found at {path}")

    with np.load(path, allow_pickle=False) as stored:
        missing = [k for k in HISTORY_KEYS if k not in stored.files]
        if missing:
            raise ValueError(
                f"history {path} is missing key(s): {missing}"
            )
        extra = [k for k in OPTIONAL_HISTORY_KEYS if k in stored.files]
        columns = {
            k: np.asarray(stored[k]) for k in (*HISTORY_KEYS, *extra)
        }

    lengths = {k: v.shape[0] for k, v in columns.items()}
    for key, values in columns.items():
        if values.ndim != 1:
            raise ValueError(
                f"history[{key!r}] must be 1-D, got shape {values.shape}"
            )
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"history entries must have equal lengths, got {lengths}"
        )
    if columns["epoch"].shape[0] == 0:
        raise ValueError("history file contains no epochs")
    if not np.issubdtype(columns["epoch"].dtype, np.integer):
        raise TypeError(
            f"history['epoch'] must contain integers, got dtype "
            f"{columns['epoch'].dtype}"
        )
    for key in ("train_loss", "val_loss", "seconds"):
        values = columns[key].astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"history[{key!r}] contains non-finite values")
        columns[key] = values
    for key in extra:
        values = columns[key].astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"history[{key!r}] contains non-finite values")
        if values.min() < 0.0 or values.max() > 100.0:
            raise ValueError(
                f"history[{key!r}] must be percentages in 0..100, got "
                f"range {values.min():.4g}..{values.max():.4g}"
            )
        columns[key] = values

    return {
        key: (
            columns[key].astype(np.int64).tolist()
            if key == "epoch"
            else columns[key].tolist()
        )
        for key in HISTORY_KEYS + tuple(extra)
    }


