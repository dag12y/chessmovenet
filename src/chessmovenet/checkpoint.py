"""Save and load PyTorch model checkpoints and per-epoch training history."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from chessmovenet.model import ChessMoveNet

CHECKPOINT_FORMAT = "chessmovenet-pytorch-v1"
CHECKPOINT_SUFFIXES = {".pt", ".pth"}
HISTORY_KEYS = ("epoch", "train_loss", "val_loss", "seconds")
OPTIONAL_HISTORY_KEYS = (
    "train_acc_top1",
    "train_acc_top5",
    "val_acc_top1",
    "val_acc_top5",
)


def _check_checkpoint_path(path, role: str) -> Path:
    path = Path(path)
    if path.suffix not in CHECKPOINT_SUFFIXES:
        raise ValueError(
            f"{role} path must end in .pt or .pth, got {str(path)!r}"
        )
    return path


def _check_npz_path(path, role: str) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"{role} path must end in .npz, got {str(path)!r}")
    return path


def _check_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it "
            f"or choose a different path"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def _torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def save_checkpoint(
    path, net: ChessMoveNet, *, overwrite: bool = False
) -> None:
    """Save architecture metadata and ``state_dict`` to a ``.pt`` file."""
    path = _check_checkpoint_path(path, "checkpoint")
    _check_writable(path, overwrite)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "input_size": int(net.input_size),
        "hidden_sizes": tuple(int(width) for width in net.hidden_sizes),
        "output_size": int(net.output_size),
        "activation": str(net.activation),
        "state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
    }
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu") -> ChessMoveNet:
    """Rebuild a :class:`ChessMoveNet` from a ``.pt`` checkpoint.

    ``map_location`` defaults to CPU so a model trained on NVIDIA GPU can be
    loaded on a machine that only has CPU.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint found at {path}")

    stored = _torch_load(path, map_location=map_location)
    if not isinstance(stored, dict):
        raise ValueError(f"checkpoint {path} is not a ChessMoveNet payload")

    required = (
        "input_size",
        "hidden_sizes",
        "output_size",
        "activation",
        "state_dict",
    )
    missing = [key for key in required if key not in stored]
    if missing:
        raise ValueError(
            f"checkpoint {path} is missing parameter array(s): {missing}"
        )

    state_dict = stored["state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"checkpoint {path} has an empty or invalid state_dict")

    hidden_sizes = tuple(int(width) for width in stored["hidden_sizes"])
    net = ChessMoveNet(
        input_size=int(stored["input_size"]),
        hidden_size=hidden_sizes[0],
        hidden_sizes=hidden_sizes,
        output_size=int(stored["output_size"]),
        seed=0,
        activation=str(stored["activation"]),
    )
    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        message = str(exc)
        if "size mismatch" in message or "Missing key" in message:
            raise ValueError(message) from exc
        raise
    net.to(torch.device(map_location) if isinstance(map_location, str) else map_location)
    net.eval()
    return net


def save_history(path, history: dict, *, overwrite: bool = False) -> None:
    """Save per-epoch training records to ``path`` (a ``.npz`` file)."""
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
    np.savez(path, epoch=epoch, **columns)


def load_history(path) -> dict[str, list]:
    """Load a history saved by :func:`save_history`."""
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
