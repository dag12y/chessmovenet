"""Plot ChessMoveNet training history from a ``.npz`` history file.

Generates two PNG graphs:

- reports/loss_curves.png     -- training & validation loss vs. epoch
- reports/accuracy_curves.png -- train/val top-1 & top-5 accuracy vs. epoch

Usage::

    PYTHONPATH=src python scripts/plot_history.py
    PYTHONPATH=src python scripts/plot_history.py artifacts/history_ep15.npz
    PYTHONPATH=src python scripts/plot_history.py /some/other/history.npz

Default history file: artifacts/history_ep15.npz.
Loaded read-only with allow_pickle=False; never modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_HISTORY = "artifacts/history_ep15.npz"
REPORTS_DIR = Path("reports")
LOSS_PLOT = REPORTS_DIR / "loss_curves.png"
ACC_PLOT = REPORTS_DIR / "accuracy_curves.png"

REQUIRED_KEYS = ("epoch", "train_loss", "val_loss")

ACC_KEYS = (
    ("train_acc_top1", "Train top-1", "#1f77b4"),
    ("val_acc_top1", "Val top-1", "#ff7f0e"),
    ("train_acc_top5", "Train top-5", "#2ca02c"),
    ("val_acc_top5", "Val top-5", "#d62728"),
)


def load_history(path: Path) -> dict:
    """Load a history ``.npz`` file with ``allow_pickle=False``.

    Returns a dict mapping key name -> 1-D numpy array.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if any required key is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"History file not found: {path}")

    with np.load(path, allow_pickle=False) as stored:
        missing = [k for k in REQUIRED_KEYS if k not in stored.files]
        if missing:
            raise ValueError(
                f"History file {path} is missing required key(s): {missing}"
            )
        data = {k: np.asarray(stored[k]) for k in stored.files}

    return data


def _validate_1d(history: dict) -> None:
    """Ensure every column in *history* is 1-D and all have the same length."""
    for key, values in history.items():
        if values.ndim != 1:
            raise ValueError(
                f"history[{key!r}] must be 1-D, got shape {values.shape}"
            )
    lengths = {v.shape[0] for v in history.values()}
    if len(lengths) != 1:
        raise ValueError(
            "History columns have inconsistent lengths: "
            f"{ {k: v.shape[0] for k, v in history.items()} }"
        )


def plot_loss(history: dict, output_path: Path) -> None:
    """Plot training and validation loss curves and save to *output_path*."""
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, "o-", label="Train loss", color="#1f77b4")
    plt.plot(epochs, val_loss, "o-", label="Validation loss", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training & Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy(history: dict, output_path: Path) -> None:
    """Plot available accuracy curves and save to *output_path*."""
    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))

    plotted = False
    for key, label, color in ACC_KEYS:
        if key in history:
            plt.plot(epochs, history[key], "o-", label=label, color=color)
            plotted = True

    if not plotted:
        available = [k for k, _, _ in ACC_KEYS if k in history]
        raise ValueError(
            "History file contains no accuracy columns "
            f"(checked {list(ACC_KEYS)}); found only: {available}. "
            "Cannot plot accuracy curves."
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Top-1 / Top-5 Accuracy Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ChessMoveNet training history from a .npz file."
    )
    parser.add_argument(
        "history_file",
        nargs="?",
        default=DEFAULT_HISTORY,
        help=f"Path to history .npz file (default: {DEFAULT_HISTORY})",
    )
    args = parser.parse_args()

    history_path = Path(args.history_file)

    # --- Load --------------------------------------------------------------
    try:
        history = load_history(history_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    _validate_1d(history)

    # --- Output directory --------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for plot_path in (LOSS_PLOT, ACC_PLOT):
        if plot_path.exists():
            print(f"Warning: {plot_path} already exists and will be overwritten.")

    # --- Loss curves -------------------------------------------------------
    plot_loss(history, LOSS_PLOT)
    print(f"Saved {LOSS_PLOT}")

    # --- Accuracy curves ---------------------------------------------------
    acc_keys_present = [k for k, _, _ in ACC_KEYS if k in history]
    if acc_keys_present:
        plot_accuracy(history, ACC_PLOT)
        print(f"Saved {ACC_PLOT}")
    else:
        print(
            "Note: no accuracy columns found in history file; "
            "skipping accuracy plot.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
