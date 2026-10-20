"""Evaluate a saved ChessMoveNet checkpoint once on the held-out test split."""

from __future__ import annotations

import argparse

import numpy as np

from chessmovenet.checkpoint import load_checkpoint
from chessmovenet.train import evaluate_accuracy, evaluate_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/checkpoint_ep15.npz")
    parser.add_argument("--data", default="data/processed/split.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    net = load_checkpoint(args.checkpoint)
    with np.load(args.data, allow_pickle=False) as data:
        X_test = data["X_test"]
        y_test = data["y_test"]

    loss = evaluate_loss(net, X_test, y_test, batch_size=args.batch_size)
    accuracy = evaluate_accuracy(net, X_test, y_test, batch_size=args.batch_size)
    print(f"checkpoint: {args.checkpoint}")
    print(f"test examples: {X_test.shape[0]}")
    print(f"test loss: {loss:.6f}")
    print(f"test top-1 accuracy: {accuracy['top1']:.4f}%")
    print(f"test top-5 accuracy: {accuracy['top5']:.4f}%")


if __name__ == "__main__":
    main()
