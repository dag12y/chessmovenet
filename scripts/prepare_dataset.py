"""Build the train/validation/test dataset from one Chess.com PGN file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chessmovenet.split import split_dataset_chronologically


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", required=True, help="Source PGN file")
    parser.add_argument("--username", required=True, help="Chess.com username")
    parser.add_argument(
        "--output", default="data/processed/split.npz", help="Output .npz file"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file"
    )
    args = parser.parse_args()

    source = Path(args.pgn)
    output = Path(args.output)
    if not source.is_file():
        parser.error(f"PGN file not found: {source}")
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output} (use --overwrite to replace it)")

    split = split_dataset_chronologically(source, args.username)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        X_test=split.X_test,
        y_test=split.y_test,
    )
    print(f"saved {output}")
    print(f"train: {split.X_train.shape}, validation: {split.X_val.shape}, test: {split.X_test.shape}")


if __name__ == "__main__":
    main()
