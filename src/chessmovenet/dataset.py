"""Build aligned feature and move-label arrays from a player's PGN games."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from chessmovenet.data import extract_examples, my_color, read_games
from chessmovenet.encode import BOARD_FEATURE_SIZE, encode_board
from chessmovenet.moves import move_to_label


def build_dataset(
    pgn_path: str | Path, username: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``X`` features and ``y`` move labels for ``username``.

    Each row encodes the position before one of the player's moves. Games
    without the player are skipped; no examples returns correctly shaped
    empty arrays.

    Args:
        pgn_path: Path to the PGN file to read (read-only; never modified).
        username: The player whose moves should become training examples.
            Matching is case-insensitive, exactly as in :mod:`data`.

    Returns:
        ``(X, y)`` with shapes ``(n, 781)`` and ``(n,)`` and dtypes
        ``float32`` and ``int64`` respectively.
    """
    # Collect rows before stacking them into one array.
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for game in read_games(pgn_path):
        if my_color(game, username) is None:
            continue

        for fen, uci in extract_examples(game, username):
            # FEN stores the position before the move.
            board = chess.Board(fen)
            feature_rows.append(encode_board(board))  # (781,) float32

            labels.append(move_to_label(chess.Move.from_uci(uci)))

    # Preserve the standard shapes and dtypes for empty datasets.
    if not feature_rows:
        empty_x = np.zeros((0, BOARD_FEATURE_SIZE), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        return empty_x, empty_y

    X = np.stack(feature_rows)
    y = np.asarray(labels, dtype=np.int64)

    return X, y
