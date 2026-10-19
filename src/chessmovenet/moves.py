"""Move encoding for ChessMoveNet.

Turns UCI move strings into fixed class indices (0..4095) and builds a legal
move mask per position. Only python-chess and NumPy are used.
"""

from __future__ import annotations

import chess
import numpy as np

OUTPUT_SIZE = 4096
"""Number of output classes: 64 from-squares x 64 to-squares."""


def move_to_index(uci: str) -> int:
    """Convert a UCI move string to a class index.

    Index = from_square * 64 + to_square (python-chess square numbering,
    a1 = 0 .. h8 = 63). Any promotion suffix (e.g. the "q" in "e7e8q") is
    ignored, so "e7e8q" and "e7e8" map to the same class.
    """
    move = chess.Move.from_uci(uci)
    return move.from_square * 64 + move.to_square


def index_to_move(index: int) -> str:
    """Convert a class index back to a "from-to" UCI string (no promotion).

    The inverse of :func:`move_to_index` for non-promotion moves.
    """
    from_square = index // 64
    to_square = index % 64
    return chess.square_name(from_square) + chess.square_name(to_square)


def legal_mask(fen: str) -> np.ndarray:
    """Return a float32 array of shape (OUTPUT_SIZE,) with 1.0 for each legal
    from-to move in the position described by ``fen``, else 0.0."""
    board = chess.Board(fen)
    mask = np.zeros(OUTPUT_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        mask[move.from_square * 64 + move.to_square] = 1.0
    return mask


def encode_moves(ucis) -> np.ndarray:
    """Encode a list of UCI move strings into an int array of class indices."""
    indices = np.zeros(len(ucis), dtype=np.int64)
    for i, uci in enumerate(ucis):
        indices[i] = move_to_index(uci)
    return indices
