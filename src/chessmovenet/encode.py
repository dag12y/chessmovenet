"""FEN encoding for ChessMoveNet.

Turns a FEN string into a fixed-length float32 vector that a NumPy neural
network can consume. Only python-chess and NumPy are used.
"""

from __future__ import annotations

import chess
import numpy as np

INPUT_SIZE = 773
"""Length of the encoded vector: 12 planes x 64 squares + turn + 4 castling flags."""


def fen_to_vector(fen):
    """Encode a FEN string as a float32 array of shape (INPUT_SIZE,).

    Layout:
      * indices 0..767:  12 planes of 64 squares.
            White piece of type t -> plane t - 1 (0..5).
            Black piece of type t -> plane 6 + (t - 1) (6..11).
            Index = plane * 64 + square (a1 = 0 .. h8 = 63).
      * index 768:       1.0 if it is White's turn, else 0.0.
      * indices 769..772: castling rights -- White kingside, White queenside,
            Black kingside, Black queenside (1.0 if available).
    """
    board = chess.Board(fen)
    vector = np.zeros(INPUT_SIZE, dtype=np.float32)

    # Piece planes: one-hot per (piece, color, square).
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type  # 1 = pawn .. 6 = king
        if piece.color == chess.WHITE:
            plane = piece_type - 1  # white planes 0..5
        else:
            plane = 6 + (piece_type - 1)  # black planes 6..11
        index = plane * 64 + square
        vector[index] = 1.0

    # Side to move.
    vector[768] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights: White kingside, White queenside, Black kingside,
    # Black queenside.
    vector[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    vector[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    vector[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    vector[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    return vector


def encode_fens(fens):
    """Encode a list of FEN strings into a float32 array of shape (N, INPUT_SIZE).

    A simple loop over fen_to_vector is used so the code stays easy to read.
    """
    encoded = np.zeros((len(fens), INPUT_SIZE), dtype=np.float32)
    for i, fen in enumerate(fens):
        encoded[i] = fen_to_vector(fen)
    return encoded
