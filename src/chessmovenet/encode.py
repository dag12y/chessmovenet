"""Encode a chess board as a fixed-length numerical feature vector.

This module turns a :class:`chess.Board` into a 1-D NumPy array of
``float32`` values with **781 features** in a fixed, deterministic layout,
so that positions can be fed to a neural network later:

======  ========  ===================================================
Index   Count     Meaning
======  ========  ===================================================
0–767   768       Piece placement: 12 binary planes x 64 squares
768     1         Side to move (1 = White, 0 = Black)
769–772 4         Castling rights (see order below)
773–780 8         En-passant target file (one feature per file a–h)
======  ========  ===================================================

Total: 768 + 1 + 4 + 8 = **781** features.

Piece planes (indices 0–767)
----------------------------
Twelve 64-square binary planes, ordered first by color then by piece type
(``chess.PAWN=1`` … ``chess.KING=6``):

=====  ===========  ===============
Plane  Color        Piece type
=====  ===========  ===============
0      White        Pawn
1      White        Knight
2      White        Bishop
3      White        Rook
4      White        Queen
5      White        King
6      Black        Pawn
7      Black        Knight
8      Black        Bishop
9      Black        Rook
10     Black        Queen
11     Black        King
=====  ===========  ===============

The feature index of a piece is ``plane * 64 + square``.

Square numbering within each plane
----------------------------------
We use python-chess's native square numbering, which already matches the
required order: ``a1 = 0``, ``b1 = 1``, … ``h1 = 7``, ``a2 = 8``, …
``h8 = 63``.  (``chess.square(file, rank) == rank * 8 + file``, with files
and ranks 0-based.)  A square is simply used as its own plane offset.

Side to move (index 768)
------------------------
``1.0`` when it is White's turn to move (``board.turn == chess.WHITE``),
``0.0`` when it is Black's.

Castling rights (indices 769–772)
---------------------------------
One binary feature each, in this fixed order:

1. index 769 — White kingside
2. index 770 — White queenside
3. index 771 — Black kingside
4. index 772 — Black queenside

They use python-chess's ``has_*_castling_rights(color)`` helpers, which
check both the castling-rights mask *and* that the king and rook are still
on their home squares — so a stale right in a hand-written FEN cannot
produce a phantom 1.

En-passant target file (indices 773–780)
----------------------------------------
One binary feature per file: index ``773 + file``, where file ``a = 0``
… ``h = 7``.  It is set to ``1.0`` for the file of the en-passant target
square, and all eight features are ``0.0`` when there is no target.

Why only the *file*?  In any valid chess position the en-passant target
square is always on rank 6 when White is to move, and on rank 3 when
Black is.  The side-to-move feature (index 768) already tells us which
case we are in, so (file + side to move) uniquely identifies the full
target square — encoding the rank as well would be redundant.

Which en-passant convention?  We read ``board.ep_square``, python-chess's
*raw* attribute: the target square of the last double pawn push as stored
in the position/FEN, **even when no capture is actually possible**
(pseudo-legal availability).  For example, after ``1. e4`` the target
``e3`` is recorded although no black pawn can capture there.  This
convention is deterministic, mirrors the FEN standard, and never hides
information (a model can learn to ignore a target no pawn can reach).
python-chess's stricter ``board.has_legal_en_passant()`` — True only when
a fully legal capture exists — was deliberately *not* used, because it
would make the encoding depend on move-legality analysis and silently
drop positions that still "look like" en-passant positions.

Known limitations
-----------------
- **Promotion is not encoded.**  A promoted queen on d5 and a queen that
  started there encode identically; the board vector only describes
  piece *placement*.  Which piece a pawn promotes to is part of the move
  *target* and will be handled by the later move-encoding feature.
- The halfmove clock and fullmove number are not encoded.
- Only the en-passant *file* is encoded (see above — the rank is
  recoverable, so this is not a loss in valid positions).
- Castling features describe standard chess; Chess960 castling-to-file
  nuances are out of scope for this project.
"""

from __future__ import annotations

import chess
import numpy as np

# --- Fixed layout sizes and offsets -----------------------------------------

SQUARES_PER_PLANE = 64
PIECE_PLANES = 12
PIECE_FEATURE_SIZE = PIECE_PLANES * SQUARES_PER_PLANE  # 768

SIDE_TO_MOVE_INDEX = PIECE_FEATURE_SIZE  # 768
WHITE_KINGSIDE_INDEX = SIDE_TO_MOVE_INDEX + 1  # 769
WHITE_QUEENSIDE_INDEX = SIDE_TO_MOVE_INDEX + 2  # 770
BLACK_KINGSIDE_INDEX = SIDE_TO_MOVE_INDEX + 3  # 771
BLACK_QUEENSIDE_INDEX = SIDE_TO_MOVE_INDEX + 4  # 772
EN_PASSANT_FILE_OFFSET = SIDE_TO_MOVE_INDEX + 5  # 773
EN_PASSANT_FILE_COUNT = 8

BOARD_FEATURE_SIZE = (
    PIECE_FEATURE_SIZE + 1 + 4 + EN_PASSANT_FILE_COUNT
)  # 781


def _build_piece_plane_index() -> dict[tuple[bool, int], int]:
    """Map ``(color, piece_type)`` to its plane index 0–11.

    White's six pieces occupy planes 0–5 (pawn … king), Black's occupy
    planes 6–11, in the same piece-type order.
    """
    mapping: dict[tuple[bool, int], int] = {}
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in range(chess.PAWN, chess.KING + 1):
            color_offset = 0 if color == chess.WHITE else 6
            mapping[(color, piece_type)] = color_offset + (
                piece_type - chess.PAWN
            )
    return mapping


# Public constant documenting the exact plane layout (also handy in tests).
PIECE_PLANE_INDEX: dict[tuple[bool, int], int] = _build_piece_plane_index()


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode ``board`` as a 781-element ``float32`` feature vector.

    The layout is fully documented in the module docstring: 768 piece-plane
    features (12 planes x 64 squares, ``plane * 64 + square``), then side to
    move, then four castling rights, then eight en-passant file features.

    Args:
        board: The position to encode.  The board is only read, never
            modified.

    Returns:
        A 1-D array of shape ``(781,)`` with dtype ``float32``.  Every
        value is ``0.0`` or ``1.0``; the layout is identical for every
        board, so element ``i`` always means the same thing.
    """
    features = np.zeros(BOARD_FEATURE_SIZE, dtype=np.float32)

    # --- 1. Piece placement: one 1.0 per piece, in its (color, type) plane ---
    # board.piece_map() returns {square: Piece}, e.g. {4: Piece('P', WHITE)}.
    # python-chess square numbers are already a1=0 ... h8=63, which is the
    # within-plane index we want.
    for square, piece in board.piece_map().items():
        plane = PIECE_PLANE_INDEX[(piece.color, piece.piece_type)]
        features[plane * SQUARES_PER_PLANE + square] = 1.0

    # --- 2. Side to move: 1.0 for White, 0.0 for Black ---
    features[SIDE_TO_MOVE_INDEX] = 1.0 if board.turn == chess.WHITE else 0.0

    # --- 3. Castling rights, in the documented order ---
    # The helpers verify the rights mask *and* that king/rook are home.
    features[WHITE_KINGSIDE_INDEX] = float(
        board.has_kingside_castling_rights(chess.WHITE)
    )
    features[WHITE_QUEENSIDE_INDEX] = float(
        board.has_queenside_castling_rights(chess.WHITE)
    )
    features[BLACK_KINGSIDE_INDEX] = float(
        board.has_kingside_castling_rights(chess.BLACK)
    )
    features[BLACK_QUEENSIDE_INDEX] = float(
        board.has_queenside_castling_rights(chess.BLACK)
    )

    # --- 4. En-passant target FILE (a=0 ... h=7); all zeros if none ---
    # We use the raw board.ep_square (set after every double push), not the
    # stricter has_legal_en_passant() — see the module docstring.
    if board.ep_square is not None:
        file_index = chess.square_file(board.ep_square)
        features[EN_PASSANT_FILE_OFFSET + file_index] = 1.0

    return features

