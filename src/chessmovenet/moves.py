"""Encode chess moves as integer class labels for a neural network.

This module defines a fixed, deterministic mapping between
:class:`chess.Move` objects and integer labels in ``0 .. OUTPUT_SIZE-1``.
A classifier network will have one output neuron per label, so the label
space must be identical for every position — exactly like the fixed
781-feature board layout in :mod:`chessmovenet.encode`.

Trade-offs considered
---------------------

*A collapsed from/to space* (``label = from_square * 64 + to_square``,
4096 classes) is smaller and simpler, but ``e7e8q`` and ``e7e8n`` would
share one label: the network could never learn underpromotion, and
decoding would have to guess the promotion piece.

*A full from x to x promotion-product space* (64 x 64 x 4 = 16384) would
waste ~75% of the output neurons on impossible combinations.

We use the sweet spot: **4096 from/to classes plus 176 explicitly
enumerated promotion classes = 4272 labels**.  Only the physically
possible promotion moves exist as labels (a promoting pawn moves one
forward step or one diagonal capture onto the last rank), so every label
decodes to a distinct, sensible move and promotion choice is fully
preserved.

Label mapping (exact)
---------------------

**Labels 0-4095 — ordinary moves (no promotion):**

``label = from_square * 64 + to_square``

using python-chess square numbers ``a1 = 0 ... h8 = 63``.  This block
naturally covers quiet moves, captures, castling (encoded as the king's
move, e.g. white O-O is ``e1g1``), and en passant captures (the pawn's
from/to squares, e.g. ``e5f6``).  It also contains geometrically
impossible entries such as ``a1a1`` or rook-like jumps; they are inert
placeholders.  Whether a decoded move is *legal* depends on the board
position and is checked there (e.g. via ``board.is_legal``), never here.

**Labels 4096-4271 — promotions (176 classes):**

Every physically possible ``(from_square, to_square, promotion_piece)``
triple, enumerated in a fixed order:

1. promotion piece: queen, rook, bishop, knight;
2. within a piece: from-square ascending (a1=0 ... h8=63);
3. within a from-square: to-square ascending.

Eligible triples: from rank 7 to rank 8 (White) or from rank 2 to rank
1 (Black), with the to-square on the adjacent file or the same file.
Because from-squares run a1 (0) to h8 (63), Black's rank-2 squares
(8-15) are enumerated before White's rank-7 squares (48-55), so the
very first promotion label, 4096, is Black's ``a2a1q``.

``chess.Move`` does **not** validate its promotion field (a king
"promotion" constructs fine), so :func:`move_to_label` rejects any
promotion piece other than queen/rook/bishop/knight, and rejects
promotion moves whose geometry is not a legal promotion shape.
"""

from __future__ import annotations

import chess
import numpy as np

# Number of from-square x to-square combinations (no promotion).
FROM_TO_CLASS_COUNT = 64 * 64  # 4096

# Fixed promotion-piece order used for enumerating promotion classes.
PROMOTION_PIECES: tuple[int, ...] = (
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
)


def _build_promotion_labels() -> tuple[
    list[tuple[int, int, int]], dict[tuple[int, int, int], int]
]:
    """Enumerate every physically possible promotion move once.

    Returns:
        ``(promotion_list, promotion_index)`` where ``promotion_list[i]``
        is the ``(from_square, to_square, promotion_piece)`` triple with
        label ``FROM_TO_CLASS_COUNT + i``, and ``promotion_index`` maps
        each triple back to its label.  Both are built from the same loop,
        so encode and decode can never disagree.
    """
    promotion_list: list[tuple[int, int, int]] = []
    promotion_index: dict[tuple[int, int, int], int] = {}

    for promotion_piece in PROMOTION_PIECES:
        for from_square in chess.SQUARES:  # 0 (a1) ... 63 (h8)
            from_rank = chess.square_rank(from_square)
            # White promotes on rank 7 (0-based 6), Black on rank 2.
            if from_rank == 6:
                to_rank = 7
            elif from_rank == 1:
                to_rank = 0
            else:
                continue  # Not a rank a promotion can start from.

            from_file = chess.square_file(from_square)
            for to_square in chess.SQUARES:
                to_rank_actual = chess.square_rank(to_square)
                to_file = chess.square_file(to_square)
                if to_rank_actual != to_rank:
                    continue
                if abs(to_file - from_file) > 1:
                    continue  # Pawn cannot jump sideways.
                triple = (from_square, to_square, promotion_piece)
                promotion_index[triple] = FROM_TO_CLASS_COUNT + len(
                    promotion_list
                )
                promotion_list.append(triple)

    return promotion_list, promotion_index


def move_to_label(move: chess.Move) -> int:
    """Encode a move as its integer class label.

    Ordinary moves (including captures, castling, and en passant) use
    ``from_square * 64 + to_square``.  Promotion moves are looked up in
    the enumerated promotion block, so the promotion piece is preserved.

    Args:
        move: The move to encode.

    Returns:
        The integer label in ``0 .. OUTPUT_SIZE - 1``.

    Raises:
        TypeError: If ``move`` is not a :class:`chess.Move`.
        ValueError: If ``move.promotion`` is set to something other than
            queen/rook/bishop/knight, or if a promotion move does not
            have promotion geometry (one-step forward or diagonal capture
            onto the last rank).  python-chess itself does not validate
            the promotion field, so this check lives here.
    """
    if not isinstance(move, chess.Move):
        raise TypeError(
            f"expected a chess.Move, got {type(move).__name__}"
        )

    if move.promotion is None:
        return move.from_square * 64 + move.to_square

    # Promotion move: validate piece and geometry, then look up the label.
    if move.promotion not in PROMOTION_PIECES:
        raise ValueError(
            f"invalid promotion piece {move.promotion!r}; "
            "expected queen, rook, bishop, or knight"
        )

    triple = (move.from_square, move.to_square, move.promotion)
    label = PROMOTION_INDEX.get(triple)
    if label is None:
        raise ValueError(
            f"{move.uci()} is not a possible promotion move "
            "(promotions move one step forward or one diagonal capture "
            "onto the last rank)"
        )
    return label


def label_to_move(label: int | np.integer) -> chess.Move:
    """Decode an integer class label back into a move.

    The returned move carries its promotion piece when the label is in
    the promotion block.  **Legality is not checked**: whether the move
    can actually be played depends on the board position, so callers
    should verify with e.g. ``board.is_legal(move)`` or filter against
    the legal moves before using it.

    Args:
        label: An integer in ``0 .. OUTPUT_SIZE - 1``.  NumPy integer
            types (e.g. the output of ``np.argmax``) are accepted.

    Returns:
        The :class:`chess.Move` the label represents.

    Raises:
        TypeError: If ``label`` is not an integer.
        ValueError: If ``label`` is outside ``0 .. OUTPUT_SIZE - 1``.
    """
    if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
        raise TypeError(
            f"expected an integer label, got {type(label).__name__}"
        )
    label = int(label)

    if label < 0 or label >= OUTPUT_SIZE:
        raise ValueError(
            f"label {label} is outside the valid range "
            f"0 .. {OUTPUT_SIZE - 1}"
        )

    if label < FROM_TO_CLASS_COUNT:
        from_square, to_square = divmod(label, 64)
        return chess.Move(from_square, to_square)

    from_square, to_square, promotion_piece = PROMOTION_LIST[
        label - FROM_TO_CLASS_COUNT
    ]
    return chess.Move(from_square, to_square, promotion=promotion_piece)


# The full label space: 4096 ordinary moves + 176 promotion classes.
PROMOTION_LIST, PROMOTION_INDEX = _build_promotion_labels()
OUTPUT_SIZE = FROM_TO_CLASS_COUNT + len(PROMOTION_LIST)  # 4272
