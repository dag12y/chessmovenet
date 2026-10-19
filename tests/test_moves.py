"""Tests for :mod:`chessmovenet.moves` (move <-> integer class label).

Every position is hand-constructed (short move sequences or FEN strings).
No test reads personal game data.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessmovenet.moves import (
    FROM_TO_CLASS_COUNT,
    OUTPUT_SIZE,
    PROMOTION_LIST,
    label_to_move,
    move_to_label,
)

# The full label space: 4096 from/to classes + 176 promotion classes.
EXPECTED_OUTPUT_SIZE = 4272
EXPECTED_PROMOTION_CLASSES = 176


# ---------------------------------------------------------------------------
# Constants and layout
# ---------------------------------------------------------------------------


def test_output_size_is_4272() -> None:
    """4096 from/to classes plus exactly 176 promotion classes."""
    assert FROM_TO_CLASS_COUNT == 4096
    assert len(PROMOTION_LIST) == EXPECTED_PROMOTION_CLASSES
    assert OUTPUT_SIZE == EXPECTED_OUTPUT_SIZE


def test_first_promotion_label_is_a2a1_queen() -> None:
    """Label 4096 is the first enumerated promotion: a2a1q.

    Black's rank-2 squares (8-15) sort before White's rank-7 squares
    (48-55), so a Black promotion comes first in the enumeration.
    """
    move = chess.Move.from_uci("a2a1q")

    assert move_to_label(move) == FROM_TO_CLASS_COUNT


# ---------------------------------------------------------------------------
# Ordinary moves, captures, castling, en passant
# ---------------------------------------------------------------------------


def test_ordinary_move_uses_from_times_64_plus_to() -> None:
    """1. e2e4 encodes as E2 * 64 + E4."""
    move = chess.Move.from_uci("e2e4")

    assert move_to_label(move) == chess.E2 * 64 + chess.E4  # 796
    assert label_to_move(796) == move


def test_capture_is_just_from_to() -> None:
    """A capture (exd5) uses the same from/to formula; no special case."""
    move = chess.Move.from_uci("e4d5")  # White pawn e4 captures d5.

    assert move_to_label(move) == chess.E4 * 64 + chess.D5  # 1827
    assert label_to_move(1827) == move


def test_castling_is_encoded_as_the_kings_move() -> None:
    """White O-O is the king move e1g1; black O-O is e8g8."""
    white_castle = chess.Move.from_uci("e1g1")
    black_castle = chess.Move.from_uci("e8g8")

    assert move_to_label(white_castle) == chess.E1 * 64 + chess.G1  # 262
    assert move_to_label(black_castle) == chess.E8 * 64 + chess.G8
    assert label_to_move(262) == white_castle


def test_en_passant_capture_is_the_pawns_from_to() -> None:
    """An en-passant capture (e.g. e5xf6 e.p.) is the pawn's from/to."""
    move = chess.Move.from_uci("e5f6")

    assert move_to_label(move) == chess.E5 * 64 + chess.F6  # 2349
    assert label_to_move(2349) == move


# ---------------------------------------------------------------------------
# Promotions
# ---------------------------------------------------------------------------


def test_four_promotion_choices_have_distinct_labels() -> None:
    """e7e8 promotions to Q/R/B/N map to four different labels."""
    labels = [
        move_to_label(chess.Move.from_uci(f"e7e8{piece}"))
        for piece in "qrbn"
    ]

    assert len(set(labels)) == 4
    assert all(FROM_TO_CLASS_COUNT <= label < OUTPUT_SIZE for label in labels)


def test_promotion_label_round_trips_with_correct_piece() -> None:
    """Decoding each promotion label restores the exact promotion piece."""
    for uci in ("e7e8q", "e7e8r", "e7e8b", "e7e8n"):
        move = chess.Move.from_uci(uci)

        decoded = label_to_move(move_to_label(move))

        assert decoded == move
        assert decoded.promotion == move.promotion
        assert decoded.uci() == uci


def test_black_promotion_is_encoded() -> None:
    """Black underpromotion (a2a1n) round-trips with the knight kept."""
    move = chess.Move.from_uci("a2a1n")

    decoded = label_to_move(move_to_label(move))

    assert decoded == move
    assert decoded.promotion == chess.KNIGHT


def test_capture_promotion_is_encoded() -> None:
    """A capture-promotion (g7xf8q) round-trips; from-square plus piece
    alone would be ambiguous here, so to-square is part of the label."""
    move = chess.Move.from_uci("g7h8q")

    decoded = label_to_move(move_to_label(move))

    assert decoded == move
    assert decoded.uci() == "g7h8q"


# ---------------------------------------------------------------------------
# Encode/decode consistency
# ---------------------------------------------------------------------------


def test_every_label_round_trips_through_decode_encode() -> None:
    """label -> move -> label is the identity for the whole label space."""
    for label in range(OUTPUT_SIZE):
        assert move_to_label(label_to_move(label)) == label


def test_decoded_moves_are_all_distinct() -> None:
    """The label space is a true bijection: no two labels share a move."""
    decoded = [label_to_move(label) for label in range(OUTPUT_SIZE)]

    assert len(set(decoded)) == OUTPUT_SIZE


def test_sample_of_real_game_moves_round_trips() -> None:
    """Moves from a short legal game all round-trip exactly."""
    board = chess.Board()
    for uci in ("g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"):
        move = chess.Move.from_uci(uci)
        assert label_to_move(move_to_label(move)) == move


def test_decode_accepts_numpy_integers() -> None:
    """Labels coming from NumPy (e.g. np.argmax) decode fine."""
    expected = chess.Move.from_uci("e2e4")
    label = np.int64(move_to_label(expected))

    assert label_to_move(label) == expected


# ---------------------------------------------------------------------------
# Legality is position-dependent, not encoded
# ---------------------------------------------------------------------------


def test_decoded_move_is_not_assumed_legal() -> None:
    """Decoding works for any label; legality needs a board.

    The move e2e5 (a pawn jumping two squares from e2 to e5) decodes
    cleanly, but no board position makes it legal — the label space
    contains geometrically impossible moves on purpose, and legality is
    the caller's job to check.
    """
    move = label_to_move(chess.E2 * 64 + chess.E5)

    assert move == chess.Move.from_uci("e2e5")
    assert not chess.Board().is_legal(move)


def test_promotion_label_decodes_to_legal_move_on_right_board() -> None:
    """The same promotion label is legal on a board where it belongs."""
    move = label_to_move(move_to_label(chess.Move.from_uci("e7e8q")))

    # Black king off the e-file so the e7 pawn can really promote.
    board = chess.Board("6k1/4P3/8/8/8/8/8/4K3 w - - 0 1")

    assert board.is_legal(move)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_negative_label_raises_value_error() -> None:
    with pytest.raises(ValueError):
        label_to_move(-1)


def test_label_beyond_output_size_raises_value_error() -> None:
    with pytest.raises(ValueError):
        label_to_move(OUTPUT_SIZE)


def test_huge_label_raises_value_error() -> None:
    with pytest.raises(ValueError):
        label_to_move(999_999)


@pytest.mark.parametrize("bad_label", ["5", None, 5.0, True, [5]])
def test_non_integer_label_raises_type_error(bad_label) -> None:
    with pytest.raises(TypeError):
        label_to_move(bad_label)


def test_non_move_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        move_to_label("e2e4")  # type: ignore[arg-type]


def test_promotion_with_king_piece_raises_value_error() -> None:
    """python-chess accepts promotion=KING, so we must reject it here."""
    bogus = chess.Move(chess.E7, chess.E8, promotion=chess.KING)

    with pytest.raises(ValueError):
        move_to_label(bogus)


def test_promotion_from_impossible_square_raises_value_error() -> None:
    """A 'promotion' starting on rank 4 has no possible label."""
    bogus = chess.Move(chess.E4, chess.E5, promotion=chess.QUEEN)

    with pytest.raises(ValueError):
        move_to_label(bogus)


def test_promotion_that_jumps_two_ranks_raises_value_error() -> None:
    """e7f9-style geometry (off board) or e6e8 style must be rejected."""
    bogus = chess.Move(chess.E6, chess.E8, promotion=chess.QUEEN)

    with pytest.raises(ValueError):
        move_to_label(bogus)

    """A capture-promotion (g7xf8q) round-trips; from-square plus piece
    alone would be ambiguous here, so to-square is part of the label."""
    move = chess.Move.from_uci("g7h8q")

    decoded = label_to_move(move_to_label(move))

    assert decoded == move
    assert decoded.uci() == "g7h8q"
