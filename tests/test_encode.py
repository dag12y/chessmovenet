"""Tests for :mod:`chessmovenet.encode` (board -> feature vector).

All positions are hand-constructed ``chess.Board`` objects (from FEN
strings or short move sequences).  No test reads personal game data.
"""

from __future__ import annotations

import chess
import numpy as np

from chessmovenet.encode import (
    BLACK_KINGSIDE_INDEX,
    BLACK_QUEENSIDE_INDEX,
    BOARD_FEATURE_SIZE,
    EN_PASSANT_FILE_COUNT,
    EN_PASSANT_FILE_OFFSET,
    PIECE_PLANE_INDEX,
    SIDE_TO_MOVE_INDEX,
    SQUARES_PER_PLANE,
    WHITE_KINGSIDE_INDEX,
    WHITE_QUEENSIDE_INDEX,
    encode_board,
)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Total feature count: 768 piece planes + 1 side to move + 4 castling
# + 8 en-passant files.
EXPECTED_FEATURE_COUNT = 781


def piece_feature_index(color: bool, piece_type: int, square: int) -> int:
    """Return the global feature index for a piece on a square.

    Mirrors the documented layout: ``plane * 64 + square``.
    """
    plane = PIECE_PLANE_INDEX[(color, piece_type)]
    return plane * SQUARES_PER_PLANE + square


# ---------------------------------------------------------------------------
# Shape, dtype, and value sanity
# ---------------------------------------------------------------------------


def test_output_shape_is_exactly_781() -> None:
    """The vector has the documented length of 781."""
    board = chess.Board(STARTING_FEN)

    features = encode_board(board)

    assert features.shape == (EXPECTED_FEATURE_COUNT,)
    assert BOARD_FEATURE_SIZE == EXPECTED_FEATURE_COUNT


def test_output_dtype_is_float32() -> None:
    """The dtype is a clear float type suitable for neural-network input."""
    features = encode_board(chess.Board(STARTING_FEN))

    assert features.dtype == np.float32


def test_values_are_numeric_binary_and_finite() -> None:
    """Every value is a finite number, here always exactly 0.0 or 1.0."""
    features = encode_board(chess.Board(STARTING_FEN))

    assert np.issubdtype(features.dtype, np.number)
    assert np.isfinite(features).all()
    assert np.isin(features, [0.0, 1.0]).all()


# ---------------------------------------------------------------------------
# Starting position
# ---------------------------------------------------------------------------


def test_starting_position_piece_planes() -> None:
    """All 32 pieces sit on their expected squares of the right planes."""
    features = encode_board(chess.Board(STARTING_FEN))
    piece_area = features[: 12 * SQUARES_PER_PLANE]

    # Exactly 32 pieces -> exactly 32 ones in the piece planes.
    assert piece_area.sum() == 32

    # Spot-check one piece of each color on both wings.
    assert features[piece_feature_index(chess.WHITE, chess.PAWN, chess.E2)] == 1.0
    assert features[piece_feature_index(chess.WHITE, chess.KING, chess.E1)] == 1.0
    assert features[piece_feature_index(chess.WHITE, chess.ROOK, chess.A1)] == 1.0
    assert features[piece_feature_index(chess.BLACK, chess.PAWN, chess.E7)] == 1.0
    assert features[piece_feature_index(chess.BLACK, chess.KING, chess.E8)] == 1.0
    assert features[piece_feature_index(chess.BLACK, chess.QUEEN, chess.D8)] == 1.0

    # Squares that start empty carry no feature.
    assert features[piece_feature_index(chess.WHITE, chess.PAWN, chess.E4)] == 0.0
    assert features[piece_feature_index(chess.BLACK, chess.KNIGHT, chess.C3)] == 0.0


def test_starting_position_side_to_move_is_white() -> None:
    """White moves first, so the side-to-move feature is 1.0."""
    features = encode_board(chess.Board(STARTING_FEN))

    assert features[SIDE_TO_MOVE_INDEX] == 1.0


def test_starting_position_has_all_castling_rights() -> None:
    """All four castling features start at 1.0."""
    features = encode_board(chess.Board(STARTING_FEN))

    assert features[WHITE_KINGSIDE_INDEX] == 1.0
    assert features[WHITE_QUEENSIDE_INDEX] == 1.0
    assert features[BLACK_KINGSIDE_INDEX] == 1.0
    assert features[BLACK_QUEENSIDE_INDEX] == 1.0


def test_starting_position_has_no_en_passant_target() -> None:
    """No double pawn push has happened, so all eight features are 0.0."""
    features = encode_board(chess.Board(STARTING_FEN))

    en_passant_area = features[
        EN_PASSANT_FILE_OFFSET : EN_PASSANT_FILE_OFFSET + EN_PASSANT_FILE_COUNT
    ]
    assert (en_passant_area == 0.0).all()



# ---------------------------------------------------------------------------
# Individual feature sensitivity
# ---------------------------------------------------------------------------


def test_known_piece_on_known_square_activates_correct_feature() -> None:
    """A lone queen on d4 lights up exactly plane 4, square d4."""
    board = chess.Board("8/8/8/8/3Q4/8/8/8 w - - 0 1")

    features = encode_board(board)

    expected_index = piece_feature_index(chess.WHITE, chess.QUEEN, chess.D4)
    ones = np.flatnonzero(features)
    # The only ones are the queen and the side-to-move feature (White).
    assert list(ones) == [expected_index, SIDE_TO_MOVE_INDEX]
    assert features[expected_index] == 1.0


def test_side_to_move_changes_only_its_feature() -> None:
    """Flipping the turn changes exactly one element: index 768."""
    white_to_move = encode_board(chess.Board(STARTING_FEN))
    black_to_move = encode_board(
        chess.Board(STARTING_FEN.replace(" w ", " b "))
    )

    differing = np.flatnonzero(white_to_move != black_to_move)

    assert list(differing) == [SIDE_TO_MOVE_INDEX]
    assert white_to_move[SIDE_TO_MOVE_INDEX] == 1.0
    assert black_to_move[SIDE_TO_MOVE_INDEX] == 0.0


def test_removing_white_kingside_right_changes_only_its_feature() -> None:
    """Dropping the K right flips exactly index 769."""
    full = chess.Board(STARTING_FEN)
    no_white_kingside = chess.Board(
        STARTING_FEN.replace(" w KQkq ", " w Qkq ")
    )

    differing = np.flatnonzero(
        encode_board(full) != encode_board(no_white_kingside)
    )

    assert list(differing) == [WHITE_KINGSIDE_INDEX]
    assert encode_board(no_white_kingside)[WHITE_KINGSIDE_INDEX] == 0.0


def test_removing_white_queenside_right_changes_only_its_feature() -> None:
    """Dropping the Q right flips exactly index 770."""
    full = chess.Board(STARTING_FEN)
    no_white_queenside = chess.Board(
        STARTING_FEN.replace(" w KQkq ", " w Kkq ")
    )

    differing = np.flatnonzero(
        encode_board(full) != encode_board(no_white_queenside)
    )

    assert list(differing) == [WHITE_QUEENSIDE_INDEX]


def test_removing_black_kingside_right_changes_only_its_feature() -> None:
    """Dropping the k right flips exactly index 771."""
    full = chess.Board(STARTING_FEN)
    no_black_kingside = chess.Board(
        STARTING_FEN.replace(" w KQkq ", " w KQq ")
    )

    differing = np.flatnonzero(
        encode_board(full) != encode_board(no_black_kingside)
    )

    assert list(differing) == [BLACK_KINGSIDE_INDEX]


def test_removing_black_queenside_right_changes_only_its_feature() -> None:
    """Dropping the q right flips exactly index 772."""
    full = chess.Board(STARTING_FEN)
    no_black_queenside = chess.Board(
        STARTING_FEN.replace(" w KQkq ", " w KQk ")
    )

    differing = np.flatnonzero(
        encode_board(full) != encode_board(no_black_queenside)
    )

    assert list(differing) == [BLACK_QUEENSIDE_INDEX]


# ---------------------------------------------------------------------------
# En passant
# ---------------------------------------------------------------------------


def test_en_passant_target_sets_correct_file_feature() -> None:
    """After 1. e4 d5 2. e5 f5 the en-passant target is f6: file f = 5."""
    board = chess.Board()
    for move in ("e2e4", "d7d5", "e4e5", "f7f5"):
        board.push_uci(move)
    # Sanity: the position really has an en-passant target on f6.
    assert board.ep_square == chess.F6

    features = encode_board(board)

    # file f is the 6th file (a=0 ... f=5).
    file_f_index = EN_PASSANT_FILE_OFFSET + 5
    en_passant_area = features[
        EN_PASSANT_FILE_OFFSET : EN_PASSANT_FILE_OFFSET + EN_PASSANT_FILE_COUNT
    ]
    assert features[file_f_index] == 1.0
    assert en_passant_area.sum() == 1.0


def test_en_passant_recorded_even_when_capture_impossible() -> None:
    """Documented convention: the raw ep_square is used, so after 1. e4
    (target e3, but no black pawn can capture) the e-file feature is set."""
    board = chess.Board()
    board.push_uci("e2e4")
    # The raw target square exists although no capture is possible.
    assert board.ep_square == chess.E3
    assert not board.has_legal_en_passant()

    features = encode_board(board)

    file_e_index = EN_PASSANT_FILE_OFFSET + 4  # file e = index 4
    assert features[file_e_index] == 1.0


def test_board_without_en_passant_has_all_eight_features_zero() -> None:
    """After 1. e3 (single step, no double push) no en-passant target."""
    board = chess.Board()
    board.push_uci("e2e3")
    assert board.ep_square is None

    features = encode_board(board)

    en_passant_area = features[
        EN_PASSANT_FILE_OFFSET : EN_PASSANT_FILE_OFFSET + EN_PASSANT_FILE_COUNT
    ]
    assert (en_passant_area == 0.0).all()


# ---------------------------------------------------------------------------
# Sparse boards
# ---------------------------------------------------------------------------


def test_encoder_works_with_only_a_few_pieces() -> None:
    """A bare king-vs-king endgame encodes cleanly: only kings and rights."""
    board = chess.Board("8/8/8/8/8/4k3/8/4K3 w - - 0 1")

    features = encode_board(board)

    assert features.shape == (EXPECTED_FEATURE_COUNT,)
    ones = np.flatnonzero(features)
    # Only three features are on: the two kings and side to move (White).
    # The FEN has no castling rights ("-"), so all four rights features
    # are 0.0 despite the kings standing on their home squares.
    expected = [
        piece_feature_index(chess.WHITE, chess.KING, chess.E1),
        piece_feature_index(chess.BLACK, chess.KING, chess.E3),
        SIDE_TO_MOVE_INDEX,
    ]
    assert sorted(ones) == expected


def test_encoding_is_deterministic() -> None:
    """Encoding the same position twice gives identical arrays."""
    board = chess.Board(STARTING_FEN)

    first = encode_board(board)
    second = encode_board(board)

    np.testing.assert_array_equal(first, second)
