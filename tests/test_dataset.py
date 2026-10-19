"""Tests for :mod:`chessmovenet.dataset` (PGN -> ``(X, y)`` arrays).

Every test writes a tiny hand-made PGN string to a pytest ``tmp_path``
file and reads it back — no test touches the real personal PGN in
``data/raw``.  The fixtures are deliberately small so each expected
label can be written out by hand.
"""

from __future__ import annotations

import numpy as np

import chess

from chessmovenet.dataset import build_dataset
from chessmovenet.encode import PIECE_PLANE_INDEX, SIDE_TO_MOVE_INDEX, SQUARES_PER_PLANE
from chessmovenet.moves import move_to_label

# ---------------------------------------------------------------------------
# Small PGN fixtures (written to tmp_path by each test)
# ---------------------------------------------------------------------------

# Two complete games.  "dag12yi" plays White in the first game and Black
# in the second, so one username yields examples from both games.
TWO_GAME_PGN = """\
[Event "Casual Game"]
[Site "Chess.com"]
[Date "2024.01.01"]
[White "dag12yi"]
[Black "OpponentOne"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

[Event "Casual Game"]
[Site "Chess.com"]
[Date "2024.01.02"]
[White "OpponentTwo"]
[Black "dag12yi"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1
"""

# dag12yi plays Black in a single short game: their moves are e7e5 and
# b8c6, both decided from positions where it is Black to move.
SINGLE_BLACK_GAME_PGN = """\
[Event "Casual Game"]
[White "OpponentOne"]
[Black "dag12yi"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
"""

# Total feature count of encode_board: 768 piece planes + 1 side to move
# + 4 castling rights + 8 en-passant files.
EXPECTED_FEATURE_COUNT = 781


def write_pgn(tmp_path, text: str, name: str = "games.pgn"):
    """Write ``text`` to a temporary PGN file and return its path."""
    pgn_path = tmp_path / name
    pgn_path.write_text(text, encoding="utf-8")
    return pgn_path


def piece_feature_index(color: bool, piece_type: int, square: int) -> int:
    """Global feature index of a piece on a square (plane * 64 + square)."""
    plane = PIECE_PLANE_INDEX[(color, piece_type)]
    return plane * SQUARES_PER_PLANE + square


# ---------------------------------------------------------------------------
# Shapes and dtypes
# ---------------------------------------------------------------------------


def test_shapes_and_dtypes(tmp_path) -> None:
    """X is (n, 781) float32 and y is (n,) integer with the same n."""
    pgn_path = write_pgn(tmp_path, TWO_GAME_PGN)

    X, y = build_dataset(pgn_path, "dag12yi")

    # dag12yi played 3 moves as White (e4, Nf3, Bb5) in game 1 and
    # 3 moves as Black (d5, e6, Nf6) in game 2 -> 6 examples.
    assert X.shape == (6, EXPECTED_FEATURE_COUNT)
    assert X.dtype == np.float32
    assert y.shape == (6,)
    assert np.issubdtype(y.dtype, np.integer)


def test_x_values_are_binary_and_finite(tmp_path) -> None:
    """Every feature is a finite 0.0/1.0 value (board encodings are binary)."""
    pgn_path = write_pgn(tmp_path, TWO_GAME_PGN)

    X, _ = build_dataset(pgn_path, "dag12yi")

    assert np.isfinite(X).all()
    assert np.isin(X, [0.0, 1.0]).all()


# ---------------------------------------------------------------------------
# Target labels
# ---------------------------------------------------------------------------


def test_y_contains_the_expected_labels_in_order(tmp_path) -> None:
    """Each row's label is exactly the move played from that position."""
    pgn_path = write_pgn(tmp_path, TWO_GAME_PGN)

    _, y = build_dataset(pgn_path, "dag12yi")

    expected_uci = [
        # game 1 (dag12yi = White): White's moves only
        "e2e4", "g1f3", "f1b5",
        # game 2 (dag12yi = Black): Black's moves only
        "d7d5", "e7e6", "g8f6",
    ]
    expected = [move_to_label(chess.Move.from_uci(uci)) for uci in expected_uci]

    assert y.tolist() == expected


def test_first_row_is_the_starting_position(tmp_path) -> None:
    """Row 0 encodes the standard start position, before 1. e2e4."""
    pgn_path = write_pgn(tmp_path, TWO_GAME_PGN)

    X, y = build_dataset(pgn_path, "dag12yi")

    # White to move, pawn still on e2, king still on e1.
    assert X[0][SIDE_TO_MOVE_INDEX] == 1.0
    assert X[0][piece_feature_index(chess.WHITE, chess.PAWN, chess.E2)] == 1.0
    assert X[0][piece_feature_index(chess.WHITE, chess.KING, chess.E1)] == 1.0
    # The first target is 1. e2e4.
    assert y[0] == move_to_label(chess.Move.from_uci("e2e4"))


def test_second_row_is_position_before_not_after_the_move(tmp_path) -> None:
    """Row 1 is the board after 1...e5 but before 2. Nf3: knight still on g1."""
    pgn_path = write_pgn(tmp_path, TWO_GAME_PGN)

    X, y = build_dataset(pgn_path, "dag12yi")

    # If the FEN had been captured *after* the move, g1 would be empty.
    assert X[1][piece_feature_index(chess.WHITE, chess.KNIGHT, chess.G1)] == 1.0
    # Black has already answered 1...e5, so the side-to-move flag is White.
    assert X[1][SIDE_TO_MOVE_INDEX] == 1.0
    assert y[1] == move_to_label(chess.Move.from_uci("g1f3"))



def test_black_players_rows_show_black_to_move(tmp_path) -> None:
    """For a Black player the first row is the position after 1. e4."""
    pgn_path = write_pgn(tmp_path, SINGLE_BLACK_GAME_PGN)

    X, y = build_dataset(pgn_path, "dag12yi")

    # dag12yi's first move is 1...e5, decided with Black to move.
    assert X.shape == (2, EXPECTED_FEATURE_COUNT)
    assert X[0][SIDE_TO_MOVE_INDEX] == 0.0  # Black to move.
    # White's pawn has already advanced to e4; e2 is empty again.
    assert X[0][piece_feature_index(chess.WHITE, chess.PAWN, chess.E4)] == 1.0
    assert X[0][piece_feature_index(chess.WHITE, chess.PAWN, chess.E2)] == 0.0
    assert y[0] == move_to_label(chess.Move.from_uci("e7e5"))
    assert y[1] == move_to_label(chess.Move.from_uci("b8c6"))


# ---------------------------------------------------------------------------
# Only the requested player's moves
# ---------------------------------------------------------------------------


def test_only_the_requested_players_moves_are_included(tmp_path) -> None:
    """Asking for the opponent yields only the opponent's moves."""
    pgn_path = write_pgn(tmp_path, TWO_GAME_PGN)

    # OpponentOne plays only game 1 (as Black): 1...e5, 2...Nc6, 3...a6.
    X, y = build_dataset(pgn_path, "OpponentOne")

    assert X.shape == (3, EXPECTED_FEATURE_COUNT)
    expected = [
        move_to_label(chess.Move.from_uci("e7e5")),
        move_to_label(chess.Move.from_uci("b8c6")),
        move_to_label(chess.Move.from_uci("a7a6")),
    ]
    assert y.tolist() == expected


def test_dataset_skips_games_the_player_is_not_in(tmp_path) -> None:
    """A player active in only one game gets only that game's examples."""
    pgn_path = write_pgn(tmp_path, TWO_GAME_PGN)

    # OpponentTwo appears solely in game 2 (as White): d4, c4, Nc3.
    X, y = build_dataset(pgn_path, "OpponentTwo")

    assert X.shape == (3, EXPECTED_FEATURE_COUNT)
    expected = [
        move_to_label(chess.Move.from_uci("d2d4")),
        move_to_label(chess.Move.from_uci("c2c4")),
        move_to_label(chess.Move.from_uci("b1c3")),
    ]
    assert y.tolist() == expected


def test_username_matching_is_case_insensitive(tmp_path) -> None:
    """'DAG12YI' finds the same examples as 'dag12yi'."""
    pgn_path = write_pgn(tmp_path, TWO_GAME_PGN)

    X_lower, y_lower = build_dataset(pgn_path, "dag12yi")
    X_upper, y_upper = build_dataset(pgn_path, "DAG12YI")

    np.testing.assert_array_equal(X_lower, X_upper)
    np.testing.assert_array_equal(y_lower, y_upper)


# ---------------------------------------------------------------------------
# Empty / no-player cases
# ---------------------------------------------------------------------------


def test_non_player_username_gives_empty_well_formed_arrays(tmp_path) -> None:
    """A user who played nothing gets (0, 781) float32 and (0,) integer."""
    pgn_path = write_pgn(tmp_path, TWO_GAME_PGN)

    X, y = build_dataset(pgn_path, "someone-else")

    assert X.shape == (0, EXPECTED_FEATURE_COUNT)
    assert X.dtype == np.float32
    assert y.shape == (0,)
    assert np.issubdtype(y.dtype, np.integer)


def test_empty_pgn_file_gives_empty_well_formed_arrays(tmp_path) -> None:
    """A file with zero games is handled without error."""
    pgn_path = write_pgn(tmp_path, "", name="empty.pgn")

    X, y = build_dataset(pgn_path, "dag12yi")

    assert X.shape == (0, EXPECTED_FEATURE_COUNT)
    assert X.dtype == np.float32
    assert y.shape == (0,)
    assert np.issubdtype(y.dtype, np.integer)


def test_game_without_moves_contributes_nothing(tmp_path) -> None:
    """A game record with no moves at all is skipped safely."""
    pgn_path = write_pgn(
        tmp_path,
        """\
[Event "Aborted Game"]
[White "dag12yi"]
[Black "OpponentOne"]
[Result "*"]

*
""",
    )

    X, y = build_dataset(pgn_path, "dag12yi")

    assert X.shape == (0, EXPECTED_FEATURE_COUNT)
    assert y.shape == (0,)