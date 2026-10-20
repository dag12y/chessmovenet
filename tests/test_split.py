"""Tests for :mod:`chessmovenet.split` (chronological whole-game splitting).

Every test writes tiny hand-made PGN files to pytest ``tmp_path``.
The central fixture is a ten-game PGN where each game has a distinct
opening, so the expected labels of every game can be written out by
hand and any row-level leakage between splits would be visible.

The ten fixture games (oldest first); dag12yi plays White and has two
moves in each:

1  2024.01.01  1. e4 e5 2. Nf3 Nc6     ->  e2e4, g1f3
2  2024.01.02  1. d4 d5 2. c4 e6       ->  d2d4, c2c4
3  2024.01.03  1. c4 e5 2. Nc3 Nf6     ->  c2c4, b1c3
4  2024.01.04  1. f4 e5 2. Nf3 Nc6     ->  f2f4, g1f3
5  2024.01.05  1. b3 e5 2. Bb2 d6      ->  b2b3, c1b2
6  2024.01.06  1. g3 d5 2. Bg2 Nf6     ->  g2g3, f1g2
7  2024.01.07  1. e3 e5 2. d4 exd4     ->  e2e3, d2d4
8  2024.01.08  1. d3 d5 2. Nf3 Nf6     ->  d2d3, g1f3
9  2024.01.09  1. Nf3 d5 2. Nc3 Nf6    ->  g1f3, b1c3
10 2024.01.10  1. h3 e5 2. Nc3 Nc6     ->  h2h3, b1c3
"""

from __future__ import annotations

import warnings

import chess
import numpy as np
import pytest

from chessmovenet.dataset import build_dataset
from chessmovenet.encode import BOARD_FEATURE_SIZE
from chessmovenet.moves import move_to_label
from chessmovenet.split import (
    build_per_game_dataset,
    sort_games_chronologically,
    split_dataset_chronologically,
    split_games,
)

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def game_block(
    date: str,
    moves: str,
    white: str = "dag12yi",
    black: str = "Opponent",
    time: str | None = None,
) -> str:
    """Render one PGN game header + movetext block as a string."""
    time_line = f'[Time "{time}"]\n' if time is not None else ""
    return (
        '[Event "Casual Game"]\n'
        '[Site "Chess.com"]\n'
        f'[Date "{date}"]\n'
        f"{time_line}"
        f'[White "{white}"]\n'
        f'[Black "{black}"]\n'
        '[Result "*"]\n'
        f"\n{moves} *\n\n"
    )


def write_pgn(tmp_path, blocks: list[str], name: str = "games.pgn"):
    """Write concatenated PGN blocks to a temporary file."""
    pgn_path = tmp_path / name
    pgn_path.write_text("".join(blocks), encoding="utf-8")
    return pgn_path


# (date, full movetext) for the ten fixture games, oldest first.
EVEN_TEN_GAMES: list[tuple[str, str]] = [
    ("2024.01.01", "1. e4 e5 2. Nf3 Nc6"),
    ("2024.01.02", "1. d4 d5 2. c4 e6"),
    ("2024.01.03", "1. c4 e5 2. Nc3 Nf6"),
    ("2024.01.04", "1. f4 e5 2. Nf3 Nc6"),
    ("2024.01.05", "1. b3 e5 2. Bb2 d6"),
    ("2024.01.06", "1. g3 d5 2. Bg2 Nf6"),
    ("2024.01.07", "1. e3 e5 2. d4 exd4"),
    ("2024.01.08", "1. d3 d5 2. Nf3 Nf6"),
    ("2024.01.09", "1. Nf3 d5 2. Nc3 Nf6"),
    ("2024.01.10", "1. h3 e5 2. Nc3 Nc6"),
]

# The two dag12yi UCI labels of each fixture game, in chronological order.
EXPECTED_LABELS_PER_GAME: list[list[str]] = [
    ["e2e4", "g1f3"],
    ["d2d4", "c2c4"],
    ["c2c4", "b1c3"],
    ["f2f4", "g1f3"],
    ["b2b3", "c1b2"],
    ["g2g3", "f1g2"],
    ["e2e3", "d2d4"],
    ["d2d3", "g1f3"],
    ["g1f3", "b1c3"],
    ["h2h3", "b1c3"],
]


def labels_to_ints(uci_labels: list[str]) -> list[int]:
    """Convert UCI strings to their integer class labels."""
    return [move_to_label(chess.Move.from_uci(u)) for u in uci_labels]


def even_ten_games_blocks() -> list[str]:
    """The ten fixture games as PGN blocks, in chronological order."""
    return [game_block(date, moves) for date, moves in EVEN_TEN_GAMES]

# ---------------------------------------------------------------------------
# Game counts and proportions (10 whole games -> 8 / 1 / 1)
# ---------------------------------------------------------------------------


def test_ten_even_games_split_8_1_1_whole_games(tmp_path) -> None:
    """Ten two-move games give 16/2/2 rows: 8 games train, 1 val, 1 test."""
    pgn_path = write_pgn(tmp_path, even_ten_games_blocks())

    split = split_dataset_chronologically(pgn_path, "dag12yi")

    assert split.X_train.shape == (16, BOARD_FEATURE_SIZE)
    assert split.y_train.shape == (16,)
    assert split.X_val.shape == (2, BOARD_FEATURE_SIZE)
    assert split.y_val.shape == (2,)
    assert split.X_test.shape == (2, BOARD_FEATURE_SIZE)
    assert split.y_test.shape == (2,)


def test_dtypes_match_dataset_conventions(tmp_path) -> None:
    """All six arrays follow build_dataset's float32 / int64 dtypes."""
    pgn_path = write_pgn(tmp_path, even_ten_games_blocks())

    split = split_dataset_chronologically(pgn_path, "dag12yi")

    for X in (split.X_train, split.X_val, split.X_test):
        assert X.dtype == np.float32
    for y in (split.y_train, split.y_val, split.y_test):
        assert np.issubdtype(y.dtype, np.integer)


def test_seven_uneven_games_use_closest_whole_game_counts(tmp_path) -> None:
    """7 games of sizes 3,1,2,3,1,2,2 split into 5/1/1 games = 10/2/2 rows."""
    blocks = [
        game_block("2024.02.01", "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"),  # 3
        game_block("2024.02.02", "1. d4 d5"),  # 1
        game_block("2024.02.03", "1. c4 e5 2. Nc3 Nf6"),  # 2
        # NB: after 1. f4 the e2 pawn still blocks the bishop, so the
        # third move must be d4, not Bb5 (unlike the Ruy Lopez game 1).
        game_block("2024.02.04", "1. f4 e5 2. Nf3 Nc6 3. d4 exd4"),  # 3
        game_block("2024.02.05", "1. b3 e5"),  # 1
        game_block("2024.02.06", "1. g3 d5 2. Bg2 Nf6"),  # 2
        game_block("2024.02.07", "1. e3 e5 2. d4 exd4"),  # 2
    ]
    pgn_path = write_pgn(tmp_path, blocks)

    split = split_dataset_chronologically(pgn_path, "dag12yi")

    # 5 oldest games train (3+1+2+3+1 = 10 rows), then 1 val game (2
    # rows) and 1 test game (2 rows).  Whole games cannot do better.
    assert split.X_train.shape == (10, BOARD_FEATURE_SIZE)
    assert split.X_val.shape == (2, BOARD_FEATURE_SIZE)
    assert split.X_test.shape == (2, BOARD_FEATURE_SIZE)


# ---------------------------------------------------------------------------
# Chronological ordering
# ---------------------------------------------------------------------------


def test_games_are_split_by_date_not_by_file_order(tmp_path) -> None:
    """Games stored newest-first still split oldest-train/newest-test."""
    # Same ten games, written in exactly reverse chronological order.
    reversed_blocks = [
        game_block(date, moves) for date, moves in reversed(EVEN_TEN_GAMES)
    ]
    pgn_path = write_pgn(tmp_path, reversed_blocks)

    split = split_dataset_chronologically(pgn_path, "dag12yi")

    # The validation game is the 9th oldest (2024.01.09: Nf3, Nc3),
    # NOT the 9th game in the file (which would be 2024.01.02).
    assert split.y_val.tolist() == labels_to_ints(["g1f3", "b1c3"])
    # The test game is the newest: 2024.01.10 (h3, Nc3).
    assert split.y_test.tolist() == labels_to_ints(["h2h3", "b1c3"])


def test_training_rows_are_the_eight_oldest_games(tmp_path) -> None:
    """y_train is exactly the first 8 games' labels, in chronological order."""
    pgn_path = write_pgn(tmp_path, even_ten_games_blocks())

    split = split_dataset_chronologically(pgn_path, "dag12yi")

    expected = labels_to_ints(
        [uci for game in EXPECTED_LABELS_PER_GAME[:8] for uci in game]
    )
    assert split.y_train.tolist() == expected


def test_sort_uses_date_then_time_and_keeps_file_order_for_ties(tmp_path) -> None:
    """Same date -> Time decides; identical Date+Time -> file order kept."""
    blocks = [
        game_block("2024.03.01", "1. e4 e5", time="18:00:00"),
        game_block("2024.03.01", "1. d4 d5", time="09:00:00"),
        game_block("2024.02.28", "1. c4 e5"),  # Oldest date overall.
        game_block("2024.03.01", "1. f4 e5", time="09:00:00"),  # Tie.
    ]
    pgn_path = write_pgn(tmp_path, blocks)

    games = build_per_game_dataset(pgn_path, "dag12yi")
    ordered = sort_games_chronologically(games)

    first_moves = [game.y[0] for game in ordered]
    assert first_moves == [
        move_to_label(chess.Move.from_uci("c2c4")),  # Feb 28 first.
        move_to_label(chess.Move.from_uci("d2d4")),  # 09:00 before 18:00.
        move_to_label(chess.Move.from_uci("f2f4")),  # Tie: file order.
        move_to_label(chess.Move.from_uci("e2e4")),  # 18:00 last.
    ]


# ---------------------------------------------------------------------------
# Game-level separation: no game may straddle two sets
# ---------------------------------------------------------------------------


def test_each_split_equals_the_concatenation_of_whole_games(tmp_path) -> None:
    """Every split is exactly its whole games' labels, so no game can straddle.

    With the ten fixture games the expected assignment is known by
    construction: games 1-8 (16 rows) train, game 9 validation,
    game 10 test.  Rebuilding those label lists independently and
    comparing proves the boundaries fall between games, not inside one.
    """
    pgn_path = write_pgn(tmp_path, even_ten_games_blocks())

    split = split_dataset_chronologically(pgn_path, "dag12yi")

    def flat(game_indices: range) -> list[int]:
        return labels_to_ints(
            [uci for i in game_indices for uci in EXPECTED_LABELS_PER_GAME[i]]
        )

    assert split.y_train.tolist() == flat(range(0, 8))  # Games 1-8.
    assert split.y_val.tolist() == flat(range(8, 9))  # Game 9 whole.
    assert split.y_test.tolist() == flat(range(9, 10))  # Game 10 whole.


def test_game_aware_split_differs_from_naive_row_split(tmp_path) -> None:
    """On uneven games the split must follow game boundaries, not rows.

    Seven games with sizes 3,1,2,3,1,2,2 (14 rows).  A naive row-wise
    80/10/10 split would cut game 6 in half (train = rows 0-10, which
    ends inside game 6).  Our whole-game split instead trains on the
    five oldest games (10 rows), validates on all of game 6 (2 rows),
    and tests on all of game 7 (2 rows).  This test is the module's
    reason to exist.
    """
    blocks = [
        game_block("2024.02.01", "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"),  # 3
        game_block("2024.02.02", "1. d4 d5"),  # 1
        game_block("2024.02.03", "1. c4 e5 2. Nc3 Nf6"),  # 2
        # NB: after 1. f4 the e2 pawn still blocks the bishop, so the
        # third move is d4, not Bb5 (unlike the Ruy Lopez game 1).
        game_block("2024.02.04", "1. f4 e5 2. Nf3 Nc6 3. d4 exd4"),  # 3
        game_block("2024.02.05", "1. b3 e5"),  # 1
        game_block("2024.02.06", "1. g3 d5 2. Bg2 Nf6"),  # 2
        game_block("2024.02.07", "1. e3 e5 2. d4 exd4"),  # 2
    ]
    pgn_path = write_pgn(tmp_path, blocks)

    split = split_dataset_chronologically(pgn_path, "dag12yi")

    # The whole-game answer: validation is all of game 6 (g3, Bg2).
    assert split.y_val.tolist() == labels_to_ints(["g2g3", "f1g2"])
    # Test is all of game 7 (e3, d4).
    assert split.y_test.tolist() == labels_to_ints(["e2e3", "d2d4"])
    # A naive row split would have had 11 train rows (one from game 6)
    # and only 1 test row; ours has 10 and 2 — the game boundary wins.
    assert split.X_train.shape[0] == 10
    assert split.X_test.shape[0] == 2


# ---------------------------------------------------------------------------
# X / y alignment: each board row keeps its own move label
# ---------------------------------------------------------------------------


def test_xy_pairs_survive_stacking_inside_each_split(tmp_path) -> None:
    """Row k of every X split still pairs with label k of its y split."""
    pgn_path = write_pgn(tmp_path, even_ten_games_blocks())

    split = split_dataset_chronologically(pgn_path, "dag12yi")

    # Validation is game 9: 1. Nf3 d5 2. Nc3 Nf6.  Row 0 must be the
    # starting position (knight still on g1) with target g1f3; row 1
    # the position after 1...d5 (knight now on f3) with target b1c3.
    from chessmovenet.encode import PIECE_PLANE_INDEX, SIDE_TO_MOVE_INDEX

    def piece_index(color: bool, piece_type: int, square: int) -> int:
        return PIECE_PLANE_INDEX[(color, piece_type)] * 64 + square

    assert split.X_val[0][piece_index(chess.WHITE, chess.KNIGHT, chess.G1)] == 1.0
    assert split.y_val[0] == move_to_label(chess.Move.from_uci("g1f3"))
    assert split.X_val[1][piece_index(chess.WHITE, chess.KNIGHT, chess.F3)] == 1.0
    assert split.X_val[1][piece_index(chess.WHITE, chess.KNIGHT, chess.G1)] == 0.0
    assert split.X_val[1][SIDE_TO_MOVE_INDEX] == 1.0  # White to move again.
    assert split.y_val[1] == move_to_label(chess.Move.from_uci("b1c3"))


def test_concatenated_splits_reproduce_build_dataset(tmp_path) -> None:
    """Train+val+test rows equal the flat build_dataset output in order.

    On a chronologically ordered file, stacking the three splits in
    order must reproduce exactly what ``dataset.build_dataset`` returns
    — the strongest possible check that no row was lost, duplicated,
    or reordered by the splitting machinery.
    """
    pgn_path = write_pgn(tmp_path, even_ten_games_blocks())

    X_all, y_all = build_dataset(pgn_path, "dag12yi")
    split = split_dataset_chronologically(pgn_path, "dag12yi")

    stacked_X = np.concatenate(
        [split.X_train, split.X_val, split.X_test], axis=0
    )
    stacked_y = np.concatenate(
        [split.y_train, split.y_val, split.y_test], axis=0
    )

    np.testing.assert_array_equal(stacked_X, X_all)
    np.testing.assert_array_equal(stacked_y, y_all)


# ---------------------------------------------------------------------------
# build_per_game_dataset: the game-aware reader
# ---------------------------------------------------------------------------


def test_build_per_game_dataset_returns_one_bundle_per_game(tmp_path) -> None:
    """Ten games in, ten separate (X_i, y_i) bundles out, file order."""
    pgn_path = write_pgn(tmp_path, even_ten_games_blocks())

    games = build_per_game_dataset(pgn_path, "dag12yi")

    assert len(games) == 10
    for game, expected_uci in zip(games, EXPECTED_LABELS_PER_GAME):
        assert game.X.shape == (2, BOARD_FEATURE_SIZE)
        assert game.X.dtype == np.float32
        assert game.y.tolist() == labels_to_ints(expected_uci)


def test_bundle_dates_come_from_pgn_headers(tmp_path) -> None:
    """Each bundle remembers its Date/Time headers for later sorting."""
    pgn_path = write_pgn(
        tmp_path,
        [
            game_block("2024.05.01", "1. e4 e5", time="10:00:00"),
            game_block("2024.05.02", "1. d4 d5"),  # No Time header.
        ],
    )

    games = build_per_game_dataset(pgn_path, "dag12yi")

    assert [g.date for g in games] == ["2024.05.01", "2024.05.02"]
    assert [g.time for g in games] == ["10:00:00", ""]


def test_non_player_games_are_skipped(tmp_path) -> None:
    """A game the user did not play adds no bundle and no rows."""
    pgn_path = write_pgn(
        tmp_path,
        [
            game_block("2024.06.01", "1. e4 e5"),
            game_block(
                "2024.06.02",
                "1. d4 d5",
                white="StrangerA",
                black="StrangerB",
            ),
        ],
    )

    games = build_per_game_dataset(pgn_path, "dag12yi")

    assert len(games) == 1
    assert games[0].date == "2024.06.01"


# ---------------------------------------------------------------------------
# Edge cases: empty inputs and very small PGNs
# ---------------------------------------------------------------------------


def test_empty_pgn_file_yields_three_empty_well_formed_splits(tmp_path) -> None:
    """A zero-game file is handled without error or warning."""
    pgn_path = write_pgn(tmp_path, [], name="empty.pgn")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # No warning expected for n = 0.
        split = split_dataset_chronologically(pgn_path, "dag12yi")

    for X in (split.X_train, split.X_val, split.X_test):
        assert X.shape == (0, BOARD_FEATURE_SIZE)
        assert X.dtype == np.float32
    for y in (split.y_train, split.y_val, split.y_test):
        assert y.shape == (0,)
        assert np.issubdtype(y.dtype, np.integer)


def test_non_player_username_yields_empty_splits(tmp_path) -> None:
    """Asking for someone who never played gives empty, well-formed sets."""
    pgn_path = write_pgn(tmp_path, even_ten_games_blocks())

    split = split_dataset_chronologically(pgn_path, "nobody-here")

    assert split.X_train.shape == (0, BOARD_FEATURE_SIZE)
    assert split.X_val.shape == (0, BOARD_FEATURE_SIZE)
    assert split.X_test.shape == (0, BOARD_FEATURE_SIZE)
    assert split.y_train.shape == (0,)


def test_game_without_moves_does_not_count_toward_proportions(tmp_path) -> None:
    """An aborted (moveless) game is skipped, so 10 real games still split 8/1/1."""
    blocks = even_ten_games_blocks() + [
        game_block("2024.01.11", "*")  # Aborted: no moves at all.
    ]
    pgn_path = write_pgn(tmp_path, blocks)

    split = split_dataset_chronologically(pgn_path, "dag12yi")

    # Still 10 usable games -> 16/2/2 rows, not 11 games' worth.
    assert split.X_train.shape == (16, BOARD_FEATURE_SIZE)
    assert split.X_val.shape == (2, BOARD_FEATURE_SIZE)
    assert split.X_test.shape == (2, BOARD_FEATURE_SIZE)
    # And the aborted game (newest date) did not sneak into test.
    assert split.y_test.tolist() == labels_to_ints(["h2h3", "b1c3"])


def test_too_few_games_warns_but_still_returns(tmp_path) -> None:
    """With 3 games the 80/10/10 cut leaves val empty: warn, don't crash."""
    blocks = [
        game_block("2024.07.01", "1. e4 e5"),
        game_block("2024.07.02", "1. d4 d5"),
        game_block("2024.07.03", "1. c4 e5"),
    ]
    pgn_path = write_pgn(tmp_path, blocks)

    with pytest.warns(UserWarning, match="only 3 games"):
        split = split_dataset_chronologically(pgn_path, "dag12yi")

    # int(0.8*3) = 2 train games, int(0.9*3) - 2 = 0 val, 1 test.
    # Each game "1. e4 e5" gives White exactly one move (e4).
    assert split.X_train.shape == (2, BOARD_FEATURE_SIZE)
    assert split.X_val.shape == (0, BOARD_FEATURE_SIZE)
    assert split.X_test.shape == (1, BOARD_FEATURE_SIZE)
