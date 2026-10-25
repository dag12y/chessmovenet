"""Tests for :mod:`chessmovenet.data` (PGN reading and example extraction).

All tests use tiny hand-written PGN strings, either parsed directly from an
in-memory string or written to a pytest ``tmp_path`` file. No test reads the
real personal PGN in ``data/raw``.
"""

from __future__ import annotations

import io

import chess
import chess.pgn

from chessmovenet.data import (
    extract_examples,
    is_standard_chess,
    my_color,
    read_games,
)

# A complete two-game PGN document, as two Chess.com-style games.
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

# One game as a plain string, for tests that only need a single game.
SINGLE_GAME_PGN = """\
[Event "Casual Game"]
[White "dag12yi"]
[Black "OpponentOne"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
"""


def parse_single_game(pgn_text: str) -> chess.pgn.Game:
    """Parse exactly one game from an in-memory PGN string."""
    return chess.pgn.read_game(io.StringIO(pgn_text))


# ---------------------------------------------------------------------------
# read_games
# ---------------------------------------------------------------------------


def test_read_games_yields_each_game_in_order(tmp_path) -> None:
    """A multi-game PGN file yields one game object per game, in order."""
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(TWO_GAME_PGN, encoding="utf-8")

    games = list(read_games(pgn_path))

    assert len(games) == 2
    assert games[0].headers["White"] == "dag12yi"
    assert games[1].headers["Black"] == "dag12yi"


def test_read_games_empty_file_yields_nothing(tmp_path) -> None:
    """An empty PGN file is valid and yields zero games."""
    pgn_path = tmp_path / "empty.pgn"
    pgn_path.write_text("", encoding="utf-8")

    assert list(read_games(pgn_path)) == []


def test_read_games_is_lazy(tmp_path) -> None:
    """Taking only the first game from a generator parses at most one game."""
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(TWO_GAME_PGN, encoding="utf-8")

    first_game = next(iter(read_games(pgn_path)))

    assert first_game.headers["White"] == "dag12yi"



# ---------------------------------------------------------------------------
# my_color
# ---------------------------------------------------------------------------


def test_my_color_returns_white_when_user_is_white() -> None:
    """A user listed in the White header plays White."""
    game = parse_single_game(SINGLE_GAME_PGN)

    assert my_color(game, "dag12yi") == chess.WHITE


def test_my_color_returns_black_when_user_is_black() -> None:
    """A user listed in the Black header plays Black."""
    game = parse_single_game(
        """\
[Event "Casual Game"]
[White "OpponentOne"]
[Black "dag12yi"]
[Result "*"]

1. e4 e5 *
"""
    )

    assert my_color(game, "dag12yi") == chess.BLACK


def test_my_color_matches_username_case_insensitively() -> None:
    """Header and query may differ only in capitalization."""
    game = parse_single_game(SINGLE_GAME_PGN)

    assert my_color(game, "DAG12YI") == chess.WHITE
    assert my_color(game, "Dag12yi") == chess.WHITE
    assert my_color(game, "dag12YI") == chess.WHITE


def test_my_color_returns_none_when_user_is_not_a_player() -> None:
    """A username absent from both headers is not a player."""
    game = parse_single_game(SINGLE_GAME_PGN)

    assert my_color(game, "someone-else") is None


# ---------------------------------------------------------------------------
# extract_examples
# ---------------------------------------------------------------------------


def test_extract_examples_returns_only_white_moves() -> None:
    """White's examples contain White's moves in order, and only those."""
    game = parse_single_game(SINGLE_GAME_PGN)

    examples = extract_examples(game, "dag12yi")

    assert [uci for _, uci in examples] == ["e2e4", "g1f3", "f1b5"]


def test_extract_examples_returns_only_black_moves() -> None:
    """Black's examples contain Black's moves in order, and only those."""
    game = parse_single_game(
        """\
[Event "Casual Game"]
[White "OpponentOne"]
[Black "dag12yi"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
"""
    )

    examples = extract_examples(game, "dag12yi")

    assert [uci for _, uci in examples] == ["e7e5", "b8c6"]


def test_extract_examples_matches_username_case_insensitively() -> None:
    """Example extraction uses the same case-insensitive matching."""
    game = parse_single_game(SINGLE_GAME_PGN)

    examples = extract_examples(game, "Dag12YI")

    assert [uci for _, uci in examples] == ["e2e4", "g1f3", "f1b5"]


def test_extract_examples_returns_empty_list_for_non_player() -> None:
    """A user who did not play in the game gets no examples."""
    game = parse_single_game(SINGLE_GAME_PGN)

    assert extract_examples(game, "someone-else") == []


def test_is_standard_chess_rejects_chess_com_variants() -> None:
    """3-check and Chess960 headers are not standard chess."""
    three_check = parse_single_game(
        """\
[Event "Live Chess - Three-Check"]
[White "dag12yi"]
[Black "OpponentOne"]
[Variant "3-check"]
[SetUp "1"]
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 +0+0"]
[Result "*"]

1. e4 e5 *
"""
    )
    chess960 = parse_single_game(
        """\
[Event "Live Chess - Chess960"]
[White "dag12yi"]
[Black "OpponentOne"]
[Variant "Chess960"]
[Result "*"]

1. e4 e5 *
"""
    )

    assert is_standard_chess(parse_single_game(SINGLE_GAME_PGN))
    assert not is_standard_chess(three_check)
    assert not is_standard_chess(chess960)


def test_extract_examples_skips_three_check_variant() -> None:
    """Chess.com 3-check FENs must not be replayed as standard chess."""
    game = parse_single_game(
        """\
[Event "Live Chess - Three-Check"]
[White "dag12yi"]
[Black "OpponentOne"]
[Variant "3-check"]
[SetUp "1"]
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 +0+0"]
[Result "*"]

1. e4 e5 *
"""
    )

    assert extract_examples(game, "dag12yi") == []


def test_extract_examples_fen_is_captured_before_each_move() -> None:
    """Each FEN is the position *before* the recorded move, not after."""
    game = parse_single_game(SINGLE_GAME_PGN)

    examples = extract_examples(game, "dag12yi")

    # First White move: FEN is the standard starting position and the move
    # is 1. e4 — if the FEN were captured after the move, the pawn would
    # already be on e4 here.
    first_fen, first_move = examples[0]
    assert first_fen == chess.STARTING_FEN
    assert first_move == "e2e4"

    # Second White move: FEN is the position after 1...e5 but before
    # 2. Nf3 — the knight must still be on g1.
    second_fen, second_move = examples[1]
    board = chess.Board(second_fen)
    assert board.piece_at(chess.G1) == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert board.turn == chess.WHITE
    assert second_move == "g1f3"


def test_extract_examples_every_recorded_move_is_legal_in_its_fen() -> None:
    """Sanity check: the move is legal in the FEN recorded alongside it,
    which can only hold if the FEN precedes the move."""
    game = parse_single_game(
        """\
[Event "Casual Game"]
[White "dag12yi"]
[Black "OpponentOne"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 *
"""
    )

    examples = extract_examples(game, "dag12yi")

    for fen, uci in examples:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        assert move in list(board.legal_moves)
