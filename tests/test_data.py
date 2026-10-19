"""Tests for chessmovenet.data using a tiny hand-written PGN."""

import io

import chess
import chess.pgn

from chessmovenet.data import extract_examples, my_color

PGN_WHITE = """\
[White "dag12yi"]
[Black "opponent"]

1. e4 e5 2. Nf3 Nc6
"""

PGN_BLACK = """\
[White "opponent"]
[Black "dag12yi"]

1. e4 e5 2. Nf3 Nc6
"""

PGN_STRANGER = """\
[White "alice"]
[Black "bob"]

1. e4 e5 2. Nf3 Nc6
"""


def _load_game(pgn_text):
    return chess.pgn.read_game(io.StringIO(pgn_text))


def test_my_color_white():
    game = _load_game(PGN_WHITE)
    assert my_color(game, "dag12yi") is chess.WHITE


def test_my_color_black():
    game = _load_game(PGN_BLACK)
    assert my_color(game, "dag12yi") is chess.BLACK


def test_my_color_not_a_player():
    game = _load_game(PGN_STRANGER)
    assert my_color(game, "dag12yi") is None


def test_extract_examples_only_my_moves():
    game = _load_game(PGN_WHITE)
    examples = extract_examples(game, "dag12yi")
    # dag12yi (White) moves: 1. e4 and 2. Nf3 -> 2 examples.
    assert [uci for _, uci in examples] == ["e2e4", "g1f3"]


def test_extract_examples_fen_before_move():
    game = _load_game(PGN_WHITE)
    examples = extract_examples(game, "dag12yi")
    # First FEN must be the starting position (before 1. e4), not after it.
    assert examples[0][0] == chess.Board().fen()
    # The recorded move must match that FEN's side to move (White).
    assert chess.Board(examples[0][0]).turn is chess.WHITE


def test_extract_examples_skips_non_player_game():
    game = _load_game(PGN_STRANGER)
    assert extract_examples(game, "dag12yi") == []
