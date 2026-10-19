"""Pytest suite for chessmovenet.prepare.

Uses tiny hand-written PGNs written to temporary files so the tests are
self-contained and never touch the real (personal) raw PGN.
"""

import chess
import chess.pgn
import io

from chessmovenet.data import my_color
from chessmovenet.prepare import (
    build_examples,
    split_records,
    validate_pgn,
)

WHITE_PGN = """\
[Event "Test"]
[White "dag12yi"]
[Black "opponent"]
[UTCDate "2024.01.01"]
[UTCTime "10:00:00"]
[Result "1-0"]
[TimeControl "600"]
[WhiteElo "1000"]
[BlackElo "900"]
[Link "https://www.chess.com/game/live/00000000001"]

1. e4 e5 2. Nf3 Nc6
"""

BLACK_PGN = """\
[Event "Test"]
[White "opponent"]
[Black "dag12yi"]
[UTCDate "2024.01.02"]
[UTCTime "11:00:00"]
[Result "0-1"]
[TimeControl "600"]
[WhiteElo "900"]
[BlackElo "1000"]
[Link "https://www.chess.com/game/live/00000000002"]

1. e4 e5 2. Nf3 Nc6
"""

VARIANT_PGN = """\
[Event "Test"]
[White "dag12yi"]
[Black "opponent"]
[Variant "Chess960"]
[UTCDate "2024.01.03"]
[UTCTime "12:00:00"]
[Result "1-0"]
[TimeControl "600"]
[WhiteElo "1000"]
[BlackElo "900"]
[Link "https://www.chess.com/game/live/00000000003"]

1. e4 e5
"""

DUPLICATE_PGN = """\
[Event "Test"]
[White "dag12yi"]
[Black "opponent"]
[UTCDate "2024.01.04"]
[UTCTime "13:00:00"]
[Result "1-0"]
[TimeControl "600"]
[WhiteElo "1000"]
[BlackElo "900"]
[Link "https://www.chess.com/game/live/00000000009"]

1. e4 e5

[Event "Test"]
[White "dag12yi"]
[Black "opponent"]
[UTCDate "2024.01.05"]
[UTCTime "14:00:00"]
[Result "1-0"]
[TimeControl "600"]
[WhiteElo "1000"]
[BlackElo "900"]
[Link "https://www.chess.com/game/live/00000000009"]

1. d4 d5
"""

NOMOVES_PGN = """\
[Event "Test"]
[White "dag12yi"]
[Black "opponent"]
[UTCDate "2024.01.06"]
[UTCTime "15:00:00"]
[Result "*"]
[TimeControl "600"]
[WhiteElo "1000"]
[BlackElo "900"]
[Link "https://www.chess.com/game/live/00000000007"]

"""


def _write_pgn(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _load(path, username="dag12yi"):
    records, stats, _ = validate_pgn(path, username)
    return records, stats


def test_my_color_white():
    game = chess.pgn.read_game(io.StringIO(WHITE_PGN))
    assert my_color(game, "dag12yi") is chess.WHITE
    assert my_color(game, "DAG12YI") is chess.WHITE  # case-insensitive


def test_my_color_black():
    game = chess.pgn.read_game(io.StringIO(BLACK_PGN))
    assert my_color(game, "dag12yi") is chess.BLACK


def test_my_color_not_a_player():
    game = chess.pgn.read_game(io.StringIO('[White "alice"]\n[Black "bob"]\n\n1. e4\n'))
    assert my_color(game, "dag12yi") is None
def test_valid_game_white(tmp_path):
    path = _write_pgn(tmp_path, "white.pgn", WHITE_PGN)
    records, stats = _load(path)
    assert stats["valid"] == 1
    assert stats["not_our_identity"] == 0
    assert stats["non_standard_variant"] == 0
    assert records[0]["color"] is chess.WHITE


def test_valid_game_black(tmp_path):
    path = _write_pgn(tmp_path, "black.pgn", BLACK_PGN)
    records, stats = _load(path)
    assert stats["valid"] == 1
    assert records[0]["color"] is chess.BLACK


def test_variant_skipped(tmp_path):
    path = _write_pgn(tmp_path, "variant.pgn", VARIANT_PGN)
    records, stats = _load(path)
    assert stats["valid"] == 0
    assert stats["non_standard_variant"] == 1
    assert records == []


def test_duplicate_removed(tmp_path):
    path = _write_pgn(tmp_path, "dup.pgn", DUPLICATE_PGN)
    records, stats = _load(path)
    assert stats["valid"] == 1
    assert stats["duplicate"] == 1
    assert len(records) == 1


def test_no_moves_invalid(tmp_path):
    path = _write_pgn(tmp_path, "nomoves.pgn", NOMOVES_PGN)
    records, stats = _load(path)
    assert stats["valid"] == 0
    assert stats["invalid_no_moves"] == 1


def test_examples_only_user_moves_and_legal(tmp_path):
    path = _write_pgn(tmp_path, "white.pgn", WHITE_PGN)
    records, _ = _load(path)
    examples = build_examples(records)
    # dag12yi is White: moves are 1. e4 and 2. Nf3.
    assert [e["uci"] for e in examples] == ["e2e4", "g1f3"]
    for e in examples:
        board = chess.Board(e["fen"])
        move = chess.Move.from_uci(e["uci"])
        assert board.is_legal(move), f"illegal {e['uci']} in {e['fen']}"
        expected_turn = chess.WHITE if e["color"] == 1 else chess.BLACK
        assert board.turn == expected_turn
        assert e["game_id"] == "https://www.chess.com/game/live/00000000001"


def test_black_examples_only_user_moves(tmp_path):
    path = _write_pgn(tmp_path, "black.pgn", BLACK_PGN)
    records, _ = _load(path)
    examples = build_examples(records)
    # dag12yi is Black: moves are 1... e5 and 2... Nc6 (b8-c6).
    assert [e["uci"] for e in examples] == ["e7e5", "b8c6"]
    for e in examples:
        board = chess.Board(e["fen"])
        assert board.turn == chess.BLACK


def _make_record(game_id, date_str, color=chess.WHITE):
    return {
        "game_id": game_id,
        "date": date_str,
        "result": "1-0",
        "time_control": "600",
        "color": color,
        "initial_fen": chess.Board().fen(),
        "moves": ["e2e4"],
    }


def test_split_chronological_and_leakage_free():
    games = [
        _make_record(f"g{i}", f"2024.{i:02d}.01 10:00:00") for i in range(1, 11)
    ]
    train, val, test = split_records(games)
    all_ids = [g["game_id"] for split in (train, val, test) for g in split]
    assert len(set(all_ids)) == len(all_ids) == 10  # no shared game ids
    assert max(g["date"] for g in train) < min(g["date"] for g in val)
    assert max(g["date"] for g in val) < min(g["date"] for g in test)


def test_split_recent_train():
    games = [
        _make_record(f"g{i}", f"2024.01.{i:02d} 10:00:00") for i in range(1, 21)
    ]
    train, val, test = split_records(games, recent_train=3)
    assert len(train) == 3
    assert len(val) > 0 and len(test) > 0
    assert max(g["date"] for g in train) < min(g["date"] for g in val)

