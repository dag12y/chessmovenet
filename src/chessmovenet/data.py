"""Read PGN games and extract supervised move examples.

This is the first data-reading module of ChessMoveNet. It answers three
questions about a PGN file of chess games:

1. Which games are in the file?      -> :func:`read_games`
2. Which color did a player play?    -> :func:`my_color`
3. What positions did they see?      -> :func:`extract_examples`

Everything here is deliberately simple: plain functions, descriptive names,
and no board encoding or machine-learning code. The only dependency is
``python-chess``.

Key ``python-chess`` concepts used in this module:

- ``chess.pgn.read_game(handle)`` reads **one** game from a text file that is
  already open, and returns ``None`` once the end of the file is reached.
  Calling it repeatedly therefore walks a PGN file game by game without ever
  loading the whole file into memory.
- ``game.headers`` is a mapping of PGN tag names (``"White"``, ``"Black"``,
  ``"Date"``, ...) to their string values.
- ``game.board()`` returns a fresh :class:`chess.Board` set up in the game's
  starting position (from the ``FEN`` tag if present, otherwise the standard
  starting position).
- ``game.mainline_moves()`` iterates over the moves actually played in the
  game, in order.
- ``board.turn`` is ``chess.WHITE`` (``True``) when it is White's turn to
  move and ``chess.BLACK`` (``False``) when it is Black's.
- ``board.fen()`` renders the current position as a
  `FEN <https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation>`_
  string, and ``move.uci()`` renders a move in UCI notation (``"e2e4"``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn


def read_games(path: str | Path) -> Iterator[chess.pgn.Game]:
    """Lazily yield each game stored in a PGN file.

    The file is opened in text mode and parsed one game at a time by
    :func:`chess.pgn.read_game`, which returns ``None`` at end of file.
    Because this is a generator, callers can stream through tens of
    thousands of games while only one game is in memory at a time.

    Args:
        path: Path to the PGN file (UTF-8 text).

    Yields:
        One :class:`chess.pgn.Game` per game in the file, in file order.
    """
    with open(path, "r", encoding="utf-8") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:  # End of file.
                return
            yield game


def my_color(game: chess.pgn.Game, username: str) -> bool | None:
    """Return which side ``username`` played in ``game``.

    Reads the ``White`` and ``Black`` PGN headers and compares them to
    ``username`` ignoring case, because Chess.com usernames may differ only
    in capitalization between game records.

    Args:
        game: A parsed PGN game.
        username: The player name to look for.

    Returns:
        ``chess.WHITE`` (``True``) if the username matches the ``White``
        header, ``chess.BLACK`` (``False``) if it matches the ``Black``
        header, or ``None`` if the username is not either player.
    """
    white_player = game.headers.get("White", "")
    black_player = game.headers.get("Black", "")

    if username.lower() == white_player.lower():
        return chess.WHITE
    if username.lower() == black_player.lower():
        return chess.BLACK
    return None


def extract_examples(
    game: chess.pgn.Game, username: str
) -> list[tuple[str, str]]:
    """Extract ``(fen_before_move, uci_move)`` pairs for one player's moves.

    Replays the game from its starting position on a fresh board. Before
    every move the position to move is snapshotted with ``board.fen()``, so
    each pair records *the position the player actually saw when choosing
    the move* — exactly the (position -> move) input/target pair a
    supervised model needs. Only the moves of the requested player are
    recorded; the opponent's moves are still pushed onto the board so the
    positions stay in sync, but they are not returned.

    Args:
        game: A parsed PGN game.
        username: The player whose moves should be extracted.

    Returns:
        A list of ``(fen_before_move, uci_move)`` tuples for the player's
        moves in chronological order. Empty if the username is not a player
        in this game.
    """
    color = my_color(game, username)
    if color is None:
        return []

    board = game.board()  # Fresh board in the game's starting position.
    examples: list[tuple[str, str]] = []

    for move in game.mainline_moves():
        if board.turn == color:
            examples.append((board.fen(), move.uci()))
        board.push(move)  # Apply the move, whoever played it.

    return examples
