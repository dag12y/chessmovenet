"""Read PGN games and extract a player's positions and moves."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn


def read_games(path: str | Path) -> Iterator[chess.pgn.Game]:
    """Yield games from a PGN file in file order.

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


# Chess.com labels standard games with no Variant header. Anything else
# (3-check, Chess960, Crazyhouse, ...) uses a different FEN dialect.
_STANDARD_VARIANTS = {"", "standard", "chess"}


def is_standard_chess(game: chess.pgn.Game) -> bool:
    """Return whether ``game`` is standard chess, not a Chess.com variant.

    Args:
        game: A parsed PGN game.

    Returns:
        ``True`` when the ``Variant`` header is missing or names standard
        chess; ``False`` for 3-check, Chess960, Crazyhouse, and similar.
    """
    variant = game.headers.get("Variant", "").strip().lower()
    return variant in _STANDARD_VARIANTS


def my_color(game: chess.pgn.Game, username: str) -> bool | None:
    """Return the player's color, or ``None`` if they are absent.

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
    """Return ``(fen_before_move, uci_move)`` pairs for the player.

    Opponent moves update the board but are not included.

    Args:
        game: A parsed PGN game.
        username: The player whose moves should be extracted.

    Returns:
        A list of ``(fen_before_move, uci_move)`` tuples for the player's
        moves in chronological order. Empty if the username is not a player
        in this game.
    """
    if not is_standard_chess(game):
        return []

    color = my_color(game, username)
    if color is None:
        return []

    board = game.board()
    examples: list[tuple[str, str]] = []

    for move in game.mainline_moves():
        if board.turn == color:
            examples.append((board.fen(), move.uci()))
        board.push(move)  # Apply the move, whoever played it.

    return examples
