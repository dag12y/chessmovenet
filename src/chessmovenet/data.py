"""PGN reading and move-example extraction for ChessMoveNet.

Reads a player's games from a PGN file and turns the games where that player
took part into supervised learning examples: ``(FEN before the move, UCI move)``
pairs recorded only for the moves the player actually made.
"""

from __future__ import annotations

import chess
import chess.pgn


def read_games(path):
    """Yield one ``chess.pgn.Game`` at a time from a PGN file.

    Reads the file lazily so we never load all games into memory at once.
    """
    with open(path, encoding="utf-8") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                # End of file: no more games to read.
                return
            yield game


def my_color(game, username):
    """Return the color this user plays in ``game``.

    Looks at the ``White`` and ``Black`` PGN headers. Comparison is
    case-insensitive. Returns ``None`` when the user is not a player.
    """
    username_lower = username.lower()
    white = game.headers.get("White", "")
    black = game.headers.get("Black", "")
    if white.lower() == username_lower:
        return chess.WHITE
    if black.lower() == username_lower:
        return chess.BLACK
    return None


def extract_examples(game, username="dag12yi"):
    """Return ``(fen_before_move, uci_move)`` pairs for the user's moves only.

    Replays the game move by move. On each of the user's turns we snapshot the
    FEN *before* pushing the move and record the move in UCI notation. Games
    where the user is not a player yield an empty list.
    """
    color = my_color(game, username)
    if color is None:
        # Not our game: nothing to learn from.
        return []

    board = game.board()
    examples = []
    for move in game.mainline_moves():
        if board.turn == color:
            # It is our turn: capture the position before we move.
            fen_before = board.fen()
            uci_move = move.uci()
            examples.append((fen_before, uci_move))
        board.push(move)
    return examples
