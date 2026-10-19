"""Build a supervised-learning dataset ``(X, y)`` from a PGN file of games.

This module is the glue between three earlier ChessMoveNet modules:

- :mod:`chessmovenet.data` reads the PGN file and extracts, for one player,
  the ``(fen_before_move, uci_move)`` pairs of every move that player made.
- :mod:`chessmovenet.encode` turns a :class:`chess.Board` into a fixed
  781-element ``float32`` feature vector.
- :mod:`chessmovenet.moves` turns a :class:`chess.Move` into a single
  integer class label.

The single public function, :func:`build_dataset`, wires those three pieces
together into the two arrays every supervised model needs:

``X`` — one row per move the player made, holding the encoding of the
       board position **immediately before** that move;
``y`` — one integer label per row, naming the move that was played.

Why the position *before* the move?  A chess engine that imitates a player
must learn "given the position I am looking at, what would this player
play?".  If we stored the position *after* the move, the answer would
already be visible on the board and the network would learn nothing
transferable.  :func:`chessmovenet.data.extract_examples` already
guarantees this ordering, so this module only has to keep each FEN paired
with its own move while converting both to numbers.

The two returned arrays are always aligned: ``len(X) == len(y) == the
number of moves the player made`` across all games in the file, in
chronological order (game order first, move order within each game).
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from chessmovenet.data import extract_examples, my_color, read_games
from chessmovenet.encode import BOARD_FEATURE_SIZE, encode_board
from chessmovenet.moves import move_to_label


def build_dataset(
    pgn_path: str | Path, username: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build input features ``X`` and target labels ``y`` from a PGN file.

    Every move played by ``username`` in the file becomes exactly one
    training example:

    1. The game is replayed by :func:`chessmovenet.data.extract_examples`,
       which records the FEN of the position *before* each of the user's
       moves together with the move itself (in UCI notation).
    2. Each FEN string is turned back into a :class:`chess.Board` and
       encoded with :func:`chessmovenet.encode.encode_board` into a 781
       element ``float32`` row.
    3. Each UCI string is turned into a :class:`chess.Move` and encoded
       with :func:`chessmovenet.moves.move_to_label` into one integer.

    Games where ``username`` is not one of the two players are skipped
    (their headers simply do not match), and a file with no usable games
    yields empty arrays of the correct shapes rather than an error.

    Args:
        pgn_path: Path to the PGN file to read (read-only; never modified).
        username: The player whose moves should become training examples.
            Matching is case-insensitive, exactly as in :mod:`data`.

    Returns:
        A ``(X, y)`` tuple where:

        - ``X`` has shape ``(number_of_examples, 781)`` and dtype
          ``np.float32`` — one encoded board position per row.
        - ``y`` has shape ``(number_of_examples,)`` and an integer dtype
          (``np.int64``) — the class label of the move played from that
          position.

        If the player made no moves in the file, ``X`` has shape
        ``(0, 781)``, ``y`` has shape ``(0,)``, and dtypes are unchanged.
    """
    # Buffers that grow one entry per example.  A plain Python list of
    # rows is the simplest way to collect an unknown number of examples
    # before stacking them into one 2-D array at the end.
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    # read_games yields one game at a time, so memory use stays flat no
    # matter how many games the file contains.
    for game in read_games(pgn_path):
        # Explicit guard: a game the user did not play in contributes
        # nothing.  (extract_examples would return an empty list for such
        # a game anyway; checking my_color first makes the skip visible
        # and costs one header lookup.)
        if my_color(game, username) is None:
            continue

        # (fen_before_move, uci_move) pairs for the user's moves only,
        # in the order they were played.
        for fen, uci in extract_examples(game, username):
            # Requirement: the input row is the position *before* the
            # move.  The stored FEN is exactly that snapshot, so we only
            # need to parse it back into a board object.
            board = chess.Board(fen)
            feature_rows.append(encode_board(board))  # (781,) float32

            # The target is the move played from that position, as one
            # integer class label.  chess.Move.from_uci parses "e2e4",
            # "e7e8q", etc.
            labels.append(move_to_label(chess.Move.from_uci(uci)))

    # Safe empty case: no games, or no moves by this player.  We still
    # return correctly shaped and typed arrays so downstream code can do
    # X.shape[0] / arithmetic without special-casing None.
    if not feature_rows:
        empty_x = np.zeros((0, BOARD_FEATURE_SIZE), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        return empty_x, empty_y

    # np.stack turns the list of (781,) float32 rows into one
    # (number_of_examples, 781) float32 matrix.
    X = np.stack(feature_rows)
    # One integer label per row; dtype fixed to int64 for consistency.
    y = np.asarray(labels, dtype=np.int64)

    return X, y