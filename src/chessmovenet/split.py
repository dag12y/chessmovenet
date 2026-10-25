"""Chronological whole-game 80/10/10 train/validation/test split.

Three sets, each with a distinct role:

- Train (~80%): the only set the network learns from (gradient descent).
- Validation (~10%): held-out games used to measure generalization and
  guide tuning (early stopping, capacity, learning rate).
- Test (~10%): untouched until the final, one-shot estimate of how the
  model does on brand-new games.

Golden rule: a game the model trained on never appears in validation
or test.

Why whole games, oldest-first?
Splitting flat (X, y) rows row-by-row leaks moves of the same game
across sets — consecutive moves are near-identical, so validation
looks great while nothing transfers. Shuffling games mixes career
eras, breaking the "predict what comes next" use case. This module
therefore keeps one (X_i, y_i) bundle per game, sorts by Date/Time
headers, and moves whole games: oldest 80% train, next 10% validate,
newest 10% test.

Ratios apply to whole games via floor math (e.g. 7 games -> 5/1/1),
so exact percentages aren't guaranteed; empty sets raise a
UserWarning instead of failing.

Functions
---------
- build_per_game_dataset — PGN file -> per-game example bundles
- sort_games_chronologically — order bundles by Date/Time headers
- split_games — cut ordered games at the 80%/90% marks into (X, y) arrays
- split_dataset_chronologically — end-to-end wrapper
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from chessmovenet.data import extract_examples, my_color, read_games
from chessmovenet.encode import BOARD_FEATURE_SIZE, encode_board
from chessmovenet.moves import move_to_label

# Ratios apply to whole games, not individual moves.
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


@dataclass
class GameExamples:
    """All the training examples of one game, kept together.

    Attributes:
        date: The game's ``Date`` PGN header (e.g. ``"2024.01.31"``),
            or ``""`` if the header is missing.  Used for ordering.
        time: The game's ``Time`` PGN header (e.g. ``"14:32:05"``), or
            ``""`` if missing.  Breaks ties between same-date games.
        X: This game's board encodings, shape ``(n_i, 781)``,
            ``float32`` — one row per move by the tracked player.
        y: This game's move labels, shape ``(n_i,)``, ``int64`` —
            row ``k`` of ``X`` is paired with label ``k`` of ``y``.
    """

    date: str
    time: str
    X: np.ndarray
    y: np.ndarray

    @property
    def n_examples(self) -> int:
        """How many of this game's moves became training examples."""
        return int(self.X.shape[0])


@dataclass
class DatasetSplit:
    """The three ``(X, y)`` pairs of a train/validation/test split.

    Every array follows the ``dataset.build_dataset`` conventions:
    ``X_*`` is ``(n, 781)`` ``float32``, ``y_*`` is ``(n,)`` ``int64``.
    Any of the three may have ``n == 0`` for very small PGNs.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _empty_arrays() -> tuple[np.ndarray, np.ndarray]:
    """A correctly shaped/typed empty ``(X, y)`` pair (0 examples)."""
    empty_x = np.zeros((0, BOARD_FEATURE_SIZE), dtype=np.float32)
    empty_y = np.zeros((0,), dtype=np.int64)
    return empty_x, empty_y


def _encode_one_game(
    game: chess.pgn.Game, username: str
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one game's moves as an ``(X, y)`` pair.

    This is the per-game core of ``dataset.build_dataset``: for every
    ``(fen_before_move, uci_move)`` pair that
    :func:`chessmovenet.data.extract_examples` records for ``username``,
    the FEN becomes a :class:`chess.Board` encoded by
    :func:`chessmovenet.encode.encode_board`, and the UCI move becomes
    an integer via :func:`chessmovenet.moves.move_to_label`.  (We
    duplicate those ~10 glue lines here rather than reuse
    ``build_dataset`` because that function flattens games together —
    exactly the boundary information this module must preserve.  If
    ``dataset.py`` is ever refactored, this helper is the piece that
    would move there and be shared by both APIs.)

    Args:
        game: A parsed PGN game the player took part in.
        username: The player whose moves are encoded.

    Returns:
        ``(X_i, y_i)`` for this game alone; empty arrays of the standard
        shapes/dtypes if the game contains no moves by the player.
    """
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for fen, uci in extract_examples(game, username):
        board = chess.Board(fen)
        feature_rows.append(encode_board(board))  # (781,) float32
        labels.append(move_to_label(chess.Move.from_uci(uci)))

    if not feature_rows:
        return _empty_arrays()

    return np.stack(feature_rows), np.asarray(labels, dtype=np.int64)


def build_per_game_dataset(
    pgn_path: str | Path, username: str
) -> list[GameExamples]:
    """Read usable games into separate example bundles.

    Streams the file with :func:`chessmovenet.data.read_games` (one game
    in memory at a time) and keeps games **separate** — the game-aware
    counterpart of ``dataset.build_dataset``, whose flat output cannot
    support game-level splitting.

    Skipped silently:

    - games where ``username`` is not one of the two players;
    - Chess.com variants (3-check, Chess960, Crazyhouse, ...);
    - games with zero moves by the player (e.g. an aborted game record),
      because an empty game would distort the 80/10/10 *game* counts
      without adding any data.

    Games appear in **file order**; chronological order is applied later
    by :func:`sort_games_chronologically`.

    Args:
        pgn_path: Path to the PGN file (read-only).
        username: The player to build examples for (case-insensitive).

    Returns:
        One :class:`GameExamples` per usable game, in file order.
    """
    games: list[GameExamples] = []

    for game in read_games(pgn_path):
        # Same explicit guard as dataset.build_dataset: a game the user
        # did not play contributes nothing.
        if my_color(game, username) is None:
            continue

        X, y = _encode_one_game(game, username)
        if X.shape[0] == 0:
            continue  # No moves by this player: nothing to split later.

        games.append(
            GameExamples(
                date=game.headers.get("Date", ""),
                time=game.headers.get("Time", ""),
                X=X,
                y=y,
            )
        )

    return games


def sort_games_chronologically(games: list[GameExamples]) -> list[GameExamples]:
    """Return games oldest-first by their ``Date`` and ``Time`` headers.

    The sort key is the tuple ``(date, time)`` of the PGN headers, both
    plain strings.  ``"2024.01.31"`` sorts before ``"2024.02.01"`` because
    strings compare character by character and the year comes first.
    Within the same date, the ``Time`` header (``"14:32:05"``) decides;
    if both are identical — or missing — the original file order is
    kept, because Python's ``sorted`` is *stable* (equal keys never swap
    relative order).  Games without a Date header get ``""``, which
    sorts first; that is an explicit, documented convention rather than
    a crash.

    Args:
        games: Per-game bundles in any order (typically file order).

    Returns:
        A new list of the same bundles, oldest game first.
    """
    return sorted(games, key=lambda game: (game.date, game.time))


def _stack(games: list[GameExamples]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate game arrays in order, or return correctly shaped empties.

    Row ``k`` of the output ``X`` still pairs with label ``k`` of the
    output ``y``, because each game's ``X_i``/``y_i`` were built
    together and we stack the games in the same order for both arrays.
    An empty game list yields the standard ``(0, 781)`` / ``(0,)``
    arrays.
    """
    if not games:
        return _empty_arrays()

    X = np.concatenate([game.X for game in games], axis=0)
    y = np.concatenate([game.y for game in games], axis=0)
    return X, y


def split_games(games: list[GameExamples]) -> DatasetSplit:
    """Split chronologically at whole-game boundaries using 80/10/10 ratios.

    The games are sorted oldest-first, then divided at *game*
    boundaries: the oldest ``int(0.8 * n)`` games become training, the
    next ``int(0.9 * n) - int(0.8 * n)`` games become validation, and
    the newest ``n - int(0.9 * n)`` games become test.  Because we
    slice the *list of games* — never a flat array of rows — no game
    can straddle two sets.

    Whole-game arithmetic cannot always reach exactly 80/10/10 (7 games
    become 5/1/1); we accept that trade-off.  If the PGN is so small
    that a set would be empty (and there is at least one game), a
    ``UserWarning`` explains the actual counts instead of hiding the
    problem.

    Args:
        games: Per-game bundles, e.g. from
            :func:`build_per_game_dataset`.  File order is fine —
            this function sorts them itself.

    Returns:
        A :class:`DatasetSplit` of ``(X, y)`` pairs; all six arrays use
        the standard shapes/dtypes and may have zero rows.
    """
    ordered = sort_games_chronologically(games)
    n = len(ordered)

    n_train = int(TRAIN_RATIO * n)  # floor: whole games only.
    n_train_and_val = int((TRAIN_RATIO + VAL_RATIO) * n)
    n_val = n_train_and_val - n_train
    n_test = n - n_train_and_val

    if n > 0 and (n_train == 0 or n_val == 0 or n_test == 0):
        warnings.warn(
            f"only {n} games available: split is "
            f"{n_train} train / {n_val} val / {n_test} test games, "
            "so at least one set is empty; add more games for a "
            "usable 80/10/10 split",
            stacklevel=2,
        )

    # Stack each game slice into its split arrays.
    X_train, y_train = _stack(ordered[:n_train])
    X_val, y_val = _stack(ordered[n_train:n_train_and_val])
    X_test, y_test = _stack(ordered[n_train_and_val:])

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


def split_dataset_chronologically(
    pgn_path: str | Path, username: str
) -> DatasetSplit:
    """Build a player's PGN dataset and split it chronologically.

    Combines the three building blocks:

    1. :func:`build_per_game_dataset` — read every usable game of
       ``username`` as its own ``(X_i, y_i)`` bundle;
    2. :func:`sort_games_chronologically` (inside :func:`split_games`) —
       order the bundles oldest first by their Date/Time headers;
    3. :func:`split_games` — cut whole games at the 80% and 90% marks.

    Args:
        pgn_path: Path to the PGN file (read-only).
        username: The player whose moves become examples.

    Returns:
        A :class:`DatasetSplit` whose training games are the oldest 80%,
        validation the next 10%, and test the newest 10% — measured in
        whole games, never rows.
    """
    games = build_per_game_dataset(pgn_path, username)
    return split_games(games)
