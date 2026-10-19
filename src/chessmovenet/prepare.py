"""Dataset preparation for ChessMoveNet.

Validates a raw PGN (read-only), builds supervised examples from the user's
moves, splits games chronologically so there is no leakage between splits, and
saves train/validation/test splits plus a short text report.

Only python-chess and NumPy are used. No training happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chess
import numpy as np

from .data import my_color, read_games

logger = logging.getLogger("chessmovenet.prepare")

TEST_FRACTION = 0.10
VAL_FRACTION = 0.10
MIN_GAMES_WARN = 20


def game_id_for(game: chess.pgn.Game) -> str:
    """Return a stable id for a game: the Chess.com Link header, else a hash.

    The Link header is unique per Chess.com game. When it is absent we fall
    back to a SHA-1 of the player headers, dates, result, and move list so
    duplicate detection still works.
    """
    link = (game.headers.get("Link") or "").strip()
    if link:
        return link
    digest = hashlib.sha1()
    for key in ("White", "Black", "UTCDate", "UTCTime", "Result"):
        digest.update(str(game.headers.get(key, "")).encode("utf-8"))
    moves = "|".join(m.uci() for m in game.mainline_moves())
    digest.update(moves.encode("utf-8"))
    return "sha1:" + digest.hexdigest()[:16]


def parse_datetime(game: chess.pgn.Game) -> Optional[datetime]:
    """Parse UTCDate (+ UTCTime) headers into a timezone-aware UTC datetime.

    Chess.com PGNs use ``YYYY.MM.DD`` for the date and ``HH:MM:SS`` for the
    time. Returns ``None`` when the date is missing or unparseable.
    """
    date_s = (game.headers.get("UTCDate") or "").strip()
    time_s = (game.headers.get("UTCTime") or "").strip()
    if not date_s:
        return None
    text = date_s + " " + time_s if time_s else date_s
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def is_standard_variant(game: chess.pgn.Game) -> bool:
    """Return True only for standard chess.

    Standard chess games have no ``Variant`` header (or an empty one).
    Chess960, King of the Hill, etc. have a non-empty Variant header and are
    skipped because the network learns standard-chess move patterns.
    """
    return (game.headers.get("Variant") or "").strip() == ""
def validate_pgn(path, username):
    """Validate the PGN and collect lightweight records for valid games.

    The raw PGN file is only read, never modified. Each yielded game is
    classified into exactly one bucket by priority:
    incomplete (no moves) > non-standard variant > not our identity >
    duplicate > valid.

    Returns ``(records, stats, summary)`` where ``records`` hold everything
    needed to build examples (so full python-chess Game objects do not stay in
    memory), ``stats`` holds classification counts, and ``summary`` holds
    date/rating/time-control/result aggregates.
    """
    path = Path(path)

    # Cheap pre-pass: count how many games the file declares via [Event tags.
    total_declared = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("[Event"):
                total_declared += 1

    records = []
    seen_ids: set[str] = set()
    stats = {
        "total_declared": total_declared,
        "yielded": 0,
        "parse_errors_or_truncated": 0,
        "invalid_no_moves": 0,
        "non_standard_variant": 0,
        "not_our_identity": 0,
        "duplicate": 0,
        "valid": 0,
    }
    time_controls: Counter[str] = Counter()
    results: Counter[str] = Counter()
    dates: list[datetime] = []
    my_ratings: list[tuple[datetime, int]] = []

    for game in read_games(path):
        stats["yielded"] += 1

        # 1. The game must involve us; otherwise there is nothing to imitate.
        color = my_color(game, username)
        if color is None:
            stats["not_our_identity"] += 1
            continue

        # 2. Standard chess only.
        if not is_standard_variant(game):
            stats["non_standard_variant"] += 1
            continue

        moves = list(game.mainline_moves())
        if not moves:
            stats["invalid_no_moves"] += 1
            continue

        # 3. Deduplicate by game id.
        gid = game_id_for(game)
        if gid in seen_ids:
            stats["duplicate"] += 1
            continue
        seen_ids.add(gid)

        # Valid game.
        stats["valid"] += 1

        result = (game.headers.get("Result") or "?").strip()
        time_control = (game.headers.get("TimeControl") or "?").strip()
        results[result] += 1
        time_controls[time_control] += 1

        dt = parse_datetime(game)
        if dt is not None:
            dates.append(dt)

        my_elo: Optional[int] = None
        elo_header = "WhiteElo" if color == chess.WHITE else "BlackElo"
        elo_s = (game.headers.get(elo_header) or "").strip()
        if elo_s:
            try:
                my_elo = int(elo_s)
            except ValueError:
                my_elo = None
        if my_elo is not None and dt is not None:
            my_ratings.append((dt, my_elo))

        records.append(
            {
                "game_id": gid,
                "date": dt.isoformat() if dt else "",
                "result": result,
                "time_control": time_control,
                "color": color,
                "initial_fen": game.board().fen(),
                "moves": [m.uci() for m in moves],
            }
        )

    stats["parse_errors_or_truncated"] = total_declared - stats["yielded"]

    summary = {
        "date_range": (min(dates), max(dates)) if dates else (None, None),
        "time_controls": dict(time_controls.most_common()),
        "results": dict(results.most_common()),
        "rating_min": min((r for _, r in my_ratings), default=None),
        "rating_max": max((r for _, r in my_ratings), default=None),
        "rating_first": my_ratings[0][1] if my_ratings else None,
        "rating_last": my_ratings[-1][1] if my_ratings else None,
        "n_my_ratings": len(my_ratings),
    }
    return records, stats, summary

def split_records(records, recent_train: Optional[int] = None):
    """Split GAMES chronologically: oldest 80% train, next 10% val, newest 10% test.

    Splitting by whole game (never by example) guarantees no game appears in
    more than one split. Sorting by UTC date+time makes the split
    chronological, so the model never trains on the future and tests on the
    past.

    If ``recent_train`` is given, training uses only the latest N games from
    the train portion; validation and test are unchanged. This mimics training
    on the most recent form while keeping fixed val/test sets.

    Returns ``(train_records, val_records, test_records)``.
    """
    ordered = sorted(records, key=lambda r: r["date"])
    n = len(ordered)
    if n == 0:
        return [], [], []
    if n < 3:
        logger.warning("Only %d game(s); cannot make three non-empty splits", n)
        return ordered, [], []

    n_test = max(1, int(round(TEST_FRACTION * n)))
    n_val = max(1, int(round(VAL_FRACTION * n)))
    n_train = n - n_val - n_test
    if n_train < 1:
        logger.warning("Too few games (%d) for an 80/10/10 split", n)
        return ordered, [], []

    if n < MIN_GAMES_WARN:
        logger.warning(
            "Only %d games; train/validation/test splits may be too small", n
        )

    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]

    if recent_train is not None:
        if recent_train < 1:
            raise ValueError("--recent must be a positive integer")
        if recent_train > len(train):
            logger.warning(
                "--recent %d exceeds train games %d; using all train games",
                recent_train,
                len(train),
            )
            recent_train = len(train)
        train = train[-recent_train:]

    return train, val, test


def build_examples(records):
    """Build one supervised example per user move from valid game records.

    Each example is a dict: fen (position before the move), uci (the move),
    color (my color as 1=White/0=Black), game_id, date, result, time_control.

    The position is replayed move by move. On the user's turns the FEN is
    snapshotted *before* the move is pushed, so the input to the network is
    the position the user actually saw when deciding. Every recorded move is
    checked for legality; if an illegal move is met, the rest of that game is
    dropped because later positions would no longer be trustworthy. Only the
    user's moves are recorded -- never the opponent's.
    """
    examples = []
    for rec in records:
        board = chess.Board(rec["initial_fen"])
        color = rec["color"]
        for uci in rec["moves"]:
            move = chess.Move.from_uci(uci)
            if not board.is_legal(move):
                logger.warning(
                    "Illegal move %s in game %s; stopping this game",
                    uci,
                    rec["game_id"],
                )
                break
            if board.turn == color:
                examples.append(
                    {
                        "fen": board.fen(),
                        "uci": uci,
                        "color": int(color),
                        "game_id": rec["game_id"],
                        "date": rec["date"],
                        "result": rec["result"],
                        "time_control": rec["time_control"],
                    }
                )
            board.push(move)
    return examples


def save_split(examples, out_path):
    """Save one split as a NumPy ``.npz`` file (object arrays for strings)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not examples:
        empty = np.array([], dtype=object)
        np.savez(
            out_path,
            fen=empty.copy(),
            uci=empty.copy(),
            color=np.array([], dtype=np.int8),
            game_id=empty.copy(),
            date=empty.copy(),
            result=empty.copy(),
            time_control=empty.copy(),
        )
        logger.warning("Saved empty split to %s (0 examples)", out_path)
        return 0
    np.savez(
        out_path,
        fen=np.array([e["fen"] for e in examples], dtype=object),
        uci=np.array([e["uci"] for e in examples], dtype=object),
        color=np.array([e["color"] for e in examples], dtype=np.int8),
        game_id=np.array([e["game_id"] for e in examples], dtype=object),
        date=np.array([e["date"] for e in examples], dtype=object),
        result=np.array([e["result"] for e in examples], dtype=object),
        time_control=np.array(
            [e["time_control"] for e in examples], dtype=object
        ),
    )
    logger.info("Saved %d examples to %s", len(examples), out_path)
    return len(examples)

def write_report(report_path, stats, summary, splits_info, username, pgn_path):
    """Write a short human-readable text report and return its text."""
    lines = []
    lines.append("ChessMoveNet dataset preparation report")
    lines.append("=" * 45)
    lines.append(f"PGN: {pgn_path}")
    lines.append(f"Username: {username}")
    lines.append("")
    lines.append("Validation (raw PGN is never modified):")
    lines.append(f"  total games declared ([Event]): {stats['total_declared']}")
    lines.append(f"  games yielded / parsed OK:      {stats['yielded']}")
    lines.append(
        f"  parse errors / truncated:       {stats['parse_errors_or_truncated']}"
    )
    lines.append(f"  invalid (no moves):             {stats['invalid_no_moves']}")
    lines.append(
        f"  non-standard variants skipped:  {stats['non_standard_variant']}"
    )
    lines.append(f"  not our identity:               {stats['not_our_identity']}")
    lines.append(f"  duplicates removed:             {stats['duplicate']}")
    lines.append(f"  valid games:                    {stats['valid']}")
    lines.append("")
    lines.append("Examples (one per user move):")
    for name in ("train", "val", "test"):
        n_rec, n_ex = splits_info[name]
        lines.append(f"  {name:5s}: {n_rec:5d} games, {n_ex:7d} examples")
    lines.append("")
    lines.append(
        "Chronological split (oldest 80% train, next 10% val, newest 10% test):"
    )
    dr = summary["date_range"]
    if dr[0] is not None:
        lines.append(f"  date range: {dr[0].date()} -> {dr[1].date()}")
    lines.append("")
    lines.append("Time controls:")
    for key, value in summary["time_controls"].items():
        lines.append(f"  {key}: {value}")
    lines.append("Results:")
    for key, value in summary["results"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append(
        f"My rating over time ({summary['n_my_ratings']} games with my Elo):"
    )
    lines.append(f"  min: {summary['rating_min']}   max: {summary['rating_max']}")
    lines.append(
        f"  first: {summary['rating_first']}   last: {summary['rating_last']}"
    )
    lines.append("")
    lines.append("Files:")
    for name in ("train", "val", "test"):
        lines.append(f"  data/processed/{name}.npz")
    lines.append(f"  {report_path}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare the ChessMoveNet dataset from a raw PGN."
    )
    parser.add_argument("--pgn", required=True, help="Path to the raw PGN file")
    parser.add_argument(
        "--username",
        required=True,
        help="Your Chess.com username (case-insensitive)",
    )
    parser.add_argument(
        "--out", default="data/processed", help="Output directory for the splits"
    )
    parser.add_argument(
        "--report",
        default="reports/prepare_report.txt",
        help="Path for the text report",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=None,
        help="Train only on the latest N games (validation/test unchanged)",
    )
    return parser


def main(argv=None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
    )
    args = build_arg_parser().parse_args(argv)

    pgn_path = Path(args.pgn)
    if not pgn_path.exists():
        logger.error("PGN not found: %s", pgn_path)
        return 2

    records, stats, summary = validate_pgn(pgn_path, args.username)
    logger.info(
        "Validation complete: %d valid games of %d yielded",
        stats["valid"],
        stats["yielded"],
    )

    train_rec, val_rec, test_rec = split_records(
        records, recent_train=args.recent
    )
    splits_rec = {"train": train_rec, "val": val_rec, "test": test_rec}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    splits_info = {}
    for name in ("train", "val", "test"):
        examples = build_examples(splits_rec[name])
        splits_info[name] = (
            len(splits_rec[name]),
            save_split(examples, out_dir / f"{name}.npz"),
        )

    report = write_report(
        report_path, stats, summary, splits_info, args.username, str(pgn_path)
    )
    report_path.write_text(report, encoding="utf-8")
    print(report)
    logger.info("Report written to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

