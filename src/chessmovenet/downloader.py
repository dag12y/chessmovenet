"""Download a player's public Chess.com game history.

Uses the official Chess.com Published Data API:

- Archives:  GET https://api.chess.com/pub/player/{username}/games/archives
- Monthly:   GET https://api.chess.com/pub/player/{username}/games/{yyyy}/{mm}

Requests are sequential (no parallelism), carry a descriptive User-Agent,
and are spaced by ``REQUEST_DELAY_SECONDS`` to stay polite to the API.
No password, cookies, or API credentials are required or requested.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

API_BASE_URL = "https://api.chess.com/pub"
USER_AGENT = (
    "ChessMoveNet/0.1 "
    "(personal chess imitation learning project)"
)
REQUEST_TIMEOUT: tuple[float, float] = (10.0, 30.0)
REQUEST_DELAY_SECONDS = 1.1
MAX_RETRIES = 3
DEFAULT_DATA_DIR = Path("data") / "raw"
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class DownloadError(Exception):
    """Raised when a Chess.com API request cannot be completed."""


@dataclass
class DownloadResult:
    """Outcome of a game-history download."""

    username: str
    archives_total: int = 0
    archives_processed: int = 0
    games_downloaded: int = 0
    failed_requests: list[str] = field(default_factory=list)
    pgn: str = ""


def build_archives_url(username: str) -> str:
    """Return the URL listing a player's monthly game archives."""
    return f"{API_BASE_URL}/player/{username}/games/archives"


def month_label(archive_url: str) -> str:
    """Extract a human-readable ``yyyy/mm`` label from an archive URL."""
    parts = archive_url.rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else archive_url


def parse_archives(payload: Any) -> list[str]:
    """Extract monthly archive URLs from an archives API response."""
    if not isinstance(payload, dict):
        raise DownloadError(
            f"Unexpected archives response: expected a JSON object, "
            f"got {type(payload).__name__}"
        )
    archives = payload.get("archives")
    if not isinstance(archives, list) or not archives:
        raise DownloadError("API response contains no monthly archives")
    urls: list[str] = []
    for entry in archives:
        if not isinstance(entry, str) or not entry.startswith("http"):
            raise DownloadError(f"Unexpected archive URL in API response: {entry!r}")
        urls.append(entry)
    return urls


def extract_game_pdns(monthly_payload: Any) -> list[str]:
    """Extract the PGN of every game in a monthly games API response."""
    if not isinstance(monthly_payload, dict):
        return []
    games = monthly_payload.get("games")
    if not isinstance(games, list):
        return []
    pdns: list[str] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        pgn = game.get("pgn")
        if isinstance(pgn, str) and pgn.strip():
            pdns.append(pgn.strip())
    return pdns


def join_games(monthly_pdns: list[list[str]]) -> str:
    """Assemble per-month PGN lists into one document of separate games."""
    games = [pgn for month in monthly_pdns for pgn in month]
    return "\n\n".join(games)


def _retry_after_seconds(response: Any) -> float | None:
    """Parse the ``Retry-After`` header, if present and numeric."""
    headers = getattr(response, "headers", None)
    if not isinstance(headers, dict):
        return None
    raw = headers.get("Retry-After")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def fetch_json(
    session: requests.Session,
    url: str,
    *,
    timeout: tuple[float, float] = REQUEST_TIMEOUT,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET a JSON document, retrying transient failures and rate limiting.

    Raises :class:`DownloadError` on invalid usernames (404), client
    errors, invalid JSON, or after exhausting retries on 429/5xx and
    network problems.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=timeout
            )
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise DownloadError(f"Network error requesting {url}: {exc}") from exc
            sleep(2 ** attempt)
            continue
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise DownloadError(f"Invalid JSON response from {url}") from exc
        if response.status_code == 404:
            raise DownloadError(
                f"Not found (HTTP 404) at {url} — check the username is correct"
            )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == MAX_RETRIES - 1:
                raise DownloadError(
                    f"HTTP {response.status_code} for {url} after {MAX_RETRIES} retries"
                )
            wait = _retry_after_seconds(response)
            sleep(wait if wait is not None else 2 ** attempt)
            continue
        raise DownloadError(f"HTTP {response.status_code} for {url}")
    raise DownloadError(f"Failed to fetch {url}")  # safeguard; loop raises above


def download_player_games(
    username: str,
    *,
    session: requests.Session,
    sleep: Callable[[float], None] = time.sleep,
    delay: float = REQUEST_DELAY_SECONDS,
    progress: Callable[[str], None] = print,
) -> DownloadResult:
    """Download every available monthly archive for ``username``.

    Transient per-archive failures are recorded in ``failed_requests``
    and do not abort the whole download.
    """
    archives = parse_archives(fetch_json(session, build_archives_url(username), sleep=sleep))
    monthly_pdns: list[list[str]] = []
    failed: list[str] = []
    total = len(archives)

    for index, url in enumerate(archives, start=1):
        progress(f"[{index}/{total}] Downloading {month_label(url)} ...")
        try:
            payload = fetch_json(session, url, sleep=sleep)
            monthly_pdns.append(extract_game_pdns(payload))
        except DownloadError as exc:
            failed.append(f"{url} ({exc})")
        sleep(delay)

    return DownloadResult(
        username=username,
        archives_total=total,
        archives_processed=total - len(failed),
        games_downloaded=sum(len(month) for month in monthly_pdns),
        failed_requests=failed,
        pgn=join_games(monthly_pdns),
    )


def output_path_for(username: str, data_dir: Path | None = None) -> Path:
    """Return the destination path for a player's combined PGN file."""
    return (data_dir or DEFAULT_DATA_DIR) / f"{username}.pgn"


def save_pgn(pgn: str, path: Path) -> None:
    """Write the combined PGN document, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pgn.rstrip() + "\n", encoding="utf-8")


def confirm_overwrite(path: Path, *, input_fn: Callable[[str], str] | None = None) -> bool:
    """Ask before overwriting an existing download file."""
    if not path.exists():
        return True
    ask = input_fn or input
    answer = ask(f"{path} already exists. Overwrite? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _summarize(result: DownloadResult, path: Path) -> None:
    print(
        f"Downloaded {result.games_downloaded} game(s) from "
        f"{result.archives_processed}/{result.archives_total} monthly archive(s)."
    )
    print(f"Saved to {path}")
    if result.failed_requests:
        print("Failed requests:", file=sys.stderr)
        for failure in result.failed_requests:
            print(f"  - {failure}", file=sys.stderr)


def main() -> int:
    """Prompt for a Chess.com username and download the player's games."""
    try:
        username = input("Chess.com username: ").strip()
    except EOFError:
        print("error: no username provided", file=sys.stderr)
        return 2
    if not USERNAME_PATTERN.fullmatch(username):
        print(f"error: '{username}' is not a valid Chess.com username", file=sys.stderr)
        return 2

    path = output_path_for(username)
    if not confirm_overwrite(path):
        print(f"Kept existing file: {path}")
        return 0

    try:
        with requests.Session() as session:
            result = download_player_games(username, session=session)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not result.pgn:
        print(f"error: no games found for '{username}'", file=sys.stderr)
        return 1

    save_pgn(result.pgn, path)
    _summarize(result, path)
    return 1 if result.failed_requests else 0


if __name__ == "__main__":
    raise SystemExit(main())
