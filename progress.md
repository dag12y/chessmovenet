# ChessMoveNet — Progress Summary

A NumPy neural network built from scratch (no PyTorch/TensorFlow) that learns
to imitate the user's chess moves. Code is kept simple, readable, and
explainable line by line. This document summarizes what has been built so far.

## Project layout

```
src/chessmovenet/
    __init__.py      (empty package marker)
    downloader.py    (pre-existing: downloads games from Chess.com API)
    data.py          (PGN reading + move-example extraction)
    prepare.py       (validation, dataset prep, chronological split, CLI)
    encode.py        (FEN -> fixed-length vector)
    moves.py         (UCI move <-> class index, legal move mask)
    network.py       (two-layer network: init, forward, loss)
tests/
    test_data.py     (6 tests)
    test_encode.py   (6 tests)
    test_moves.py    (7 tests)
    test_network.py  (8 tests)
data/raw/dag12yi.pgn     (4,799 Chess.com games, read-only, never committed)
data/processed/          (train/val/test .npz splits, generated, never committed)
reports/                 (text preparation report, generated, never committed)
progress.md              (this file)
```

All `.venv/`, `data/`, `data/processed/`, `reports/`, `__pycache__/`, and
`.pytest_cache/` are gitignored so no personal game data is ever committed.

## Phase 1 — Data reading (`data.py`)

**Functions:**
- `read_games(path)` — lazy generator yielding one `chess.pgn.Game` at a time
  from a PGN file; never loads all games into memory.
- `my_color(game, username)` — returns `chess.WHITE` or `chess.BLACK` by
  case-insensitive matching of the `White`/`Black` PGN headers; `None` if the
  user is not a player.
- `extract_examples(game, username="dag12yi")` — replays the game and returns
  `(fen_before_move, uci_move)` pairs **only for the user's moves**. The FEN is
  snapshotted *before* pushing the move, so the input is the position the user
  actually saw when deciding.

**Key design decision:** FEN before the move because supervised imitation
learning needs (position → move). Taking the FEN after the move would teach the
network on a position that already reflects the move's consequences.

**Tests (tests/test_data.py):** White/Black identification (case-insensitive),
not-a-player returns None, only my moves returned, FEN is from before the move,
non-player games yield empty list.


## Phase 2 — Data validation & preparation (`prepare.py`)

**Validation (`validate_pgn`):** reads the raw PGN read-only (never modifies
it) and classifies every game into exactly one bucket by priority:
incomplete (no moves) → non-standard variant → not our identity → duplicate →
valid. Deduplication uses the Chess.com `Link` header, falling back to a SHA-1
hash of headers + moves. Non-standard variants (Chess960, King of the Hill) are
skipped. Collects date range, time controls, results, and the user's rating over
time.

**Splitting (`split_records`):** splits **by whole game, chronologically**
(oldest 80% train, next 10% val, newest 10% test), sorted by UTC date + time.
No game appears in more than one split, so there is no data leakage. Optional
`--recent N` trains only on the latest N games from the train portion while
keeping val/test unchanged; warns if there are too few games.

**Example building (`build_examples`):** one supervised example per user move
with `fen`, `uci`, `color`, `game_id`, `date`, `result`, `time_control`. Every
recorded move is checked for legality; if an illegal move is met the rest of
that game is dropped. Only the user's moves are recorded.

**Saving (`save_split`):** each split saved as `data/processed/{split}.npz`
(object arrays for strings, int8 for color).

**Report (`write_report`):** short text report to `reports/prepare_report.txt`
with validation counts, examples per split, date range, time controls, results,
rating range, and file list.

**CLI:** `python -m chessmovenet.prepare --pgn ... --username ...` with
logging, type hints, clear errors, and optional `--recent`.

**Real-data run results:**
- 4,799 games declared / 4,799 parsed OK / 0 parse errors
- 6 invalid (no moves), 9 non-standard variants skipped, 4,784 valid games
- 0 games where identity could not be established, 0 duplicates
- Examples: train 3,828 games / 117,969 examples; val 478 / 16,177; test 478 /
  16,223 (total 150,369)
- Date range: 2024-03-16 → 2026-10-16
- Rating: min 0, max 1,691; first 591, last 1,665
- Verified: 0 game_ids appear in multiple splits (leakage-free)

**Tests (tests/test_prepare.py):** White/Black ID, legal-move extraction,
invalid-game handling, duplicate removal, variant skipping, leakage-free
chronological split, `--recent` train mode.


## Phase 3 — FEN encoding (`encode.py`)

- `INPUT_SIZE = 773` named constant.
- `fen_to_vector(fen)` → float32 array of shape (773,):
  - Indices 0..767: 12 planes × 64 squares. White piece type `t` → plane
    `t-1` (0..5); black → plane `6+(t-1)` (6..11). Index = plane×64 + square.
  - Index 768: 1.0 if White to move, else 0.0.
  - Indices 769..772: castling rights (White kingside, White queenside, Black
    kingside, Black queenside).
- `encode_fens(fens)` → float32 array of shape (N, 773) via a simple loop.

**Tests (tests/test_encode.py):** shape (773,) and dtype float32; starting
position has exactly 32 ones; white pawn on e2 sets index `0*64+E2`; black king
on e8 sets index `11*64+E8`; turn flag is 1.0 at start and 0.0 after 1. e4;
all four castling flags are 1.0 at start, white ones 0.0 when white loses
castling rights; `encode_fens` on 3 FENs returns shape (3, 773).

**Real-data check:** first 1,000 training FENs encode to shape (1000, 773);
average ~22.75 ones per row in the piece planes (drops from 32 as pieces are
captured).


## Phase 4 — Move encoding (`moves.py`)

- `OUTPUT_SIZE = 4096` constant (64 from-squares × 64 to-squares).
- `move_to_index(uci)` → `from_square * 64 + to_square`; promotion suffix
  ignored (`e7e8q` == `e7e8`).
- `index_to_move(index)` → "from-to" UCI string (no promotion).
- `legal_mask(fen)` → float32 array (4096,): 1.0 for each legal from-to move.
- `encode_moves(ucis)` → int array (N,) of class indices.

**Tests (tests/test_moves.py):** `d7d5 -> 3299`; `e2e4` round-trips; promotion
suffix ignored; start position's legal_mask has exactly 20 ones; the played
move's index is set in legal_mask for the first 1,000 train examples; encode
shape/type correct.

**Real-data class distribution (train):**
- Promotions: train 448 (0.38%), val 53 (0.33%), test 66 (0.41%) — a known
  limitation since promotion piece type is collapsed into the from-to class.
- Top 10 classes: e7e5, e2e4, g1f3, b8c6, g8f6, b1c3, f1c4, d2d3, d7d6, e1g1
  (each ~0.7–1.65% of train).
- Coverage: top 10 → 12.84%, top 50 → 29.46%, top 200 → 56.58% of train
  examples.

## Phase 5 — Neural network (`network.py`)

A tiny two-layer fully connected network, NumPy only. No backpropagation yet.

- `init_params(input_size=773, hidden_size=256, output_size=4096, seed=0)`
  → dict with `W1, b1, W2, b2`. Weights: `normal * sqrt(2/fan_in)` (He init),
  biases: zeros, all float32, using `np.random.default_rng(seed)`.
- `relu(z)` → `max(0, z)` element-wise.
- `softmax(z)` → row-wise, subtracts row max first for numerical stability.
- `forward(params, X)` → `(P, cache)`; cache holds `X, Z1, A1, Z2` for later
  backpropagation.
- `cross_entropy(P, y)` → `-mean(log(P[range(N), y] + 1e-12))` for int class
  targets.

**Tests (tests/test_network.py):** parameter shapes/dtypes, zero biases, forward
output shapes, probability rows sum to 1, softmax invariance to `z + 1000`
(no overflow), ReLU zeroes negatives, cross-entropy ≈ 0 on certain prediction,
initial loss on 1,000 real examples close to `ln(4096) = 8.318` (within 0.5;
actual 8.3324).

**Forward-pass shapes (5 examples):** X (5,773), Z1 (5,256), A1 (5,256), Z2
(5,4096), P (5,4096); P row sums = 1.

**Initial loss (1,000 train examples):** 8.3324 vs ln(4096) = 8.3178 — exactly
what is expected for a randomly initialized network (uniform over 4,096 classes).

---

## Test suite status

```
40 passed in 0.73s
  test_data.py:      6 passed
  test_encode.py:    6 passed
  test_moves.py:     7 passed
  test_network.py:   8 passed
  test_prepare.py:   13 passed
```

All tests use tiny hand-written PGNs or temp files — none touch the real
personal PGN except the intentional real-data checks.

---

## Next steps (not yet implemented)

- Backpropagation (`backward(params, cache, dP)`) and gradient descent update
  (`update(params, grads, lr)`).
- Training loop over train/val splits with accuracy tracking.
- Move prediction: `argmax` over the 4,096 output classes, masked by
  `legal_mask` so illegal moves are never suggested at inference.
- Promotion handling: since `move_to_index` drops the promotion suffix, the
  network cannot yet distinguish `e7e8q` from `e7e8r`; a follow-up could expand
  the output space or add a second head for promotion piece type.
- Integration: load `data/processed/train.npz`, encode FENs with
  `encode_fens`, encode moves with `encode_moves`, train, and evaluate on val/
  test.

