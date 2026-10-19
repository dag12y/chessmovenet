# ChessMoveNet

A NumPy neural network built from scratch (no PyTorch/TensorFlow) that learns
to imitate my chess moves. Course assignment — code is kept simple, readable,
and explainable line by line.

## Layout

```
src/chessmovenet/   package (data.py, downloader.py, prepare.py)
data/raw/dag12yi.pgn   my 4,799 Chess.com games (read-only, never committed)
data/processed/    train/val/test .npz splits (generated, never committed)
reports/           text preparation report (generated, never committed)
tests/             pytest suite
```

## Commands

All commands run from the project root with the venv active and
`PYTHONPATH=src`:

```bash
# 1. Validate the raw PGN, build examples, split chronologically,
#    and save train/val/test splits plus a report.
PYTHONPATH=src python -m chessmovenet.prepare \
  --pgn data/raw/dag12yi.pgn \
  --username dag12yi

# 2. Train only on the latest N games (validation/test games unchanged).
PYTHONPATH=src python -m chessmovenet.prepare \
  --pgn data/raw/dag12yi.pgn \
  --username dag12yi \
  --recent 5000

# 3. Run the test suite.
PYTHONPATH=src python -m pytest tests/ -v
```

## Data safety

`data/`, `data/processed/`, `reports/`, and `.venv` are in `.gitignore`. The
raw PGN and all processed personal game data are never committed.
