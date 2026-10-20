# ChessMoveNet

A chess move-prediction neural network built from scratch with NumPy only — no
PyTorch, TensorFlow, or autograd. Given the encoding of a board position, it
learns to imitate one player's moves. Every line is kept simple, readable, and
explainable. Course project.

## How it fits together

```
781 board features  ->  256 hidden units (ReLU)  ->  4272 move-class logits
   encode.py               model.py                      moves.py
```

- **Input**: `encode.py` turns a `chess.Board` into a 781-float vector
  (piece planes, side to move, castling, en passant).
- **Output**: `moves.py` maps each UCI move to one integer label in
  `0 .. 4271` (from-square × 64 + to-square).
- **Learning**: `model.py` holds the network (`forward`, `backward`),
  `loss.py` holds the numerically stable cross-entropy loss.
- **Data**: `data.py` reads PGN games, `dataset.py` builds `(X, y)` pairs,
  `split.py` cuts them chronologically into train/val/test **by whole game**
  (oldest 80% train, next 10% val, newest 10% test — no game in two sets).
- `downloader.py` fetches games from the Chess.com API (pre-existing).

## Status

| Phase | Module | State |
|-------|--------|-------|
| PGN reading & move extraction | `data.py` | done |
| 781-feature board encoding | `encode.py` | done |
| 4,272-class move labels | `moves.py` | done |
| Per-game dataset + chronological split | `dataset.py`, `split.py` | done |
| Network: init, forward, predict, top-k | `model.py` | done |
| Stable cross-entropy loss | `loss.py` | done |
| Backpropagation (gradients, no updates) | `model.py` `backward` | done |
| Training loop, evaluation, saving | — | **next** |

The numerical gradient check (analytical vs. finite-difference gradients on a
tiny network) passes, so the forward/backward math is verified correct before
any training begins.

Known limitations, deliberate for now: promotion piece type is collapsed into
the from-to class (`e7e8q` == `e7e8`), and predictions are not filtered for
legality — combining with `moves.legal_mask` is a later phase.

## Layout

```
src/chessmovenet/   package: data, dataset, split, encode, moves, model, loss
tests/              pytest suite (150 tests)
data/raw/           my Chess.com PGN (read-only, never committed)
data/processed/     split.npz — train/val/test arrays (generated, never committed)
reports/            preparation report (generated, never committed)
progress.md         running summary of the project
```

## Commands

All commands run from the project root with the venv active and `PYTHONPATH=src`:

```bash
# Run the test suite.
PYTHONPATH=src python -m pytest tests/ -v

# Regenerate data/processed/split.npz from the raw PGN
# (~4,800 games; takes a few minutes).
PYTHONPATH=src python - <<'EOF'
from pathlib import Path
import numpy as np
from chessmovenet.split import split_dataset_chronologically

split = split_dataset_chronologically("data/raw/dag12yi.pgn", "dag12yi")
np.savez(
    "data/processed/split.npz",
    X_train=split.X_train, y_train=split.y_train,
    X_val=split.X_val, y_val=split.y_val,
    X_test=split.X_test, y_test=split.y_test,
)
for name in ("train.npz", "val.npz", "test.npz"):
    (Path("data/processed") / name).unlink(missing_ok=True)
print(split.X_train.shape, split.X_val.shape, split.X_test.shape)
EOF

# Smoke-test the math on real data (loss is finite, gradients have
# the right shapes; nothing is trained or saved).
PYTHONPATH=src python - <<'EOF'
import numpy as np
from chessmovenet.loss import cross_entropy_loss
from chessmovenet.model import ChessMoveNet

data = np.load("data/processed/split.npz")
X, y = data["X_train"][:64], data["y_train"][:64]
net = ChessMoveNet(seed=0)
print("loss:", cross_entropy_loss(net.forward(X), y))
print({k: v.shape for k, v in net.backward(X, y).items()})
EOF
```

## Data safety

`data/`, `data/processed/`, `reports/`, and `.venv` are in `.gitignore`. The
raw PGN and all processed personal game data are never committed.

