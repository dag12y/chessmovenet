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
  `loss.py` holds the numerically stable cross-entropy loss,
  `train.py` runs the mini-batch gradient-descent loop, and
  `checkpoint.py` saves/reloads trained weights and training history.
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
| Network: init, forward, predict, backprop | `model.py` | done |
| Stable cross-entropy loss | `loss.py` | done |
| Training loop (mini-batch gradient descent) | `train.py` | done |
| Checkpoints & history (no pickle, safe reload) | `checkpoint.py` | done |
| Accuracy tracking (top-1 / top-5, raw classes) | `train.py` `evaluate_accuracy` | done |
| Required 3-layer, one-neuron gradient-descent demonstration | `scripts/three_layer_gradient_descent.py` | done |
| Final held-out test-set evaluation | `scripts/evaluate_checkpoint.py` | done |
| Bonus: Adam optimizer | `train.py` | done |
| Bonus: sigmoid/tanh, configurable layer experiments | — | optional / remaining |

The numerical gradient check (analytical vs. finite-difference gradients on a
tiny network) passes, so the forward/backward math is verified correct before
any training begins.

The completed 15-epoch checkpoint was evaluated once on the held-out test
split: loss **7.196366**, top-1 **5.9210%**, and top-5 **10.8900%** over
16,281 positions. Re-run `scripts/evaluate_checkpoint.py` to reproduce the
result from the saved checkpoint.

Known limitations, deliberate for now: promotion piece type is collapsed into
the from-to class (`e7e8q` == `e7e8`), and predictions are not filtered for
legality — combining with `moves.legal_mask` is a later phase.

## Metrics: how to read top-1 and top-5

Training reports two raw move-class accuracies per epoch, on both the
training and the validation split:

- **top-1** — the model's single highest-scoring move class is the move the
  player actually played. With 4,272 classes this is a hard test; even a strong
  model will often prefer a different (also reasonable) move.
- **top-5** — the played move appears among the five highest-scoring classes.
  This measures "was the right move at least *considered*?"

Both are computed over **all 4,272 classes with no legality filtering** — a
"hit" means the recorded label matched, *not* that the model suggested a
playable move (the top-5 set may contain illegal moves for that position).
This is intentional for Step 11: the metric is exactly `argmax`/`top-k` over
the network's raw output, easy to verify by hand.

The stricter **legal-move accuracy** — ranking only the moves allowed by the
position (via `moves.legal_mask`) — is a *different* metric and is not
silently mixed in. It requires per-position FENs, which the processed arrays
don't carry, so it would need a dataset change first; it remains a candidate
for a later step or the presentation.

Rules of thumb while reading numbers: top-5 ≥ top-1 always; random guessing
over 4,272 classes gives ≈ 0.02% top-1 and ≈ 0.12% top-5; the class-imbalanced
baseline ("always guess the most common move") sits around 1.7% top-1 on this
dataset — beating that baseline meaningfully is the first real milestone.

## Layout

```
src/chessmovenet/   package: data, dataset, split, encode, moves, model,
                    loss, train, checkpoint
tests/              pytest suite (194 tests)
artifacts/          checkpoint.npz + history.npz after training
                    (generated, never committed)
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

# Run the small course demonstration: three dense layers, one neuron in each,
# batched forward/backpropagation, mean-squared-error loss, and gradient descent.
PYTHONPATH=src python scripts/three_layer_gradient_descent.py

# Train on the real dataset (training + validation only; the test set
# is never loaded). Defaults: 3 epochs, batch 128, lr 0.001, seed 0.
# Writes artifacts/checkpoint.npz and artifacts/history.npz; refuses to
# overwrite existing outputs unless you add --overwrite.
PYTHONPATH=src python -m chessmovenet.train --epochs 3 --seed 0

# 15-epoch experiment (Step 11) -- NEW output paths, so the verified
# 3-epoch baseline files are left untouched.  If either target file
# already exists the command refuses to start (add --overwrite only
# when you mean to replace THAT file):
PYTHONPATH=src python -m chessmovenet.train \
  --epochs 15 --batch-size 128 --lr 0.001 --seed 0 \
  --checkpoint artifacts/checkpoint_ep15.npz \
  --history artifacts/history_ep15.npz

# Bonus: train with Adam. Use new output paths so baseline results are kept.
PYTHONPATH=src python -m chessmovenet.train \
  --epochs 15 --batch-size 128 --lr 0.001 --optimizer adam --seed 0 \
  --checkpoint artifacts/checkpoint_adam_ep15.npz \
  --history artifacts/history_adam_ep15.npz

# Expected output format (numbers illustrative -- nothing is pre-
# claimed about them; epoch time now includes accuracy measurement,
# roughly +40% over loss-only epochs):
#
# loaded data/processed/split.npz: train (118006, 781), val (16246, 781) (test set untouched)
# training: epochs=15 batch_size=128 lr=0.001 seed=0
# epoch 1/15  train loss 8.3xxx top1 x.x% top5 xx.x%  val loss 8.3xxx top1 x.x% top5 xx.x%  3x.xs
# ...
# epoch 15/15  train loss x.xxxx top1 x.x% top5 xx.x%  val loss x.xxxx top1 x.x% top5 xx.x%  3x.xs
# finished 15 epoch(s) in xxx.xs  |  final train loss x.xxxx top1 x.x%, val loss x.xxxx top1 x.x% top5 x.x%
# saved checkpoint -> artifacts/checkpoint_ep15.npz
# saved history    -> artifacts/history_ep15.npz
#
# Baseline reference (3-epoch run): train 8.3341 -> 8.2557 -> 8.1704,
# val 8.3035 -> 8.2293 -> 8.1402, ~22 s/epoch (loss only, no accuracy).

# Replace outputs from a previous run:
PYTHONPATH=src python -m chessmovenet.train --epochs 3 --overwrite

# Plot training history (Step 12) -- generates reports/loss_curves.png and
# reports/accuracy_curves.png from the .npz history file.  Defaults to the
# 15-epoch history produced above; pass a different path to plot another run.
PYTHONPATH=src python scripts/plot_history.py
PYTHONPATH=src python scripts/plot_history.py artifacts/history_ep15.npz

# Evaluate the selected final checkpoint on the held-out test split.
# Do not use this result to tune hyperparameters; it is the final score.
PYTHONPATH=src python scripts/evaluate_checkpoint.py \
  --checkpoint artifacts/checkpoint_ep15.npz

# How to read the graphs:
# - Loss curves: train and val loss should both trend downward.  If val loss
#   stops improving while train loss keeps dropping, the model is overfitting.
# - Accuracy curves: top-1 and top-5 are percentages (0..100).  Higher is better.
#   Top-5 > top-1 means the correct move was usually among the top few guesses
#   even when it wasn't the #1 pick.

# Reload a trained checkpoint for inference (no retraining):
PYTHONPATH=src python - <<'EOF'
import numpy as np
from chessmovenet.checkpoint import load_checkpoint, load_history

net = load_checkpoint("artifacts/checkpoint.npz")
history = load_history("artifacts/history.npz")
print("epochs:", history["epoch"])
print("val losses:", [round(v, 4) for v in history["val_loss"]])

X = np.load("data/processed/split.npz")["X_val"][:1]   # one position
label, confidence = net.predict(X[0])
print("top prediction:", label, "confidence:", confidence)
EOF

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

`data/`, `data/processed/`, `reports/`, `artifacts/`, and `.venv` are in
`.gitignore`. The raw PGN, all processed personal game data, and every
trained checkpoint/history file are never committed.

