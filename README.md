# ChessMoveNet

ChessMoveNet is a chess move-prediction neural network built with **PyTorch**. Given a chess position, it learns to predict the move a player is likely to make. The forward pass, loss, backpropagation, and parameter updates are handled by PyTorch autograd, and training runs on CPU or NVIDIA CUDA GPU.

The repository also contains a FastAPI backend and React/Vite interface for playing against a trained checkpoint locally.

## How it works

```text
PGN games → board/move examples → numeric encoding → neural-network training → predicted move
```

The default classifier architecture is:

```text
781 board features → 256 ReLU hidden units → 4,272 move logits → softmax probabilities
```

| Component | Implementation |
| --- | --- |
| Input | 781 `float32` features: 12 piece planes (768 values), side to move, castling rights, and en-passant file |
| Output | 4,272 move classes: 4,096 from/to combinations plus 176 promotion moves |
| Forward pass | PyTorch dense layers with autograd; runs on CPU or CUDA |
| Loss | Numerically stable multiclass cross-entropy (`F.cross_entropy`) |
| Learning | Mini-batch SGD or Adam via PyTorch optimizers |
| Data split | Chronological 80% / 10% / 10% train/validation/test split by complete game |
| Verification | Automated tests, safe `.pt` checkpoints with `weights_only` loading |

## Neural-network math

For a mini-batch `X`, the network computes (via PyTorch `nn.Linear` layers):

```text
Z₁ = XW₁ + b₁
A₁ = ReLU(Z₁)
logits = A₁W₂ + b₂
```

Cross-entropy measures the probability given to the move that was actually played. PyTorch autograd computes gradients via backpropagation, and the optimizer updates every parameter:

```text
parameter ← parameter − learning_rate × gradient
```

The model supports configurable hidden-layer widths and `relu`, `sigmoid`, or `tanh` activations. The default is a single 256-unit ReLU hidden layer.

## Repository layout

```text
src/chessmovenet/
  data.py          Read chess games from PGN files
  downloader.py    Fetch Chess.com game data
  dataset.py       Build board / move-label examples
  split.py         Chronological, whole-game data splitting
  encode.py        Convert a chess.Board to 781 features
  moves.py         Convert moves to and from 4,272 class labels
  model.py         Network, forward propagation, and backpropagation
  loss.py          Stable cross-entropy
  train.py         Mini-batches, SGD/Adam, and evaluation
  checkpoint.py    Save and load models/history
backend/            FastAPI game service
frontend/           React/Vite chess interface
scripts/            Evaluation, plots, and gradient-descent demonstration
tests/              Automated tests
```

## Setup and testing

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-api.txt pytest
```

On the machine with the NVIDIA GPU, install the CUDA build of PyTorch instead:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Run the test suite from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

## Short commands

After setup, the easiest way to use the project is with `make`:

```bash
make help       # List all available commands
make test       # Run tests
make demo       # Run the small gradient-descent example
make download   # Download public Chess.com games
make prepare USERNAME=your_name  # Build a dataset from the downloaded PGN
make train      # Train for 3 epochs
make plot       # Create training charts
make evaluate   # Evaluate the latest checkpoint on the test set
```

To change the training length, pass `EPOCHS`. For example:

```bash
make train EPOCHS=15
```

`make train` saves to `artifacts/checkpoint_latest.npz` and
`artifacts/history_latest.npz`. If those files already exist, choose new names
instead of accidentally replacing a previous result:

```bash
make train EPOCHS=15 \
  CHECKPOINT=artifacts/checkpoint_ep15_new.npz \
  HISTORY=artifacts/history_ep15_new.npz
```

## Update with new Chess.com games

To train on more recent public games, download a fresh PGN and rebuild the
processed split before training:

```bash
# Enter the Chess.com username when prompted.
make download

# Rebuild data/processed/split.npz from data/raw/your_name.pgn.
make prepare USERNAME=your_name

# Train the new dataset.
make train
```

`make download` uses Chess.com’s public API and stores the PGN in
`data/raw/<username>.pgn`. It asks before replacing an existing PGN.
`make prepare` deliberately replaces the current processed split, because it
represents the newly selected training data. Both the raw PGN and processed
arrays are ignored by Git.

Run the small three-layer gradient-descent demonstration. It learns `y = 2x + 1` with batched forward passes, backpropagation, mean-squared error, and plain gradient descent:

```bash
PYTHONPATH=src .venv/bin/python scripts/three_layer_gradient_descent.py
```

## Training and evaluation

Training expects a processed split at `data/processed/split.npz`. Data and trained-model artifacts are ignored by Git because they can be large or personal.

```bash
PYTHONPATH=src .venv/bin/python -m chessmovenet.train \
  --epochs 15 --batch-size 128 --lr 0.001 --seed 0 \
  --checkpoint artifacts/checkpoint_ep15.npz \
  --history artifacts/history_ep15.npz
```

Useful options:

```bash
--optimizer adam          # Adam optimizer
--activation tanh         # Select hidden activation
--hidden-sizes 256,128    # Use two hidden layers
```

Evaluate a selected checkpoint once on the held-out test set:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint artifacts/checkpoint_ep15.npz
```

## Results

One completed 15-epoch checkpoint was evaluated on the held-out test split:

| Metric | Result |
| --- | ---: |
| Test positions | 16,281 |
| Cross-entropy loss | 7.196366 |
| Top-1 accuracy | 5.9210% |
| Top-5 accuracy | 10.8900% |

Top-1 means the highest-scoring class matches the recorded move. Top-5 means the recorded move appears among the five highest-scoring classes. These are raw classification metrics across all 4,272 output classes, without legality filtering.

## Play locally

Install frontend packages once:

```bash
cd frontend
npm install
```

Start the API from the repository root. A checkpoint must be available; the default location is `artifacts/checkpoint_adam_ep6.npz`.

```bash
PYTHONPATH=src .venv/bin/uvicorn backend.app:app --reload
```

In another terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open the local Vite URL (normally `http://127.0.0.1:5173`). You play White; the backend validates your move, scores the board, and selects the highest-scoring **legal** reply. To select another model file, set `CHESSMOVENET_CHECKPOINT` before starting the API.

## Limitations and next steps

- Raw evaluation ranks all 4,272 move classes, including moves illegal in a particular position; the game app filters to legal moves before playing.
- Chess move prediction is a difficult, high-class-count task; this compact dense model cannot represent every strategic pattern.
- Natural next steps include richer board features, legality-aware training/evaluation, more data, and convolutional or attention-based architectures.

## Technologies

Python, PyTorch, NumPy, python-chess, FastAPI, Uvicorn, React, Vite, and pytest.
