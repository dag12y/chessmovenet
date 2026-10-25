# GPU Training Workflow

ChessMoveNet is implemented in **PyTorch**. Train on a machine with an NVIDIA GPU, then copy the trained checkpoint back to your local machine to play against it.

## Two-machine setup

| Machine | Role | GPU | PyTorch install |
|---------|------|-----|-----------------|
| PC A (your local) | Play against model | Not needed | CPU build |
| PC B (GPU machine) | Train the model | NVIDIA | CUDA build |

---

## Step 1: Clone on both machines

```bash
# On both machines
git clone https://github.com/dag12y/chessmovenet.git
cd chessmovenet
git checkout gpu-training-ready
```

---

## Step 2: PC B (GPU machine) — Install CUDA PyTorch

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-api.txt pytest
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA is visible:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True` followed by your GPU name.

---

## Step 3: PC B — Prepare the dataset

If you already have `data/processed/split.npz` (e.g. copied from PC A), skip this step.

Otherwise, download your Chess.com games and build the split:

```bash
# Download your games (enter username when prompted)
.venv/bin/python -m chessmovenet.downloader

# Build train/val/test split
PYTHONPATH=src .venv/bin/python scripts/prepare_dataset.py \
    --pgn data/raw/your_name.pgn --username your_name \
    --output data/processed/split.npz --overwrite
```

---

## Step 4: PC B — Train the model

```bash
PYTHONPATH=src .venv/bin/python -m chessmovenet.train \
    --epochs 15 --batch-size 128 --lr 0.001 --seed 0 \
    --optimizer adam --activation relu --hidden-sizes 256 \
    --device auto \
    --checkpoint artifacts/checkpoint_latest.pt \
    --history artifacts/history_latest.npz
```

Key options:
- `--device auto` — uses CUDA if available, falls back to CPU
- `--epochs 15` — more epochs = better model, longer training
- `--batch-size 128` — larger batches train faster on GPU
- `--optimizer adam` — Adam optimizer (recommended)
- `--hidden-sizes 256` — or `256,128` for two hidden layers
- `--checkpoint` — where to save the trained weights (`.pt` file)
- `--history` — per-epoch training history (`.npz` file)

Training output will show per-epoch train/val loss and accuracy.

---

## Step 5: PC B — Copy the trained checkpoint to PC A

Copy only the small files (the data/ and artifacts/ directories are gitignored):

```bash
# On PC B
scp artifacts/checkpoint_latest.pt user@pc-a:~/chessmovenet/artifacts/
scp artifacts/history_latest.npz user@pc-a:~/chessmovenet/artifacts/
```

Or copy via USB drive, etc. You only need `checkpoint_latest.pt` to play.

---

## Step 6: PC A (local) — Install CPU PyTorch

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-api.txt pytest
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 7: PC A — Play against the model

### Option A: Start the API + web frontend

```bash
# Start the FastAPI server (uses the checkpoint you copied)
CHESSMOVENET_CHECKPOINT=artifacts/checkpoint_latest.pt \
  PYTHONPATH=src .venv/bin/uvicorn backend.app:app --reload

# In another terminal, start the frontend
cd frontend && npm install  # first time only
cd frontend && npm run dev
```

Open `http://127.0.0.1:5173` — you play White; the model plays Black.

### Option B: Use `make play`

```bash
make play PLAY_CHECKPOINT=artifacts/checkpoint_latest.pt
```

This starts both the API and frontend together.

### Option C: Evaluate on the test set

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_checkpoint.py \
    --checkpoint artifacts/checkpoint_latest.pt
```

---

## File summary

| File | Purpose | Size | Git-tracked? |
|------|---------|------|--------------|
| `artifacts/checkpoint_latest.pt` | Trained model weights (copy this!) | ~5–50 MB | No (gitignored) |
| `artifacts/history_latest.npz` | Per-epoch training history | Small | No (gitignored) |
| `data/processed/split.npz` | Train/val/test split | Large | No (gitignored) |
| `src/chessmovenet/model.py` | PyTorch model definition | Small | Yes |
| `src/chessmovenet/train.py` | Training loop (CPU/CUDA) | Small | Yes |
| `src/chessmovenet/checkpoint.py` | Save/load `.pt` checkpoints | Small | Yes |
| `backend/app.py` | FastAPI play service | Small | Yes |

---

## Notes

- **CPU vs GPU**: A checkpoint trained on CUDA loads fine on CPU via `map_location="cpu"` — no conversion needed.
- **`.pt` format**: Checkpoints use PyTorch's `state_dict` format, loadable with `weights_only=True` for safety.
- **Large data**: `data/` and `artifacts/` are gitignored — copy them manually between machines.
