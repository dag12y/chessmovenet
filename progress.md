# ChessMoveNet — Completion Summary

This is a NumPy-only neural-network project for chess move prediction. It
implements the core course requirements and includes a separate, small
gradient-descent example intended for hand calculation.

## Completed core requirements

- **Dense layers:** `ChessMoveNet` performs fully connected affine transforms
  with `X @ W + b`; its production architecture is `781 -> 256 -> 4272`.
- **Activations:** ReLU is used in the hidden layer and numerically stable
  Softmax produces class probabilities.
- **Forward propagation and backpropagation:** `model.py` computes logits and
  exact cross-entropy gradients for every parameter. A central finite-
  difference gradient check verifies the implementation on a tiny network.
- **Loss:** `loss.py` provides stable multiclass cross-entropy using
  log-sum-exp.
- **Batch processing and gradient descent:** `train.py` shuffles mini-batches,
  applies `parameter -= learning_rate * gradient`, and tracks loss and top-1 /
  top-5 accuracy.
- **Requested simple demonstration:**
  `scripts/three_layer_gradient_descent.py` contains exactly three dense
  layers with one neuron each. It trains the batched toy function `y = 2x + 1`
  with mean-squared error and plain gradient descent.

## Supporting project work

- Board positions are encoded into **781** float32 features, including piece
  planes, side to move, castling rights, and en-passant file.
- Move labels cover **4,272** classes: 4,096 ordinary from-to moves plus 176
  distinct promotion moves.
- The dataset is split chronologically by complete game into train, validation,
  and held-out test partitions.
- Checkpoints and histories use safe numeric `.npz` files; training charts are
  generated in `reports/`.

## Verification and final result

```text
209 passed
```

The completed 15-epoch checkpoint (`artifacts/checkpoint_ep15.npz`) was
evaluated once against the untouched test split:

| Metric | Result |
|---|---:|
| Test positions | 16,281 |
| Cross-entropy loss | 7.196366 |
| Top-1 accuracy | 5.9210% |
| Top-5 accuracy | 10.8900% |

The reproducible command is:

```bash
PYTHONPATH=src python scripts/evaluate_checkpoint.py \
  --checkpoint artifacts/checkpoint_ep15.npz
```

## Bonus extensions

- **Adam optimizer:** implemented in `train.py`; select it with
  `--optimizer adam`.
- **Sigmoid and Tanh:** implemented in `model.py`, including their
  backpropagation derivatives; choose an activation with `--activation`.
- **Configurable hidden layers:** pass a width list such as
  `--hidden-sizes 256,128`; forward propagation, backpropagation, Adam/SGD,
  and checkpoint loading/saving support the chosen layer count.

The listed bonus extensions are implemented.
