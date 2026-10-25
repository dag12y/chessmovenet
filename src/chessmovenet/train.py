"""Mini-batch training with PyTorch on CPU or NVIDIA GPU."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from chessmovenet.checkpoint import save_checkpoint, save_history
from chessmovenet.loss import cross_entropy_loss
from chessmovenet.model import ChessMoveNet


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Pick CUDA when available, otherwise CPU. ``cuda`` fails if no GPU."""
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}")
    chosen = torch.device(device)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no NVIDIA GPU is visible to PyTorch. "
            "Install a CUDA build of torch on the GPU machine, or pass --device cpu."
        )
    return chosen


def _check_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"expected {name} to be an integer, got {type(value).__name__}"
        )
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _check_learning_rate(lr) -> float:
    if isinstance(lr, bool) or not isinstance(
        lr, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"expected lr to be a number, got {type(lr).__name__}")
    lr = float(lr)
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"lr must be a finite positive number, got {lr}")
    return lr


def _check_unit_interval(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"expected {name} to be a number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1), got {value}")
    return value


def _check_epsilon(value) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"expected adam_epsilon to be a number, got {type(value).__name__}"
        )
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"adam_epsilon must be finite and positive, got {value}")
    return value


def _check_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(
            f"expected seed to be an integer, got {type(seed).__name__}"
        )
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return seed


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden sizes must be comma-separated integers, e.g. 256,128"
        ) from exc
    if not widths or any(width < 1 for width in widths):
        raise argparse.ArgumentTypeError(
            "hidden sizes must be positive integers, e.g. 256,128"
        )
    return widths


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int = 128, seed: int = 0
):
    """Yield reproducibly shuffled mini-batches of ``(X_batch, y_batch)``."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(
            f"expected X to be 2-D (N, features), but got "
            f"{X.ndim} dimension(s) with shape {X.shape}"
        )
    if y.ndim != 1:
        raise ValueError(f"expected y to be 1-D (N,), but got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of examples, got "
            f"{X.shape[0]} and {y.shape[0]}"
        )
    if X.shape[0] == 0:
        raise ValueError("cannot iterate over an empty dataset")
    batch_size = _check_positive_int(batch_size, "batch_size")
    seed = _check_seed(seed)

    n = X.shape[0]
    order = np.random.default_rng(seed).permutation(n)

    def batches():
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield X[idx], y[idx]

    return batches()


def _validated_gradients(
    net: ChessMoveNet, grads: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if not isinstance(grads, dict):
        raise TypeError(
            f"expected grads to be a dict, got {type(grads).__name__}"
        )
    parameter_names = net.parameter_names
    if set(grads) != set(parameter_names):
        raise ValueError(
            f"grads must have exactly the keys {parameter_names}, "
            f"got {tuple(sorted(grads))}"
        )

    checked: dict[str, np.ndarray] = {}
    for name in parameter_names:
        gradient = np.asarray(grads[name])
        parameter = getattr(net, name)
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"gradient {name} has shape {gradient.shape}, but the "
                f"parameter has shape {parameter.shape}"
            )
        if not np.issubdtype(gradient.dtype, np.number):
            raise TypeError(
                f"gradient {name} must contain numbers, got dtype "
                f"{gradient.dtype}"
            )
        if not np.isfinite(gradient).all():
            raise ValueError(f"gradient {name} must be finite")
        checked[name] = gradient.astype(np.asarray(parameter).dtype, copy=False)
    return checked


class AdamOptimizer:
    """Adam optimizer with per-parameter first and second moments."""

    def __init__(
        self,
        net: ChessMoveNet,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.lr = _check_learning_rate(lr)
        self.beta1 = _check_unit_interval(beta1, "adam_beta1")
        self.beta2 = _check_unit_interval(beta2, "adam_beta2")
        self.epsilon = _check_epsilon(epsilon)
        self.steps = 0
        self.m = {
            name: np.zeros_like(np.asarray(getattr(net, name)))
            for name in net.parameter_names
        }
        self.v = {
            name: np.zeros_like(np.asarray(getattr(net, name)))
            for name in net.parameter_names
        }

    def step(self, net: ChessMoveNet, grads: dict[str, np.ndarray]) -> None:
        grads = _validated_gradients(net, grads)
        self.steps += 1
        bias1 = 1.0 - self.beta1**self.steps
        bias2 = 1.0 - self.beta2**self.steps

        for name in net.parameter_names:
            gradient = grads[name]
            self.m[name] *= self.beta1
            self.m[name] += (1.0 - self.beta1) * gradient
            self.v[name] *= self.beta2
            self.v[name] += (1.0 - self.beta2) * np.square(gradient)

            m_hat = self.m[name] / bias1
            v_hat = self.v[name] / bias2
            getattr(net, name)[:] -= self.lr * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )


def update_parameters(
    net: ChessMoveNet, grads: dict[str, np.ndarray], lr
) -> None:
    """Apply one NumPy gradient-descent step (used by tests)."""
    lr = _check_learning_rate(lr)
    grads = _validated_gradients(net, grads)
    for name in net.parameter_names:
        getattr(net, name)[:] -= lr * grads[name]


def _features_and_labels(X, y) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError(
            f"expected 2-D X and 1-D y, got {X.ndim}-D and {y.ndim}-D"
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]}"
        )
    if X.shape[0] == 0:
        raise ValueError("cannot train or evaluate on 0 rows")
    return X, y


def evaluate_loss(
    net: ChessMoveNet,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
) -> float:
    """Average loss of ``net`` on ``(X, y)`` -- forward pass only."""
    total_loss = 0.0
    total_rows = 0
    for X_batch, y_batch in iterate_minibatches(
        X, y, batch_size=batch_size, seed=0
    ):
        loss = cross_entropy_loss(net.forward(X_batch), y_batch)
        total_loss += loss * X_batch.shape[0]
        total_rows += X_batch.shape[0]
    return total_loss / total_rows


def evaluate_accuracy(
    net,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
) -> dict[str, float]:
    """Top-1 and top-5 accuracy of ``net`` on ``(X, y)`` -- read-only."""
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError(f"y must contain integer labels, got dtype {y.dtype}")
    n_classes = int(net.output_size)
    if n_classes < 1:
        raise ValueError(
            f"net.output_size must be positive, got {n_classes}"
        )
    if y.size and (int(y.min()) < 0 or int(y.max()) >= n_classes):
        raise ValueError(
            f"labels must be in 0 .. {n_classes - 1}, "
            f"got a value outside this range"
        )
    k5 = min(5, n_classes)

    rows = 0
    top1_hits = 0
    top5_hits = 0

    was_training = isinstance(net, torch.nn.Module) and net.training
    if isinstance(net, torch.nn.Module):
        net.eval()
    try:
        for X_batch, y_batch in iterate_minibatches(
            X, y, batch_size=batch_size, seed=0
        ):
            logits = net.forward(X_batch)
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError(
                    "logits contain NaN or infinite values; the network's "
                    "weights may have diverged"
                )

            best = np.argmax(logits, axis=-1)
            top1_hits += int(np.count_nonzero(best == y_batch))
            top5_idx = np.argsort(logits, axis=-1)[:, -k5:]
            in_top5 = np.any(top5_idx == y_batch[:, np.newaxis], axis=-1)
            top5_hits += int(np.count_nonzero(in_top5))
            rows += y_batch.shape[0]
    finally:
        if was_training:
            net.train()

    return {"top1": 100.0 * top1_hits / rows, "top5": 100.0 * top5_hits / rows}


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    features = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    labels = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
    dataset = TensorDataset(features, labels)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=pin_memory,
        drop_last=False,
    )


def train(
    net: ChessMoveNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 0.001,
    seed: int = 0,
    optimizer: str = "sgd",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    verbose: bool = True,
    device: str | torch.device | None = "cpu",
) -> dict[str, list]:
    """Train ``net`` with mini-batch SGD or Adam on ``device``."""
    epochs = _check_positive_int(epochs, "epochs")
    batch_size = _check_positive_int(batch_size, "batch_size")
    lr = _check_learning_rate(lr)
    seed = _check_seed(seed)
    if not isinstance(optimizer, str):
        raise TypeError(
            f"optimizer must be a string, got {type(optimizer).__name__}"
        )
    optimizer = optimizer.lower()
    if optimizer not in {"sgd", "adam"}:
        raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer!r}")

    for tag, X, y in (
        ("X_train", X_train, y_train),
        ("X_val", X_val, y_val),
    ):
        try:
            _features_and_labels(X, y)
        except ValueError as exc:
            raise ValueError(f"{tag}: {exc}") from exc

    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)

    torch_device = resolve_device(device)
    torch.manual_seed(seed)
    net.to(torch_device)
    if optimizer == "adam":
        torch_opt = torch.optim.Adam(
            net.parameters(),
            lr=lr,
            betas=(
                _check_unit_interval(adam_beta1, "adam_beta1"),
                _check_unit_interval(adam_beta2, "adam_beta2"),
            ),
            eps=_check_epsilon(adam_epsilon),
        )
    else:
        torch_opt = torch.optim.SGD(net.parameters(), lr=lr)

    pin_memory = torch_device.type == "cuda"
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "seconds": [],
        "train_acc_top1": [],
        "train_acc_top5": [],
        "val_acc_top1": [],
        "val_acc_top5": [],
    }

    for epoch in range(epochs):
        started = time.perf_counter()
        net.train()
        loader = _make_loader(
            X_train, y_train, batch_size, seed + epoch, pin_memory
        )
        total_loss = 0.0
        total_rows = 0
        for features, labels in loader:
            features = features.to(torch_device, non_blocking=pin_memory)
            labels = labels.to(torch_device, non_blocking=pin_memory)
            torch_opt.zero_grad(set_to_none=True)
            logits = net.forward_tensor(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch_opt.step()
            total_loss += float(loss.detach().cpu()) * features.shape[0]
            total_rows += features.shape[0]

        train_loss = total_loss / total_rows
        net.eval()
        val_loss = evaluate_loss(net, X_val, y_val, batch_size=batch_size)
        train_acc = evaluate_accuracy(
            net, X_train, y_train, batch_size=batch_size
        )
        val_acc = evaluate_accuracy(net, X_val, y_val, batch_size=batch_size)

        seconds = time.perf_counter() - started
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["seconds"].append(seconds)
        history["train_acc_top1"].append(train_acc["top1"])
        history["train_acc_top5"].append(train_acc["top5"])
        history["val_acc_top1"].append(val_acc["top1"])
        history["val_acc_top5"].append(val_acc["top5"])

        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs}  "
                f"train loss {train_loss:.4f} "
                f"top1 {train_acc['top1']:.1f}% top5 {train_acc['top5']:.1f}%  "
                f"val loss {val_loss:.4f} "
                f"top1 {val_acc['top1']:.1f}% top5 {val_acc['top5']:.1f}%  "
                f"{seconds:.1f}s",
                flush=True,
            )

    return history


def main() -> None:
    """Command-line entry point: train on ``split.npz`` (train + val only)."""
    parser = argparse.ArgumentParser(
        description=(
            "Train ChessMoveNet with PyTorch mini-batch SGD/Adam on the "
            "prepared dataset (training + validation only; the test set "
            "is never loaded)."
        )
    )
    parser.add_argument(
        "--data",
        default="data/processed/split.npz",
        help="path to the prepared split archive (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="full passes over the training set (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="examples per gradient update (default: %(default)s)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="base seed for weight init and shuffling (default: %(default)s)",
    )
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam"), default="sgd",
        help="weight-update rule (default: %(default)s)",
    )
    parser.add_argument(
        "--activation", choices=("relu", "sigmoid", "tanh"), default="relu",
        help="hidden-layer activation (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-sizes", type=_parse_hidden_sizes, default=(256,),
        help="comma-separated hidden layer widths (default: 256)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, or cuda (default: auto = CUDA when available)",
    )
    parser.add_argument(
        "--adam-beta1", type=float, default=0.9,
        help="Adam first-moment decay (default: %(default)s)",
    )
    parser.add_argument(
        "--adam-beta2", type=float, default=0.999,
        help="Adam second-moment decay (default: %(default)s)",
    )
    parser.add_argument(
        "--adam-epsilon", type=float, default=1e-8,
        help="Adam numerical stabilizer (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint", default="artifacts/checkpoint.pt",
        help="where to save trained weights (default: %(default)s)",
    )
    parser.add_argument(
        "--history", default="artifacts/history.npz",
        help="where to save the per-epoch history (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace the checkpoint/history files if they already exist",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    history_path = Path(args.history)
    if checkpoint.suffix not in {".pt", ".pth"}:
        parser.error(f"--checkpoint must end in .pt, got {args.checkpoint!r}")
    if history_path.suffix != ".npz":
        parser.error(f"--history must end in .npz, got {args.history!r}")
    for flag, out in (("--checkpoint", checkpoint), ("--history", history_path)):
        if out.exists() and not args.overwrite:
            parser.error(
                f"{flag} file {out} already exists; re-run with "
                f"--overwrite or choose another path"
            )

    device = resolve_device(args.device)
    data = np.load(args.data)
    try:
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
    finally:
        data.close()
    print(
        f"loaded {args.data}: "
        f"train {X_train.shape}, val {X_val.shape} "
        f"(test set untouched)",
        flush=True,
    )

    net = ChessMoveNet(
        seed=args.seed,
        activation=args.activation,
        hidden_sizes=args.hidden_sizes,
        device=device,
    )
    print(
        f"training on {device} "
        f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}): "
        f"epochs={args.epochs} batch_size={args.batch_size} "
        f"lr={args.lr} optimizer={args.optimizer} activation={args.activation} "
        f"hidden_sizes={args.hidden_sizes} seed={args.seed}",
        flush=True,
    )
    history = train(
        net,
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        optimizer=args.optimizer,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        device=device,
    )

    total = sum(history["seconds"])
    print(
        f"finished {len(history['seconds'])} epoch(s) in {total:.1f}s  |  "
        f"final train loss {history['train_loss'][-1]:.4f} "
        f"top1 {history['train_acc_top1'][-1]:.1f}%, "
        f"val loss {history['val_loss'][-1]:.4f} "
        f"top1 {history['val_acc_top1'][-1]:.1f}% "
        f"top5 {history['val_acc_top5'][-1]:.1f}%",
        flush=True,
    )

    save_checkpoint(args.checkpoint, net, overwrite=args.overwrite)
    save_history(args.history, history, overwrite=args.overwrite)
    print(f"saved checkpoint -> {args.checkpoint}", flush=True)
    print(f"saved history    -> {args.history}", flush=True)


if __name__ == "__main__":
    main()
