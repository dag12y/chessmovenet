"""PyTorch feed-forward classifier that scores chess move classes."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chessmovenet.encode import BOARD_FEATURE_SIZE
from chessmovenet.moves import OUTPUT_SIZE

# Model dimensions shared with the board and move encoders.
INPUT_SIZE = BOARD_FEATURE_SIZE  # 781 features per encoded position
HIDDEN_SIZE = 256  # neurons in the single hidden layer

# Parameter order used by training and checkpointing for the default net.
PARAMETER_NAMES = ("W1", "b1", "W2", "b2")

_PARAM_NAME = re.compile(r"^([Wb])([1-9][0-9]*)$")


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified linear unit: ``max(0, z)``, applied element-wise."""
    return np.maximum(z, 0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid, applied element-wise."""
    z = np.asarray(z)
    result = np.empty_like(z, dtype=np.result_type(z, np.float32))
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


def tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent, applied element-wise, with range ``[-1, 1]``."""
    return np.tanh(z)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


class _ParamArray(np.ndarray):
    """NumPy view of a PyTorch parameter that writes mutations back."""

    def __new__(cls, array: np.ndarray, writeback: Callable[[np.ndarray], None]):
        obj = np.array(array, copy=True).view(cls)
        obj._writeback = writeback
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self._writeback = getattr(obj, "_writeback", None)

    def copy(self, order="C"):  # noqa: ANN001
        return np.array(self, copy=True, order=order)

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        return np.ndarray.astype(
            np.asarray(self), dtype, order=order, casting=casting, subok=False, copy=copy
        )

    def __setitem__(self, key, value) -> None:
        np.ndarray.__setitem__(self, key, value)
        if self._writeback is not None:
            self._writeback(np.asarray(self))

    def __isub__(self, other):
        np.ndarray.__isub__(self, other)
        if self._writeback is not None:
            self._writeback(np.asarray(self))
        return self

    def __iadd__(self, other):
        np.ndarray.__iadd__(self, other)
        if self._writeback is not None:
            self._writeback(np.asarray(self))
        return self

    def __imul__(self, other):
        np.ndarray.__imul__(self, other)
        if self._writeback is not None:
            self._writeback(np.asarray(self))
        return self


class ChessMoveNet(nn.Module):
    """Fully connected move classifier implemented with PyTorch autograd.

    Dense layers live on CPU or CUDA. Tests and the play backend still see the
    NumPy layout ``W`` shape ``(in, out)`` used by the original from-scratch net.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = OUTPUT_SIZE,
        seed: int | None = None,
        activation: str = "relu",
        hidden_sizes: tuple[int, ...] | list[int] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = (hidden_size,)
        elif not isinstance(hidden_sizes, (tuple, list)) or not hidden_sizes:
            raise ValueError("hidden_sizes must be a non-empty list or tuple")
        else:
            hidden_sizes = tuple(hidden_sizes)

        for name, value in (("input_size", input_size), ("output_size", output_size)):
            is_int = isinstance(value, (int, np.integer))
            if isinstance(value, bool) or not is_int:
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        for index, value in enumerate(hidden_sizes):
            is_int = isinstance(value, (int, np.integer))
            if isinstance(value, bool) or not is_int:
                raise TypeError(
                    f"hidden_sizes[{index}] must be an integer, "
                    f"got {type(value).__name__}"
                )
            if value < 1:
                raise ValueError(
                    f"hidden_sizes[{index}] must be at least 1, got {value}"
                )

        self.input_size = int(input_size)
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)
        self.hidden_size = self.hidden_sizes[0]
        self.output_size = int(output_size)
        self.layer_sizes = (self.input_size, *self.hidden_sizes, self.output_size)
        self.parameter_names = tuple(
            name
            for layer in range(1, len(self.layer_sizes))
            for name in (f"W{layer}", f"b{layer}")
        )
        if not isinstance(activation, str):
            raise TypeError(
                f"activation must be a string, got {type(activation).__name__}"
            )
        self.activation = activation.lower()
        if self.activation not in {"relu", "sigmoid", "tanh"}:
            raise ValueError(
                "activation must be 'relu', 'sigmoid', or 'tanh', "
                f"got {activation!r}"
            )

        layers: list[nn.Linear] = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            layers.append(nn.Linear(fan_in, fan_out, bias=True))
        self.layers = nn.ModuleList(layers)
        self._init_parameters(seed)
        if device is not None:
            self.to(device)

    def _init_parameters(self, seed: int | None) -> None:
        if seed is not None:
            torch.manual_seed(int(seed))
        for layer in self.layers:
            fan_in = layer.weight.size(1)
            std = float(np.sqrt((2.0 if self.activation == "relu" else 1.0) / fan_in))
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            nn.init.zeros_(layer.bias)

    def _parse_param_name(self, name: str) -> tuple[str, int] | None:
        match = _PARAM_NAME.fullmatch(name)
        if match is None:
            return None
        kind, index = match.group(1), int(match.group(2))
        if index > len(self.layers):
            return None
        return kind, index

    def _numpy_param(self, kind: str, index: int) -> np.ndarray:
        layer = self.layers[index - 1]
        if kind == "W":
            return layer.weight.detach().T.contiguous().cpu().numpy()
        return layer.bias.detach().contiguous().cpu().numpy()

    def _write_numpy_param(self, kind: str, index: int, value: np.ndarray) -> None:
        array = np.asarray(value)
        layer = self.layers[index - 1]
        if array.dtype == np.float64 and layer.weight.dtype != torch.float64:
            self.double()
            layer = self.layers[index - 1]
        elif array.dtype == np.float32 and layer.weight.dtype == torch.float64:
            self.float()
            layer = self.layers[index - 1]
        tensor = torch.as_tensor(array, device=layer.weight.device)
        with torch.no_grad():
            if kind == "W":
                if tensor.ndim != 2:
                    raise ValueError(f"W{index} must be 2-D, got {tensor.ndim}-D")
                layer.weight.copy_(tensor.T.contiguous().to(dtype=layer.weight.dtype))
            else:
                layer.bias.copy_(tensor.to(dtype=layer.bias.dtype))

    def _module_layers(self):
        modules = self.__dict__.get("_modules") or {}
        return modules.get("layers")

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            layers = self._module_layers()
            match = _PARAM_NAME.fullmatch(name) if layers is not None else None
            if match is None:
                raise
            kind, index = match.group(1), int(match.group(2))
            if not 1 <= index <= len(layers):
                raise
            return _ParamArray(
                self._numpy_param(kind, index),
                lambda array, k=kind, i=index: self._write_numpy_param(k, i, array),
            )

    def __setattr__(self, name: str, value) -> None:
        layers = self._module_layers() if "_modules" in self.__dict__ else None
        if layers is not None:
            match = _PARAM_NAME.fullmatch(name)
            if match is not None:
                kind, index = match.group(1), int(match.group(2))
                if 1 <= index <= len(layers):
                    self._write_numpy_param(kind, index, np.asarray(value))
                    return
        super().__setattr__(name, value)

    @property
    def device(self) -> torch.device:
        return self.layers[0].weight.device

    def _activate(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(z)
        if self.activation == "sigmoid":
            return torch.sigmoid(z)
        return torch.tanh(z)

    def _check_input(self, X: np.ndarray) -> np.ndarray:
        try:
            X = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"X must contain numbers convertible to float32, got: {exc}"
            ) from exc

        if X.ndim not in (1, 2):
            raise ValueError(
                f"expected a 1-D board vector of shape ({self.input_size},) "
                f"or a 2-D batch of shape (N, {self.input_size}), "
                f"but got an array with {X.ndim} dimension(s) "
                f"and shape {X.shape}"
            )
        if X.shape[-1] != self.input_size:
            raise ValueError(
                f"each position must have exactly {self.input_size} "
                f"features, but the last axis of the input has "
                f"{X.shape[-1]}"
            )
        if X.ndim == 2 and X.shape[0] == 0:
            raise ValueError("X must contain at least one position")
        n_bad = int(X.size - np.count_nonzero(np.isfinite(X)))
        if n_bad:
            raise ValueError(
                f"X must be finite (no NaN or infinity), but "
                f"{n_bad} feature value(s) are not"
            )
        return X

    def _as_tensor(self, X: np.ndarray) -> tuple[torch.Tensor, bool]:
        X = self._check_input(X)
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]
        tensor = torch.from_numpy(np.ascontiguousarray(X)).to(
            device=self.device, dtype=self.layers[0].weight.dtype
        )
        return tensor, single

    def forward_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Batched logits from a feature tensor already on this module's device."""
        hidden = X
        for layer in self.layers[:-1]:
            hidden = self._activate(layer(hidden))
        return self.layers[-1](hidden)

    def forward(self, X, apply_softmax: bool = False):
        """NumPy-friendly forward used by tests, evaluation, and the play API.

        Accepts a board vector ``(input_size,)`` or a batch ``(N, input_size)``.
        Returns logits by default, or probabilities when ``apply_softmax=True``.
        """
        if isinstance(X, torch.Tensor):
            logits = self.forward_tensor(X)
            if apply_softmax:
                logits = torch.softmax(logits, dim=-1)
            return logits

        tensor, single = self._as_tensor(X)
        logits = self.forward_tensor(tensor)
        if apply_softmax:
            logits = torch.softmax(logits, dim=-1)
        output = logits.detach().cpu().numpy()
        return output[0] if single else output

    def _check_labels(self, labels: np.ndarray, n: int) -> np.ndarray:
        labels = np.asarray(labels)
        if labels.ndim == 0:
            labels = labels.reshape(1)

        if labels.ndim != 1:
            raise ValueError(
                f"expected labels of shape ({n},), got shape {labels.shape}"
            )
        if labels.shape[0] != n:
            raise ValueError(
                f"expected {n} label(s) to match the input batch, "
                f"got {labels.shape[0]}"
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(
                f"labels must be integers, got dtype {labels.dtype}"
            )

        labels = labels.astype(np.int64)
        if np.any(labels < 0) or np.any(labels >= self.output_size):
            raise ValueError(
                f"labels must be in 0 .. {self.output_size - 1}, "
                f"got a value outside this range"
            )
        return labels

    def backward(self, X: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        """Mean cross-entropy gradients via autograd, as NumPy arrays."""
        tensor, _single = self._as_tensor(X)
        labels = self._check_labels(labels, tensor.shape[0])
        label_tensor = torch.from_numpy(labels).to(device=self.device)

        self.zero_grad(set_to_none=True)
        logits = self.forward_tensor(tensor)
        loss = F.cross_entropy(logits, label_tensor, reduction="mean")
        loss.backward()

        gradients: dict[str, np.ndarray] = {}
        for index, layer in enumerate(self.layers, start=1):
            if layer.weight.grad is None or layer.bias.grad is None:
                raise RuntimeError(f"missing gradient for layer {index}")
            gradients[f"W{index}"] = (
                layer.weight.grad.detach().T.contiguous().cpu().numpy()
            )
            gradients[f"b{index}"] = layer.bias.grad.detach().contiguous().cpu().numpy()
        return gradients

    def predict(self, X: np.ndarray):
        """Return the most probable move class and its probability."""
        probabilities = self.forward(X, apply_softmax=True)
        labels = np.argmax(probabilities, axis=-1)
        confidences = np.max(probabilities, axis=-1)
        if probabilities.ndim == 1:
            return int(labels), float(confidences)
        return labels, confidences

    def predict_top_k(self, X: np.ndarray, k: int = 5):
        """Return the ``k`` most probable move classes, best first."""
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError(f"expected an integer k, got {type(k).__name__}")
        k = int(k)
        if k < 1 or k > self.output_size:
            raise ValueError(
                f"k must be between 1 and {self.output_size} (the number "
                f"of move classes), got {k}"
            )

        probabilities = self.forward(X, apply_softmax=True)
        single = probabilities.ndim == 1
        if single:
            probabilities = probabilities[np.newaxis, :]

        order = np.argsort(probabilities, axis=-1)
        top_labels = order[:, -k:][:, ::-1]
        top_probs = np.take_along_axis(probabilities, top_labels, axis=-1)

        if single:
            return top_labels[0], top_probs[0]
        return top_labels, top_probs
