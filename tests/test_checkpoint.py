"""Tests for :mod:`chessmovenet.checkpoint` (saving and loading).

Everything here is synthetic: tiny seeded networks, hand-built history
dicts, and ``tmp_path`` files.  **No personal game data is read and no real
training run happens.**  Checkpoints are PyTorch ``.pt`` files (state_dict
plus architecture metadata); history files remain ``.npz``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chessmovenet.checkpoint import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    load_history,
    save_checkpoint,
    save_history,
)
from chessmovenet.model import PARAMETER_NAMES, ChessMoveNet
from chessmovenet.train import train


def _fresh_net(seed: int = 0) -> ChessMoveNet:
    """A small 6 -> 5 -> 4 network (keeps files and matmuls tiny)."""
    return ChessMoveNet(input_size=6, hidden_size=5, output_size=4, seed=seed)


def _torch_state_dict(net: ChessMoveNet) -> dict[str, torch.Tensor]:
    """Return the raw PyTorch state_dict (``layers.0.weight`` style keys)."""
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


# ---------------------------------------------------------------------------
# Checkpoint: parameters round-trip exactly
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_restores_identical_parameters(
    tmp_path,
) -> None:
    """A saved net reloaded into a *fresh* net must be bit-identical --
    and produce identical outputs (usable for inference)."""
    original = _fresh_net(seed=0)
    other = _fresh_net(seed=99)  # different starting weights
    path = tmp_path / "model.pt"

    save_checkpoint(path, original)
    loaded = load_checkpoint(path)

    for name in PARAMETER_NAMES:
        np.testing.assert_array_equal(
            getattr(loaded, name), getattr(original, name)
        )
        assert getattr(loaded, name).dtype == np.float32
    # The fresh net really was different before loading (sanity check).
    assert not np.array_equal(other.W1, original.W1)

    # Same weights -> same predictions on the same input.
    X = np.random.default_rng(7).normal(size=(4, 6)).astype(np.float32)
    np.testing.assert_array_equal(loaded.forward(X), original.forward(X))


def test_checkpoint_file_has_expected_structure(tmp_path) -> None:
    """A saved checkpoint is a loadable PyTorch payload with architecture
    metadata and a state_dict, loadable with ``weights_only=True``."""
    path = tmp_path / "model.pt"
    save_checkpoint(path, _fresh_net(seed=1))

    stored = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(stored, dict)
    assert stored["format"] == CHECKPOINT_FORMAT
    assert set(stored) == {
        "format",
        "input_size",
        "hidden_sizes",
        "output_size",
        "activation",
        "state_dict",
    }
    assert stored["input_size"] == 6
    assert stored["hidden_sizes"] == (5,)
    assert stored["output_size"] == 4
    assert stored["activation"] == "relu"
    state_dict = stored["state_dict"]
    assert isinstance(state_dict, dict) and state_dict
    for tensor in state_dict.values():
        assert isinstance(tensor, torch.Tensor)
        assert torch.isfinite(tensor).all()


def test_checkpoint_round_trip_preserves_hidden_activation(tmp_path) -> None:
    original = ChessMoveNet(6, 5, 4, seed=0, activation="tanh")
    path = tmp_path / "tanh_model.pt"

    save_checkpoint(path, original)
    loaded = load_checkpoint(path)

    assert loaded.activation == "tanh"


def test_save_checkpoint_creates_missing_parent_directories(tmp_path) -> None:
    """artifacts/ may not exist yet on a fresh clone; saving must
    create it instead of failing."""
    path = tmp_path / "deep" / "nested" / "model.pt"

    save_checkpoint(path, _fresh_net())

    assert path.is_file()


def test_save_checkpoint_rejects_paths_without_pt_suffix(tmp_path) -> None:
    """torch.save would write to whatever path is given; refuse paths
    that don't end in .pt or .pth so the caller's expectations match."""
    with pytest.raises(ValueError, match=r"\.pt"):
        save_checkpoint(tmp_path / "model", _fresh_net())


def test_save_checkpoint_refuses_to_overwrite_by_default(tmp_path) -> None:
    """An existing checkpoint must survive an accidental re-save unless
    overwrite=True is explicit."""
    path = tmp_path / "model.pt"
    save_checkpoint(path, _fresh_net(seed=0))

    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_checkpoint(path, _fresh_net(seed=5))

    # The original file was not touched: it still loads seed-0 weights.
    reloaded = load_checkpoint(path)
    np.testing.assert_array_equal(
        reloaded.W1, _fresh_net(seed=0).W1
    )


def test_save_checkpoint_overwrite_flag_replaces_the_file(tmp_path) -> None:
    """overwrite=True is the explicit, working escape hatch."""
    path = tmp_path / "model.pt"
    save_checkpoint(path, _fresh_net(seed=0))
    replacement = _fresh_net(seed=5)

    save_checkpoint(path, replacement, overwrite=True)
    reloaded = load_checkpoint(path)

    np.testing.assert_array_equal(reloaded.W1, replacement.W1)
    assert not np.array_equal(reloaded.W1, _fresh_net(seed=0).W1)


def test_load_checkpoint_missing_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_checkpoint(tmp_path / "absent.pt")


# ---------------------------------------------------------------------------
# Checkpoint: malformed files fail with specific messages
# ---------------------------------------------------------------------------


def test_load_checkpoint_reports_missing_parameter_arrays(tmp_path) -> None:
    path = tmp_path / "incomplete.pt"
    net = _fresh_net()
    state_dict = _torch_state_dict(net)
    # Drop one layer's bias to simulate a truncated checkpoint.
    del state_dict["layers.1.bias"]
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "input_size": 6,
            "hidden_sizes": (5,),
            "output_size": 4,
            "activation": "relu",
            "state_dict": state_dict,
        },
        path,
    )

    with pytest.raises(ValueError, match="Missing key"):
        load_checkpoint(path)


def test_load_checkpoint_reports_shapes_that_do_not_chain(tmp_path) -> None:
    good = _fresh_net()
    base_payload = {
        "format": CHECKPOINT_FORMAT,
        "input_size": 6,
        "hidden_sizes": (5,),
        "output_size": 4,
        "activation": "relu",
    }

    # b1 sized for some other hidden width.
    bad_b1 = tmp_path / "bad_b1.pt"
    sd = _torch_state_dict(good)
    sd["layers.0.bias"] = torch.zeros(99, dtype=torch.float32)
    torch.save({**base_payload, "state_dict": sd}, bad_b1)
    with pytest.raises(ValueError, match="size mismatch"):
        load_checkpoint(bad_b1)

    # W2's input width does not match W1's hidden width.
    bad_w2 = tmp_path / "bad_w2.pt"
    sd = _torch_state_dict(good)
    sd["layers.1.weight"] = torch.zeros((7, 4), dtype=torch.float32)
    torch.save({**base_payload, "state_dict": sd}, bad_w2)
    with pytest.raises(ValueError, match="size mismatch"):
        load_checkpoint(bad_w2)

    # W1 saved as a 1-D vector (not a matrix at all).
    flat_w1 = tmp_path / "flat_w1.pt"
    sd = _torch_state_dict(good)
    sd["layers.0.weight"] = torch.zeros(30, dtype=torch.float32)
    torch.save({**base_payload, "state_dict": sd}, flat_w1)
    with pytest.raises(ValueError, match="size mismatch"):
        load_checkpoint(flat_w1)
