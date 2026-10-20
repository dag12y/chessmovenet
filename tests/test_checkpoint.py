"""Tests for :mod:`chessmovenet.checkpoint` (saving and loading).

Everything here is synthetic: tiny seeded networks, hand-built history
dicts, and ``tmp_path`` files.  **No real dataset is read and no real
training run happens.**
"""

from __future__ import annotations

import numpy as np
import pytest

from chessmovenet.checkpoint import (
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
    path = tmp_path / "model.npz"

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


def test_checkpoint_file_is_plain_numeric_npz_without_pickle(
    tmp_path,
) -> None:
    """The file contains exactly the four parameter arrays, readable
    with allow_pickle=False (i.e. no pickle anywhere)."""
    path = tmp_path / "model.npz"
    save_checkpoint(path, _fresh_net(seed=1))

    with np.load(path, allow_pickle=False) as stored:
        assert set(stored.files) == set(PARAMETER_NAMES)
        for name in PARAMETER_NAMES:
            assert stored[name].dtype == np.float32
            assert np.isfinite(stored[name]).all()


def test_save_checkpoint_creates_missing_parent_directories(tmp_path) -> None:
    """artifacts/ may not exist yet on a fresh clone; saving must
    create it instead of failing."""
    path = tmp_path / "deep" / "nested" / "model.npz"

    save_checkpoint(path, _fresh_net())

    assert path.is_file()


def test_save_checkpoint_rejects_paths_without_npz_suffix(tmp_path) -> None:
    """np.savez would silently append .npz, making the path you checked
    differ from the file written -- refuse instead."""
    with pytest.raises(ValueError, match=r"\.npz"):
        save_checkpoint(tmp_path / "model", _fresh_net())


def test_save_checkpoint_refuses_to_overwrite_by_default(tmp_path) -> None:
    """An existing checkpoint must survive an accidental re-save unless
    overwrite=True is explicit."""
    path = tmp_path / "model.npz"
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
    path = tmp_path / "model.npz"
    save_checkpoint(path, _fresh_net(seed=0))
    replacement = _fresh_net(seed=5)

    save_checkpoint(path, replacement, overwrite=True)
    reloaded = load_checkpoint(path)

    np.testing.assert_array_equal(reloaded.W1, replacement.W1)
    assert not np.array_equal(reloaded.W1, _fresh_net(seed=0).W1)


def test_load_checkpoint_missing_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_checkpoint(tmp_path / "absent.npz")


# ---------------------------------------------------------------------------
# Checkpoint: malformed files fail with specific messages
# ---------------------------------------------------------------------------


def test_load_checkpoint_reports_missing_parameter_arrays(tmp_path) -> None:
    path = tmp_path / "incomplete.npz"
    net = _fresh_net()
    np.savez(path, W1=net.W1, b1=net.b1, W2=net.W2)  # b2 forgotten

    with pytest.raises(ValueError, match="missing parameter.*b2"):
        load_checkpoint(path)


def test_load_checkpoint_reports_shapes_that_do_not_chain(tmp_path) -> None:
    good = _fresh_net()

    # b1 sized for some other hidden width.
    bad_b1 = tmp_path / "bad_b1.npz"
    np.savez(
        bad_b1, W1=good.W1, b1=np.zeros(99, np.float32),
        W2=good.W2, b2=good.b2,
    )
    with pytest.raises(ValueError, match="b1 has shape"):
        load_checkpoint(bad_b1)

    # W2's input width does not match W1's hidden width.
    bad_w2 = tmp_path / "bad_w2.npz"
    np.savez(
        bad_w2, W1=good.W1, b1=good.b1,
        W2=np.zeros((7, 4), np.float32), b2=good.b2,
    )
    with pytest.raises(ValueError, match="W2 expects"):
        load_checkpoint(bad_w2)

    # W1 saved as a 1-D vector (not a matrix at all).
    flat_w1 = tmp_path / "flat_w1.npz"
    np.savez(
        flat_w1, W1=np.zeros(30, np.float32), b1=good.b1,
        W2=good.W2, b2=good.b2,
    )
    with pytest.raises(ValueError, match="2-D"):
        load_checkpoint(flat_w1)


# ---------------------------------------------------------------------------
# History: per-epoch records round-trip
# ---------------------------------------------------------------------------


def _fake_history(n: int = 3) -> dict[str, list]:
    """A dict shaped exactly like train()'s return value."""
    return {
        "train_loss": [8.3 - 0.1 * i for i in range(n)],
        "val_loss": [8.4 - 0.05 * i for i in range(n)],
        "seconds": [1.5 + 0.1 * i for i in range(n)],
    }


def test_save_history_derives_epoch_and_round_trips(tmp_path) -> None:
    """train() has no 'epoch' key; saving derives 1..N and loading
    returns it together with identical values."""
    history = _fake_history(4)
    path = tmp_path / "history.npz"

    save_history(path, history)
    loaded = load_history(path)

    assert loaded["epoch"] == [1, 2, 3, 4]
    assert loaded["train_loss"] == history["train_loss"]
    assert loaded["val_loss"] == history["val_loss"]
    assert loaded["seconds"] == history["seconds"]
    # Ints come back as ints, floats as floats.
    assert all(isinstance(e, int) for e in loaded["epoch"])
    assert all(isinstance(v, float) for v in loaded["train_loss"])


def test_save_history_keeps_an_explicit_epoch_column(tmp_path) -> None:
    history = _fake_history(2)
    history["epoch"] = [7, 8]

    save_history(tmp_path / "history.npz", history)

    assert load_history(tmp_path / "history.npz")["epoch"] == [7, 8]


def test_save_history_validates_before_writing(tmp_path) -> None:
    """A bad history raises and leaves NO file behind."""
    path = tmp_path / "history.npz"

    with pytest.raises(TypeError, match="dict"):
        save_history(path, [1, 2, 3])
    with pytest.raises(ValueError, match="missing required key"):
        save_history(path, {"train_loss": [1.0]})
    with pytest.raises(ValueError, match="equal lengths"):
        save_history(
            path,
            {"train_loss": [1.0, 2.0], "val_loss": [1.0], "seconds": [1.0, 2.0]},
        )
    with pytest.raises(ValueError, match="at least one epoch"):
        save_history(path, {"train_loss": [], "val_loss": [], "seconds": []})
    with pytest.raises(ValueError, match="non-finite"):
        save_history(
            path,
            {
                "train_loss": [1.0, float("nan")],
                "val_loss": [1.0, 2.0],
                "seconds": [1.0, 2.0],
            },
        )
    with pytest.raises(ValueError, match=r"\.npz"):
        save_history(tmp_path / "history", _fake_history())

    assert not path.exists()  # nothing was written for any failure


def test_save_history_refuses_overwrite_unless_asked(tmp_path) -> None:
    path = tmp_path / "history.npz"
    save_history(path, _fake_history(2))

    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_history(path, _fake_history(3))

    # File untouched: still the original 2 epochs.
    assert load_history(path)["epoch"] == [1, 2]

    save_history(path, _fake_history(3), overwrite=True)
    assert load_history(path)["epoch"] == [1, 2, 3]


def test_load_history_reports_missing_file_and_missing_keys(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no history"):
        load_history(tmp_path / "absent.npz")

    path = tmp_path / "incomplete.npz"
    np.savez(path, epoch=np.array([1]), train_loss=np.array([1.0]))
    with pytest.raises(ValueError, match="missing key"):
        load_history(path)


# ---------------------------------------------------------------------------
# Integration: synthetic training -> save -> reload -> identical behavior
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Optional accuracy columns (Step 11)
# ---------------------------------------------------------------------------


def _history_with_accuracy(n: int = 2) -> dict[str, list]:
    """A dict exactly as train() now returns it (losses + accuracy)."""
    history = _fake_history(n)
    history["train_acc_top1"] = [10.0 + i for i in range(n)]
    history["train_acc_top5"] = [40.0 + i for i in range(n)]
    history["val_acc_top1"] = [11.0 + i for i in range(n)]
    history["val_acc_top5"] = [41.0 + i for i in range(n)]
    return history


def test_history_round_trips_with_accuracy_columns(tmp_path) -> None:
    """A Step-11 history keeps its four accuracy columns on reload."""
    history = _history_with_accuracy(3)
    path = tmp_path / "history.npz"

    save_history(path, history)
    loaded = load_history(path)

    assert set(loaded) == {
        "epoch", "train_loss", "val_loss", "seconds",
        "train_acc_top1", "train_acc_top5",
        "val_acc_top1", "val_acc_top5",
    }
    for key, values in history.items():
        assert loaded[key] == values


def test_history_without_accuracy_columns_still_round_trips(
    tmp_path,
) -> None:
    """Old-format files (the pre-Step-11 baseline) have no accuracy
    columns and must keep working unchanged."""
    history = _fake_history(3)  # no accuracy keys at all
    path = tmp_path / "history.npz"

    save_history(path, history)
    loaded = load_history(path)

    assert set(loaded) == {"epoch", "train_loss", "val_loss", "seconds"}
    assert loaded["epoch"] == [1, 2, 3]
    assert loaded["train_loss"] == history["train_loss"]


def test_save_history_rejects_accuracy_outside_0_to_100(tmp_path) -> None:
    """Accuracy values are percentages; 101.0% is a bug, not data."""
    history = _history_with_accuracy(2)
    history["val_acc_top1"] = [50.0, 101.0]

    with pytest.raises(ValueError, match="percentages"):
        save_history(tmp_path / "history.npz", history)


def test_trained_model_and_history_survive_a_full_round_trip(
    tmp_path,
) -> None:
    """Train a tiny net on invented data for 3 epochs, save both
    artifacts, reload into fresh objects, and prove:

    - the reloaded model reproduces the final validation loss exactly
      (weights round-tripped bit-for-bit);
    - the reloaded history matches the in-memory one value for value.
    """
    from chessmovenet.train import evaluate_loss

    rng = np.random.default_rng(42)
    X = rng.normal(size=(128, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    X_train, y_train, X_val, y_val = X[:96], y[:96], X[96:], y[96:]

    net = ChessMoveNet(input_size=6, hidden_size=8, output_size=4, seed=0)
    history = train(
        net, X_train, y_train, X_val, y_val,
        epochs=3, batch_size=32, lr=0.01, seed=0, verbose=False,
    )

    ckpt_path = tmp_path / "checkpoint.npz"
    hist_path = tmp_path / "history.npz"
    save_checkpoint(ckpt_path, net)
    save_history(hist_path, history)

    reloaded_net = load_checkpoint(ckpt_path)
    reloaded_hist = load_history(hist_path)

    # Same weights -> same validation loss, to the last bit.
    reloaded_val = evaluate_loss(reloaded_net, X_val, y_val, batch_size=32)
    assert reloaded_val == pytest.approx(history["val_loss"][-1], rel=1e-12)

    assert reloaded_hist["epoch"] == [1, 2, 3]
    assert reloaded_hist["train_loss"] == history["train_loss"]
    assert reloaded_hist["val_loss"] == history["val_loss"]
    assert reloaded_hist["seconds"] == history["seconds"]
    assert all(s > 0 for s in reloaded_hist["seconds"])


