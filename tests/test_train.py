"""Tests for :mod:`chessmovenet.train` (the training loop).

Everything here is synthetic: tiny hand-built arrays or tiny networks
with seeded RNGs.  **No personal game data is touched, and no real
training run happens in the test suite** -- the end-to-end test below
trains a 6-8-5 network on a few hundred invented rows for a few
milliseconds.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from chessmovenet.checkpoint import load_checkpoint, load_history
from chessmovenet.model import ChessMoveNet
from chessmovenet.train import (
    AdamOptimizer,
    evaluate_accuracy,
    evaluate_loss,
    iterate_minibatches,
    main,
    train,
    update_parameters,
)


def _toy(n: int = 10, features: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Rows encode their own index: X[i, 0] == i * features, so any
    batch can prove that features and labels stayed paired."""
    X = np.arange(n * features, dtype=np.float32).reshape(n, features)
    y = np.arange(n, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Mini-batches: coverage, pairing, partial final batch, reproducibility
# ---------------------------------------------------------------------------


def test_batches_cover_every_example_exactly_once() -> None:
    """The batches are a partition of the dataset: nothing lost,
    nothing duplicated, pairs aligned, dtypes preserved."""
    X, y = _toy(10)
    seen: list[int] = []

    for X_batch, y_batch in iterate_minibatches(X, y, batch_size=4, seed=0):
        assert X_batch.ndim == 2 and X_batch.shape[1] == 5
        assert y_batch.shape == (X_batch.shape[0],)
        assert X_batch.dtype == np.float32
        assert y_batch.dtype == np.int64
        # Row k of X_batch must still carry label k of y_batch.
        np.testing.assert_array_equal(X_batch[:, 0] // 5, y_batch)
        seen.extend(y_batch.tolist())

    assert sorted(seen) == list(range(10))


def test_partial_final_batch() -> None:
    """10 examples with batch_size=4 -> batches of 4, 4, and 2."""
    X, y = _toy(10)

    sizes = [
        X_batch.shape[0]
        for X_batch, _ in iterate_minibatches(X, y, batch_size=4, seed=0)
    ]

    assert sizes == [4, 4, 2]


def test_same_seed_same_order_different_seed_differs() -> None:
    """Reproducibility: identical seeds replay identical batches;
    another seed must actually shuffle differently."""
    X, y = _toy(50)

    first = np.concatenate(
        [yb for _, yb in iterate_minibatches(X, y, 8, seed=7)]
    )
    replay = np.concatenate(
        [yb for _, yb in iterate_minibatches(X, y, 8, seed=7)]
    )
    other = np.concatenate(
        [yb for _, yb in iterate_minibatches(X, y, 8, seed=8)]
    )

    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, other)


def test_shuffle_is_not_the_original_order() -> None:
    """With 100 examples and seed 0 the first batch must not be the
    chronological order -- otherwise 'shuffling' would be a lie."""
    X, y = _toy(100)

    first_batch_labels = next(
        iterate_minibatches(X, y, batch_size=100, seed=0)
    )[1]

    assert not np.array_equal(first_batch_labels, np.arange(100))


def test_input_arrays_are_never_modified_in_place() -> None:
    """Iteration reads X and y; it must not permute the caller's
    arrays (the same arrays are reused for validation later)."""
    X, y = _toy(20)
    X_before, y_before = X.copy(), y.copy()

    for _ in iterate_minibatches(X, y, batch_size=6, seed=1):
        pass  # iterate through everything

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


def test_rejects_invalid_batch_settings() -> None:
    """Clear errors for the mistakes a training loop can make."""
    X, y = _toy(10)

    with pytest.raises(ValueError, match="positive integer"):
        iterate_minibatches(X, y, batch_size=0)
    with pytest.raises(ValueError, match="positive integer"):
        iterate_minibatches(X, y, batch_size=-3)
    with pytest.raises(TypeError, match="integer"):
        iterate_minibatches(X, y, batch_size=4.5)
    with pytest.raises(TypeError, match="integer"):
        iterate_minibatches(X, y, batch_size=True)
    with pytest.raises(ValueError, match="non-negative"):
        iterate_minibatches(X, y, batch_size=4, seed=-1)

    with pytest.raises(ValueError, match="2-D"):
        iterate_minibatches(np.zeros(10, dtype=np.float32), y)
    with pytest.raises(ValueError, match="1-D"):
        iterate_minibatches(X, np.zeros((10, 1), dtype=np.int64))
    with pytest.raises(ValueError, match="same number"):
        iterate_minibatches(X, y[:-1])
    with pytest.raises(ValueError, match="empty"):
        iterate_minibatches(
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# Gradient-descent parameter updates
# ---------------------------------------------------------------------------


def _tiny_net() -> tuple[ChessMoveNet, np.ndarray, np.ndarray]:
    """A 6 -> 5 -> 4 network and a 3-example batch for it."""
    net = ChessMoveNet(input_size=6, hidden_size=5, output_size=4, seed=3)
    rng = np.random.default_rng(11)
    X = rng.normal(size=(3, 6)).astype(np.float32)
    y = np.array([0, 1, 3], dtype=np.int64)
    return net, X, y


def test_update_moves_each_parameter_against_its_gradient() -> None:
    """After one step, parameter == old parameter - lr * gradient,
    for all four parameters, and nothing else changed shape/dtype."""
    net, X, y = _tiny_net()
    grads = net.backward(X, y)
    lr = 0.1
    before = {n: getattr(net, n).copy() for n in ("W1", "b1", "W2", "b2")}

    update_parameters(net, grads, lr)

    for name in ("W1", "b1", "W2", "b2"):
        after = getattr(net, name)
        assert after.shape == before[name].shape
        assert after.dtype == np.float32
        np.testing.assert_allclose(
            after, before[name] - lr * grads[name], rtol=1e-6, atol=1e-8
        )
        assert not np.array_equal(after, before[name])  # actually moved


def test_update_step_scales_linearly_with_learning_rate() -> None:
    """The same gradients with twice the learning rate must move each
    parameter exactly twice as far -- the 'steepness dial' behaves."""
    net_a, X, y = _tiny_net()
    net_b, _, _ = _tiny_net()  # identical weights (same seed)
    grads = net_a.backward(X, y)
    before = net_a.W1.copy()

    update_parameters(net_a, grads, 0.01)
    update_parameters(net_b, grads, 0.02)

    delta_a = before - net_a.W1
    delta_b = before - net_b.W1
    # Generous tolerances: parameters live in float32, so each stored
    # step carries ~1e-7 absolute rounding noise.
    np.testing.assert_allclose(delta_b, 2.0 * delta_a, rtol=1e-4, atol=1e-7)


def test_zero_gradients_leave_the_weights_untouched() -> None:
    """A zero gradient means 'nowhere to go': the update must be a
    perfect no-op (this is what a converged or dead network sees)."""
    net, _, _ = _tiny_net()
    before = {n: getattr(net, n).copy() for n in ("W1", "b1", "W2", "b2")}
    zero_grads = {n: np.zeros_like(before[n]) for n in before}

    update_parameters(net, zero_grads, 0.5)

    for name, original in before.items():
        np.testing.assert_array_equal(getattr(net, name), original)


def test_update_rejects_invalid_grads_and_learning_rates() -> None:
    """Malformed gradients or nonsense learning rates fail loudly
    instead of silently corrupting the weights."""
    net, X, y = _tiny_net()
    good = net.backward(X, y)

    with pytest.raises(TypeError, match="dict"):
        update_parameters(net, [1, 2, 3], 0.1)
    with pytest.raises(ValueError, match="keys"):
        update_parameters(net, {"W1": good["W1"]}, 0.1)  # missing b1,W2,b2
    wrong_shape = dict(good, W2=np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        update_parameters(net, wrong_shape, 0.1)

    with pytest.raises(ValueError, match="finite positive"):
        update_parameters(net, good, 0.0)
    with pytest.raises(ValueError, match="finite positive"):
        update_parameters(net, good, -0.1)
    with pytest.raises(ValueError, match="finite positive"):
        update_parameters(net, good, float("nan"))
    with pytest.raises(TypeError, match="number"):
        update_parameters(net, good, "0.1")
    with pytest.raises(TypeError, match="number"):
        update_parameters(net, good, True)


def test_adam_first_step_uses_bias_corrected_gradient_direction() -> None:
    """On step one, Adam's corrected moments move each nonzero entry
    approximately one learning-rate unit in its gradient's direction."""
    net, X, y = _tiny_net()
    grads = net.backward(X, y)
    before = net.W1.copy()
    adam = AdamOptimizer(net, lr=0.01)

    adam.step(net, grads)

    expected = before - 0.01 * grads["W1"] / (
        np.abs(grads["W1"]) + 1e-8
    )
    np.testing.assert_allclose(net.W1, expected, rtol=1e-5, atol=1e-7)
    assert adam.steps == 1


def test_adam_reduces_loss_on_synthetic_data() -> None:
    """Adam must work end-to-end with the same batched training API."""
    X_tr, y_tr, X_v, y_v = _learnable_data()
    net = ChessMoveNet(input_size=6, hidden_size=8, output_size=4, seed=0)
    before = evaluate_loss(net, X_v, y_v, batch_size=32)

    history = train(
        net, X_tr, y_tr, X_v, y_v,
        epochs=8, batch_size=32, lr=0.01, seed=0,
        optimizer="adam", verbose=False,
    )

    assert history["val_loss"][-1] < before


# ---------------------------------------------------------------------------
# Validation / evaluation: measuring without learning
# ---------------------------------------------------------------------------


def test_evaluate_loss_equals_the_full_batch_loss() -> None:
    """Weighted mean over batches (including a short final one) must
    equal the loss computed over all rows in a single forward pass."""
    net, X, y = _tiny_net()  # 3 examples, batch_size=2 -> 2 + 1

    batched = evaluate_loss(net, X, y, batch_size=2)
    whole = evaluate_loss(net, X, y, batch_size=3)

    assert batched == pytest.approx(whole, rel=1e-9)


def test_evaluate_loss_never_changes_the_weights() -> None:
    """Validation must measure, not modify: after any number of
    evaluations the parameters are bit-for-bit identical."""
    net, X, y = _tiny_net()
    before = {n: getattr(net, n).copy() for n in ("W1", "b1", "W2", "b2")}

    for _ in range(3):
        evaluate_loss(net, X, y, batch_size=2)

    for name, original in before.items():
        np.testing.assert_array_equal(getattr(net, name), original)


# ---------------------------------------------------------------------------
# The training loop itself: synthetic end-to-end run
# ---------------------------------------------------------------------------


def _learnable_data(
    n: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic classification data the tiny net can actually learn:
    the label depends on feature 0 of the input (a real signal, not
    noise), split into train (96) and val (24)."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)  # class 0 or 1, learnable rule
    return X[:96], y[:96], X[96:], y[96:]


def test_training_reduces_loss_on_synthetic_data() -> None:
    """The whole point: after a few epochs of gradient descent on a
    learnable synthetic task, the training loss must be lower than it
    started.  (Fully deterministic: fixed seeds everywhere.)"""
    X_tr, y_tr, X_v, y_v = _learnable_data()
    net = ChessMoveNet(input_size=6, hidden_size=8, output_size=4, seed=0)

    history = train(
        net, X_tr, y_tr, X_v, y_v,
        epochs=40, batch_size=32, lr=0.01, seed=0, verbose=False,
    )

    assert len(history["train_loss"]) == 40
    assert len(history["val_loss"]) == 40
    assert len(history["seconds"]) == 40
    assert all(math.isfinite(v) for v in history["train_loss"])
    assert all(s > 0.0 for s in history["seconds"])
    # The loss curve goes down (strictly, with a real margin).
    assert history["train_loss"][-1] < history["train_loss"][0] * 0.9


def test_training_is_reproducible_with_the_same_seed() -> None:
    """Two identical runs (same weights, same shuffle) must produce
    identical loss curves -- otherwise experiments cannot be compared."""
    X_tr, y_tr, X_v, y_v = _learnable_data()

    def run() -> dict[str, list]:
        net = ChessMoveNet(input_size=6, hidden_size=8, output_size=4, seed=0)
        return train(
            net, X_tr, y_tr, X_v, y_v,
            epochs=5, batch_size=32, lr=0.01, seed=0, verbose=False,
        )

    first, replay = run(), run()

    np.testing.assert_array_equal(first["train_loss"], replay["train_loss"])
    np.testing.assert_array_equal(first["val_loss"], replay["val_loss"])
    # (seconds are wall-clock, so they are intentionally not compared.)


def test_val_loss_matches_a_post_training_evaluation() -> None:
    """The last epoch's validation loss must equal evaluating the
    trained network afterwards -- proof that the in-loop measurement
    used the final weights and added no updates of its own."""
    X_tr, y_tr, X_v, y_v = _learnable_data()
    net = ChessMoveNet(input_size=6, hidden_size=8, output_size=4, seed=0)

    history = train(
        net, X_tr, y_tr, X_v, y_v,
        epochs=3, batch_size=32, lr=0.01, seed=0, verbose=False,
    )

    after = evaluate_loss(net, X_v, y_v, batch_size=32)
    assert history["val_loss"][-1] == pytest.approx(after, rel=1e-12)


def test_train_rejects_invalid_settings() -> None:
    """Bad hyperparameters fail immediately -- before any epoch runs."""
    X_tr, y_tr, X_v, y_v = _learnable_data()
    net = ChessMoveNet(input_size=6, hidden_size=8, output_size=4, seed=0)

    def expect(kind, match, **overrides):
        settings = dict(
            epochs=1, batch_size=32, lr=0.01, seed=0, verbose=False
        )
        settings.update(overrides)
        with pytest.raises(kind, match=match):
            train(net, X_tr, y_tr, X_v, y_v, **settings)

    expect(ValueError, "positive integer", epochs=0)
    expect(TypeError, "integer", epochs=2.5)
    expect(ValueError, "positive integer", batch_size=0)
    expect(ValueError, "finite positive", lr=0.0)
    expect(ValueError, "finite positive", lr=-0.1)
    expect(TypeError, "number", lr="fast")
    expect(ValueError, "non-negative", seed=-5)

    with pytest.raises(ValueError, match="X_train"):
        train(  # label count disagrees with feature count
            net, X_tr, y_tr[:-1], X_v, y_v,
            epochs=1, batch_size=32, lr=0.01, seed=0, verbose=False,
        )
    with pytest.raises(ValueError, match="X_val"):
        train(  # empty validation set
            net, X_tr, y_tr, X_tr[:0], y_tr[:0],
            epochs=1, batch_size=32, lr=0.01, seed=0, verbose=False,
        )


# ---------------------------------------------------------------------------
# CLI (main): run on a miniature synthetic split -- NEVER the real data
# ---------------------------------------------------------------------------


def _write_tiny_split(tmp_path):
    """A miniature split.npz with the real 781-feature width, because
    main() builds the default 781 -> 256 -> 4272 architecture."""
    rng = np.random.default_rng(0)

    def block(n: int):
        X = rng.normal(size=(n, 781)).astype(np.float32)
        y = rng.integers(0, 4272, size=n).astype(np.int64)
        return X, y

    X_tr, y_tr = block(40)
    X_v, y_v = block(16)
    X_t, y_t = block(8)  # present so we could notice if it were touched;
    # main() must simply never index these keys.
    path = tmp_path / "split.npz"
    np.savez(
        path,
        X_train=X_tr, y_train=y_tr,
        X_val=X_v, y_val=y_v,
        X_test=X_t, y_test=y_t,
    )
    return path


def _run_main(tmp_path, monkeypatch, capsys, *extra: str):
    """Invoke main() with controlled argv; return (stdout, stderr)."""
    monkeypatch.setattr(sys, "argv", ["chessmovenet.train", *extra])
    main()
    captured = capsys.readouterr()
    return captured.out, captured.err


def test_main_trains_saves_and_reports_on_synthetic_data(
    tmp_path, monkeypatch, capsys
) -> None:
    """Happy path: 1 epoch on the tiny split, both artifacts written
    and loadable, informative output."""
    data = _write_tiny_split(tmp_path)
    ckpt = tmp_path / "out" / "model.npz"  # 'out/' does not exist yet
    hist = tmp_path / "out" / "history.npz"

    out, _ = _run_main(
        tmp_path, monkeypatch, capsys,
        "--data", str(data),
        "--epochs", "1",
        "--batch-size", "16",
        "--seed", "0",
        "--checkpoint", str(ckpt),
        "--history", str(hist),
    )

    assert "test set untouched" in out
    assert "epoch 1/1" in out
    assert "finished 1 epoch(s)" in out
    assert "saved checkpoint ->" in out
    assert "saved history ->" in out or "saved history    ->" in out

    loaded = load_checkpoint(ckpt)
    assert loaded.W1.shape == (781, 256)
    assert loaded.W2.shape == (256, 4272)

    reloaded_hist = load_history(hist)
    assert reloaded_hist["epoch"] == [1]
    assert len(reloaded_hist["train_loss"]) == 1
    assert reloaded_hist["seconds"][0] > 0


def test_main_refuses_existing_outputs_before_touching_any_data(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fail fast: an existing checkpoint without --overwrite must abort
    with a clear message BEFORE loading data (the --data path here does
    not even exist, so any attempt to load it would crash instead)."""
    ckpt = tmp_path / "model.npz"
    ckpt.write_bytes(b"precious previous run")
    hist = tmp_path / "history.npz"

    with pytest.raises(SystemExit) as exc:
        _run_main(
            tmp_path, monkeypatch, capsys,
            "--data", str(tmp_path / "nonexistent.npz"),
            "--checkpoint", str(ckpt),
            "--history", str(hist),
        )

    assert exc.value.code == 2  # argparse's error exit code
    err = capsys.readouterr().err
    assert "already exists" in err
    assert "--overwrite" in err
    assert ckpt.read_bytes() == b"precious previous run"  # untouched


def test_main_rejects_output_paths_without_npz_suffix(
    tmp_path, monkeypatch, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        _run_main(
            tmp_path, monkeypatch, capsys,
            "--data", str(tmp_path / "nonexistent.npz"),
            "--checkpoint", str(tmp_path / "model"),
            "--history", str(tmp_path / "history.npz"),
        )

    assert exc.value.code == 2
    assert ".npz" in capsys.readouterr().err


def test_main_overwrite_flag_replaces_existing_outputs(
    tmp_path, monkeypatch, capsys
) -> None:
    """--overwrite is the explicit escape hatch and really replaces."""
    data = _write_tiny_split(tmp_path)
    ckpt = tmp_path / "model.npz"
    hist = tmp_path / "history.npz"
    np.savez(ckpt, junk=np.zeros(1))  # stale files from an old run
    np.savez(hist, junk=np.zeros(1))

    out, _ = _run_main(
        tmp_path, monkeypatch, capsys,
        "--data", str(data),
        "--epochs", "1",
        "--batch-size", "16",
        "--checkpoint", str(ckpt),
        "--history", str(hist),
        "--overwrite",
    )

    assert "saved checkpoint ->" in out
    with np.load(ckpt, allow_pickle=False) as stored:
            assert set(stored.files) == {"W1", "b1", "W2", "b2", "activation"}
    with np.load(hist, allow_pickle=False) as stored:
        assert set(stored.files) == {
            "epoch", "train_loss", "val_loss", "seconds",
            "train_acc_top1", "train_acc_top5",
            "val_acc_top1", "val_acc_top5",
        }


# ---------------------------------------------------------------------------
# Accuracy tracking (Step 11): top-1 / top-5, raw move classes
# ---------------------------------------------------------------------------


class _ScriptedNet:
    """Stand-in network with exactly known predictions.

    ``forward`` returns fixed logits looked up by the row id stored in
    ``X[:, 0]``, so the accuracy of any batching is hand-computable --
    no weights, no randomness, and no ties at top-5 boundaries in the
    data below.
    """

    def __init__(self, logits: np.ndarray) -> None:
        self.logits = np.asarray(logits, dtype=np.float32)
        self.output_size = int(self.logits.shape[1])

    def forward(self, X, apply_softmax=False):
        return self.logits[X[:, 0].astype(np.int64)]


# 4 rows x 6 classes.  Hand-computed expectations:
#   row 0: argmax=0 (label 0) -> top1 HIT ; value 9        -> top5 HIT
#   row 1: argmax=1 (label 0) -> top1 miss; value 5 (2nd)  -> top5 HIT
#   row 2: argmax=1 (label 5) -> top1 miss; value .1 (min) -> top5 MISS
#   row 3: argmax=5 (label 3) -> top1 miss; value 3 (4th)  -> top5 HIT
_SCRIPTED_LOGITS = np.array(
    [
        [9.0, 1.0, 0.5, 0.4, 0.3, 0.2],
        [5.0, 9.0, 1.0, 0.5, 0.2, 0.1],
        [5.0, 9.0, 1.0, 0.5, 0.2, 0.1],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ],
    dtype=np.float32,
)
_SCRIPTED_LABELS = np.array([0, 0, 5, 3, 5], dtype=np.int64)
# (5th label pairs with the extra row 4 used only by the ragged-batch
# test: logits [9,8,7,6,5,0] -> argmax miss, unique-minimum value ->
# top5 miss, so rows 4 contributes a miss/miss.)


def _scripted_xy(n: int | None = None):
    """Row-id features for the first ``n`` scripted rows (default all)."""
    n = _SCRIPTED_LOGITS.shape[0] if n is None else n
    X = np.arange(n, dtype=np.float32).reshape(n, 1)
    return X, _SCRIPTED_LABELS[:n]


def test_evaluate_accuracy_matches_hand_computed_values() -> None:
    """Known logits, known labels -> exactly 25% top-1, 75% top-5."""
    net = _ScriptedNet(_SCRIPTED_LOGITS)
    X, y = _scripted_xy()

    acc = evaluate_accuracy(net, X, y, batch_size=4)

    assert acc["top1"] == pytest.approx(25.0)
    assert acc["top5"] == pytest.approx(75.0)
    assert set(acc) == {"top1", "top5"}
    assert all(isinstance(v, float) for v in acc.values())


def test_accuracy_is_identical_across_batch_sizes() -> None:
    """One batch, ragged mini-batches (5 = 2+2+1), and singleton
    batches must produce identical percentages -- the final partial
    batch counts each row exactly once."""
    net = _ScriptedNet(
        np.vstack(
            [
                _SCRIPTED_LOGITS,
                [9.0, 8.0, 7.0, 6.0, 5.0, 0.0],  # row 4: miss/miss
            ]
        )
    )
    X, y = _scripted_xy(5)  # expected: top1 1/5, top5 3/5

    whole = evaluate_accuracy(net, X, y, batch_size=5)
    ragged = evaluate_accuracy(net, X, y, batch_size=2)  # 2, 2, 1
    single = evaluate_accuracy(net, X, y, batch_size=1)

    assert whole["top1"] == pytest.approx(20.0)
    assert whole["top5"] == pytest.approx(60.0)
    assert ragged == whole
    assert single == whole


def test_top5_uses_all_classes_when_output_size_below_five() -> None:
    """With 3 classes the 'top-5' set is really all classes, so any
    valid label scores 100% top-5 (documents the min(5, C) clamp)."""
    net = _ScriptedNet(
        np.array([[1.0, 9.0, 2.0], [8.0, 0.0, 1.0]], dtype=np.float32)
    )
    X = np.arange(2, dtype=np.float32).reshape(2, 1)
    y = np.array([2, 1], dtype=np.int64)  # never the argmax, but in-range

    acc = evaluate_accuracy(net, X, y, batch_size=1)

    assert acc["top1"] == 0.0
    assert acc["top5"] == 100.0


def test_evaluate_accuracy_never_changes_real_parameters() -> None:
    """Read-only contract: forward passes only; weights bit-identical."""
    net, X, y = _tiny_net()
    before = {
        name: getattr(net, name).copy() for name in ("W1", "b1", "W2", "b2")
    }

    evaluate_accuracy(net, X, y, batch_size=2)

    for name, original in before.items():
        np.testing.assert_array_equal(getattr(net, name), original)


def test_evaluate_accuracy_rejects_bad_inputs() -> None:
    """Float labels, out-of-range labels, empty data, and non-finite
    logits all fail loudly."""
    net = _ScriptedNet(_SCRIPTED_LOGITS)
    X, y = _scripted_xy()

    with pytest.raises(TypeError, match="integer"):
        evaluate_accuracy(net, X, y.astype(np.float64))
    with pytest.raises(ValueError, match="0 \\.\\. 5"):
        evaluate_accuracy(net, X, np.array([0, 1, 2, 6], dtype=np.int64))
    with pytest.raises(ValueError, match="empty"):
        evaluate_accuracy(net, X[:0], y[:0])

    diverged = _ScriptedNet(
        np.where(np.arange(6) == 2, np.inf, _SCRIPTED_LOGITS[0]).reshape(1, 6)
    )
    with pytest.raises(ValueError, match="NaN or infinite"):
        evaluate_accuracy(
            diverged,
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
        )


def test_train_records_accuracy_for_every_epoch() -> None:
    """train() gains four accuracy histories, all finite percentages,
    final val accuracy reproducible by a fresh evaluate_accuracy call,
    and top-5 is trivially 100% for a 4-class toy network."""
    X_tr, y_tr, X_v, y_v = _learnable_data()
    net = ChessMoveNet(input_size=6, hidden_size=8, output_size=4, seed=0)

    history = train(
        net, X_tr, y_tr, X_v, y_v,
        epochs=3, batch_size=32, lr=0.01, seed=0, verbose=False,
    )

    for key in (
        "train_acc_top1",
        "train_acc_top5",
        "val_acc_top1",
        "val_acc_top5",
    ):
        assert key in history
        assert len(history[key]) == 3
        assert all(0.0 <= v <= 100.0 for v in history[key])

    # 4 output classes <= 5 -> every valid label is in the "top-5".
    assert history["val_acc_top5"] == [100.0, 100.0, 100.0]

    # The loop's last val accuracy equals a fresh read-only evaluation
    # of the trained net (same weights, same batching -> exact match).
    fresh = evaluate_accuracy(net, X_v, y_v, batch_size=32)
    assert history["val_acc_top1"][-1] == fresh["top1"]
    assert history["val_acc_top5"][-1] == fresh["top5"]





