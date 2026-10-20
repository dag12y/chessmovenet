"""Tests for ``scripts/plot_history.py`` using synthetic history data.

No real training data or model is used.  All history files are created
synthetically in ``tmp_path`` and loaded read-only with ``allow_pickle=False``.
"""

from __future__ import annotations

import numpy as np
import pytest

# Import the module under test by path so we don't need the package installed.
import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "plot_history.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_history", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


def _write_history(tmp_path, n=5, with_accuracy=True, **extra_cols):
    """Write a synthetic history .npz and return its path."""
    path = tmp_path / "history.npz"
    columns = {
        "epoch": np.arange(1, n + 1, dtype=np.int64),
        "train_loss": np.linspace(9.0, 7.0, n),
        "val_loss": np.linspace(8.8, 6.9, n),
        "seconds": np.full(n, 10.0),
    }
    if with_accuracy:
        columns.update(
            {
                "train_acc_top1": np.linspace(1.0, 6.0, n),
                "train_acc_top5": np.linspace(5.0, 11.0, n),
                "val_acc_top1": np.linspace(0.9, 5.8, n),
                "val_acc_top5": np.linspace(4.8, 10.5, n),
            }
    )
    columns.update(extra_cols)
    np.savez(path, **columns)
    return path


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------

def test_load_history_reads_npz_without_pickle(tmp_path, mod):
    path = _write_history(tmp_path)
    data = mod.load_history(path)
    assert set(data) == {
        "epoch", "train_loss", "val_loss", "seconds",
        "train_acc_top1", "train_acc_top5", "val_acc_top1", "val_acc_top5",
    }
    assert data["epoch"].dtype == np.int64
    assert len(data["epoch"]) == 5


def test_load_history_missing_file(tmp_path, mod):
    missing = tmp_path / "does_not_exist.npz"
    with pytest.raises(FileNotFoundError, match="not found"):
        mod.load_history(missing)


def test_load_history_missing_required_key(tmp_path, mod):
    path = tmp_path / "bad.npz"
    np.savez(path, epoch=np.array([1]), train_loss=np.array([8.0]))
    with pytest.raises(ValueError, match="missing required key"):
        mod.load_history(path)


def test_load_history_allow_pickle_false(tmp_path, mod):
    """The file must load with allow_pickle=False (no pickle objects)."""
    path = _write_history(tmp_path)
    with np.load(path, allow_pickle=False) as stored:
        assert stored.files  # readable without pickle


# ---------------------------------------------------------------------------
# _validate_1d
# ---------------------------------------------------------------------------

def test_validate_1d_rejects_2d_column(tmp_path, mod):
    path = _write_history(tmp_path, with_accuracy=False)
    data = mod.load_history(path)
    data["train_loss"] = data["train_loss"].reshape(-1, 1)  # make it 2-D
    with pytest.raises(ValueError, match="must be 1-D"):
        mod._validate_1d(data)


def test_validate_1d_rejects_mismatched_lengths(tmp_path, mod):
    path = _write_history(tmp_path, with_accuracy=False)
    data = mod.load_history(path)
    data["val_loss"] = data["val_loss"][:3]  # shorter than epoch
    with pytest.raises(ValueError, match="inconsistent lengths"):
        mod._validate_1d(data)


# ---------------------------------------------------------------------------
# plot_loss
# ---------------------------------------------------------------------------

def test_plot_loss_creates_png(tmp_path, mod):
    path = _write_history(tmp_path, with_accuracy=False)
    history = mod.load_history(path)
    out = tmp_path / "loss.png"
    mod.plot_loss(history, out)
    assert out.is_file()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_accuracy
# ---------------------------------------------------------------------------

def test_plot_accuracy_creates_png(tmp_path, mod):
    path = _write_history(tmp_path)
    history = mod.load_history(path)
    out = tmp_path / "acc.png"
    mod.plot_accuracy(history, out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_accuracy_raises_when_no_accuracy(tmp_path, mod):
    path = _write_history(tmp_path, with_accuracy=False)
    history = mod.load_history(path)
    out = tmp_path / "acc.png"
    with pytest.raises(ValueError, match="no accuracy columns"):
        mod.plot_accuracy(history, out)


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------

def test_main_default_history(tmp_path, monkeypatch, mod):
    """CLI with default path works when the default file exists."""
    # Point REPORTS_DIR and plots into tmp_path by monkeypatching module attrs.
    mod.REPORTS_DIR = tmp_path
    mod.LOSS_PLOT = tmp_path / "loss_curves.png"
    mod.ACC_PLOT = tmp_path / "accuracy_curves.png"

    # Create a fake default history file where the script expects it.
    default = Path("artifacts/history_ep15.npz")
    if not default.is_file():
        pytest.skip("real default history file not present in this env")

    # Run main with no arguments -> uses default path.
    import sys
    old_argv = sys.argv
    sys.argv = ["plot_history.py"]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    assert mod.LOSS_PLOT.is_file()
    assert mod.ACC_PLOT.is_file()


def test_main_custom_history_file(tmp_path, mod):
    """CLI with an explicit history file path."""
    mod.REPORTS_DIR = tmp_path
    mod.LOSS_PLOT = tmp_path / "loss_curves.png"
    mod.ACC_PLOT = tmp_path / "accuracy_curves.png"

    hist = _write_history(tmp_path)

    import sys
    old_argv = sys.argv
    sys.argv = ["plot_history.py", str(hist)]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    assert mod.LOSS_PLOT.is_file()
    assert mod.ACC_PLOT.is_file()


def test_main_missing_file_exits(mod):
    """CLI with a missing file exits with code 1 and prints an error."""
    import sys
    old_argv = sys.argv
    old_stderr = sys.stderr
    sys.argv = ["plot_history.py", "/nonexistent/history.npz"]
    captured = []
    class FakeStderr:
        def write(self, s):
            captured.append(s)
        def flush(self):
            pass
    sys.stderr = FakeStderr()
    try:
        with pytest.raises(SystemExit) as exc:
            mod.main()
        assert exc.value.code == 1
    finally:
        sys.argv = old_argv
        sys.stderr = old_stderr
    assert any("not found" in c for c in captured)


def test_main_warns_on_existing_output(tmp_path, mod, capsys):
    """CLI warns when output files already exist but still proceeds."""
    mod.REPORTS_DIR = tmp_path
    mod.LOSS_PLOT = tmp_path / "loss_curves.png"
    mod.ACC_PLOT = tmp_path / "accuracy_curves.png"
    mod.LOSS_PLOT.touch()  # pre-existing file

    hist = _write_history(tmp_path, with_accuracy=False)

    import sys
    old_argv = sys.argv
    sys.argv = ["plot_history.py", str(hist)]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    assert mod.LOSS_PLOT.is_file()
    # Accuracy is skipped because with_accuracy=False, but loss was written.
    combined = captured.out + captured.err
    assert any("Warning" in c for c in combined.splitlines())
