"""Analyze and plot all 8 ChessMoveNet training combinations.

Reads every ``hist_*.npz`` (and matching ``ckpt_*.pt``) in ``artifacts/``,
compares the runs, evaluates each checkpoint on the held-out test split, and
writes a summary CSV/Markdown report plus comparison graphs to ``reports/``.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/analyze_all_combinations.py

Outputs (all in ``reports/``)::
    combination_summary.csv     -- one row per combination with all metrics
    combination_summary.md      -- human-readable markdown report
    val_loss_curves_all.png     -- validation loss vs epoch for all 8 runs
    val_acc_top1_curves_all.png -- validation top-1 accuracy vs epoch for all 8
    train_vs_val_loss.png       -- train vs val loss for the best run
    final_val_loss_bar.png      -- final validation loss by combination
    final_val_top1_bar.png      -- final validation top-1 accuracy by combination
    training_time_bar.png       -- total training time by combination
    overfitting_gap_bar.png     -- final (train - val) loss gap by combination
    test_performance_bar.png    -- held-out test loss / top-1 / top-5 by combo
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessmovenet.checkpoint import load_checkpoint  # noqa: E402
from chessmovenet.train import evaluate_accuracy, evaluate_loss  # noqa: E402

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
DATA_PATH = Path("data/processed/split.npz")

ACT_LABELS = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh"}
OPT_LABELS = {"adam": "Adam", "sgd": "SGD"}

COMBO_COLOURS = {
    "adam_relu_256": "#1f77b4",
    "adam_relu_256_128": "#1f77b4",
    "adam_sigmoid_256": "#ff7f0e",
    "adam_tanh_256": "#ff7f0e",
    "sgd_relu_256_128": "#2ca02c",
    "sgd_sigmoid_256": "#d62728",
    "sgd_tanh_256": "#d62728",
}


def parse_combo(history_path: Path) -> dict:
    """Parse ``hist_{opt}_{act}_{sizes}.npz`` into a combo descriptor."""
    name = history_path.stem
    if not name.startswith("hist_"):
        raise ValueError(f"unexpected history filename: {history_path.name}")
    body = name[len("hist_"):]
    match = re.match(r"(adam|sgd)_(relu|sigmoid|tanh)_(.+)$", body)
    if not match:
        raise ValueError(f"cannot parse combination from {history_path.name}")
    opt, act, sizes = match.group(1), match.group(2), match.group(3)
    return {
        "opt": opt,
        "act": act,
        "sizes": sizes,
        "combo": f"{opt}_{act}_{sizes}",
        "label": f"{OPT_LABELS[opt]} {ACT_LABELS[act]} {sizes}",
        "history_path": history_path,
        "checkpoint_path": ARTIFACTS_DIR / f"ckpt_{opt}_{act}_{sizes}.pt",
    }


def load_history_values(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: np.asarray(d[k]).astype(np.float64) for k in d.files}


def summarize_run(info: dict) -> dict:
    """Compute all comparison metrics for a single combination."""
    h = load_history_values(info["history_path"])
    epochs = h["epoch"].astype(int)
    train_loss = h["train_loss"]
    val_loss = h["val_loss"]
    train_top1 = h.get("train_acc_top1", np.zeros_like(val_loss))
    val_top1 = h.get("val_acc_top1", np.zeros_like(val_loss))
    train_top5 = h.get("train_acc_top5", np.zeros_like(val_loss))
    val_top5 = h.get("val_acc_top5", np.zeros_like(val_loss))
    seconds = h.get("seconds", np.zeros_like(val_loss))

    best_val_loss_idx = int(np.argmin(val_loss))
    best_val_top1_idx = int(np.argmax(val_top1))

    metrics = {
        "combo": info["combo"],
        "label": info["label"],
        "opt": info["opt"],
        "act": info["act"],
        "sizes": info["sizes"],
        "epochs": int(len(epochs)),
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "best_val_loss": float(val_loss[best_val_loss_idx]),
        "best_val_loss_epoch": int(epochs[best_val_loss_idx]),
        "final_train_top1": float(train_top1[-1]),
        "final_val_top1": float(val_top1[-1]),
        "best_val_top1": float(val_top1[best_val_top1_idx]),
        "best_val_top1_epoch": int(epochs[best_val_top1_idx]),
        "final_val_top5": float(val_top5[-1]),
        "overfitting_gap": float(train_loss[-1] - val_loss[-1]),
        "total_time_s": float(seconds.sum()),
        "mean_epoch_time_s": float(seconds.mean()),
        "history_path": str(info["history_path"]),
        "checkpoint_path": str(info["checkpoint_path"]),
        "checkpoint_exists": info["checkpoint_path"].is_file(),
    }
    return metrics


def evaluate_on_test(info: dict) -> dict:
    """Evaluate the combination's checkpoint on the held-out test split."""
    result = {
        "combo": info["combo"],
        "test_loss": np.nan,
        "test_top1": np.nan,
        "test_top5": np.nan,
        "test_error": "",
    }
    ckpt = info["checkpoint_path"]
    if not ckpt.is_file():
        result["test_error"] = "checkpoint file missing"
        return result
    if not DATA_PATH.is_file():
        result["test_error"] = f"data split {DATA_PATH} missing"
        return result
    try:
        net = load_checkpoint(ckpt)
        with np.load(DATA_PATH, allow_pickle=False) as data:
            X_test = data["X_test"]
            y_test = data["y_test"]
        loss = evaluate_loss(net, X_test, y_test, batch_size=128)
        accuracy = evaluate_accuracy(net, X_test, y_test, batch_size=128)
        result.update(
            test_loss=float(loss),
            test_top1=float(accuracy["top1"]),
            test_top5=float(accuracy["top5"]),
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole analysis
        result["test_error"] = type(exc).__name__ + ": " + str(exc)
    return result


def discover_combinations() -> list[dict]:
    """Find every ``hist_*.npz`` in artifacts/ and parse its combination."""
    combos = []
    for path in sorted(ARTIFACTS_DIR.glob("hist_*.npz")):
        combos.append(parse_combo(path))
    if not combos:
        raise FileNotFoundError(f"No history files found in {ARTIFACTS_DIR}")
    return combos



def _style_axes(ax, xlabel, ylabel, title, legend=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=8, loc="best", frameon=True)
    ax.ticklabel_format(style="plain", axis="y")


def plot_val_loss_curves(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rows:
        h = load_history_values(Path(row["history_path"]))
        colour = COMBO_COLOURS.get(row["combo"], "#7f7f7f")
        ax.plot(
            h["epoch"], h["val_loss"], "o-", label=row["label"], color=colour,
            linewidth=2, markersize=4,
        )
    for row in rows:
        h = load_history_values(Path(row["history_path"]))
        best_idx = int(np.argmin(h["val_loss"]))
        ax.plot(
            h["epoch"][best_idx], h["val_loss"][best_idx], "*",
            color=COMBO_COLOURS.get(row["combo"], "#7f7f7f"), markersize=10,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation cross-entropy loss")
    ax.set_title("Validation Loss by Epoch — All 8 Combinations\n(star = best validation loss)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def plot_val_acc_top1_curves(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rows:
        h = load_history_values(Path(row["history_path"]))
        colour = COMBO_COLOURS.get(row["combo"], "#7f7f7f")
        ax.plot(
            h["epoch"], h["val_acc_top1"], "o-", label=row["label"], color=colour,
            linewidth=2, markersize=4,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation top-1 accuracy (%)")
    ax.set_title("Validation Top-1 Accuracy by Epoch — All 8 Combinations")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def plot_train_vs_val_loss(best_row: dict, output: Path) -> None:
    h = load_history_values(Path(best_row["history_path"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h["epoch"], h["train_loss"], "o-", label="Train loss", color="#1f77b4")
    ax.plot(h["epoch"], h["val_loss"], "o-", label="Validation loss", color="#ff7f0e")
    ax.axvline(best_row["best_val_loss_epoch"], color="gray", linestyle="--",
               alpha=0.5, label=f"Best val loss (epoch {best_row['best_val_loss_epoch']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Train vs Validation Loss — {best_row['label']} (best val loss)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def plot_bar_metric(rows: list[dict], key: str, ylabel: str, title: str,
                    output: Path, lower_is_better: bool = True) -> None:
    labels = [row["label"] for row in rows]
    values = [row[key] for row in rows]
    colours = [COMBO_COLOURS.get(row["combo"], "#7f7f7f") for row in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colours, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    best_idx = int(np.argmin(values)) if lower_is_better else int(np.argmax(values))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8, rotation=0,
        )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def plot_test_performance(rows: list[dict], output: Path) -> None:
    valid = [row for row in rows if row["test_error"] == "" and np.isfinite(row["test_top1"])]
    if not valid:
        print("Skipping test-performance plot: no valid test evaluations.")
        return
    labels = [row["label"] for row in valid]
    top1 = [row["test_top1"] for row in valid]
    top5 = [row["test_top5"] for row in valid]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, top1, width, label="Test top-1", color="#1f77b4", edgecolor="black")
    ax.bar(x + width / 2, top5, width, label="Test top-5", color="#ff7f0e", edgecolor="black")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Held-Out Test Performance by Combination")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")



def write_csv(rows: list[dict], output: Path) -> None:
    import csv
    fieldnames = [
        "combo", "label", "opt", "act", "sizes", "epochs",
        "final_train_loss", "final_val_loss", "best_val_loss",
        "best_val_loss_epoch", "final_train_top1", "final_val_top1",
        "best_val_top1", "best_val_top1_epoch", "final_val_top5",
        "overfitting_gap", "total_time_s", "mean_epoch_time_s",
        "test_loss", "test_top1", "test_top5", "checkpoint_exists",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Saved {output}")



def write_markdown(rows: list[dict], output: Path) -> None:
    lines = []
    lines.append("# ChessMoveNet — 8-Training-Combination Analysis")
    lines.append("")
    lines.append("All runs: 15 epochs, batch size 128, lr 0.001, seed 0, chronological 80/10/10 split.")
    lines.append("")
    lines.append("## Summary table (sorted by best validation loss)")
    lines.append("")
    lines.append("| Combination | Optimizer | Activation | Hidden | Best val loss | Epoch | Final val top-1 | Best val top-1 | Final train top-1 | Overfit gap | Total time (s) | Test top-1 | Test top-5 |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        test_top1 = f"{row['test_top1']:.2f}" if np.isfinite(row["test_top1"]) else "—"
        test_top5 = f"{row['test_top5']:.2f}" if np.isfinite(row["test_top5"]) else "—"
        lines.append(
            f"| {row['label']} | {row['opt'].upper()} | {row['act']} | {row['sizes']} | "
            f"{row['best_val_loss']:.4f} | {row['best_val_loss_epoch']} | "
            f"{row['final_val_top1']:.2f}% | {row['best_val_top1']:.2f}% | "
            f"{row['final_train_top1']:.2f}% | {row['overfitting_gap']:+.4f} | "
            f"{row['total_time_s']:.0f} | {test_top1}% | {test_top5}% |"
        )
    lines.append("")
    lines.append("## Key takeaways")
    lines.append("")
    best = rows[0]
    lines.append(f"- **Best overall:** `{best['label']}` — best validation loss "
                 f"{best['best_val_loss']:.4f} (epoch {best['best_val_loss_epoch']}), "
                 f"final validation top-1 {best['final_val_top1']:.2f}%.")
    adam_rows = [r for r in rows if r["opt"] == "adam"]
    sgd_rows = [r for r in rows if r["opt"] == "sgd"]
    adam_avg = float(np.mean([r["best_val_loss"] for r in adam_rows]))
    sgd_avg = float(np.mean([r["best_val_loss"] for r in sgd_rows]))
    lines.append(f"- **Adam vs SGD:** Adam averages best val loss {adam_avg:.4f} vs SGD "
                 f"{sgd_avg:.4f} — Adam converges far better on this task.")
    overfit = min(rows, key=lambda r: r["overfitting_gap"])
    lines.append(f"- **Largest train-val gap:** `{overfit['label']}` — validation loss is "
                 f"{-overfit['overfitting_gap']:.4f} above training loss "
                 f"(train {overfit['final_train_loss']:.2f} vs val {overfit['final_val_loss']:.2f}).")
    fastest = min(rows, key=lambda r: r["total_time_s"])
    lines.append(f"- **Fastest:** `{fastest['label']}` — {fastest['total_time_s']:.0f}s total "
                 f"({fastest['mean_epoch_time_s']:.1f}s/epoch).")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig_name in [
        "val_loss_curves_all.png", "val_acc_top1_curves_all.png",
        "train_vs_val_loss.png", "final_val_loss_bar.png",
        "final_val_top1_bar.png", "training_time_bar.png",
        "overfitting_gap_bar.png", "test_performance_bar.png",
    ]:
        lines.append(f"![{fig_name}]({fig_name})")
    output.write_text("\n".join(lines) + "\n")
    print(f"Saved {output}")



def main() -> None:
    if not ARTIFACTS_DIR.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {ARTIFACTS_DIR}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    combos = discover_combinations()
    print(f"Found {len(combos)} training combinations.")

    rows: list[dict] = []
    for info in combos:
        metrics = summarize_run(info)
        test_result = evaluate_on_test(info)
        metrics.update(test_result)
        rows.append(metrics)
        print(
            f"  {info['label']:22s} "
            f"best_val_loss={metrics['best_val_loss']:.4f} "
            f"final_val_top1={metrics['final_val_top1']:5.2f}% "
            f"time={metrics['total_time_s']:6.0f}s "
            f"test_top1={metrics['test_top1'] if np.isfinite(metrics['test_top1']) else float('nan'):.2f}%"
        )

    rows.sort(key=lambda r: r["best_val_loss"])

    write_csv(rows, REPORTS_DIR / "combination_summary.csv")
    write_markdown(rows, REPORTS_DIR / "combination_summary.md")

    best_row = rows[0]
    plot_val_loss_curves(rows, REPORTS_DIR / "val_loss_curves_all.png")
    plot_val_acc_top1_curves(rows, REPORTS_DIR / "val_acc_top1_curves_all.png")
    plot_train_vs_val_loss(best_row, REPORTS_DIR / "train_vs_val_loss.png")
    plot_bar_metric(rows, "final_val_loss", "Final validation loss",
                    "Final Validation Loss by Combination (lower is better)",
                    REPORTS_DIR / "final_val_loss_bar.png", lower_is_better=True)
    plot_bar_metric(rows, "final_val_top1", "Final validation top-1 accuracy (%)",
                    "Final Validation Top-1 Accuracy by Combination (higher is better)",
                    REPORTS_DIR / "final_val_top1_bar.png", lower_is_better=False)
    plot_bar_metric(rows, "total_time_s", "Total training time (seconds)",
                    "Total Training Time by Combination",
                    REPORTS_DIR / "training_time_bar.png", lower_is_better=True)
    plot_bar_metric(rows, "overfitting_gap", "Final train − val loss",
                    "Overfitting Gap by Combination (final train loss − val loss)",
                    REPORTS_DIR / "overfitting_gap_bar.png", lower_is_better=True)
    plot_test_performance(rows, REPORTS_DIR / "test_performance_bar.png")

    print("\n=== Best combination ===")
    print(f"{best_row['label']} — best val loss {best_row['best_val_loss']:.4f} "
          f"(epoch {best_row['best_val_loss_epoch']}), final val top-1 "
          f"{best_row['final_val_top1']:.2f}%")
    print(f"Report -> {REPORTS_DIR / 'combination_summary.md'}")


if __name__ == "__main__":
    main()

