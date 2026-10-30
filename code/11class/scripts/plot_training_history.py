#!/usr/bin/env python
"""Plot a publication-ready training history from metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    history = pd.DataFrame(metrics.get("history", []))
    required = {"epoch", "train_loss", "val_accuracy", "val_macro_f1"}
    if not required.issubset(history.columns) or history.empty:
        raise ValueError(f"History must contain {sorted(required)}")

    best_index = int(history["val_accuracy"].idxmax())
    best = history.loc[best_index]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(
        args.output_prefix.with_suffix(".csv"),
        index=False,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        color="#D55E00",
        linewidth=1.8,
    )
    axes[0].set_title("(a) Training loss", loc="left", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.22)
    axes[0].set_xlim(1, int(history["epoch"].max()))

    axes[1].plot(
        history["epoch"],
        history["val_accuracy"] * 100.0,
        color="#0072B2",
        linewidth=1.8,
        label="Validation accuracy",
    )
    axes[1].plot(
        history["epoch"],
        history["val_macro_f1"] * 100.0,
        color="#009E73",
        linewidth=1.8,
        label="Validation macro-F1",
    )
    axes[1].axvline(
        best["epoch"],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[1].scatter(
        [best["epoch"]],
        [best["val_accuracy"] * 100.0],
        color="#0072B2",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    axes[1].annotate(
        f"Best validation: {best['val_accuracy'] * 100:.2f}%",
        xy=(best["epoch"], best["val_accuracy"] * 100.0),
        xytext=(8, -22),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "#555555"},
        fontsize=8,
    )
    axes[1].set_title("(b) Validation metrics", loc="left", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_xlim(1, int(history["epoch"].max()))
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, loc="lower right")

    figure.tight_layout(w_pad=2.2)
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print(f"Best epoch: {int(best['epoch'])}")
    print(f"Best validation accuracy: {best['val_accuracy'] * 100:.2f}%")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
