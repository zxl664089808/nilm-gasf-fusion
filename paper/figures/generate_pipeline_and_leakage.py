#!/usr/bin/env python
"""Regenerate the pipeline and leakage-audit figures from repository assets."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


FIGURE_ROOT = Path(__file__).resolve().parent
EVIDENCE = FIGURE_ROOT / "evidence"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def save(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURE_ROOT / f"{stem}.png", dpi=400, bbox_inches="tight")
    figure.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.0,
            edgecolor=color,
            facecolor=f"{color}18",
        )
    )
    axis.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#1f2937",
    )
    axis.text(
        x + width / 2,
        y + height * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=7.5,
        color="#374151",
        linespacing=1.35,
    )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.15,
            color="#4b5563",
            shrinkA=2,
            shrinkB=2,
        )
    )


def pipeline() -> None:
    figure, axis = plt.subplots(figsize=(13.2, 4.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    box(
        axis,
        0.015,
        0.38,
        0.14,
        0.25,
        "Acquisition",
        "Voltage and current\n30 kHz recordings\nzero-load reference",
        "#0072B2",
    )
    box(
        axis,
        0.20,
        0.38,
        0.17,
        0.25,
        "Event-aligned cycles",
        "Upward voltage crossings\n500 samples per cycle\nV, I or dI, P or dP",
        "#D55E00",
    )
    box(
        axis,
        0.415,
        0.66,
        0.18,
        0.20,
        "Gramian image",
        "Per-cycle min-max scaling\nGASF matrices\n64 x 64 RGB stack",
        "#009E73",
    )
    box(
        axis,
        0.65,
        0.66,
        0.17,
        0.20,
        "Image encoder",
        "Four convolutional stages\n16-32-64-128 filters\n0.108M parameters",
        "#009E73",
    )
    box(
        axis,
        0.415,
        0.14,
        0.18,
        0.20,
        "Physical descriptors",
        "RMS, power, factor,\ncrest factor and harmonics\n14 scalar features",
        "#CC79A7",
    )
    box(
        axis,
        0.65,
        0.14,
        0.17,
        0.20,
        "Physical encoder",
        "14 to 32 embedding\nbatch normalization\nand dropout",
        "#CC79A7",
    )
    box(
        axis,
        0.865,
        0.36,
        0.12,
        0.28,
        "Fusion",
        "Concatenate image\nand physical features\n96-unit classifier\nappliance/state",
        "#1f2937",
    )

    arrow(axis, (0.155, 0.505), (0.20, 0.505))
    arrow(axis, (0.37, 0.55), (0.415, 0.69))
    arrow(axis, (0.37, 0.46), (0.415, 0.31))
    arrow(axis, (0.595, 0.76), (0.65, 0.76))
    arrow(axis, (0.595, 0.24), (0.65, 0.24))
    arrow(axis, (0.82, 0.76), (0.865, 0.56))
    arrow(axis, (0.82, 0.24), (0.865, 0.43))

    axis.text(
        0.50,
        0.025,
        "Leakage controls: numerical images contain no text; in the primary "
        "protocol, all cycles from one recording remain in one split.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#374151",
    )
    save(figure, "fig_pipeline")


def leakage() -> None:
    historical = Image.open(EVIDENCE / "historical_label_marked.png").convert("RGB")
    corrected = Image.open(EVIDENCE / "corrected_pure_gadf.png").convert("RGB")

    figure = plt.figure(figsize=(12.4, 4.15))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.35), wspace=0.16)
    panels = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    for axis, image, title in (
        (panels[0], historical, "(a) Historical image"),
        (panels[1], corrected, "(b) Corrected numerical image"),
    ):
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(title, loc="left", fontweight="bold", fontsize=10)

    panels[0].text(
        0.5,
        -0.035,
        "Class and state rendered in the title band",
        transform=panels[0].transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#8b1a1a",
    )
    panels[1].text(
        0.5,
        -0.035,
        "Only numerical RGB pixels",
        transform=panels[1].transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#234f36",
    )

    axis = figure.add_subplot(grid[0, 2])
    conditions = ["Title-bearing\nrows", "Cropped-image\nrows"]
    accuracy = np.asarray([87.83, 76.42])
    macro_f1 = np.asarray([87.45, 73.02])
    x = np.arange(2)
    width = 0.34
    axis.bar(
        x - width / 2,
        accuracy,
        width,
        color="#D55E00",
        label="Accuracy",
    )
    axis.bar(
        x + width / 2,
        macro_f1,
        width,
        color="#0072B2",
        label="Macro-F1",
    )
    axis.set_xticks(x, conditions)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Score (%)")
    axis.set_title(
        "(c) Top 12 rows only",
        loc="left",
        fontweight="bold",
        fontsize=10,
        pad=23,
    )
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for positions, values in ((x - width / 2, accuracy), (x + width / 2, macro_f1)):
        for xpos, value in zip(positions, values):
            axis.text(
                xpos,
                value + 1.7,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
    )
    axis.text(
        0.5,
        -0.24,
        "Image-level split; seed 42; one run per condition",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.7,
        color="#4b5563",
    )
    save(figure, "fig_leakage_audit")


def main() -> None:
    configure()
    pipeline()
    leakage()
    print(f"Wrote pipeline and leakage figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
