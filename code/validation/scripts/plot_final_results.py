#!/usr/bin/env python
"""Generate publication-ready summary figures from finished experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_final_results import (
    aggregate,
    read_json,
    summarize_device,
    summarize_recording,
)


def style_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def save(figure, output_root: Path, name: str):
    figure.savefig(output_root / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_root / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def channel_figure(frame: pd.DataFrame, output_root: Path):
    order = ["i", "p", "vi", "vip"]
    frame = frame.set_index("channels").loc[order].reset_index()
    x = np.arange(len(frame))
    width = 0.34
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.bar(
        x - width / 2,
        frame["test_accuracy_mean"] * 100,
        width,
        yerr=frame["test_accuracy_std"] * 100,
        capsize=3,
        label="Accuracy",
        color="#0072B2",
    )
    axis.bar(
        x + width / 2,
        frame["test_macro_f1_mean"] * 100,
        width,
        yerr=frame["test_macro_f1_std"] * 100,
        capsize=3,
        label="Macro-F1",
        color="#009E73",
    )
    axis.set_xticks(x, [value.upper() for value in frame["channels"]])
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 100)
    axis.legend(frameon=False)
    style_axes(axis)
    figure.tight_layout()
    save(figure, output_root, "fig_11class_recording_channels")


def device_figure(frame: pd.DataFrame, output_root: Path):
    label_map = {
        "image_i": "Image I",
        "image_vip": "Image V+I+P",
        "hybrid_i": "Hybrid I",
        "hybrid_vip": "Hybrid V+I+P",
    }
    order = ["image_i", "hybrid_i", "image_vip", "hybrid_vip"]
    frame = frame.set_index("run").loc[order].reset_index()
    x = np.arange(len(frame))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for axis, mean_col, std_col, title, color in (
        (
            axes[0],
            "image_accuracy_mean",
            "image_accuracy_std",
            "(a) Image-weighted accuracy",
            "#0072B2",
        ),
        (
            axes[1],
            "device_weighted_accuracy_mean",
            "device_weighted_accuracy_std",
            "(b) Device-balanced accuracy",
            "#D55E00",
        ),
    ):
        errors = frame[std_col].fillna(0) * 100
        axis.bar(x, frame[mean_col] * 100, width=0.58, color=color, yerr=errors, capsize=3)
        axis.set_xticks(x, [label_map[value] for value in frame["run"]], rotation=20, ha="right")
        axis.set_ylabel("Score (%)")
        axis.set_ylim(0, 100)
        axis.set_title(title, loc="left", fontweight="bold")
        style_axes(axis)
    figure.tight_layout()
    save(figure, output_root, "fig_11class_device_models")


def model_comparison_figure(
    image_frame: pd.DataFrame,
    fusion_runs: pd.DataFrame,
    multitask: dict,
    output_root: Path,
):
    vip = image_frame[image_frame["channels"] == "vip"].iloc[0]
    rows = [
        {
            "model": "Image V+I+P",
            "accuracy": vip["test_accuracy_mean"],
            "accuracy_sd": vip["test_accuracy_std"],
            "f1": vip["test_macro_f1_mean"],
            "f1_sd": vip["test_macro_f1_std"],
        },
        {
            "model": "Physics fusion",
            "accuracy": fusion_runs["test_accuracy"].mean(),
            "accuracy_sd": fusion_runs["test_accuracy"].std(),
            "f1": fusion_runs["test_macro_f1"].mean(),
            "f1_sd": fusion_runs["test_macro_f1"].std(),
        },
        {
            "model": "Multitask",
            "accuracy": multitask["test_accuracy"],
            "accuracy_sd": 0.0,
            "f1": multitask["test_macro_f1"],
            "f1_sd": 0.0,
        },
    ]
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    width = 0.34
    figure, axis = plt.subplots(figsize=(6.8, 3.7))
    axis.bar(
        x - width / 2,
        frame["accuracy"] * 100,
        width,
        yerr=frame["accuracy_sd"] * 100,
        capsize=3,
        color="#0072B2",
        label="Accuracy",
    )
    axis.bar(
        x + width / 2,
        frame["f1"] * 100,
        width,
        yerr=frame["f1_sd"] * 100,
        capsize=3,
        color="#009E73",
        label="Macro-F1",
    )
    axis.set_xticks(x, frame["model"])
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 100)
    axis.legend(frameon=False)
    style_axes(axis)
    figure.tight_layout()
    save(figure, output_root, "fig_26state_model_comparison")


def class_recall_figure(
    suite_root: Path,
    output_root: Path,
):
    selected = [
        "Coffee maker_OFF",
        "Coffee maker_ON",
        "Water kettle_OFF",
        "Water kettle_ON",
        "Fridge_OFF",
        "Fan_ON",
        "Incandescent Light Bulb_ON",
    ]
    baseline_root = suite_root / "07_seed_robustness" / "runs" / "26state"
    fusion_root = suite_root / "05_physics_fusion" / "runs"
    values = {}
    for name, directories in (
        (
            "Image V+I+P",
            sorted(baseline_root.glob("seed_vip_*")),
        ),
        (
            "Physics fusion",
            sorted(fusion_root.glob("seed_*")),
        ),
    ):
        recalls = {label: [] for label in selected}
        for directory in directories:
            metrics_path = directory / "metrics.json"
            confusion_path = directory / "confusion_matrix.csv"
            if not metrics_path.exists() or not confusion_path.exists():
                continue
            metrics = read_json(metrics_path)
            names = metrics["class_names"]
            matrix = pd.read_csv(confusion_path, index_col=0).to_numpy()
            for label in selected:
                index = names.index(label)
                recalls[label].append(matrix[index, index] / matrix[index].sum())
        values[name] = {
            label: (float(np.mean(recalls[label])), float(np.std(recalls[label])))
            for label in selected
        }
    labels = [label.replace("Incandescent Light Bulb", "Incand. Bulb") for label in selected]
    x = np.arange(len(selected))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.5, 4.1))
    for offset, (name, color) in zip(
        (-width / 2, width / 2),
        (("Image V+I+P", "#0072B2"), ("Physics fusion", "#009E73")),
    ):
        mean = np.asarray([values[name][label][0] for label in selected]) * 100
        sd = np.asarray([values[name][label][1] for label in selected]) * 100
        axis.bar(x + offset, mean, width, yerr=sd, capsize=2, color=color, label=name)
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylabel("Recall (%)")
    axis.set_ylim(0, 105)
    axis.legend(frameon=False)
    style_axes(axis)
    figure.tight_layout()
    save(figure, output_root, "fig_26state_class_recall")


def main():
    suite_root = Path(__file__).resolve().parents[1]
    workspace_root = suite_root.parent
    output_root = workspace_root / "04_figures"
    output_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    rec11 = aggregate(
        summarize_recording(suite_root / "07_seed_robustness" / "runs" / "11class"),
        ["channels"],
    )
    dev11 = aggregate(
        summarize_device(
            suite_root / "08_device_validation" / "runs" / "11class"
        ),
        ["run"],
    )
    rec26 = aggregate(
        summarize_recording(suite_root / "07_seed_robustness" / "runs" / "26state"),
        ["channels"],
    )
    fusion_runs = pd.DataFrame(
        [
            read_json(directory / "metrics.json")
            for directory in sorted(
                (suite_root / "05_physics_fusion" / "runs").glob("seed_*")
            )
            if (directory / "metrics.json").exists()
        ]
    )
    multitask = read_json(
        suite_root
        / "06_multitask_type_state"
        / "runs"
        / "multitask_cnn"
        / "metrics.json"
    )
    channel_figure(rec11, output_root)
    device_figure(dev11, output_root)
    model_comparison_figure(rec26, fusion_runs, multitask, output_root)
    class_recall_figure(suite_root, output_root)
    print(f"Wrote figures to {output_root}")


if __name__ == "__main__":
    main()
