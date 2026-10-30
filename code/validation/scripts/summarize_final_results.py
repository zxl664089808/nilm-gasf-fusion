#!/usr/bin/env python
"""Build paper-ready result tables from every experiment directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_recording(root: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^seed_(v|i|p|vi|vip)_(\d+)$")
    for directory in sorted(root.glob("seed_*")):
        match = pattern.match(directory.name)
        if not match or not (directory / "metrics.json").exists():
            continue
        metrics = read_json(directory / "metrics.json")
        best = max(metrics["history"], key=lambda item: item["val_accuracy"])
        rows.append(
            {
                "channels": match.group(1),
                "seed": int(match.group(2)),
                "best_epoch": best["epoch"],
                "val_accuracy": best["val_accuracy"],
                "val_macro_f1": best["val_macro_f1"],
                "test_accuracy": metrics["test_accuracy"],
                "test_macro_f1": metrics["test_macro_f1"],
                "parameters": metrics["parameters"],
                "epochs": len(metrics["history"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_device(root: Path) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(root.glob("device_seed*")):
        seed = int(seed_dir.name.replace("device_seed", ""))
        for run_name in ("image_i", "image_vip", "hybrid_i", "hybrid_vip"):
            run_dir = seed_dir / run_name
            metrics_path = run_dir / "metrics.json"
            summary_path = run_dir / "prediction_summary.json"
            if not metrics_path.exists() or not summary_path.exists():
                continue
            metrics = read_json(metrics_path)
            summary = read_json(summary_path)
            per_device_path = run_dir / "per_device_accuracy.csv"
            class_balanced = None
            zero_devices = None
            if per_device_path.exists():
                per_device = pd.read_csv(per_device_path)
                class_balanced = float(
                    per_device.groupby("true_label")["accuracy"].mean().mean()
                )
                zero_devices = int((per_device["accuracy"] == 0).sum())
            rows.append(
                {
                    "split_seed": seed,
                    "run": run_name,
                    "channels": metrics["channels"],
                    "model": metrics["model"],
                    "val_accuracy": metrics["best_val_accuracy"],
                    "image_accuracy": metrics["test_accuracy"],
                    "image_macro_f1": metrics["test_macro_f1"],
                    "device_weighted_accuracy": summary[
                        "device_weighted_accuracy"
                    ],
                    "device_accuracy_sd": summary["device_accuracy_sd"],
                    "class_balanced_device_accuracy": class_balanced,
                    "zero_accuracy_devices": zero_devices,
                    "parameters": metrics["parameters"],
                    "epochs": len(metrics["history"]),
                }
            )
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    metrics = [
        column
        for column in (
            "val_accuracy",
            "test_accuracy",
            "test_macro_f1",
            "image_accuracy",
            "image_macro_f1",
            "device_weighted_accuracy",
            "class_balanced_device_accuracy",
            "parameters",
            "epochs",
        )
        if column in frame.columns
    ]
    result = frame.groupby(group)[metrics].agg(["mean", "std"]).reset_index()
    result.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in result.columns
    ]
    return result


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def mean_sd(mean: float, standard_deviation: float) -> str:
    if pd.isna(standard_deviation):
        return percent(mean)
    return f"{percent(mean)} ± {percent(standard_deviation)}"


def markdown_summary(
    rec11: pd.DataFrame,
    dev11: pd.DataFrame,
    rec26: pd.DataFrame,
    fusion26: pd.DataFrame,
    multitask26: dict,
) -> str:
    lines = ["# Final result summary", ""]
    lines.extend(["## 11-class unseen recording", ""])
    lines.append("| Channels | Test accuracy | Test macro-F1 |")
    lines.append("|---|---:|---:|")
    for row in rec11.itertuples(index=False):
        lines.append(
            f"| {row.channels} | "
            f"{mean_sd(row.test_accuracy_mean, row.test_accuracy_std)} | "
            f"{mean_sd(row.test_macro_f1_mean, row.test_macro_f1_std)} |"
        )
    lines.extend(["", "## 11-class leave-device-out", ""])
    lines.append("| Run | Image accuracy | Macro-F1 | Device mean | Class-balanced device |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in dev11.itertuples(index=False):
        lines.append(
            f"| {row.run} | "
            f"{mean_sd(row.image_accuracy_mean, row.image_accuracy_std)} | "
            f"{mean_sd(row.image_macro_f1_mean, row.image_macro_f1_std)} | "
            f"{mean_sd(row.device_weighted_accuracy_mean, row.device_weighted_accuracy_std)} | "
            f"{mean_sd(row.class_balanced_device_accuracy_mean, row.class_balanced_device_accuracy_std)} |"
        )
    lines.extend(["", "## 26-state unseen recording", ""])
    lines.append("| Channels | Test accuracy | Test macro-F1 |")
    lines.append("|---|---:|---:|")
    for row in rec26.itertuples(index=False):
        lines.append(
            f"| {row.channels} | "
            f"{mean_sd(row.test_accuracy_mean, row.test_accuracy_std)} | "
            f"{mean_sd(row.test_macro_f1_mean, row.test_macro_f1_std)} |"
        )
    lines.extend(["", "## 26-state model comparison", ""])
    lines.append("| Model | Test accuracy | Test macro-F1 | Parameters |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Physics fusion | {percent(fusion26['test_accuracy'].mean())} ± "
        f"{percent(fusion26['test_accuracy'].std())} | "
        f"{percent(fusion26['test_macro_f1'].mean())} ± "
        f"{percent(fusion26['test_macro_f1'].std())} | "
        f"{int(fusion26['parameters'].iloc[0]):,} |"
    )
    lines.append(
        f"| Multitask seed 42 | {percent(multitask26['test_accuracy'])} | "
        f"{percent(multitask26['test_macro_f1'])} | "
        f"{multitask26['parameters']:,} |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "05_tables",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rec11 = summarize_recording(
        args.suite_root / "07_seed_robustness" / "runs" / "11class"
    )
    dev11 = summarize_device(
        args.suite_root / "08_device_validation" / "runs" / "11class"
    )
    rec26 = summarize_recording(
        args.suite_root / "07_seed_robustness" / "runs" / "26state"
    )
    fusion_dirs = sorted(
        (args.suite_root / "05_physics_fusion" / "runs").glob("seed_*")
    )
    fusion_rows = []
    for directory in fusion_dirs:
        metrics_path = directory / "metrics.json"
        if metrics_path.exists():
            metrics = read_json(metrics_path)
            fusion_rows.append(
                {
                    "seed": metrics["seed"],
                    "test_accuracy": metrics["test_accuracy"],
                    "test_macro_f1": metrics["test_macro_f1"],
                    "best_val_accuracy": metrics["best_val_accuracy"],
                    "parameters": metrics["parameters"],
                    "epochs": len(metrics["history"]),
                }
            )
    fusion26 = pd.DataFrame(fusion_rows)
    multitask_path = (
        args.suite_root
        / "06_multitask_type_state"
        / "runs"
        / "multitask_cnn"
        / "metrics.json"
    )
    multitask26 = read_json(multitask_path)

    rec11.to_csv(args.output_root / "11class_recording_runs.csv", index=False)
    aggregate(rec11, ["channels"]).to_csv(
        args.output_root / "11class_recording_summary.csv",
        index=False,
    )
    dev11.to_csv(args.output_root / "11class_device_runs.csv", index=False)
    aggregate(dev11, ["run"]).to_csv(
        args.output_root / "11class_device_summary.csv",
        index=False,
    )
    rec26.to_csv(args.output_root / "26state_image_runs.csv", index=False)
    aggregate(rec26, ["channels"]).to_csv(
        args.output_root / "26state_image_summary.csv",
        index=False,
    )
    fusion26.to_csv(args.output_root / "26state_fusion_runs.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "physics_fusion",
                "test_accuracy_mean": fusion26["test_accuracy"].mean(),
                "test_accuracy_std": fusion26["test_accuracy"].std(),
                "test_macro_f1_mean": fusion26["test_macro_f1"].mean(),
                "test_macro_f1_std": fusion26["test_macro_f1"].std(),
                "parameters": int(fusion26["parameters"].iloc[0]),
            },
            {
                "model": "multitask_seed42",
                "test_accuracy_mean": multitask26["test_accuracy"],
                "test_accuracy_std": 0.0,
                "test_macro_f1_mean": multitask26["test_macro_f1"],
                "test_macro_f1_std": 0.0,
                "parameters": multitask26["parameters"],
            },
        ]
    ).to_csv(args.output_root / "26state_model_comparison.csv", index=False)
    (args.output_root / "final_result_summary.md").write_text(
        markdown_summary(
            aggregate(rec11, ["channels"]),
            aggregate(dev11, ["run"]),
            aggregate(rec26, ["channels"]),
            fusion26,
            multitask26,
        ),
        encoding="utf-8",
    )
    print(f"Wrote final result tables to {args.output_root}")


if __name__ == "__main__":
    main()
