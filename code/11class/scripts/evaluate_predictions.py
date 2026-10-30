#!/usr/bin/env python
"""Evaluate image-weighted and device-weighted prediction performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column", default="device_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    required = {"true_label", "predicted_label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Predictions must contain {sorted(required)}")
    group_column = args.group_column
    if group_column not in frame.columns:
        group_column = "source_id"
    if group_column not in frame.columns:
        raise ValueError("Predictions need device_id or source_id")
    correct = frame["true_label"] == frame["predicted_label"]
    device_rows = []
    for group, part in frame.groupby(group_column, sort=True):
        device_rows.append(
            {
                group_column: group,
                "true_label": part["true_label"].mode().iat[0],
                "images": len(part),
                "accuracy": float(
                    (part["true_label"] == part["predicted_label"]).mean()
                ),
            }
        )
    device_frame = pd.DataFrame(device_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_frame.to_csv(
        args.output_dir / "per_device_accuracy.csv",
        index=False,
    )
    summary = {
        "images": len(frame),
        "groups": int(device_frame.shape[0]),
        "group_column": group_column,
        "image_accuracy": float(accuracy_score(
            frame["true_label"],
            frame["predicted_label"],
        )),
        "image_macro_f1": float(f1_score(
            frame["true_label"],
            frame["predicted_label"],
            average="macro",
        )),
        "device_weighted_accuracy": float(device_frame["accuracy"].mean()),
        "device_accuracy_sd": float(device_frame["accuracy"].std()),
        "device_accuracy_min": float(device_frame["accuracy"].min()),
        "device_accuracy_max": float(device_frame["accuracy"].max()),
        "correct_images": int(correct.sum()),
    }
    (args.output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
