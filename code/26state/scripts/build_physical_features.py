#!/usr/bin/env python
"""Build one physical-feature row for every image in a 26-state manifest."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nilm_pipeline.datasets import cycle_arrays
from nilm_pipeline.physics import FEATURE_NAMES, physical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    required = {"image_path", "source_pkl", "source_cycle_index"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    values = np.full((len(frame), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for source_pkl, indices in tqdm(
        frame.groupby("source_pkl", sort=False).groups.items(),
        desc="source recordings",
    ):
        with Path(source_pkl).open("rb") as handle:
            cycles = pickle.load(handle)["diff_results"]
        for row_index in indices:
            row = frame.iloc[row_index]
            cycle_index = int(row["source_cycle_index"])
            voltage, current, power = cycle_arrays(cycles[cycle_index])
            values[row_index] = physical_features(voltage, current, power)
    if np.isnan(values).any():
        raise RuntimeError("Physical feature generation left missing values")
    result = frame[["image_path", "label", "split", "source_id"]].copy()
    for column, name in enumerate(FEATURE_NAMES):
        result[name] = values[:, column]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(result):,} rows")


if __name__ == "__main__":
    main()
