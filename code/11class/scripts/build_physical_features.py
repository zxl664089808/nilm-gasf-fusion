#!/usr/bin/env python
"""Build 14 physical features for every image in an 11-class manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nilm_11class.extraction import extract_aligned_cycles
from nilm_11class.physics import FEATURE_NAMES, physical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-cycle", type=int, default=500)
    return parser.parse_args()


def parse_source(source_id: str) -> tuple[str, int]:
    if "|sample_" not in source_id:
        raise ValueError(f"Unexpected source_id: {source_id}")
    filename, sample_index = source_id.rsplit("|sample_", 1)
    return filename, int(sample_index)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    required = {"image_path", "label", "source_id", "source_cycle_index"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    values = np.full((len(frame), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for source_id, indices in tqdm(
        frame.groupby("source_id", sort=False).groups.items(),
        desc="source samples",
    ):
        filename, _ = parse_source(str(source_id))
        csv_path = args.csv_root / filename
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        data = np.genfromtxt(csv_path, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Unexpected CSV shape in {csv_path}")
        current = np.asarray(data[:, 0], dtype=np.float64)
        voltage = np.asarray(data[:, 1], dtype=np.float64)
        cycles = extract_aligned_cycles(
            current,
            voltage,
            samples_per_cycle=args.samples_per_cycle,
        )
        for row_index in indices:
            cycle_index = int(frame.loc[row_index, "source_cycle_index"])
            if cycle_index >= len(cycles):
                raise IndexError(
                    f"Cycle {cycle_index} out of range for {source_id}"
                )
            current_cycle, voltage_cycle = cycles[cycle_index]
            power_cycle = current_cycle * voltage_cycle
            values[row_index] = physical_features(
                voltage_cycle,
                current_cycle,
                power_cycle,
            )
    if np.isnan(values).any():
        raise RuntimeError("Physical feature generation left missing values")
    result = frame[["image_path", "label", "source_id"]].copy()
    for column, name in enumerate(FEATURE_NAMES):
        result[name] = values[:, column]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(result):,} rows")


if __name__ == "__main__":
    main()
