#!/usr/bin/env python
"""Cache 3x500 waveforms in manifest order for encoding ablations."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nilm_pipeline.datasets import cycle_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=500)
    return parser.parse_args()


def pad_or_trim(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) >= length:
        return values[:length]
    result = np.zeros(length, dtype=np.float32)
    result[: len(values)] = values
    return result


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    output = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.float32,
        shape=(len(frame), 3, args.length),
    )
    for source_pkl, indices in tqdm(
        frame.groupby("source_pkl", sort=False).groups.items(),
        desc="source recordings",
    ):
        with Path(source_pkl).open("rb") as handle:
            cycles = pickle.load(handle)["diff_results"]
        for row_index in indices:
            row = frame.iloc[row_index]
            voltage, current, power = cycle_arrays(
                cycles[int(row["source_cycle_index"])]
            )
            output[row_index, 0] = pad_or_trim(voltage, args.length)
            output[row_index, 1] = pad_or_trim(current, args.length)
            output[row_index, 2] = pad_or_trim(power, args.length)
    output.flush()
    del output
    print(f"Wrote {args.output} with shape ({len(frame)}, 3, {args.length})")


if __name__ == "__main__":
    main()
