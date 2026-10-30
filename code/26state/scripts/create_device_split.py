#!/usr/bin/env python
"""Create a leave-device-out split from a source-to-device mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nilm_pipeline.splits import assert_no_group_overlap, assign_grouped_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    mapping = pd.read_csv(args.device_map)
    if mapping["source_id"].duplicated().any():
        raise ValueError("Device map contains duplicate source_id values")
    merged = manifest.merge(
        mapping[["source_id", "device_id"]],
        on="source_id",
        how="left",
        validate="many_to_one",
    )
    if merged["device_id"].isna().any():
        raise ValueError("Some images lack a device mapping")
    result = assign_grouped_splits(
        merged.drop(columns=["split"], errors="ignore"),
        source_column="device_id",
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    assert_no_group_overlap(result, source_column="device_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.groupby(["label", "split"]).size().unstack(fill_value=0))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
