#!/usr/bin/env python
"""Create a source-recording-grouped train/validation/test manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nilm_pipeline.splits import assert_no_group_overlap, assign_grouped_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-column", default="source_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    result = assign_grouped_splits(
        frame,
        source_column=args.source_column,
        label_column=args.label_column,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    assert_no_group_overlap(
        result,
        source_column=args.source_column,
        split_column="split",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8")
    print(result.groupby(["label", "split"]).size().unstack(fill_value=0))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
