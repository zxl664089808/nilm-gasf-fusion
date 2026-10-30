#!/usr/bin/env python
"""Validate manifests, splits, and pure image content."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nilm_pipeline.image_io import title_band_report
from nilm_pipeline.splits import assert_no_group_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-per-label", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    missing = [path for path in frame["image_path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing; e.g. {missing[:3]}")
    if "source_id" not in frame or "split" not in frame:
        raise ValueError("Manifest must contain source_id and split")
    assert_no_group_overlap(frame)
    sampled = (
        frame.groupby("label", group_keys=False)
        .head(args.sample_per_label)
        .reset_index(drop=True)
    )
    suspicious = []
    for row in sampled.itertuples(index=False):
        report = title_band_report(row.image_path)
        if report["suspicious_title_band"]:
            suspicious.append(report)
    if suspicious:
        raise RuntimeError(
            f"{len(suspicious)} sampled images contain a suspicious white title "
            f"band; examples: {suspicious[:3]}"
        )
    print(frame.groupby(["label", "split"]).size().unstack(fill_value=0))
    print(f"Validated {len(frame):,} images and {len(sampled)} sampled pixels.")
    print("Grouped split has no source-recording overlap.")
    print("No suspicious title band was detected in the sample.")


if __name__ == "__main__":
    main()
