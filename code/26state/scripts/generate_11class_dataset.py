#!/usr/bin/env python
"""Generate pure 11-class images from cycle PKLs that retain source IDs."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nilm_pipeline.gramian import make_rgb_gramian
from nilm_pipeline.image_io import resize_pure_rgb, save_pure_rgb
from nilm_pipeline.records import safe_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-cycles-per-label", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gramian", choices=("gasf", "gadf"), default="gasf")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--limit-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.cycles_root.glob("*_fundamental_cycles.pkl"))
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        raise RuntimeError(f"No cycle PKLs found under {args.cycles_root}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for path in tqdm(files, desc="labels"):
        with path.open("rb") as handle:
            data = pickle.load(handle)
        voltage_cycles = data.get("voltage_cycles", [])
        current_cycles = data.get("current_cycles", [])
        source_ids = data.get("source_ids", [])
        if not source_ids:
            raise RuntimeError(
                f"{path} has no source_ids. Re-run the corrected 11-class cycle "
                "extraction before generating images; synthetic source IDs would "
                "reintroduce recording-level leakage."
            )
        if not (len(voltage_cycles) == len(current_cycles) == len(source_ids)):
            raise ValueError(f"Cycle/source lengths differ in {path}")
        count = min(len(voltage_cycles), args.max_cycles_per_label)
        selected = rng.choice(len(voltage_cycles), size=count, replace=False)
        label = str(data.get("device_type", path.stem))
        label_dir = safe_component(label)
        for cycle_index in tqdm(selected, desc=label, leave=False):
            voltage = np.asarray(voltage_cycles[cycle_index], dtype=np.float64)
            current = np.asarray(current_cycles[cycle_index], dtype=np.float64)
            power = voltage * current
            rgb = make_rgb_gramian(voltage, current, power, method=args.gramian)
            rgb = resize_pure_rgb(rgb, (args.image_size, args.image_size))
            source_id = str(source_ids[cycle_index])
            image_path = (
                args.output_root
                / "images"
                / label_dir
                / f"{safe_component(source_id)}__{int(cycle_index):06d}.png"
            )
            save_pure_rgb(image_path, rgb)
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "source_id": source_id,
                    "source_cycle_index": int(cycle_index),
                    "image_size": args.image_size,
                    "gramian": args.gramian,
                }
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "dataset_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False, encoding="utf-8")
    print(f"Wrote {manifest}")
    print(f"Generated {len(rows):,} pure RGB images")


if __name__ == "__main__":
    main()
