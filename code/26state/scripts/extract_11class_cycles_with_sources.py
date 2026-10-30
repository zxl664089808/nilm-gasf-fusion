#!/usr/bin/env python
"""Re-extract 11-class cycles while retaining CSV/sample provenance."""

from __future__ import annotations

import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from nilm_pipeline.extraction import extract_aligned_cycles
from nilm_pipeline.records import safe_component


DEFAULT_APPLIANCES = [
    "Air Conditioner",
    "Compact Fluorescent Lamp",
    "Fan",
    "Fridge",
    "Hairdryer",
    "Heater",
    "Incandescent Light Bulb",
    "Laptop",
    "Microwave",
    "Vacuum",
    "Washing Machine",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-cycle", type=int, default=500)
    parser.add_argument(
        "--appliances",
        default=",".join(DEFAULT_APPLIANCES),
        help="Comma-separated appliance names.",
    )
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.sample_index.open("rb") as handle:
        samples = pickle.load(handle)
    appliances = {item.strip() for item in args.appliances.split(",") if item.strip()}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        appliance = str(sample.get("appliance", "")).strip()
        if appliance in appliances:
            grouped[appliance].append(sample)

    args.output_root.mkdir(parents=True, exist_ok=True)
    for appliance, appliance_samples in sorted(grouped.items()):
        if args.limit_samples is not None:
            appliance_samples = appliance_samples[: args.limit_samples]
        voltage_cycles: list[np.ndarray] = []
        current_cycles: list[np.ndarray] = []
        source_ids: list[str] = []
        retained_samples: list[dict] = []
        for sample in tqdm(
            appliance_samples,
            desc=appliance,
            leave=False,
        ):
            csv_file = str(sample.get("csv_file", ""))
            csv_path = args.csv_root / csv_file
            if not csv_path.exists():
                continue
            data = np.genfromtxt(csv_path, delimiter=",")
            if data.ndim != 2 or data.shape[1] < 2:
                continue
            current = data[:, 0]
            voltage = data[:, 1]
            cycles = extract_aligned_cycles(
                current,
                voltage,
                samples_per_cycle=args.samples_per_cycle,
            )
            if not cycles:
                continue
            source_id = f"{csv_file}|sample_{sample.get('original_index', 0)}"
            retained_samples.append(sample)
            for current_cycle, voltage_cycle in cycles:
                current_cycles.append(current_cycle)
                voltage_cycles.append(voltage_cycle)
                source_ids.append(source_id)
        output = args.output_root / (
            f"{safe_component(appliance)}_fundamental_cycles.pkl"
        )
        payload = {
            "voltage_cycles": voltage_cycles,
            "current_cycles": current_cycles,
            "source_ids": source_ids,
            "sample_info": retained_samples,
            "device_type": appliance,
            "total_cycles": len(voltage_cycles),
            "extraction_time": time.time(),
        }
        with output.open("wb") as handle:
            pickle.dump(payload, handle)
        print(f"{appliance}: {len(voltage_cycles):,} cycles -> {output}")


if __name__ == "__main__":
    main()
