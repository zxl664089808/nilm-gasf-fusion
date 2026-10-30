#!/usr/bin/env python
"""Extract 11-class cycles and generate leakage-free RGB Gramian images."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from nilm_11class.extraction import extract_aligned_cycles
from nilm_11class.gramian import make_rgb_gramian
from nilm_11class.image_io import resize_pure_rgb, save_pure_rgb


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
    parser.add_argument("--max-cycles-per-class", type=int, default=5000)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Use only the first N source samples per class; zero means all.",
    )
    parser.add_argument("--samples-per-cycle", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--gramian", choices=("gasf", "gadf"), default="gasf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, (os.cpu_count() or 2) // 3)),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--appliances",
        default=",".join(DEFAULT_APPLIANCES),
    )
    return parser.parse_args()


def reservoir_sample(
    samples: list[dict],
    csv_root: Path,
    maximum: int,
    samples_per_cycle: int,
    seed: int,
    label: str,
) -> list[dict]:
    """Uniformly sample cycles without retaining every waveform in memory."""
    rng = random.Random(f"{seed}:{label}")
    reservoir: list[dict] = []
    seen = 0
    for sample in samples:
        csv_file = str(sample.get("csv_file", ""))
        csv_path = csv_root / csv_file
        if not csv_path.exists():
            continue
        try:
            data = np.genfromtxt(csv_path, delimiter=",")
        except Exception:
            continue
        if data.ndim != 2 or data.shape[1] < 2:
            continue
        current_signal = np.asarray(data[:, 0], dtype=np.float64)
        voltage_signal = np.asarray(data[:, 1], dtype=np.float64)
        cycles = extract_aligned_cycles(
            current_signal,
            voltage_signal,
            samples_per_cycle=samples_per_cycle,
        )
        group_id = f"{csv_file}|sample_{sample.get('original_index', 0)}"
        for cycle_index, (current_cycle, voltage_cycle) in enumerate(cycles):
            item = {
                "voltage": voltage_cycle,
                "current": current_cycle,
                "source_id": group_id,
                "cycle_index": cycle_index,
            }
            seen += 1
            if len(reservoir) < maximum:
                reservoir.append(item)
            else:
                replacement = rng.randrange(seen)
                if replacement < maximum:
                    reservoir[replacement] = item
    reservoir.sort(key=lambda item: (item["source_id"], item["cycle_index"]))
    return reservoir


def process_class(
    label: str,
    samples: list[dict],
    csv_root: Path,
    image_root: Path,
    maximum: int,
    samples_per_cycle: int,
    gramian: str,
    image_size: int,
    seed: int,
) -> tuple[str, list[dict]]:
    selected = reservoir_sample(
        samples,
        csv_root,
        maximum,
        samples_per_cycle,
        seed,
        label,
    )
    safe_label = "".join(
        character if character.isalnum() else "_" for character in label
    ).strip("_")
    rows = []
    for item in selected:
        current = np.asarray(item["current"], dtype=np.float64)
        voltage = np.asarray(item["voltage"], dtype=np.float64)
        power = current * voltage
        rgb = make_rgb_gramian(voltage, current, power, method=gramian)
        rgb = resize_pure_rgb(rgb, (image_size, image_size))
        source_name = "".join(
            character if character.isalnum() else "_"
            for character in item["source_id"]
        ).strip("_")
        image_path = (
            image_root
            / safe_label
            / f"{source_name}__{int(item['cycle_index']):06d}.png"
        )
        save_pure_rgb(image_path, rgb)
        rows.append(
            {
                "image_path": str(image_path),
                "label": label,
                "source_id": item["source_id"],
                "source_cycle_index": int(item["cycle_index"]),
                "image_size": image_size,
                "gramian": gramian,
            }
        )
    return label, rows


def main() -> None:
    args = parse_args()
    with args.sample_index.open("rb") as handle:
        sample_infos = pickle.load(handle)
    requested = {
        item.strip() for item in args.appliances.split(",") if item.strip()
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in sample_infos:
        label = str(sample.get("appliance", "")).strip()
        if label in requested:
            grouped[label].append(sample)
    if not grouped:
        raise RuntimeError("No matching appliance samples found")

    output_root = args.output_root.resolve()
    image_root = output_root / "images"
    label_manifest_root = output_root / "label_manifests"
    output_root.mkdir(parents=True, exist_ok=True)
    label_manifest_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    tasks = []
    for label, samples in sorted(grouped.items()):
        if args.samples_per_class > 0:
            samples = samples[: args.samples_per_class]
        label_manifest = label_manifest_root / (
            "".join(
                character if character.isalnum() else "_"
                for character in label
            ).strip("_")
            + ".csv"
        )
        if not args.no_resume and label_manifest.exists():
            rows.extend(pd.read_csv(label_manifest).to_dict("records"))
            print(f"Resuming completed class: {label}", flush=True)
        else:
            tasks.append((label, samples, label_manifest))

    if tasks:
        if args.workers == 1:
            for label, samples, manifest in tasks:
                _, label_rows = process_class(
                    label,
                    samples,
                    args.csv_root,
                    image_root,
                    args.max_cycles_per_class,
                    args.samples_per_cycle,
                    args.gramian,
                    args.image_size,
                    args.seed,
                )
                pd.DataFrame(label_rows).to_csv(manifest, index=False)
                rows.extend(label_rows)
                print(f"Completed {label}: {len(label_rows)} images", flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            ) as executor:
                futures = {
                    executor.submit(
                        process_class,
                        label,
                        samples,
                        args.csv_root,
                        image_root,
                        args.max_cycles_per_class,
                        args.samples_per_cycle,
                        args.gramian,
                        args.image_size,
                        args.seed,
                    ): (label, manifest)
                    for label, samples, manifest in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    label, manifest = futures[future]
                    _, label_rows = future.result()
                    pd.DataFrame(label_rows).to_csv(manifest, index=False)
                    rows.extend(label_rows)
                    print(
                        f"Completed {label}: {len(label_rows)} images",
                        flush=True,
                    )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["label", "source_id", "source_cycle_index"]
        ).reset_index(drop=True)
    manifest = output_root / "dataset_manifest.csv"
    frame.to_csv(manifest, index=False, encoding="utf-8")
    print(f"Wrote {manifest}")
    print(f"Generated {len(frame):,} pure RGB images")


if __name__ == "__main__":
    main()
