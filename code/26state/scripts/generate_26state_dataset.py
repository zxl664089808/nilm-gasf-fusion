#!/usr/bin/env python
"""Generate leakage-free 26-state RGB Gramian images from per-recording PKLs."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import pickle
from pathlib import Path

import pandas as pd

from nilm_pipeline.datasets import (
    cycle_arrays,
    discover_26state_sources,
    map_global_indices,
    select_global_indices,
)
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
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
        help="Parallel label workers. Each worker writes a separate label directory.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recreate labels even when a completed label manifest exists.",
    )
    parser.add_argument(
        "--limit-files-per-label",
        type=int,
        default=None,
        help="Development-only limit for smoke testing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def process_label(
    label: str,
    sources,
    max_cycles_per_label: int,
    seed: int,
    gramian: str,
    image_size: int,
    image_root: Path,
) -> tuple[str, list[dict]]:
    """Generate one label in a worker process and return its manifest rows."""
    total = sum(source.cycles for source in sources)
    indices = select_global_indices(
        total_cycles=total,
        maximum=max_cycles_per_label,
        seed=seed,
        label=label,
    )
    mapped = map_global_indices(sources, indices)
    label_dir = safe_component(label)
    rows: list[dict] = []
    pkl_cache: dict[Path, list[dict]] = {}
    for source, local_index in mapped:
        cycles = pkl_cache.get(source.path)
        if cycles is None:
            with source.path.open("rb") as handle:
                cycles = pickle.load(handle)["diff_results"]
            # Selections are ordered by source, so only the current file is kept.
            pkl_cache = {source.path: cycles}
        cycle = cycles[local_index]
        voltage, current, power = cycle_arrays(cycle)
        rgb = make_rgb_gramian(
            voltage,
            current,
            power,
            method=gramian,
        )
        rgb = resize_pure_rgb(rgb, (image_size, image_size))
        source_name = safe_component(source.source_id)
        image_name = f"{source_name}__{local_index:06d}.png"
        image_path = image_root / label_dir / image_name
        save_pure_rgb(image_path, rgb)
        rows.append(
            {
                "image_path": str(image_path),
                "label": label,
                "appliance": source.appliance,
                "state": source.state.upper(),
                "source_id": source.source_id,
                "source_pkl": str(source.path),
                "source_cycle_index": int(cycle.get("cycle_index", local_index)),
                "image_size": image_size,
                "gramian": gramian,
            }
        )
    return label, rows


def main() -> None:
    args = parse_args()
    groups = discover_26state_sources(args.cycles_root)
    if not groups:
        raise RuntimeError(f"No compatible PKL files found under {args.cycles_root}")

    output_root = args.output_root.resolve()
    image_root = output_root / "images"
    label_manifest_root = output_root / "label_manifests"
    label_manifest_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    tasks = []
    for label, all_sources in sorted(groups.items()):
        sources = all_sources
        if args.limit_files_per_label is not None:
            sources = sources[: args.limit_files_per_label]
        total = sum(source.cycles for source in sources)
        if args.dry_run:
            rows.append(
                {
                    "label": label,
                    "source_recordings": len(sources),
                    "available_cycles": total,
                    "selected_cycles": min(total, args.max_cycles_per_label),
                }
            )
            continue
        label_manifest = label_manifest_root / f"{safe_component(label)}.csv"
        if not args.no_resume and label_manifest.exists():
            existing = pd.read_csv(label_manifest)
            rows.extend(existing.to_dict("records"))
            print(f"Resuming completed label: {label}", flush=True)
            continue
        tasks.append((label, sources, label_manifest))

    if tasks:
        if args.workers == 1:
            for label, sources, label_manifest in tasks:
                _, label_rows = process_label(
                    label,
                    sources,
                    args.max_cycles_per_label,
                    args.seed,
                    args.gramian,
                    args.image_size,
                    image_root,
                )
                pd.DataFrame(label_rows).to_csv(
                    label_manifest, index=False, encoding="utf-8"
                )
                rows.extend(label_rows)
                print(f"Completed {label}: {len(label_rows)} images", flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            ) as executor:
                future_map = {
                    executor.submit(
                        process_label,
                        label,
                        sources,
                        args.max_cycles_per_label,
                        args.seed,
                        args.gramian,
                        args.image_size,
                        image_root,
                    ): (label, label_manifest)
                    for label, sources, label_manifest in tasks
                }
                for future in concurrent.futures.as_completed(future_map):
                    label, label_manifest = future_map[future]
                    _, label_rows = future.result()
                    pd.DataFrame(label_rows).to_csv(
                        label_manifest, index=False, encoding="utf-8"
                    )
                    rows.extend(label_rows)
                    print(
                        f"Completed {label}: {len(label_rows)} images",
                        flush=True,
                    )

    manifest_name = "dataset_manifest.csv" if not args.dry_run else "dry_run_summary.csv"
    manifest = output_root / manifest_name
    frame = pd.DataFrame(rows)
    if not args.dry_run and not frame.empty:
        frame = frame.sort_values(
            ["label", "source_id", "source_cycle_index"]
        ).reset_index(drop=True)
    frame.to_csv(manifest, index=False, encoding="utf-8")
    print(f"Wrote {manifest}")
    if not args.dry_run:
        print(f"Generated {len(rows):,} pure RGB images")
        print("No title, axes, text, or Matplotlib figure was used.")


if __name__ == "__main__":
    main()
