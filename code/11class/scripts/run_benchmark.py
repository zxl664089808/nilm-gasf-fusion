#!/usr/bin/env python
"""Run channel, resolution, model, and seed ablations through one CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=None)
    parser.add_argument(
        "--manifest-96",
        type=Path,
        default=None,
        help="Manifest generated with --image-size 96.",
    )
    parser.add_argument(
        "--manifest-128",
        type=Path,
        default=None,
        help="Manifest generated with --image-size 128.",
    )
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--only",
        choices=("all", "channels", "resolution", "models", "seeds"),
        default="all",
    )
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--seed-channels",
        default="vip",
        help="Comma-separated channel combinations for --only seeds.",
    )
    return parser.parse_args()


def experiments(
    only: str,
    seeds: list[int],
    seed_channels: list[str],
) -> list[dict]:
    result = []
    if only in {"all", "channels"}:
        for channels in ("v", "i", "p", "vi", "vip"):
            result.append(
                {
                    "group": "channel",
                    "name": f"channel_{channels}",
                    "channels": channels,
                    "model": "simple",
                    "image_size": 64,
                    "seed": 42,
                }
            )
    if only in {"all", "resolution"}:
        for image_size in (64, 96, 128):
            result.append(
                {
                    "group": "resolution",
                    "name": f"resolution_{image_size}",
                    "channels": "vip",
                    "model": "simple",
                    "image_size": image_size,
                    "seed": 42,
                }
            )
    if only in {"all", "models"}:
        for model in ("simple", "resnet18", "efficientnet_b0"):
            result.append(
                {
                    "group": "model",
                    "name": f"model_{model}",
                    "channels": "vip",
                    "model": model,
                    "image_size": 64,
                    "seed": 42,
                }
            )
    if only in {"all", "seeds"}:
        for channels in seed_channels:
            for seed in seeds:
                result.append(
                    {
                        "group": "seed",
                        "name": f"seed_{channels}_{seed}",
                        "channels": channels,
                        "model": "simple",
                        "image_size": 64,
                        "seed": seed,
                    }
                )
    return result


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    seed_channels = [
        value.strip()
        for value in args.seed_channels.split(",")
        if value.strip()
    ]
    if not seed_channels or any(
        not set(channels).issubset({"v", "i", "p"})
        for channels in seed_channels
    ):
        raise ValueError("seed channels must be subsets of v, i, p")
    configs = experiments(args.only, seeds, seed_channels)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, config in enumerate(configs, start=1):
        manifest = args.manifest
        if config["group"] == "resolution":
            if config["image_size"] == 96:
                if args.manifest_96 is None:
                    print("Skipping resolution_96: --manifest-96 not provided")
                    continue
                manifest = args.manifest_96
            elif config["image_size"] == 128:
                if args.manifest_128 is None:
                    print("Skipping resolution_128: --manifest-128 not provided")
                    continue
                manifest = args.manifest_128
        run_dir = args.output_root / config["name"]
        command = [
            sys.executable,
            str(args.train_script),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(run_dir),
            "--channels",
            config["channels"],
            "--model",
            config["model"],
            "--image-size",
            str(config["image_size"]),
            "--seed",
            str(config["seed"]),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            "auto",
        ]
        if args.max_per_label is not None:
            command.extend(["--max-per-label", str(args.max_per_label)])
        if args.image_cache is not None:
            command.extend(["--image-cache", str(args.image_cache)])
        print(
            f"[{index}/{len(configs)}] {config['name']}",
            flush=True,
        )
        subprocess.run(command, check=True, env=os.environ.copy())
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                **config,
                "test_accuracy": metrics["test_accuracy"],
                "test_macro_f1": metrics["test_macro_f1"],
                "best_val_accuracy": metrics["best_val_accuracy"],
                "parameters": metrics["parameters"],
                "epochs_run": len(metrics["history"]),
                "split_sizes": json.dumps(metrics["split_sizes"]),
            }
        )
        pd.DataFrame(rows).to_csv(
            args.output_root / "benchmark_summary.csv",
            index=False,
        )
    print(args.output_root / "benchmark_summary.csv")


if __name__ == "__main__":
    main()
