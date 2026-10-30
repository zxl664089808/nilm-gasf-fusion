#!/usr/bin/env python
"""Cache manifest images as a resized uint8 memmap for fast ablations."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def load_image(path: str, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    cache = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.uint8,
        shape=(len(frame), args.image_size, args.image_size, 3),
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(
                load_image,
                path,
                args.image_size,
            ): index
            for index, path in enumerate(frame["image_path"])
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures)
        ):
            cache[futures[future]] = future.result()
            if (completed + 1) % 5000 == 0:
                print(f"cached {completed + 1}/{len(frame)}", flush=True)
    cache.flush()
    del cache
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
