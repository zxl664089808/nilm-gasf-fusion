"""Pure image I/O and title-band diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_pure_rgb(path: Path | str, rgb: np.ndarray) -> None:
    """Save an RGB array without Matplotlib metadata or rendered text."""
    path = Path(path)
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path, format="PNG")


def resize_pure_rgb(
    rgb: np.ndarray,
    size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Resize an RGB array with Pillow, without adding labels or margins."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    return np.asarray(
        image.resize((size[1], size[0]), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def title_band_report(path: Path | str, top_fraction: float = 0.07) -> dict:
    """Return simple evidence that the historical white title band is present."""
    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    height = max(2, int(round(array.shape[0] * top_fraction)))
    band = array[:height]
    near_white = np.all(band >= 245, axis=2)
    dark = np.all(band <= 80, axis=2)
    return {
        "path": str(path),
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "white_fraction": float(near_white.mean()),
        "dark_pixel_count": int(dark.sum()),
        "suspicious_title_band": bool(near_white.mean() > 0.55 and dark.sum() > 20),
    }
