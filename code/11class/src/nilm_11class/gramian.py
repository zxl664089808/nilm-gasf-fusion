"""Correct Gramian Angular Field implementations and RGB encoding."""

from __future__ import annotations

import numpy as np


def _as_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return vector


def normalize_signed(values: np.ndarray, name: str = "signal") -> np.ndarray:
    """Map a finite signal to [-1, 1], including constant-signal handling."""
    vector = _as_finite_vector(values, name)
    minimum = float(vector.min())
    maximum = float(vector.max())
    if maximum == minimum:
        return np.zeros_like(vector)
    return 2.0 * (vector - minimum) / (maximum - minimum) - 1.0


def angular_field(
    values: np.ndarray,
    method: str = "gasf",
    name: str = "signal",
) -> np.ndarray:
    """Compute a Gramian Angular Summation or Difference Field.

    Parameters
    ----------
    values:
        One-dimensional time series.
    method:
        ``"gasf"`` for sin(phi_i + phi_j) or ``"gadf"`` for
        sin(phi_i - phi_j).
    """
    normalized = normalize_signed(values, name=name)
    angles = np.arccos(np.clip(normalized, -1.0, 1.0))
    if method == "gasf":
        matrix = np.sin(angles[:, None] + angles[None, :])
    elif method == "gadf":
        matrix = np.sin(angles[:, None] - angles[None, :])
    else:
        raise ValueError("method must be 'gasf' or 'gadf'")
    return np.clip((matrix + 1.0) / 2.0, 0.0, 1.0)


def make_rgb_gramian(
    voltage: np.ndarray,
    current: np.ndarray,
    power: np.ndarray,
    method: str = "gasf",
) -> np.ndarray:
    """Stack voltage, current, and power fields into an RGB uint8 image.

    Each channel is normalized independently, matching the legacy pipeline.
    The output contains only numerical pixels: no axes, title, text, colorbar,
    or whitespace is added.
    """
    voltage = _as_finite_vector(voltage, "voltage")
    current = _as_finite_vector(current, "current")
    power = _as_finite_vector(power, "power")
    length = min(len(voltage), len(current), len(power))
    if length < 2:
        raise ValueError("Waveforms have no usable overlap")
    channels = [
        angular_field(voltage[:length], method=method, name="voltage"),
        angular_field(current[:length], method=method, name="current"),
        angular_field(power[:length], method=method, name="power"),
    ]
    rgb = np.stack(channels, axis=-1)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
