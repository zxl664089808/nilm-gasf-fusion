"""Cycle extraction helpers that retain recording provenance."""

from __future__ import annotations

import numpy as np


def find_zero_crossings(signal: np.ndarray) -> list[int]:
    """Return interpolated upward zero crossings for a voltage signal."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    crossings: list[int] = []
    for index in range(1, len(signal)):
        left = signal[index - 1]
        right = signal[index]
        if left <= 0 < right:
            denominator = right - left
            fraction = 0.0 if denominator == 0 else -left / denominator
            # Start at the first sample on the non-negative side of the crossing.
            crossings.append(int(np.ceil(index - 1 + fraction)))
    return crossings


def extract_aligned_cycles(
    current: np.ndarray,
    voltage: np.ndarray,
    samples_per_cycle: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract fixed-length cycles beginning at upward voltage zero crossings."""
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    voltage = np.asarray(voltage, dtype=np.float64).reshape(-1)
    if len(current) != len(voltage):
        raise ValueError("Current and voltage lengths differ")
    cycles = []
    for start in find_zero_crossings(voltage):
        end = start + samples_per_cycle
        if end > len(current):
            break
        cycles.append((current[start:end], voltage[start:end]))
    return cycles
