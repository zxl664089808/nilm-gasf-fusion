"""Physical features lost by per-channel GASF normalization."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "v_rms",
    "i_rms",
    "active_power",
    "apparent_power",
    "power_factor",
    "i_peak",
    "i_crest",
    "current_thd",
    "h3_ratio",
    "h5_ratio",
    "h7_ratio",
    "mean_abs_current",
    "mean_abs_power",
    "log_abs_active_power",
]


def physical_features(
    voltage: np.ndarray,
    current: np.ndarray,
    power: np.ndarray,
) -> np.ndarray:
    voltage = np.asarray(voltage, dtype=np.float64).reshape(-1)
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    power = np.asarray(power, dtype=np.float64).reshape(-1)
    length = min(len(voltage), len(current), len(power))
    voltage = voltage[:length]
    current = current[:length]
    power = power[:length]
    epsilon = 1e-12

    v_rms = float(np.sqrt(np.mean(voltage**2)))
    i_rms = float(np.sqrt(np.mean(current**2)))
    active_power = float(np.mean(power))
    apparent_power = float(v_rms * i_rms)
    power_factor = float(
        np.clip(active_power / (apparent_power + epsilon), -1.0, 1.0)
    )
    i_peak = float(np.max(np.abs(current)))
    i_crest = float(i_peak / (i_rms + epsilon))

    centered = current - np.mean(current)
    spectrum = np.abs(np.fft.rfft(centered))
    fundamental = float(spectrum[1]) if len(spectrum) > 1 else 0.0
    harmonic = lambda order: (
        float(spectrum[order]) if len(spectrum) > order else 0.0
    )
    h2_to_h9 = [harmonic(order) for order in range(2, 10)]
    current_thd = float(
        np.sqrt(np.sum(np.square(h2_to_h9))) / (fundamental + epsilon)
    )
    h3_ratio = float(harmonic(3) / (fundamental + epsilon))
    h5_ratio = float(harmonic(5) / (fundamental + epsilon))
    h7_ratio = float(harmonic(7) / (fundamental + epsilon))

    return np.asarray(
        [
            v_rms,
            i_rms,
            active_power,
            apparent_power,
            power_factor,
            i_peak,
            i_crest,
            current_thd,
            h3_ratio,
            h5_ratio,
            h7_ratio,
            float(np.mean(np.abs(current))),
            float(np.mean(np.abs(power))),
            float(np.log1p(abs(active_power))),
        ],
        dtype=np.float32,
    )
