"""Dataset discovery and deterministic cycle selection."""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .records import SourceFile, parse_26state_source, parse_state_from_name


def discover_26state_sources(root: Path | str) -> dict[str, list[SourceFile]]:
    root = Path(root)
    grouped: dict[str, list[SourceFile]] = defaultdict(list)
    discovered: set[Path] = set()

    # The extraction pipeline already writes cycle counts to compact CSV files.
    # Use those first instead of unpickling every large waveform file.
    for summary_path in sorted(root.rglob("processing_summary.csv")):
        frame = pd.read_csv(summary_path)
        required = {"pkl_file", "regular_file", "num_cycles"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[list(required)].drop_duplicates()
        parsed = parse_state_from_name(summary_path.parent.name)
        if parsed is None:
            continue
        appliance, state = parsed
        for row in frame.itertuples(index=False):
            pkl_path = summary_path.parent / str(row.pkl_file)
            if not pkl_path.exists():
                continue
            discovered.add(pkl_path)
            source = SourceFile(
                path=pkl_path,
                appliance=appliance,
                state=state,
                source_id=str(row.regular_file),
                cycles=int(row.num_cycles),
            )
            grouped[source.label].append(source)

    # Fall back only for files that do not have a processing summary.
    for pkl_path in sorted(root.rglob("*.pkl")):
        if pkl_path in discovered:
            continue
        with pkl_path.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict) or "diff_results" not in data:
            continue
        source = parse_26state_source(
            pkl_path=pkl_path,
            metadata=data,
            cycle_count=len(data["diff_results"]),
        )
        grouped[source.label].append(source)
    return dict(grouped)


def select_global_indices(
    total_cycles: int,
    maximum: int,
    seed: int,
    label: str,
) -> list[int]:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    selected_count = min(total_cycles, maximum)
    rng = random.Random(f"{seed}:{label}")
    return sorted(rng.sample(range(total_cycles), selected_count))


def map_global_indices(
    sources: list[SourceFile],
    global_indices: Iterable[int],
) -> list[tuple[SourceFile, int]]:
    cumulative: list[int] = []
    running = 0
    for source in sources:
        running += source.cycles
        cumulative.append(running)
    mapped: list[tuple[SourceFile, int]] = []
    source_index = 0
    for global_index in global_indices:
        while source_index < len(cumulative) and global_index >= cumulative[source_index]:
            source_index += 1
        previous = cumulative[source_index - 1] if source_index else 0
        mapped.append((sources[source_index], global_index - previous))
    return mapped


def cycle_arrays(cycle: dict) -> tuple:
    voltage = cycle.get("voltage")
    current = cycle.get("diff_current", cycle.get("differential_current"))
    power = cycle.get("diff_power", cycle.get("differential_power"))
    if voltage is None or current is None or power is None:
        raise KeyError(
            f"Cycle lacks voltage/current/power. Available keys: {sorted(cycle)}"
        )
    return voltage, current, power
