"""Parsing and source-record helpers for PLAID intermediate files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    path: Path
    appliance: str
    state: str
    source_id: str
    cycles: int

    @property
    def label(self) -> str:
        return f"{self.appliance}_{self.state.upper()}"


def parse_state_from_name(name: str) -> tuple[str, str] | None:
    """Parse names such as ``Air Conditioner_on_101441``."""
    match = re.match(r"^(.*)_(on|off)(?:_\d+)?$", name, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip(), match.group(2).lower()


def parse_26state_source(
    pkl_path: Path,
    metadata: dict,
    cycle_count: int,
) -> SourceFile:
    """Parse one per-recording PKL from the 26-state dataset."""
    appliance = str(metadata.get("appliance_type", "")).strip()
    parsed = parse_state_from_name(appliance)
    if parsed is None:
        parsed = parse_state_from_name(pkl_path.parent.name)
    if parsed is None:
        raise ValueError(f"Cannot determine appliance/state from {pkl_path}")
    appliance, state = parsed
    source_id = str(metadata.get("regular_filename", "")).strip()
    if not source_id:
        source_id = f"{pkl_path.parent.name}_{metadata.get('file_id', pkl_path.stem)}"
    return SourceFile(
        path=pkl_path,
        appliance=appliance,
        state=state,
        source_id=source_id,
        cycles=int(cycle_count),
    )


def safe_component(value: str) -> str:
    """Create a filesystem-safe path component without changing the label."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._") or "unnamed"
