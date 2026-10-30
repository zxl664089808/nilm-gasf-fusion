"""Recording-grouped data splits with no source leakage across sets."""

from __future__ import annotations

import math
import random

import pandas as pd


def assign_grouped_splits(
    frame: pd.DataFrame,
    source_column: str = "source_id",
    label_column: str = "label",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Split each label by source recording, not by generated image."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than one")
    required = {"label", source_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")

    rng = random.Random(seed)
    assignments: dict[tuple[str, str], str] = {}
    for label, label_frame in frame.groupby(label_column, sort=True):
        sizes = label_frame.groupby(source_column).size().sort_index()
        groups = list(sizes.index)
        rng.shuffle(groups)
        total = int(sizes.sum())
        test_target = max(1, math.ceil(total * test_fraction))
        val_target = max(1, math.ceil(total * val_fraction))

        test_groups: list[str] = []
        running = 0
        remaining: list[str] = []
        for group in groups:
            if running < test_target:
                test_groups.append(group)
                running += int(sizes[group])
            else:
                remaining.append(group)

        val_groups: list[str] = []
        running = 0
        train_groups: list[str] = []
        for group in remaining:
            if running < val_target:
                val_groups.append(group)
                running += int(sizes[group])
            else:
                train_groups.append(group)

        if not train_groups or not val_groups or not test_groups:
            raise ValueError(
                f"Label {label!r} has too few source recordings for a grouped split"
            )
        for group in train_groups:
            assignments[(label, group)] = "train"
        for group in val_groups:
            assignments[(label, group)] = "val"
        for group in test_groups:
            assignments[(label, group)] = "test"

    result = frame.copy()
    result["split"] = [
        assignments[(label, group)]
        for label, group in zip(result[label_column], result[source_column])
    ]
    return result


def assert_no_group_overlap(
    frame: pd.DataFrame,
    source_column: str = "source_id",
    split_column: str = "split",
) -> None:
    """Raise if one source recording appears in more than one split."""
    counts = frame.groupby(source_column)[split_column].nunique()
    leaked = counts[counts > 1]
    if len(leaked):
        raise ValueError(
            f"{len(leaked)} source recordings occur in multiple splits; "
            f"examples: {list(leaked.index[:5])}"
        )
