"""The demonstration table: one CSV row per successful episode, appended as collected.

CSV on purpose -- collection is parallel and long-running, so the table has to survive a
worker dying mid-sweep.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FEATURE_PREFIX = "feature"
PSI_PREFIX = "psi"


def row_from_episode(
    features: NDArray[np.float64],
    psi: NDArray[np.float64],
    metadata: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    """Flatten one episode into a table row."""
    row: dict[str, float | int | str] = dict(metadata)
    row.update({f"{FEATURE_PREFIX}_{i}": float(v) for i, v in enumerate(features)})
    row.update({f"{PSI_PREFIX}_{i}": float(v) for i, v in enumerate(psi.ravel())})
    return row


def append_row(path: str | Path, row: dict) -> None:
    """Append one row, writing the header only when the file is new.

    A single-line append, so concurrent workers interleave rather than corrupt rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def guard_variant(path: str | Path, name: str) -> None:
    """Refuse to mix variants in one table; a run appends to whatever is there."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    existing = pd.read_csv(path, nrows=1)
    if "variant" not in existing or existing["variant"].iloc[0] != name:
        found = existing["variant"].iloc[0] if "variant" in existing else "none recorded"
        raise SystemExit(
            f"{path} already holds variant {found!r}, not {name!r}; use a new --tag."
        )


def load_dataset(
    path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], pd.DataFrame]:
    """The features, the flattened control points, and everything else."""
    frame = pd.read_csv(path)
    feature_columns = sorted(
        (c for c in frame.columns if c.startswith(f"{FEATURE_PREFIX}_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    psi_columns = sorted(
        (c for c in frame.columns if c.startswith(f"{PSI_PREFIX}_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    features = frame[feature_columns].to_numpy(dtype=np.float64)
    labels = frame[psi_columns].to_numpy(dtype=np.float64)
    metadata = frame.drop(columns=feature_columns + psi_columns)
    return features, labels, metadata


def split_column(metadata: pd.DataFrame) -> str | None:
    """What a held-out group is: a person, or a task when the run has only one person."""
    if "person" in metadata and metadata["person"].nunique() > 1:
        return "person"
    return "task_index" if "task_index" in metadata else None


def train_test_split_by_person(
    features: NDArray[np.float64],
    labels: NDArray[np.float64],
    metadata: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split by person: every row of a test person is kept out of training.

    Splitting by task would leave a person's other tasks in training, and `theta` and
    the belief are six of the features, so the model could look them up. A single-person
    run has no such split to make, so it holds out whole tasks instead.
    """
    column = split_column(metadata)
    groups = metadata[column].to_numpy() if column else np.arange(len(features))
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut = max(1, int(round(test_fraction * len(unique))))
    held_out = set(unique[:cut].tolist())
    mask = np.array([g in held_out for g in groups])
    return features[~mask], labels[~mask], features[mask], labels[mask]
