from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io_utils import write_json


def require_specimen_level_rows(df: pd.DataFrame) -> None:
    """Reject repeated specimen rows before row-index-based train/test splitting."""
    if "specimen_id" not in df.columns:
        raise ValueError("Validation requires a specimen_id column")
    specimen_ids = df["specimen_id"]
    missing = specimen_ids.isna() | specimen_ids.astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError("Validation requires a non-empty specimen_id for every row")
    duplicated = specimen_ids.duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "Validation requires one row per specimen_id; repeated IDs could place "
            "the same specimen in both training and test sets"
        )


def save_split(split: dict, path: str | Path) -> None:
    write_json(split, path)


def summarize_splits(split_records: list[dict]) -> pd.DataFrame:
    rows = []
    for split in split_records:
        rows.append(
            {
                "fold": split.get("fold"),
                "n_train": len(split.get("train_indices", [])),
                "n_test": len(split.get("test_indices", [])),
                "lever_arm_length_left_out": split.get("lever_arm_length_left_out"),
                "generalization_type": split.get("generalization_type"),
                "stratified": split.get("stratified", False),
            }
        )
    return pd.DataFrame(rows)
