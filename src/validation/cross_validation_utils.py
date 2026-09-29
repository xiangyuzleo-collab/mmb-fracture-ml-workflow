from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io_utils import write_json


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
