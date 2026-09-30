"""Protect the public demo from accidentally using experimental rows."""

from __future__ import annotations

import pandas as pd


def validate_synthetic_specimens(specimens: pd.DataFrame) -> None:
    if specimens.empty or "sample_type" not in specimens or "specimen_id" not in specimens:
        raise ValueError("The demo requires labeled synthetic specimen rows")
    labeled_synthetic = specimens["sample_type"].eq("synthetic_demo").all()
    synthetic_ids = specimens["specimen_id"].astype("string").str.startswith("SYN-").fillna(False).all()
    if not labeled_synthetic or not synthetic_ids:
        raise ValueError("The demo accepts only synthetic_demo rows with SYN- specimen IDs")
