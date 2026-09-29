from __future__ import annotations

import pandas as pd


def leave_one_lever_arm_splits(df: pd.DataFrame, lever_col: str = "lever_arm_length"):
    for value in sorted(pd.to_numeric(df[lever_col], errors="coerce").dropna().unique()):
        test_mask = pd.to_numeric(df[lever_col], errors="coerce") == value
        train_indices = df.index[~test_mask].tolist()
        test_indices = df.index[test_mask].tolist()
        if value == min(pd.to_numeric(df[lever_col], errors="coerce").dropna()) or value == max(pd.to_numeric(df[lever_col], errors="coerce").dropna()):
            generalization_type = "extrapolation-like"
        else:
            generalization_type = "interpolation-like"
        yield {
            "fold": f"leave_c_{int(value)}",
            "lever_arm_length_left_out": float(value),
            "generalization_type": generalization_type,
            "train_indices": train_indices,
            "test_indices": test_indices,
        }
