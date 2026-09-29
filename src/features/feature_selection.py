from __future__ import annotations

import pandas as pd


def remove_constant_features(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    kept, removed = [], []
    for feature in features:
        if feature not in df.columns:
            removed.append(f"{feature}:missing")
        elif df[feature].nunique(dropna=True) <= 1:
            removed.append(f"{feature}:constant")
        else:
            kept.append(feature)
    return kept, removed
