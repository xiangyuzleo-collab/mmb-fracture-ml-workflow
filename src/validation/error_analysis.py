"""Summarize held-out prediction errors by experimental condition."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_condition_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"validation_strategy", "model", "direction", "lever_arm_length", "actual", "predicted"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")
    if predictions.empty:
        raise ValueError("Cannot summarize empty predictions")

    errors = predictions.copy()
    errors["error"] = pd.to_numeric(errors["predicted"]) - pd.to_numeric(errors["actual"])
    if not np.isfinite(errors["error"]).all():
        raise ValueError("Predictions and targets must be finite")
    errors["absolute_error"] = errors["error"].abs()
    errors["squared_error"] = errors["error"] ** 2
    group_columns = ["validation_strategy", "model", "direction", "lever_arm_length"]
    summary = (
        errors.groupby(group_columns, dropna=False)
        .agg(
            n_predictions=("error", "size"),
            mean_error=("error", "mean"),
            MAE=("absolute_error", "mean"),
            mean_squared_error=("squared_error", "mean"),
        )
        .reset_index()
    )
    summary["RMSE"] = np.sqrt(summary.pop("mean_squared_error"))
    return summary.sort_values(group_columns).reset_index(drop=True)
