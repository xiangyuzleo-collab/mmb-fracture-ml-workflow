from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_relative_errors(y_true, y_pred, eps: float = 1e-12) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, np.nan, np.abs(y_true))
    return np.abs(y_pred - y_true) / denom


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rel = safe_relative_errors(y_true, y_pred)
    finite_rel = rel[np.isfinite(rel)]
    return {
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.nanmean(rel) * 100.0) if finite_rel.size else float("nan"),
        "MaxAE": float(np.max(np.abs(y_pred - y_true))) if y_true.size else float("nan"),
        "MaxRE": float(np.nanmax(rel) * 100.0) if finite_rel.size else float("nan"),
        "MeanRE": float(np.nanmean(rel) * 100.0) if finite_rel.size else float("nan"),
        "MedianRE": float(np.nanmedian(rel) * 100.0) if finite_rel.size else float("nan"),
    }


def metric_rows(y_true, y_pred, **labels) -> dict[str, float | str | int]:
    row = dict(labels)
    row.update(regression_metrics(y_true, y_pred))
    return row
