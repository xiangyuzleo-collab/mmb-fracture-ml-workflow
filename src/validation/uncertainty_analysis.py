from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(values, n_bootstrap: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n_bootstrap)]
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
    }
