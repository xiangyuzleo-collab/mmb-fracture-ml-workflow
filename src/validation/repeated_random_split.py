from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def repeated_random_splits(indices, n_repeats: int, test_size: float, random_seed: int, strata=None):
    indices = np.asarray(list(indices))
    stratify = None if strata is None else np.asarray(list(strata))
    if stratify is not None and len(stratify) != len(indices):
        raise ValueError("strata must have the same length and order as indices")
    for repeat in range(n_repeats):
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed + repeat,
            stratify=stratify,
        )
        yield {
            "fold": f"repeat_{repeat + 1:03d}",
            "repeat": repeat + 1,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "stratified": stratify is not None,
        }
