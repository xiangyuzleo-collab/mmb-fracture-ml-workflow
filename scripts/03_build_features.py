from __future__ import annotations

import _bootstrap  # noqa: F401
from src.data.check_data_leakage import run_leakage_check
from src.features.build_features import build_feature_manifest


if __name__ == "__main__":
    build_feature_manifest()
    run_leakage_check()
