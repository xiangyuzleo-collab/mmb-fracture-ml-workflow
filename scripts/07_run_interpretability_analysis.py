from __future__ import annotations

import _bootstrap  # noqa: F401
from src.interpretation.permutation_importance_analysis import run_permutation_importance
from src.interpretation.shap_analysis import run_shap_analysis


if __name__ == "__main__":
    run_permutation_importance()
    run_shap_analysis()
