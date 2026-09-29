from __future__ import annotations

from src.models.evaluate_models import refit_full_models


def train_classical_ml_models():
    return refit_full_models(only_models=["svr", "random_forest", "extra_trees", "gradient_boosting", "xgboost"])


if __name__ == "__main__":
    train_classical_ml_models()
