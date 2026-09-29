from __future__ import annotations

from src.models.evaluate_models import refit_full_models


def train_baseline_models():
    return refit_full_models(only_models=["mean_predictor", "linear_regression", "polynomial_ridge", "ridge_regression", "lasso_regression"])


if __name__ == "__main__":
    train_baseline_models()
