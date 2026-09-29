from __future__ import annotations

from src.models.evaluate_models import refit_full_models


def train_gpr_model():
    return refit_full_models(only_models=["gaussian_process"])


if __name__ == "__main__":
    train_gpr_model()
