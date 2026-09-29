from __future__ import annotations

from src.models.evaluate_models import refit_full_models


def train_neural_network():
    return refit_full_models(only_models=["mlp"])


if __name__ == "__main__":
    train_neural_network()
