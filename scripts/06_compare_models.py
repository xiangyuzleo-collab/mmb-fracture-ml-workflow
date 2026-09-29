from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.models.evaluate_models import refit_full_models
from src.models.model_selection import build_model_ranking
from src.visualization.plotting import model_comparison_figures, prediction_and_residual_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    args = parser.parse_args()
    refit_full_models(args.models, args.tasks, args.targets)
    build_model_ranking()
    model_comparison_figures()
    prediction_and_residual_figures()
