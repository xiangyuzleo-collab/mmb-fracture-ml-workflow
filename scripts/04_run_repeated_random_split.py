from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.models.evaluate_models import run_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    args = parser.parse_args()
    run_validation("repeated_random_split", args.n_repeats, args.models, args.tasks, args.targets)
