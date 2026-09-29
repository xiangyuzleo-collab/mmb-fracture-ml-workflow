from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.models.evaluate_models import run_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    args = parser.parse_args()
    run_validation("leave_one_lever_arm_out", only_models=args.models, only_tasks=args.tasks, only_targets=args.targets)
