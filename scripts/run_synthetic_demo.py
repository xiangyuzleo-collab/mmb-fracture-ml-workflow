"""Exercise the original validation code on explicitly artificial specimens."""

from __future__ import annotations

import _bootstrap  # noqa: F401
import pandas as pd

from src.models.evaluate_models import run_validation
from src.utils.paths import project_root


def main() -> None:
    root = project_root()
    data = root / "examples" / "synthetic_specimens.csv"
    output = root / "outputs" / "synthetic_demo"
    common = {
        "only_models": ["mean_predictor", "ridge_regression", "svr"],
        "only_tasks": ["task_A_pre_test"],
        "only_targets": ["Gc_raw"],
        "save_fold_models": False,
        "dataset_path": data,
        "output_root": output,
    }
    for strategy, extra in [
        ("repeated_random_split", {"n_repeats": 3}),
        ("leave_one_lever_arm_out", {}),
    ]:
        paths = run_validation(strategy, **common, **extra)
        metrics = pd.read_csv(paths["metrics"])
        if metrics.empty:
            raise RuntimeError(f"No metrics produced for {strategy}")
        print(f"\n{strategy} (synthetic data; metrics have no scientific meaning)")
        print(metrics.groupby("model")[["RMSE", "MAE"]].mean().round(3).to_string())
        print(f"Saved: {paths['metrics'].relative_to(root)}")


if __name__ == "__main__":
    main()
