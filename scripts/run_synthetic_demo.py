"""Exercise the original validation code on explicitly artificial specimens."""

from __future__ import annotations

import platform
from importlib.metadata import version

import _bootstrap  # noqa: F401
import pandas as pd

from src.data.synthetic_checks import validate_synthetic_specimens
from src.models.evaluate_models import run_validation
from src.utils.io_utils import read_yaml, sha256_file, write_json
from src.utils.paths import config_dir
from src.utils.paths import project_root
from src.validation.error_analysis import summarize_condition_errors


def main() -> None:
    root = project_root()
    data = root / "examples" / "synthetic_specimens.csv"
    output = root / "outputs" / "synthetic_demo"
    specimens = pd.read_csv(data)
    validate_synthetic_specimens(specimens)
    common = {
        "only_models": ["mean_predictor", "ridge_regression", "svr"],
        "only_tasks": ["task_A_pre_test"],
        "only_targets": ["Gc_raw"],
        "save_fold_models": False,
        "dataset_path": data,
        "output_root": output,
    }
    recorded_outputs = []
    for strategy, extra in [
        ("repeated_random_split", {"n_repeats": 3}),
        ("leave_one_lever_arm_out", {}),
    ]:
        paths = run_validation(strategy, **common, **extra)
        metrics = pd.read_csv(paths["metrics"])
        if metrics.empty:
            raise RuntimeError(f"No metrics produced for {strategy}")
        predictions = pd.read_csv(paths["predictions"])
        diagnostics_path = output / "diagnostics" / f"{strategy}_condition_errors.csv"
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        summarize_condition_errors(predictions).to_csv(diagnostics_path, index=False)
        recorded_outputs.append(
            {
                "strategy": strategy,
                "metrics": paths["metrics"].relative_to(root).as_posix(),
                "predictions": paths["predictions"].relative_to(root).as_posix(),
                "split_summary": paths["split_summary"].relative_to(root).as_posix(),
                "condition_errors": diagnostics_path.relative_to(root).as_posix(),
                "metric_rows": len(metrics),
            }
        )
        print(f"\n{strategy} (synthetic data; metrics have no scientific meaning)")
        print(metrics.groupby("model")[["RMSE", "MAE"]].mean().round(3).to_string())
        print(f"Saved: {paths['metrics'].relative_to(root)}")
    write_json(
        {
            "dataset": "examples/synthetic_specimens.csv",
            "dataset_sha256": sha256_file(data),
            "specimen_count": len(specimens),
            "sample_type": "synthetic_demo",
            "random_seed": int(read_yaml(config_dir() / "config.yaml")["project"]["random_seed"]),
            "models": common["only_models"],
            "task": common["only_tasks"][0],
            "target": common["only_targets"][0],
            "python_version": platform.python_version(),
            "package_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
            "outputs": recorded_outputs,
            "note": "Artificial example; metrics have no scientific meaning.",
        },
        output / "run_manifest.json",
    )


if __name__ == "__main__":
    main()
